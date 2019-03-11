import argparse, os, shutil, time, warnings
from pathlib import Path
import numpy as np
import sys
import math

import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data.sampler import Sampler
import torchvision
import pickle
from tqdm import tqdm
from dist_utils import env_world_size, env_rank

def get_loaders(traindir, valdir, sz, bs, fp16=True, val_bs=None, workers=8, distributed=False):
    val_bs = val_bs or bs
    train_tfms = [
            transforms.RandomResizedCrop(sz),
            transforms.RandomHorizontalFlip()
    ]
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose(train_tfms))
    train_sampler = (DistributedSampler(train_dataset, num_replicas=env_world_size(), rank=env_rank()) if distributed else None)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, collate_fn=fast_collate, 
        sampler=train_sampler)

    val_dataset, val_sampler = create_validation_set(valdir, val_bs, sz, rect_val=rect_val, distributed=distributed)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=workers, pin_memory=True, collate_fn=fast_collate, 
        batch_sampler=val_sampler)

    train_loader = BatchTransformDataLoader(train_loader, fp16=fp16)
    val_loader = BatchTransformDataLoader(val_loader, fp16=fp16)

    return train_loader, val_loader, train_sampler, val_sampler


def create_validation_set(valdir, batch_size, target_size, distributed):    
    val_tfms = [transforms.Resize(int(target_size*1.14)), transforms.CenterCrop(target_size)]
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose(val_tfms))
    val_sampler = DistValSampler(list(range(len(val_dataset))), batch_size=batch_size, distributed=distributed)
    return val_dataset, val_sampler

class BatchTransformDataLoader():
    # Mean normalization on batch level instead of individual
    # https://github.com/NVIDIA/apex/blob/59bf7d139e20fb4fa54b09c6592a2ff862f3ac7f/examples/imagenet/main.py#L222
    def __init__(self, loader, fp16=True):
        self.loader = loader
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.fp16 = fp16
        if self.fp16: self.mean, self.std = self.mean.half(), self.std.half()

    def __len__(self): return len(self.loader)

    def process_tensors(self, input, target, non_blocking=True):
        input = input.cuda(non_blocking=non_blocking)
        if self.fp16: input = input.half()
        else: input = input.float()
        if len(input.shape) < 3: return input, target.cuda(non_blocking=non_blocking)
        return input.sub_(self.mean).div_(self.std), target.cuda(non_blocking=non_blocking)

    def update_batch_size(self, bs):
        self.loader.batch_sampler.batch_size = bs
            
    def __iter__(self):
        return (self.process_tensors(input, target, non_blocking=True) for input,target in self.loader)

def fast_collate(batch):
    if not batch: return torch.tensor([]), torch.tensor([])
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


class DistValSampler(Sampler):
    # DistValSampler distrbutes batches equally (based on batch size) to every gpu (even if there aren't enough images)
    # WARNING: Some baches will contain an empty array to signify there aren't enough images
    # Distributed=False - same validation happens on every single gpu
    def __init__(self, indices, batch_size, distributed=True):
        self.indices = indices
        self.batch_size = batch_size
        if distributed:
            self.world_size = env_world_size()
            self.global_rank = env_rank()
        else: 
            self.global_rank = 0
            self.world_size = 1
            
        # expected number of batches per sample. Need this so each distributed gpu validates on same number of batches.
        # even if there isn't enough data to go around
        self.expected_num_batches = math.ceil(len(self.indices) / self.world_size / self.batch_size)
        
        # num_samples = total images / world_size. This is what we distribute to each gpu
        self.num_samples = self.expected_num_batches * self.batch_size
        
    def __iter__(self):
        offset = self.num_samples * self.global_rank
        sampled_indices = self.indices[offset:offset+self.num_samples]
        for i in range(self.expected_num_batches):
            offset = i*self.batch_size
            yield sampled_indices[offset:offset+self.batch_size]
    def __len__(self): return self.expected_num_batches
    def set_epoch(self, epoch): return
