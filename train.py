#!/usr/bin/env python

import os
import subprocess

NUM_GPUS = 4

def get_training_schedule():
  lr = 1.0
  return [
    { 'ep':0, 'sz':224, 'bs':128 },
    { 'ep':(0, 7), 'lr':(lr, lr*2) },
    { 'ep':(7, 18), 'lr':(lr*2, lr/4) },
    { 'ep':(18, 27), 'lr':(lr/4, lr/10) },
    { 'ep':(27, 30), 'lr':(lr/10, lr/100) },
    { 'ep':(30, 33), 'lr':(lr/100,lr/1000) },
  ]

def main():
  runs = os.listdir('/data/rohan/experiments/')
  run_nums = [run[3:] for run in runs if run[:3] == 'run']
  max_run = max(map(int, run_nums))
  logdir = f'/data/rohan/experiments/run{max_run+1}'
  os.mkdir(logdir)

  training_params = [
    '/data/datasets/imagenet',
    '--fp16',
    '--logdir', logdir,
    '--distributed',
    '--init-bn0',
    '--no-bn-wd',
    '--phases', get_training_schedule(),
  ]
  format_params = lambda arg: '\"' + str(arg) + '\"' if isinstance(arg, list) else str(arg)
  training_params = ' '.join(map(format_params, training_params))

  cmd = f'CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node={NUM_GPUS} training/train_imagenet_nv.py {training_params}'
  subprocess.run(f'echo {cmd} > {logdir}/task.cmd', shell=True, check=True)
  subprocess.run(cmd, shell=True, check=True)

if __name__ == '__main__':
  main()