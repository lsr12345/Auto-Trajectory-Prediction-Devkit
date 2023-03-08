'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: PPT common functions

example:

'''

import os
from functools import partial

import torch
import torch.distributed as dist

from loguru import logger


def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def reduce_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def remove_file(file_dir, key_words=''):
    assert key_words != ''
    for fn in os.listdir(file_dir):
        if key_words in fn:
            os.remove(os.path.join(file_dir, fn))
            return True
    else:
        return False

def prepare_device(local_rank, local_world_size, distributed=False):
    if distributed:
        ngpu_per_process = torch.cuda.device_count() // local_world_size
        device_ids = list(range(local_rank * ngpu_per_process, (local_rank + 1) * ngpu_per_process))

        if torch.cuda.is_available() and local_rank != -1:
            torch.cuda.set_device(device_ids[0])
            device = 'cuda'
        else:
            device = 'cpu'
        device = torch.device(device)
        return device, device_ids
    else:
        n_gpu = torch.cuda.device_count()
        n_gpu_use = local_world_size
        if n_gpu_use > 0 and n_gpu == 0:
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            n_gpu_use = n_gpu

        list_ids = list(range(n_gpu_use))
        if n_gpu_use > 0:
            torch.cuda.set_device(list_ids[0])
            device = 'cuda'
        else:
            device = 'cpu'
        device = torch.device(device)
        return device, list_ids

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))



