# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import pickle
import time

import torch
import torch.distributed as dist

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def get_world_size():
    '''
    Return the number of GPU usage
    '''
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()