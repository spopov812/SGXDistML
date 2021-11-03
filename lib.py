import argparse
import torch
import os
import datetime

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

initialized = False

parser = argparse.ArgumentParser()
parser.add_argument("--node_id", default=0, type=int)
args = parser.parse_args()


def init(num_nodes=2, num_processes=1, address='localhost', port='1234', cuda=False):

  os.environ['MASTER_ADDR'] = address
  os.environ['MASTER_PORT'] = port

  backend = 'nccl' if cuda else 'gloo'

  torch.distributed.init_process_group(backend, init_method="env://", timeout=datetime.timedelta(0, 1800), 
      world_size=num_nodes, rank=args.node_id)

  global initialized 

  initialized = True


def distributed_sgx_run(func, *args):

  return func(*args)


def distributed_sgx_model(model):

  if not initialized:
    print("Model init")
    init()

  return torch.nn.parallel.DistributedDataParallel(model)


def distributed_sgx_dataloader(dataset):

  if not initialized:
    print("DataLoader init")
    init()

  return DataLoader(dataset, sampler=DistributedSampler(dataset))