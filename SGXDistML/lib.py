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


def distributed_sgx_model(model):

  return torch.nn.parallel.DistributedDataParallel(model)


def distributed_sgx_dataloader(dataset):

  return DataLoader(dataset, sampler=DistributedSampler(dataset))


def verify_args(model, dataloader, optimizer):

  if model == -1:
    raise TypeError("distributed_sgx_run missing model kwarg")

  if dataloader == -1:
    raise TypeError("distributed_sgx_run missing dataloader kwarg")

  if optimizer == -1:
    raise TypeError("distributed_sgx_run missing optimizer kwarg")


def distributed_sgx_run(model=-1, dataloader=-1, optimizer=-1, num_nodes=2, num_processes=1, address='localhost', port='1234', cuda=False):

  verify_args(model, dataloader, optimizer)

  init(num_nodes, num_processes, address, port, cuda)

  def inner(func):
    
    def wrapper(*args, **kwargs):
        
      args = list(args)
      
      args[model] = distributed_sgx_model(args[model])

      args[dataloader] = distributed_sgx_dataloader(args[dataloader])

      args[optimizer].params = args[model].parameters()

      return func(*args, **kwargs)
    
    return wrapper
            
  return inner
