"""run.py:"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


""" All-Reduce example."""
def run(rank, size):
    """ Simple collective communication. """

    group = dist.new_group([0, 1])
    tensor0 = torch.zeros(1)
    tensor1 = torch.ones(1)

    if rank == 0:
        dist.scatter(tensor0, scatter_list=[tensor0, tensor1], group=group)

    if rank != 0:
        dist.scatter(tensor0, group=group)
        print('Rank ', rank, ' has data ', tensor0[0])


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
