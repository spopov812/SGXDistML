import argparse
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from model import Net

#prase the local_rank argument from command line for the current process
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

#setup the distributed backend for managing the distributed training
torch.distributed.init_process_group('gloo')

#Setup the distributed sampler to split the dataset to each GPU.
transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
dist_sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=dist_sampler)

#set the cuda device to a GPU allocated to current process .
device = torch.device('cpu', args.local_rank)
model = Net().to(device)
model = torch.nn.parallel.DistributedDataParallel(model,  device_ids=None,
                                                          output_device=args.local_rank)

optimizer = optim.Adadelta(model.parameters(), lr=0.001)

torch.save(model.state_dict(), "master_init.pt")

#Start training the model normally.
for inputs, labels in tqdm(dataloader):
  inputs = inputs.to(device)
  labels = labels.to(device)

  preds = model(inputs)
  loss = F.nll_loss(preds, labels)
  loss.backward()
  optimizer.step()

torch.save(model.state_dict(), "master_final.pt")
