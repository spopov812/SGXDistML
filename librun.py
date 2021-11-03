import torch
import datetime
import os

import argparse
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from model import Net

from lib import *

def run():

    #Start training the model normally.
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        preds = model(inputs)
        loss = F.nll_loss(preds, labels)
        loss.backward()
        optimizer.step()

    return True


#Setup the distributed sampler to split the dataset to each GPU.
transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
dataloader = distributed_sgx_dataloader(dataset)

#set the cuda device to a GPU allocated to current process .
device = torch.device('cpu')
model = distributed_sgx_model(Net()).to(device)
model = torch.nn.parallel.DistributedDataParallel(model)

optimizer = optim.Adadelta(model.parameters(), lr=0.001)

distributed_sgx_run(run)
