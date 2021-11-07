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

from SGXDistML.lib import distributed_sgx_run

from pprint import pprint

@distributed_sgx_run(model=2, dataloader=0, optimizer=3)
def train(dataloader, device, model, optimizer):

    #Start training the model normally.
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        preds = model(inputs)
        loss = F.nll_loss(preds, labels)
        loss.backward()
        optimizer.step()

    return True


def main():

    #Setup the distributed sampler to split the dataset to each GPU.
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

    dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    #set the cuda device to a GPU allocated to current process .
    device = torch.device('cpu')
    model = Net().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=0.001)

    train(dataset, device, model, optimizer)

if __name__ == "__main__":
    main()
