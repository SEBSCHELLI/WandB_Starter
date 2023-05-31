import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import torch.optim as optim
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


for nw in range(20):
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    # dataset1 = Subset(dataset1, list(range(10000)))
    train_loader = torch.utils.data.DataLoader(dataset1,
                                                   batch_size=1,
                                                   pin_memory=True,
                                                num_workers=nw)

    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        print(data)
        break
    break
    print(f'{nw}: {(time.time() - start_time)}')
