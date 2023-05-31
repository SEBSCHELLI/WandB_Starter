import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self, dropout):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(9216, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        perc_processed = batch_idx / len(train_loader)
        current_epoch = epoch + perc_processed

        if batch_idx % 5 == 0:
            print(f'Epoch: {current_epoch:.4f}\tLoss: {loss.item():.6f}')


def test(model, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.0f}%\n')


# define parameters
config = {
    "lr": 5e-3,
    "epochs": 10,
    "train_batch_size": 32,
    "test_batch_size": 500,
    "model_dropout": 0.2
}


def main():
    lr = config['lr']
    epochs = config['epochs']
    train_batch_size = config['train_batch_size']
    test_batch_size = config['test_batch_size']
    model_dropout = config['model_dropout']

    # load data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset1 = Subset(dataset1, list(range(5000)))
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=train_batch_size)

    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    dataset2 = Subset(dataset2, list(range(5000)))
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=test_batch_size)

    # load model
    model = Net(dropout=model_dropout)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # training and evaluation
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader, epoch)


if __name__ == '__main__':
    main()
