import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import torch.optim as optim
from torchvision import datasets, transforms
import wandb


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

        output = model(data.cuda())

        loss = F.nll_loss(output, target.cuda())
        loss.backward()
        optimizer.step()

        perc_processed = batch_idx / len(train_loader)
        current_epoch = epoch + perc_processed

        if batch_idx % 5 == 0:
            print(f'Epoch: {current_epoch:.4f}\tLoss: {loss.item():.6f}')

            """"""""""""""""""""""""
            " Log training metrics "
            train_metrics = {
                "epoch": current_epoch,
                "train_loss": loss.item()
            }
            wandb.log(train_metrics)
            """"""""""""""""""""""""


def test(model, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    first_misclassified_img = None
    with torch.no_grad():
        for data, target in test_loader:
            target = target.cuda()
            output = model(data.cuda())
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if first_misclassified_img is None:
                is_correct = pred.eq(target.view_as(pred))
                misclassified_imgs = data[is_correct == False]
                misclassified_labels = target[(is_correct == False).flatten()]
                misclassified_preds = pred[(is_correct == False).flatten()]
                first_misclassified_img = misclassified_imgs[0]
                first_misclassified_label = misclassified_labels[0]
                first_missclassifier_pred = misclassified_preds[0].item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.0f}%\n')

    """"""""""""""""""""""""
    " Log test metrics     "
    test_metrics = {
        "epoch": epoch,
        "test_loss": test_loss,
        "test_accuracy": accuracy
    }
    wandb.log(test_metrics)
    """"""""""""""""""""""""

    if first_misclassified_img is not None:
        image = wandb.Image(
            first_misclassified_img,
            caption=f"First misclassified image with GT={first_misclassified_label} and pred={first_missclassifier_pred}"
        )
        wandb.log({"epoch": epoch,
                   "First misclassified example": image})


# define parameters
config = {
    "lr": 5e-3,
    "epochs": 2,
    "train_batch_size": 64,
    "test_batch_size": 512,
    "model_dropout": 0.2
}


def main():
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    " Create a new Weights&Biases (W&B) run within the project KTS_WB_Test "
    wandb.init(project="KTS_OS_WB_MNIST", config=config)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    " (optional) store the parameters in the run config. The run config will be available on W&B   "
    lr = wandb.config['lr']  # or wandb.config.lr = lr
    epochs = wandb.config['epochs']
    train_batch_size = wandb.config['train_batch_size']
    test_batch_size = wandb.config['test_batch_size']
    model_dropout = wandb.config['model_dropout']
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # load data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset1 = Subset(dataset1, list(range(1000)))
    train_loader = torch.utils.data.DataLoader(dataset1,
                                               batch_size=train_batch_size,
                                               pin_memory=True,
                                               num_workers=10)

    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    #dataset2 = Subset(dataset2, list(range(5000)))
    test_loader = torch.utils.data.DataLoader(dataset2,
                                              batch_size=test_batch_size,
                                              pin_memory=True,
                                              num_workers=10)

    # load model
    model = Net(dropout=model_dropout).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    " (optional) watch the model and store topology and gradients in W&B   "
    wandb.watch(model, log_freq=10, log_graph=True)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # training and evaluation
    for epoch in range(epochs):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader, epoch+1)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    " (mandatory) let W&B know that the model training is finished   "
    wandb.finish()
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

if __name__ == '__main__':
    main()
