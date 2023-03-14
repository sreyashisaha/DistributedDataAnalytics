from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
import time
from zmq import device


class Mnist(nn.Module):
  def __init__(self):
    super(Mnist, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding = 1)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride = 2, padding = 1)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1, padding = 0)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride = 2)
    self.pool3 = nn.MaxPool2d(kernel_size=5, stride = 1)
    self.fc1 = nn.Linear(in_features=64, out_features=32)
    self.fc2 = nn.Linear(in_features=32, out_features=10)
    self.dp1 = nn.Dropout(0.5)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool1(x)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)
    x = self.dp1(x)
    x = self.pool3(x)
    x = x.reshape(x.shape[0], -1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


def get_accuracy(model, data_loader, device):
    correct_pred = 0
    n = 0

    with torch.no_grad():
        for X, y_true in data_loader:
            X = X.to(device)
            y_true = y_true.to(device)

            y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n

def test_model(model, data_loader, device, criterion):
    model.eval()
    running_loss = 0

    for X, y_true in data_loader:
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

    Test_Loss = running_loss / len(data_loader.dataset)
    Test_acc = get_accuracy(model, data_loader, device)

    print(f'Test Loss: {Test_Loss},  Test Accuracy: {Test_acc} ')


def train_model(model, data_loader, rank, opt,device):
    # defining the optimizer
    # defining loss function
    loss_fn = CrossEntropyLoss()
    epochs = 15
    for epoch in range(epochs):
        model.train()
        for idx, (train_x, train_label) in enumerate(data_loader):
            opt.zero_grad()
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            loss.backward()
            opt.step()
        Train_acc = get_accuracy(model, data_loader, device) * 100
        print(f'Epoch:{epoch} Training Loss: {loss}, Training Accuracy: {Train_acc} in rank:{rank}')

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = torch.nn.CrossEntropyLoss()
    model = Mnist()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    start_time = time.time()
    num_processes = 5
    model.share_memory()
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data/MNIST', train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST('./data/MNIST', train=False,
                                  transform=transform)

    processes = []
    for rank in range(num_processes):
        train_loader = DataLoader(dataset=train_dataset,
                                  sampler=DistributedSampler(dataset=train_dataset, num_replicas=num_processes,
                                                             rank=rank), batch_size=256)
        p = mp.Process(target=train_model, args=(model, train_loader, rank, optimizer,device))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    test_model(model, DataLoader(dataset=test_dataset, batch_size=256), device, criterion)
    end_time = time.time()
    Execution_time = end_time - start_time
    print(f'Total Execution Time: {Execution_time}')