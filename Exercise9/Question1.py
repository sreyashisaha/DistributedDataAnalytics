from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import random_split

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

def train(model, criterion, optimizer, dataloader, dataset_size, training_loss, train_accuracy):
    running_loss = 0.0
    no_of_correct_predictions = no_of_samples = 0
      
 
    for data in dataloader:
        input_x, labels_y  = data
        batch_size = input_x.shape[0]
        
        input_x = input_x.to(device)
        labels_y = labels_y.to(device)
 
        optimizer.zero_grad()
        outputs_y_hat = model(input_x)
        loss = criterion(outputs_y_hat, labels_y)
 
        loss.backward()
        optimizer.step()
                      
        running_loss += loss.data * batch_size
 
        _, preds = outputs_y_hat.max(1)
        no_of_correct_predictions += (preds == labels_y).sum()
        no_of_samples += preds.size(0)

    training_loss = (running_loss / dataset_size)

    train_accuracy = (no_of_correct_predictions.item()/no_of_samples)*100

    for name, param in model.named_parameters():
        model_parameters[name] = param

    return training_loss, train_accuracy


def test(model, criterion, epoch, dataloader, dataset_size):
    valid_loss = 0.0
    no_of_correct_predictions = no_of_samples = 0

    model.eval()

    with torch.no_grad():
        for data in dataloader:
            input_x, labels_y  = data
            batch_size = input_x.shape[0]
 
            input_x = input_x.to(device)
            labels_y = labels_y.to(device)
            outputs_y_hat = model(input_x)
 
            loss = criterion(outputs_y_hat, labels_y)
            
            valid_loss += loss.data * batch_size

            _, preds = outputs_y_hat.max(1)
            no_of_correct_predictions += (preds == labels_y).sum()
            no_of_samples += preds.size(0)

 
    test_loss = (valid_loss / dataset_size)
    print("Test loss calculated after updating the weights of the model :", test_loss.item())

    test_accuracy = (no_of_correct_predictions.item()/no_of_samples)*100
    print("Test Accuracy calculated after updating the weights of the model :", test_accuracy)
    return

def gradients_average():
    for key in model_parameters.keys():
        if model_parameters[key] is not None:
            new_param = comm.reduce(model_parameters[key], op=MPI.SUM, root=0)
            if new_param is not None:
                global_gradients[key] = (new_param/size)

def new_updated_model():
    for name, param in model.named_parameters():
        param.data = global_gradients[name]


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

split_data = None
partitioned_dataset = None
training_loss = 0
train_accuracy = 0

model_parameters = global_gradients = {}

train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)

if rank == 0:
    # train_dataset = datasets.Mnist(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
    # test_dataset = datasets.Mnist(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
    split_data = random_split(train_dataset, [int(len(train_dataset)/size) for i in range(size)])
else:
    split_data = None

partitioned_dataset = comm.scatter(split_data, root=0)
print(f'rank:{rank} got this size of training Data:{len(partitioned_dataset)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.01
num_epochs = 15

model = Mnist().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train_loader = DataLoader(dataset=partitioned_dataset, batch_size=50, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=50, shuffle=True)

start_time = MPI.Wtime()
for epoch in range(0, num_epochs):
    
    training_loss, train_accuracy = train(model, criterion, optimizer, train_loader, len(partitioned_dataset), training_loss, train_accuracy)
    comm.Barrier()
    
    total_loss = comm.reduce(training_loss, op=MPI.SUM, root=0)
    if total_loss is not None:
        if epoch != num_epochs:
            print("Training Loss after ", epoch, " epoch is ", (total_loss/size).item()) 
            
        # else: 
        #     print("The Final Training Loss is ", (total_loss/size).item())

    total_accuracy = comm.reduce(train_accuracy, op=MPI.SUM, root=0)
    if total_accuracy is not None:
        if epoch != num_epochs:
            print("Train Accuracy after ", epoch, " epoch is ", total_accuracy/size)  
        # else:
        #     print("The Final Training Accuracy is :", total_accuracy/size)
    
    
    gradients_average()
    global_gradients = comm.bcast(global_gradients, root=0)
    # print("The new weights obtained after training the model is ", global_gradients)
    new_updated_model()

end_time = MPI.Wtime()
Net_time = end_time - start_time
all_processes_time = comm.gather(Net_time, root=0)

if rank == 0:
    test(model, criterion, epoch, test_loader, len(test_dataset))
    times = np.vstack(all_processes_time)
    time_sum = np.sum(times)
    print('Execution Time for processes is %.3f' % time_sum)