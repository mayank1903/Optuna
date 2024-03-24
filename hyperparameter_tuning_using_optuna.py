#general libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

#torch 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#optuna
import optuna
import optuna_dashboard

#building a simple cnn architecture
class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#function to define the training and test data loaders
def get_mnist_loaders(train_batch_size, test_batch_size):

    """Get MNIST data loaders"""
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=train_batch_size, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader

#function to train the model
def train_model(model, train_loader, optimizer, criterion, log_interval, epoch, device):

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data.to(device))

        loss = criterion(output, target.to(device))
        loss.backward()

        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

#function to validate and evaluate the model
def validate_model(model, test_loader, criterion, device, entropy_threshold=0.5):

    #evaluating the model
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)  # Move data and target to device
            output = model(data)
            total_loss += criterion(output, target).item()

            # Model prediction
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()  # No need to move predicted to device

    total_loss /= len(test_loader.dataset)
    test_accuracy = 100. * (correct / len(test_loader.dataset))

    #log the performance after every epoch
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_accuracy

#setting the objective function
def objective(trial):

    #define the configurations
    cfg = {
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'log_interval': 100,
        'seed': 0,
        'save_model': False,
        'n_epochs': 5,
        'train_batch_size': 64,
        'test_batch_size': 512,
        'lr': trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        'momentum': trial.suggest_float("momentum", 0.4, 0.9, step=0.1),
        'criterion': nn.CrossEntropyLoss()
    }

    #load the model, loaders and the optimisers
    torch.manual_seed(cfg['seed'])
    train_loader, test_loader = get_mnist_loaders(cfg['train_batch_size'], cfg['test_batch_size'])
    model = SimpleCNN().to(cfg['device'])
    optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])

    #run the epochs
    for epoch in range(1, cfg['n_epochs']+1):
        train_model(model, train_loader, optimizer, cfg['criterion'], cfg['log_interval'], epoch, cfg['device'])
        test_accuracy = validate_model(model, test_loader, cfg['criterion'], cfg['device'])

    if cfg['save_model']:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    return test_accuracy #returm the metrics you want to optimise in the study

#creating study with optuna
storage = "sqlite:///db.sqlite3" #define the storage
sampler = optuna.samplers.TPESampler() #define the sampler
study = optuna.create_study(study_name="mnist_classification", direction='maximize', storage=storage, sampler=sampler)
study.optimize(objective, n_trials=20) #define the number of trials

print(f"Parameters of the best trial: {study.best_trial}\nValue of the best trial: {study.best_value}")