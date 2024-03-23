#general libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import textwrap
import time

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
from optuna.trial import TrialState
from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import upload_artifact

#optuna dashboard packages
from optuna_dashboard import save_note, register_objective_form_widgets, ChoiceWidget
from optuna_dashboard.artifact import get_artifact_path

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

def get_mnist_loaders(train_batch_size, test_batch_size):

    """Get MNIST data loaders"""
    train_loader = DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=train_batch_size, shuffle=True)
    
    test_loader = DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader

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

        
def validate_model(model, test_loader, criterion, device, img_path, epoch, num_epochs, entropy_threshold=0.5):

    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    #get the confused images
    confused_images = []
    actual_labels = []
    predicted_labels = []

    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):
            output = model(data).to(device)
            total_loss += criterion(output, target.to(device)).item()
            
            #model prediction
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Calculate entropy
            probabilities = torch.softmax(output, dim=1)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
            
            # Find indices of confused images
            confused_mask = entropy > entropy_threshold
            confused_images.extend(data[confused_mask].cpu().numpy())

            # Store actual and predicted labels of confused images
            actual_labels.extend(target[confused_mask].cpu().numpy())
            predicted_labels.extend(predicted[confused_mask].cpu().numpy())

    total_loss /= len(test_loader.dataset)
    test_accuracy = 100. * (correct / len(test_loader.dataset))
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # Plot the top 9 confused images
    if confused_images and epoch==num_epochs:
        plot_top_confused_images(confused_images, actual_labels, predicted_labels, img_path)
        

    return test_accuracy

def plot_top_confused_images(confused_images, actual_labels, predicted_labels, output_path):
    n_confused_images = min(len(confused_images), 6)
    print(f"Plotting top {n_confused_images} confused images:")

    fig, axes = plt.subplots(2, 3, figsize=(5, 5))
    
    for i, ax in enumerate(axes.flat):
        if i < n_confused_images:
            ax.imshow(confused_images[i].squeeze(), cmap='gray')
            ax.set_title(f"Actual: {actual_labels[i]}\nPredicted: {predicted_labels[i]}")
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    plt.show()

def objective(study: optuna.Study, artifact_store: FileSystemArtifactStore):

    #ask the trial number
    trial = study.ask()

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

    torch.manual_seed(cfg['seed'])
    train_loader, test_loader = get_mnist_loaders(cfg['train_batch_size'], cfg['test_batch_size'])
    model = SimpleCNN().to(cfg['device'])
    optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
    for epoch in range(1, cfg['n_epochs']+1):
        train_model(model, train_loader, optimizer, cfg['criterion'], cfg['log_interval'], epoch, cfg['device']) #training the model

        #get the image path
        img_path = f"tmp/confused_lot-{trial.number}.png"
        test_accuracy = validate_model(model, test_loader, cfg['criterion'], cfg['device'], img_path, epoch, cfg['n_epochs'])

        #saving the artifacts to the artifacts directory
        if epoch == cfg['n_epochs']:
            artifacts_id = upload_artifact(trial, img_path, artifact_store)
            artifact_path = get_artifact_path(trial, artifacts_id)

    if cfg['save_model']:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    # 4. Save Note
    note = textwrap.dedent(
        f"""\
    ## Trial {trial.number}

    Grid of images where model is confused among different classes 🤨
    ![generated-image]({artifact_path})

    Test Accuracy of the model: {test_accuracy:.2f}
    """
    )
    save_note(trial, note)

    return test_accuracy

def start_optimization(artifact_store: FileSystemArtifactStore):
    # 1. Create Study
    storage = "sqlite:///db.sqlite3"
    study = optuna.create_study(study_name="HITL_with_optuna_for_digit_classification", 
                                direction='maximize', 
                                storage=storage,
                                load_if_exists=True)

    # 2. Set an objective name
    study.set_metric_names(["Are you satisfied with the model's validation accuracy?"])

    # 3. Register ChoiceWidget
    register_objective_form_widgets(
        study,
        widgets=[
            ChoiceWidget(
                choices=["Yes 👍", "Somewhat 👌", "No 👎"],
                values=[1, 0, -1],
                description="Please input your score!",
            ),
        ],
    )

    # 4. Start Human-in-the-loop Optimization
    n_batch = 2
    while True:
        running_trials = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
        if len(running_trials) >= n_batch:
            time.sleep(1)  # Avoid busy-loop
            continue
        objective(study, artifact_store)

def main():
    tmp_path = os.path.join(os.path.dirname(__file__), "tmp")

    # 1. Create Artifact Store
    artifact_path = os.path.join(os.path.dirname(__file__), "artifact")
    artifact_store = FileSystemArtifactStore(artifact_path)

    print(f"paths : {tmp_path}, {artifact_path}")

    if not os.path.exists(artifact_path):
        os.mkdir(artifact_path)

    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)

    # 2. Run optimize loop
    start_optimization(artifact_store)


if __name__ == "__main__":
    main()