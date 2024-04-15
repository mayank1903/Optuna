#general libraries
import torch
from torch import nn
from torchvision import datasets, transforms
import math
import time
import logging
import matplotlib.pyplot as plt
import itertools
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils
import os
import textwrap
import torch.optim as optim

#optuna
import optuna
from optuna.artifacts import FileSystemArtifactStore, upload_artifact

#optuna dashboard
import optuna_dashboard
from optuna_dashboard.artifact import get_artifact_path
from optuna_dashboard.preferential import create_study
from optuna_dashboard import register_preference_feedback_component
from optuna_dashboard.preferential.samplers.gp import PreferentialGPSampler

torch.manual_seed(111)

device = "cuda" if torch.cuda.is_available() else "cpu"

#define the loaders
def get_mnist_loaders(train_batch_size, test_batch_size):
    """Get MNIST data loaders"""
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=train_batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader

#discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output
    
#generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output
    
#train discriminator
def train_discriminator(discriminator, images, real_labels, fake_images, fake_labels, criterion, d_optimizer):
    discriminator.zero_grad()
    outputs = discriminator(images)
    real_loss = criterion(outputs, real_labels.unsqueeze(1))
    real_score = outputs

    outputs = discriminator(fake_images)
    fake_loss = criterion(outputs, fake_labels.unsqueeze(1))
    fake_score = outputs

    d_loss = real_loss + fake_loss
    d_loss.backward()

    d_optimizer.step()
    return d_loss, real_score, fake_score

#train generator
def train_generator(generator, discriminator_outputs, real_labels, criterion, g_optimizer):
    generator.zero_grad()
    g_loss = criterion(discriminator_outputs, real_labels.unsqueeze(1))
    g_loss.backward()

    g_optimizer.step()
    return g_loss

#generate grid of new images
# Plot grid of 9 images from generator after each epoch
def generate_new_images(generator, sample_images, latent_dim, img_dir):
    fixed_noise = torch.randn(sample_images, latent_dim).to(device)  # Sample 15 images
    fake_images = generator(fixed_noise).to(device)

    plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(fake_images, nrow=5, padding=1, normalize=True).cpu().numpy(),
            (1, 2, 0)
        )
    )
    plt.savefig(img_dir)
    plt.show()
    plt.close()

#gan training objective functions
def train_GANs(study: optuna.Study,
               artifact_store: FileSystemArtifactStore):

    trial = study.ask() #start a trial

    print(f"running trial number: {trial.number}")

    latent_dim = 128

    #define the generator and the discriminator
    discriminator = Discriminator().to(device=device)
    generator = Generator().to(device=device)

    cfg = {
        "train_batch_size": trial.suggest_categorical("train_batch_size", [64, 128]),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_epochs": 100,
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    }

    #define the loader
    batch_size = cfg["train_batch_size"]
    train_loader, _ = get_mnist_loaders(batch_size, batch_size)


    #define the optimizers
    lr = cfg['lr']
    optimizer_name = cfg['optimizer']
    d_optimizer = getattr(optim, optimizer_name)(discriminator.parameters(), lr=lr)  # Instantiate optimizer from name
    g_optimizer = getattr(optim, optimizer_name)(generator.parameters(), lr=lr)  # Instantiate optimizer from name

    #define the criterion
    criterion = nn.BCELoss()

    print(f"Batch Size: {batch_size}\nLearning Rate: {lr}\nOptimizer: {optimizer_name}")

    for epoch in range(cfg['num_epochs']):

        print(f"running epoch number: {epoch}")

        for n, (images, _) in tqdm(enumerate(train_loader)):
            images = images.to(device)
            real_labels = torch.ones(images.size(0)).to(device)

            noise = torch.randn(images.size(0), latent_dim).to(device)
            fake_images = generator(noise)
            fake_labels = torch.zeros(images.size(0)).to(device)

            # Train the discriminator
            d_loss, real_score, fake_score = train_discriminator(discriminator, images,
                                                                 real_labels, fake_images, fake_labels,
                                                                  criterion, d_optimizer)

            noise = torch.randn(images.size(0), latent_dim).to(device)
            fake_images = generator(noise)
            outputs = discriminator(fake_images)

            # Train the generator
            g_loss = train_generator(generator, outputs, real_labels, criterion, g_optimizer)

            if (n+1) % len(train_loader) == 0:

                print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, '
                    'D(x): %.2f, D(G(z)): %.2f'
                    % (epoch + 1, cfg['num_epochs'], n + 1, len(train_loader), d_loss.item(), g_loss.item(),
                        real_score.mean().item(), fake_score.mean().item()))

    img_path = f"tmp/generated_image-{trial.number}.png"
    generate_new_images(generator, 30, latent_dim, img_path)

    user_attr_key = "generated-image"
    artifact_id = upload_artifact(trial, img_path, artifact_store)
    register_preference_feedback_component(study, "artifact", user_attr_key)
    trial.set_user_attr(user_attr_key, artifact_id)

#start preferential optimisationo
def start_optimization(artifact_store: FileSystemArtifactStore):
    # 1. Create Study
    storage = "sqlite:///pref_opt.db"
    study = create_study(
        n_generate=4, #generate 4 images before running another trial for feedback
        study_name="Preferential Optimization", #name of the study
        storage=storage, #defining the storage
        sampler=PreferentialGPSampler(), #sampler
        load_if_exists=True,
    )

    while True:
    # If study.should_generate() returns False, the generator waits for human evaluation.
        if not study.should_generate():
            time.sleep(0.1)  # Avoid busy-loop
            continue

        train_GANs(study, artifact_store)

#main function
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

