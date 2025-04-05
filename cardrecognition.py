import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm 

import matplotlib.pyplot as plt #for data visualization
import pandas as pd
import numpy as np
import sys
from tqdm.auto import tqdm

class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes

# STEP 2 - Pytorch Model

class simpleCardClassifer(nn.Module):
    def __init__(self, num_classes = 53):

        super(simpleCardClassifer, self).__init__()

        # where we define all parts of the model
        self.base_model = timm.create_model('efficientnet_b0', pretrained = True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        # make a classifier
        enet_out_size = 1280
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        # connect these parts and return the ouput
        x = self.features(x)
        output = self.classifier(x)
        return output
    
if __name__ == "__main__": # only run this if directly running the file
    # get a dictionary associating target values with folder name
    data_dir = "/Users/rickeychiu/Desktop/Personal Coding/pytorch-stuff/cards_dataset/train"
    target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}

    # making sure it is always 128 x 128, makes it into a tensor
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = PlayingCardDataset(data_dir, transform)

    # 32 at a time makes training it faster
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

    model = simpleCardClassifer(num_classes = 53)

    # STEP 3 - Training Loop

    train_folder = '/Users/rickeychiu/Desktop/Personal Coding/pytorch-stuff/cards_dataset/train/'
    valid_folder = '/Users/rickeychiu/Desktop/Personal Coding/pytorch-stuff/cards_dataset/valid/'
    test_folder = '/Users/rickeychiu/Desktop/Personal Coding/pytorch-stuff/cards_dataset/test/'

    train_dataset = PlayingCardDataset(train_folder, transform = transform)
    val_dataset = PlayingCardDataset(valid_folder, transform = transform)
    test_dataset = PlayingCardDataset(test_folder, transform = transform)

    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)
    test_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)

    # epoch = one run through the training set
    num_epochs = 5
    train_losses, val_losses = [], []

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model.to(device)

    # loss function - what the model will optimize for
    criterion = nn.CrossEntropyLoss()
    # optimizer - adam is the best place to start for most tasks
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(num_epochs):
        # training phase
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc = "Training loop"):

            # move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # validation phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc = 'Validation loop'):
                
                # move inputs and labels to the device
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

            val_loss = running_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
            print(f"Epoch {epoch + 1} / {num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

    # Save model after training
    torch.save(model.state_dict(), "trained_model.pth")

    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.title("Loss over epochs")
    plt.show()
