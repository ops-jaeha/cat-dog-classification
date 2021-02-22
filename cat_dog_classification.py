# Import Library
import numpy as np
import pandas as pd
import os
import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torch.optim as optim
from PIL import Image


# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Your Device is {}'.format(device))
print('\n')


# Train Test Split
from sklearn.model_selection import train_test_split
dataset = ImageFolder("./data/training_set")
train_data, test_data, train_label, test_label = train_test_split(dataset.imgs, dataset.targets, test_size=0.2, random_state=42)

## ImageLoader Class
class ImageLoader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = self.checkChannel(dataset)  # some images are CMYK, Grayscale, check only RGB
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image = Image.open(self.dataset[item][0])
        classCategory = self.dataset[item][1]
        if self.transform:
            image = self.transform(image)
        return image, classCategory

    def checkChannel(self, dataset):
        datasetRGB = []
        for index in range(len(dataset)):
            if (Image.open(dataset[index][0]).getbands() == ("R", "G", "B")):  # Check Channels
                datasetRGB.append(dataset[index])
        return datasetRGB


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
]) # train transform

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
]) # test transform

"""
mission_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
"""

train_dataset = ImageLoader(train_data, train_transform)
test_dataset = ImageLoader(test_data, test_transform)
# mission_dataset = ImageLoader(mission_data, mission_transform)


# Data Loader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
# mission_loader = DataLoader(mission_dataset, batch_size=64, shuffle=True)

from tqdm import tqdm
from torchvision import models
model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train and test
def train(num_epoch, model):
    for epoch in range(0, num_epoch):
        #         current_loss = 0.0
        #         current_corrects = 0
        losses = []
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader))  # create a progress bar
        for batch_idx, (data, targets) in loop:
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = model(data)

            loss = criterion(scores, targets)
            optimizer.zero_grad()
            losses.append(loss)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(scores, 1)
            #             current_loss += loss.item() * data.size(0)
            #             current_corrects += (preds == targets).sum().item()
            #             accuracy = int(current_corrects / len(train_loader.dataset) * 100)
            loop.set_description(f"Epoch {epoch + 1}/{num_epoch} process: {int((batch_idx / len(train_loader)) * 100)}")
            loop.set_postfix(loss=loss.data.item())

        # save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'checpoint_epoch_' + str(epoch) + '.pt')


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            _, predictions = torch.max(output, 1)
            correct += (predictions == y).sum().item()
            test_loss = criterion(output, y)

    test_loss /= len(test_loader.dataset)
    print("Average Loss: ", test_loss, "  Accuracy: ", correct, " / ",
          len(test_loader.dataset), "  ", int(correct / len(test_loader.dataset) * 100), "%")

"""
def mission():
    model.eval()
    mission_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in mission_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            _, predictions = torch.max(output, 1)
            correct += (predictions == y).sum().item()
            test_loss = criterion(output, y)

    test_loss /= len(mission_loader.dataset)
    print("Test Average Loss : ", test_loss, "Test Accuracy : ", correct, " / ",
          len(mission_loader.dataset), "  ", int(correct / len(mission_loader.dataset) * 100), "%")
"""

if __name__ == "__main__":
    train(5, model)
    test()
#   Mission()           # Mission

print("----> Loading checkpoint")
checkpoint = torch.load("./checpoint_epoch_4.pt") # Try to load last checkpoint
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


# Check the test set
## Test
dataset = ImageFolder('./data/test_set/',
                      transform=transforms.Compose([
                          transforms.Resize((224, 224)),
                          transforms.ToTensor(),
                          transforms.Normalize([0.5]*3, [0.5]*3)
                      ]))
print(dataset)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

"""
## Mission
dataset = ImageFolder('./data/Mission/',
                      transform=transforms.Compose([
                          transforms.Resize((224, 224)),
                          transforms.ToTensor(),
                          transforms.Normalize([0.5]*3, [0.5]*3)
                      ]))
print(dataset)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
"""

# for j, (data, labels) in enumerate(dataloader):
with torch.no_grad():
    model.eval()
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)
        print(f"predicted ----> {predicted[0]}")


# Create a function to predict random cats and dog images
def RandomImagePrediction(filepath):
    img_array = Image.open(filepath).convert("RGB")
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = data_transforms(img_array).unsqueeze(dim=0)
    load = DataLoader(img)

    for x in load:
        x = x.to(device)
        pred = model(x)
        _, preds = torch.max(pred, 1)
        print(f"class : {preds}")
        if pred[0] == 1:
            print(f"predicted ----> Dog")
        else:
            print(f"predicted ----> Cat")


# Input
# 직접 이미지 넣기
if __name__ == "main":
    RandomImagePrediction("./data/Mission/dogs/Mission_dog_1.png")
    RandomImagePrediction("./data/Mission/dogs/Mission_dog_2.png")
    RandomImagePrediction("./data/Mission/cats/Mission_cat.png")

# Reference
# 코드 출처
# https://www.kaggle.com/adinishad/pytorch-cats-and-dogs-classification