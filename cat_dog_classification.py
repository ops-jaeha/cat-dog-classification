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


# Train Test Split
from sklearn.model_selection import train_test_split
datasets = ImageFolder("/data/training_set/training_set/")
train_data, test_data, mission_data, train_label, test_label = train_test_split(datasets.imgs, datasets.targets, test_size=0.2, random_state=42)

## ImageLoader Class
class ImageLoader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = self.checkChannel(dataset)
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
            if (Image.open(dataset[index][0]).getbands() == ("R", "G", "B")):
                datasetRGB.append(dataset[index])
                return datasetRGB

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

mission_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = ImageLoader(train_data, train_transform)
test_dataset = ImageLoader(test_data, test_transform)
mission_dataset = ImageLoader(mission_data, mission_transform)


# Data Loader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
mission_loader = DataLoader(mission_dataset, batch_size=64, shuffle=True)

# https://www.kaggle.com/adinishad/pytorch-cats-and-dogs-classification
# 6번