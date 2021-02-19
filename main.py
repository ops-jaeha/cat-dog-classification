<<<<<<< HEAD
# Import Library
import os
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Train & Validation Loader
class CustomData(Dataset):
    def __init__(self, file_list, dir, transform = None):
        self.file_list = file_list
        self.dir = dir
        self.transform = transform
        if 'dog' in self.file_list[0]:
            self.label = 1
        else:
            self.label = 0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.file_list[idx]))
        if self.transform:
            img = self.transform(img)
        img = img.numpy()
        return img.astype('float32'), self.label

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ColorJitter(),
    transforms.RandomCrop(224),
    transforms.Resize(128),
    transforms.ToTensor()])
=======
>>>>>>> 95dfcb38c4bb0dd70426dd7fbac3af1666872263
