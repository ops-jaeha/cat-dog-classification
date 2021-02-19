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


# Parameter
train_dir = '/data/training_set/training_set/'
test_dir = '/data/test_set/test_set/'
mission_dir = '/data/Mission/'

train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)
mission_files = os.listdir(mission_dir)

cat_files = [tf for tf in train_files if 'cat' in tf]
dog_files = [tf for tf in train_files if 'dog' in tf]
cats = CustomData(cat_files, train_dir, transforms = data_transform)
dogs = CustomData(dog_files, train_dir, transforms = data_transform)
catdogs = ConcatDataset([cats, dogs])
trainloader = DataLoader(catdogs, batch_size=32, shuffle=True, num_workers=0)

# https://www.secmem.org/blog/2020/03/19/Image-Classification/
# criterion 부터