import os
import shutil
import argparse

import torch
import torch.nn as nn

from net import AlexNetPlusLatent

from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.utils.data as data
from PIL import Image


transform_train = transforms.Compose(
    [transforms.Resize(256),
     transforms.RandomCrop(227),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose(
    [transforms.Resize(256),
     transforms.Resize(227),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

class MyDataset(data.Dataset):
    def __init__(self, file, dir_path, transform=None):
        imgs = []
        fw = open(file, 'r')
        lines = fw.readlines()
        for line in lines:
            words = line.strip().strip('\n').split('\t')
            imgs.append((words[0], words[1]))
        self.imgs = imgs
        self.dir_path = dir_path
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.imgs[index]
        path = self.dir_path + label + '/' + path + '.jpg'
        #path = os.path.join(self.dir_path, path)
        img = Image.open(path).convert('RGB')
        label = int(label) - 1
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

class MyTestDataset(data.Dataset):
    def __init__(self, file, dir_path, transform=None):
        imgs = []
        fw = open(file, 'r')
        lines = fw.readlines()
        for line in lines:
            words = line.strip().strip('\n').split('\t')
            imgs.append((words[0], words[1]))
        self.imgs = imgs
        self.dir_path = dir_path
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.imgs[index]
        path = self.dir_path + path
        #path = os.path.join(self.dir_path, path)
        img = Image.open(path).convert('RGB')
        label = int(label) - 1
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


def load_data():
    train_data = MyDataset('../../data/train_list_mini.txt', '../../data/train_data_mini/', transform_train)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=5,
                                              shuffle=True, num_workers=1)

    test_data = MyTestDataset('../../data/test_list_mini.txt', '../../data/test_data_mini/', transform_train)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=5,
                                             shuffle=True, num_workers=1)

    return trainloader, testloader