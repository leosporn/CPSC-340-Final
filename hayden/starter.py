from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd

import pickle

train_imgs = pickle.load(open("train_images_512.pk",'rb'), encoding='bytes')
train_labels = pickle.load(open("train_labels_512.pk",'rb'), encoding='bytes')
test_imgs = pickle.load(open("test_images_512.pk",'rb'), encoding='bytes')


class CovidDatasetTrain(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

class CovidDatasetTest(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return self.imgs[idx]  

def make_data_loaders():
    train_dataset = CovidDatasetTrain(train_imgs, train_labels)
    test_dataset = CovidDatasetTest(test_imgs)

    return {
        "train": DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1),
        "test": DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1),
    }

data_loaders = make_data_loaders()
dataset_sizes = {'train': len(data_loaders['train'].dataset), 
                 'test':len(data_loaders['test'].dataset)}

class_names = ['covid', 'background']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Training loop starter
num_epochs = 1 # Set this yourself

for epoch in range(num_epochs):
    for sample in data_loaders["train"]:
        pass
    # Image shape
    # Batch size x Channels x Width x Height
    print(sample[0].shape)
    # Labels shape
    # Batch size
    print(sample[1].shape)