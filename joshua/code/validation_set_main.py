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
import numpy as np
import pandas as pd
import pickle
from torch.utils.data.sampler import SubsetRandomSampler


plt.ion()   # interactive mode

train_imgs = pickle.load(open("../data/train_images_512.pk",'rb'), encoding='bytes')
train_labels = pickle.load(open("../data/train_labels_512.pk",'rb'), encoding='bytes')
test_imgs = pickle.load(open("../data/test_images_512.pk",'rb'), encoding='bytes')


def generate_loaders(data_imgs, data_labels, random_seed, batch_size, augment, validation_set_ratio = 0.15, shuffle = True, show_sample=False, num_workers=0, pin_memory=False):
    #TODO: this?
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if (augment):
        #TODO
        print("implement me")
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    train_dataset = CovidDatasetTrain(data_imgs, data_labels)
    valid_dataset = CovidDatasetTrain(data_imgs, data_labels)

    train_qty = len(train_dataset)
    indices = list(range(train_qty))
    split = int(np.floor(validation_set_ratio * train_qty))
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)


    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, 
        num_workers=num_workers, pin_memory=pin_memory
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)


class CovidDatasetTrain(Dataset):

    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.imgs[idx]
        label = torch.tensor([self.labels[idx]])
        if self.transform:
            self.transform(sample)
        return sample, label

class CovidDatasetTest(Dataset):

    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return self.imgs[idx] 
    

    
    

train_loader, valid_loader = generate_loaders(train_imgs, train_labels, 18, 8, False)
dataset_sizes = {'t1': len(train_loader.dataset),
                 't1_valset': len(valid_loader.dataset)}

class_names = ['covid', 'background']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 8, 20)
        self.hidden = nn.Linear(20, 30)
        self.hidden2 = nn.Linear(30, 10)
        self.hidden3 = nn.Linear(10, 1)
        self.output = nn.Linear(1,1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.hidden(x)
        x = torch.sigmoid(x)
        x = self.hidden2(x)
        x = torch.sigmoid(x)
        x = self.hidden3(x)
        x = torch.sigmoid(x)
        x = self.output(x)
        return x


model = Net()


import torch.optim as optim

loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay= 1e-4)

for epoch in range(1, 100): ## run the model for 10 epochs
    train_loss, valid_loss = [], []
    ## training part 
    model.train()
    for i, data in (train_loader):
        optimizer.zero_grad()
        ## 1. forward propagation
        output = model(data.mean(0))
        
        ## 2. loss calculation
        loss = loss_function(output, target.float())
        
        ## 3. backward propagation
        loss.backward()
        
        ## 4. weight optimization
        optimizer.step()
        
        train_loss.append(loss.item())
        
    ## evaluation part 
    model.eval()
    for data, target in valid_loader.dataset:
        output = model(data.mean(0).flatten())
        loss = loss_function(output, target.float())
        valid_loss.append(loss.item())
    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))


# Test data set

test_loader = torch.utils.data.DataLoader(
    CovidDatasetTest(test_imgs)
)

classification = model(test_loader.dataset)
