import os
import pickle
import torch
import torchvision
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Verbose
verbose = True
augment_verbose = False
model_verbose = False
train_verbose = True
# Directories
data_dir = os.path.join('..', 'data')
image_dir = os.path.join('..', 'images')
test_dir = os.path.join('test')
# Filenames
X_train_filename = 'train_images_512.pk'
y_train_filename = 'train_labels_512.pk'
X_test_filename = 'test_images_512.pk'
# Parameters to load raw data
shift, scale, downsample_factor = 1, 127.5, 2
# Parameters for data augmentation
all_transforms = torchvision.transforms.Compose([
    # transforms.ToPILImage(),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    # torchvision.transforms.RandomRotation(degrees=15, expand=False),
    torchvision.transforms.RandomAffine(degrees=15, translate=(0.02, 0.02), scale=(1, 1.15)),
    torchvision.transforms.ColorJitter(contrast=0.4, brightness=0.4),
    torchvision.transforms.RandomErasing(p=0.3, scale=(0.01, 0.03), ratio=(1 / 3, 3)),
    torchvision.transforms.ToTensor(),
])
n, p = 1000, 0.5


def load_pk(filename):
    with open(os.path.join('..', 'data', filename), 'rb') as f:
        return pickle.load(f, encoding='bytes')


def load_X(filename):
    X = load_pk(filename)
    return scale * (X.mean(1) + shift)


def load_y(filename):
    return load_pk(filename)


def downsample(X_in, f):
    n1, n2, n3 = X_in.shape
    X_out = torch.zeros((n1, n2 // f, n3 // f))
    for i1 in range(X_out.shape[1]):
        for i2 in range(X_out.shape[2]):
            X_out[:, i1, i2] = X_in[:, i1 * f:(i1 + 1) * f, i2 * f:(i2 + 1) * f].mean(2).mean(1)
    return X_out


def augment(X_in, y_in, T):
    X_out = torch.zeros((n, X_in.shape[1], X_in.shape[2]))
    y_out = torch.zeros(n)
    for i in range(n):
        label = int(torch.rand(1, 1) > p)
        idx = y_in == label
        X, y = X_in[idx], y_in[idx]
        choice = int(torch.randint(0, y.shape[0] - 1, (1, 1)))
        im = torchvision.transforms.functional.to_pil_image(X[choice])
        while True:
            try:
                im = T(im)
                break
            except AttributeError:
                if augment_verbose:
                    print(f'Something went wrong: {i:3d}')
                continue
        X_out[i] = im
        y_out[i] = label
    return X_out, y_out


def show_train_image(idx=None):
    if idx is None:
        idx = int(torch.randint(0, y_train.shape[0], (1,)))
    im = np.array(X_train[idx][None, :, :]).transpose((1, 2, 0)).repeat(3, 2)
    label = int(y_train[idx])
    print(f'Label: {label:d}\n'
          f'idx:   {idx:d}')
    plt.imshow(im)
    plt.show()


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = Sequential(
            # Layer 1
            Conv2d(in_channels=1, out_channels=10, kernel_size=3),
            BatchNorm2d(10),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Layer 2
            Conv2d(in_channels=10, out_channels=10, kernel_size=3),
            BatchNorm2d(10),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = Sequential(
            Linear(10 * (512 // downsample_factor) ** 2, 1)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


if __name__ == '__main__':
    # Load raw data
    if verbose:
        print(f'{"Loading data...":30s}', end='')
    X_train = load_X(X_train_filename)
    y_train = load_y(y_train_filename)
    X_test = load_X(X_test_filename)
    if verbose:
        print('DONE')
    # Downsampling
    if downsample_factor > 1:
        if verbose:
            print(f'{"Downsampling data...":30s}', end='')
        X_train = downsample(X_train, downsample_factor)
        X_test = downsample(X_test, downsample_factor)
        if verbose:
            print('DONE')
    # Data augmentation
    if n is not None:
        if verbose:
            print(f'{"Augmenting data...":30s}', end='')
        X_train, y_train = augment(X_train, y_train, all_transforms)
        if verbose:
            print('DONE')
    # X_train = X_train[:, None, :, :]
    # X_test = X_test[:, None, :, :]
    # # Building model
    # model = Net()
    # optimizer = Adam(model.parameters(), lr=0.07)
    # criterion = CrossEntropyLoss()
    # if model_verbose:
    #     print('Model:')
    #     print(model)
    # # Training
    # if verbose:
    #     print(f'{"Training model...":30s}', end='')
    # for epoch in range(3):
    #     x, y = Variable(X_train), Variable(y_train)
    #     optimizer.zero_grad()
    #     y_hat = model(X_train)
    #     loss_train = criterion(y_hat, y_train)
    #     loss_train.backward()
    #     optimizer.step()
    #     loss = loss_train.item()
    #     if train_verbose:
    #         print(f'\tEpoch {epoch: 3d}: loss = {loss:.2f}')