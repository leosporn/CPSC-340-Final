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
# Parameters for splitting data into training and validation sets
validation_fractions = (0.3, 0.3)
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
n, p = 100, 0.5


# Verbose
def print_verbose(msg, end, v):
    if v:
        print(msg, end=end)


verbose = True
augment_verbose = False
model_verbose = False
train_verbose = True


def load_pk(filename):
    with open(os.path.join('..', 'data', filename), 'rb') as f:
        return pickle.load(f, encoding='bytes')


def load_X(filename):
    X = load_pk(filename)
    return scale * (X.mean(1) + shift)


def load_y(filename):
    return load_pk(filename).type(torch.long)


def downsample(X, f):
    n1, n2, n3 = X.shape
    for i1 in range(n2):
        for i2 in range(n3):
            X[:, i1, i2] = X[:, i1 * f:(i1 + 1) * f, i2 * f:(i2 + 1) * f].mean(-1).mean(-1)
    return X[:, :n2 // f, :n3 // f]


def split(X, y):
    def split_(X_, y_, f):
        idx, k = torch.randperm(len(y_)), round(len(y_) * f)
        return X_[idx[:k]], X_[idx[k:]], y_[idx[:k]], y_[idx[k:]]

    Xv0, Xt0, yv0, yt0 = split_(X[y == 0], y[y == 0], validation_fractions[0])
    Xv1, Xt1, yv1, yt1 = split_(X[y == 1], y[y == 1], validation_fractions[1])
    Xv = torch.cat((Xv0, Xv1))
    Xt = torch.cat((Xt0, Xt1))
    yv = torch.cat((yv0, yv1))
    yt = torch.cat((yt0, yt1))
    return Xt, Xv, yt, yv


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
    print(f'Label: {label:d} (X_train[{idx:d}])')
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
            # Linear(10 * (512 // downsample_factor // 4) ** 2, 1)
            Linear(38440, 1)
        )

    def forward(self, X):
        X = self.cnn_layers(X)
        X = X.reshape(X.size(0), -1)
        X = self.linear_layers(X)
        return X


if __name__ == '__main__':
    # Load raw data
    print_verbose(f'{"Loading data...":30s}', '', verbose)
    X_train = load_X(X_train_filename)
    y_train = load_y(y_train_filename)
    X_test = load_X(X_test_filename)
    print_verbose('DONE', '\n', verbose)
    # Downsampling
    if downsample_factor > 1:
        print_verbose(f'{"Downsampling data...":30s}', '', verbose)
        X_train = downsample(X_train, downsample_factor)
        X_test = downsample(X_test, downsample_factor)
        print_verbose('DONE', '\n', verbose)
    # Split into training/validation fractions
    X_train, X_valid, y_train, y_valid = split(X_train, y_train)
    # Data augmentation
    if n is not None:
        print_verbose(f'{"Augmenting data...":30s}', '', verbose)
        X_train, y_train = augment(X_train, y_train, all_transforms)
        print_verbose('DONE', '\n', verbose)
    X_train = X_train[:, None, :, :]
    X_valid = X_valid[:, None, :, :]
    X_test = X_test[:, None, :, :]
    # Building model
    model = Net()
    optimizer = Adam(model.parameters(), lr=0.07)
    criterion = CrossEntropyLoss()
    if model_verbose:
        print('Model:')
        print(model)
    # Training
    if verbose:
        print(f'{"Training model...":30s}')
    for epoch in range(3):
        optimizer.zero_grad()
        y_hat = model(X_train)
        loss_train = criterion(y_hat, y_train)
        loss_train.backward()
        optimizer.step()
        loss = loss_train.item()
        if train_verbose:
            print(f'\tEpoch {epoch: 3d}: loss = {loss:.2f}')
