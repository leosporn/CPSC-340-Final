# 1. Import relevant libraries
# 2. Define constants
# 3. Define functions
# 4. Load and preprocess data
# 5. Build CNN
# 6. Fit the CNN

# ==================== 1. Import relevant libraries ====================

# Libraries to open data
import os
import pickle

# numpy and matplotlib.pyplot to visualize data
import matplotlib.pyplot as plt
import numpy as np
# Torch and PIL libraries for image preprocessing
import torch
import torchvision
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# Keras libraries for CNN
from keras.models import Sequential

# library for ensembling (mode)
from scipy import stats

# ============================================================

# ==================== 2. Define constants ====================

# Verbose levels
VERBOSE = {
    'progress': True,  # Prints progress along the way
    'augmenting': False,  # Notify whenever an image transformation goes wrong
    'verify': False,  # Shows an image of representative transformed images
}

# Pathnames and filenames
DIR = {
    'data': os.path.join('..', 'data'),
    'images': os.path.join('..', 'images')
}
FILENAMES = {
    'X_train': os.path.join(DIR['data'], 'train_images_512.pk'),
    'y_train': os.path.join(DIR['data'], 'train_labels_512.pk'),
    'X_test': os.path.join(DIR['data'], 'test_images_512.pk'),
}

# Parameters to load raw data
SHIFT, SCALE = 1, 127.5  # data = (data + 1) * 127.5
SIZE = {'raw': 512, 'final': 256}  # size of raw data and data after downsampling

# Relative amount of data used for training vs validation
TRAIN_VALID_RATIO = 2 / 1  # i.e. ratio of training data : validation data = 2:1

# Parameters for data augmentation
N_TRAIN = 1000  # Train using n = 1000
POS_NEG_RATIO = 1 / 1  # positive:negative samples in augmented training set = 1:1
# (may be backwards)

# Parameters for data augmentation
TRANSFORMATIONS = torchvision.transforms.Compose([
    # Horizontal flip with probability 50%
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    # Rotate by up to 15 degrees either way
    # Translate by ± 2% of image size
    # Expand by up to 15%
    torchvision.transforms.RandomAffine(degrees=15, translate=(0.02, 0.02), scale=(1, 1.15)),  # rotation +
    # Adjust contrast and brightness by up to 40% and 30% respectively
    torchvision.transforms.ColorJitter(contrast=0.4, brightness=0.3),
    # Should randomly add black boxes, but doesn't seem to work
    torchvision.transforms.RandomErasing(p=0.3, scale=(0.01, 0.03), ratio=(1 / 3, 3)),
    # Convert back to torch.tensor
    torchvision.transforms.ToTensor(),
])


# ============================================================

# ==================== 3. Define functions ====================

# Function to print progress
def verbose_print(message, print_flag, end='\n'):
    if print_flag:
        print(message, end=end)


# Functions to load data
def load_pk(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f, encoding='bytes')


def load_X(filename):
    X = load_pk(filename)
    assert X.size(2) == X.size(3) == SIZE['raw']
    return SCALE * (X[:, :1, :, :] + SHIFT)


def load_y(filename):
    y = load_pk(filename)
    return y.type(torch.long)  # convert to integer


# Function to downsample data (i.e. from m=512x512 to n=256x256)
def downsample(X, m, n):
    f = m // n
    for i in range(n):
        for j in range(n):
            X[:, :, i, j] = X[:, :, i * f:(i + 1) * f, j * f:(j + 1) * f].mean(-1).mean(-1)
    return X[:, :, :n, :n]


# Function to split data into training and validation sets
def split_train_valid(X, y):
    def split(X_, y_, f_):
        idx, k = torch.randperm(y_.size(0)), round(y_.size(0) * f_)
        return X_[idx[:k]], y_[idx[:k]], X_[idx[k:]], y_[idx[k:]]

    f = TRAIN_VALID_RATIO / (1 + TRAIN_VALID_RATIO)
    X_train_0, y_train_0, X_valid_0, y_valid_0 = split(X[y == 0], y[y == 0], f)
    X_train_1, y_train_1, X_valid_1, y_valid_1 = split(X[y == 1], y[y == 1], f)
    X_train_, y_train_ = torch.cat((X_train_0, X_train_1)), torch.cat((y_train_0, y_train_1))
    X_valid_, y_valid_ = torch.cat((X_valid_0, X_valid_1)), torch.cat((y_valid_0, y_valid_1))
    return X_train_, y_train_, X_valid_, y_valid_

def makeThreeFolds(X,y):
    folds = 3
    n = X.shape[0]
    a = np.arange(n)
    #a = np.arange(n - n%3)#ignore what we can't split evenly??
    def shuffle(X_, y_):
        idx = torch.randperm(y_.size(0))
        return X_[idx[:]], y_[idx[:]]
    X, y = shuffle(X,y)

    Xvalid_1 = X[a[a%folds == 1]]
    yvalid_1 = y[a[a%folds == 1]]
    Xtrain_1 = X[a[a%folds != 1]]
    ytrain_1 = y[a[a%folds != 1]]

    Xvalid_2 = X[a[a%folds == 2]]
    yvalid_2 = y[a[a%folds == 2]]
    Xtrain_2 = X[a[a%folds != 2]]
    ytrain_2 = y[a[a%folds != 2]]

    Xvalid_3 = X[a[a%folds == 0]]
    yvalid_3 = y[a[a%folds == 0]]
    Xtrain_3 = X[a[a%folds != 0]]
    ytrain_3 = y[a[a%folds != 0]]

    return [Xtrain_1, Xtrain_2, Xtrain_3], [ytrain_1, ytrain_2, ytrain_3], [Xvalid_1, Xvalid_2, Xvalid_3], [yvalid_1, yvalid_2, yvalid_3]



# Function to augment training data
def augment(X_in, y_in, transform_fn):
    # Helper to transform a specific image
    # Continues to try for each image until it is successful (takes 1-3 tries)
    def transform(im_):
        while True:
            try:
                return transform_fn(im_)
            except AttributeError:
                verbose_print(f'Something went wrong', VERBOSE['augmenting'])
                continue

    X_out, y_out = torch.zeros(N_TRAIN, 1, SIZE['final'], SIZE['final']), torch.zeros(N_TRAIN)
    f = POS_NEG_RATIO / (1 + POS_NEG_RATIO)
    for i in range(N_TRAIN):
        label = int(torch.rand(1, 1) > f)
        X = X_in[y_in == label]
        choice = int(torch.randint(0, X.size(0), (1, 1)))
        im = torchvision.transforms.functional.to_pil_image(X[choice])
        X_out[i], y_out[i] = transform(im), label
    return X_out, y_out


# Function to visualize transformed training data
def display(X, h=4, w=5):
    s = SIZE['final']
    im = np.empty((1, h * s, w * s))
    idx = np.random.choice(X.size(0), (h, w), replace=False)
    for i in range(h):
        for j in range(w):
            im[:, s * i:s * (i + 1), s * j:s * (j + 1)] = np.array(X[idx[i, j]])
    plt.imshow(im.transpose((1, 2, 0)).repeat(3, 2))
    plt.title('Augmented training data sample')
    plt.axis('off')
    plt.show()


# ============================================================

if __name__ == '__main__':
    # ==================== 4. Load and preprocess data ====================

    # Load raw data
    verbose_print(f'{"Loading data...":30s}', VERBOSE['progress'], end='')
    X_train = load_X(FILENAMES['X_train'])
    y_train = load_y(FILENAMES['y_train'])
    X_test = load_X(FILENAMES['X_test'])
    verbose_print('DONE', VERBOSE['progress'])

    # Downsample data
    verbose_print(f'{"Downsampling data...":30s}', VERBOSE['progress'], end='')
    X_train = downsample(X_train, SIZE['raw'], SIZE['final'])
    X_test = downsample(X_test, SIZE['raw'], SIZE['final'])
    verbose_print('DONE', VERBOSE['progress'])

    # Split training data into training and validation sets
    verbose_print(f'{"Splitting data...":30s}', VERBOSE['progress'], end='')
    #X_train, y_train, X_valid, y_valid = split_train_valid(X_train, y_train)
    X_trains, y_trains, X_valids, y_valids  = makeThreeFolds(X_train,y_train)
    verbose_print('DONE', VERBOSE['progress'])

    totscore = 0
    for i in [2,1,0]:
        X_train = X_trains[i]
        y_train = y_trains[i]
        X_valid = X_valids[i]
        y_valid = y_valids[i]
        # Augment data
        verbose_print(f'{"Augmenting data...":30s}', VERBOSE['progress'], end='')
        X_train, y_train = augment(X_train, y_train, TRANSFORMATIONS)
        verbose_print('DONE', VERBOSE['progress'])

        # Plot a sample of the augmented data for visual inspection
        if VERBOSE['verify']:
            display(X_train)

        # Convert data to numpy arrays and reshape
        X_train = np.array(X_train).reshape((-1, 1, SIZE['final'], SIZE['final'])).transpose((0, 2, 3, 1))
        X_valid = np.array(X_valid).reshape((-1, 1, SIZE['final'], SIZE['final'])).transpose((0, 2, 3, 1))
        X_test = np.array(X_test).reshape((-1, 1, SIZE['final'], SIZE['final'])).transpose((0, 2, 3, 1))
        y_train = np.array(y_train)
        y_valid = np.array(y_valid)

        # ============================================================

        # ==================== 5. Build CNN ====================

        # Initialize model
        model = Sequential()

        # 200 3x3 convolutional filters w/ 2x2 max pool layer
        model.add(Conv2D(200, kernel_size=3, activation='relu', input_shape=(SIZE['final'], SIZE['final'], 1)))
        model.add(MaxPooling2D(2))
        # 50 3x3 convolutional filters w/ 2x2 max pool layer
        model.add(Conv2D(50, kernel_size=3, activation='relu'))
        model.add(MaxPooling2D(2))
        # 30 3x3 convolutional filters w/ 2x2 max pool layer
        model.add(Conv2D(30, kernel_size=3, activation='relu'))
        model.add(MaxPooling2D(2))
        # Dense layer w/ 15 nodes
        model.add(Flatten())
        model.add(Dense(15, activation='relu'))
        # Final activation
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

        # ============================================================

        #========Make a set of models================
        # Initialize models
        #same model diff data?
        #X_train = load_X(FILENAMES['X_train'])
        #y_train = load_y(FILENAMES['y_train'])
        #X_train = downsample(X_train, SIZE['raw'], SIZE['final'])
        #X_train, y_train = augment(X_train, y_train, TRANSFORMATIONS)
        #bootstrap sample X train, maybe remake train data
        #not sure if we want to bootstrap before generating rotations or not
        def ensemblePredict(XtoPred,X_valid,y_valid, numEnsembling = 10):
            preds = np.zeros((XtoPred.shape[0], numEnsembling))
            for i in range(numEnsembling):
                print("making bootstrap of all training data") #we can change this to exclude validation set if we want
                np.random.seed(i)
                boot_idcs = np.random.randint(X_train.shape[0], size=X_train.shape[0])
                bootX = X_train[boot_idcs]
                bootY = y_train[boot_idcs]
                model.fit(bootX, bootY, batch_size=10, epochs=10, validation_data=(X_valid, y_valid))
                print("finished fitting model number " +str(i))
                preds[:,i] = model.predict(XtoPred)[:,0]
            return stats.mode(preds, axis=1)
        #bootstrap samples


        # ==================== 6. Fit the CNN ====================
        preds = ensemblePredict(X_valid, X_valid, y_valid) #first arg will be X_test when we actually run this - I hope having validation data doesn't bias the fitting
        score = np.mean(preds != y_valid) * 100
        print(f'Accuracy on validation set: {100 * score:.2f}%')
        totscore+=score

        #model.fit(X_train, y_train, batch_size=10, epochs=10, validation_data=(X_valid, y_valid))
        #score = model.evaluate(X_valid, y_valid)
        #print(f'Accuracy on validation set: {100 * score[-1]:.2f}%')
        #totscore += 100 * score[-1]

        # ============================================================
    print(f'Accuracy across validation sets: {(totscore/3):.2f}%')