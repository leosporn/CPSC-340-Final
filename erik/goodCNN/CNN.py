import os
import pickle
import torch
import torchvision as vision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import argparse
import glob

# Verbose
verbose = False

# Model Locations, set to "" if not used
load_path = "" # "/Users/Zoidberg/Desktop/CPSC-340-Final/erik/model"
save_path = os.path.join('..', 'models', 'densenet161_4') # "/Users/Zoidberg/Documents/CPSC-340-Final/erik/resnet18model"

# Directories
test_dir = os.path.join('..', 'testimages')
train_dir = os.path.join('..', 'trainimages')
val_dir = os.path.join('..', 'valimages')

# Filenames
TRAIN_IMAGES_FILENAME = 'train_images_512.pk'
TRAIN_LABELS_FILENAME = 'train_labels_512.pk'
TEST_IMAGES_FILENAME = 'test_images_512.pk'

# Transormations
transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model for Transfer Learning using torchvision.models
# https://pytorch.org/docs/stable/torchvision/models.html
model = models.densenet161(pretrained=True)

# Training
epochs = 8


def load_data(adjust=True, save_images=False):
    """
    Load training and test data.
    Optionally re-scale data so it is between 0 and 1 (originally -1 and -0.9921).
    Optionally save images for visualization.

    :param adjust: Whether to re-scale data.
    :param save_images: Whether to save images.
    :return: train data, train labels, test data.
    """
    def load_pk(filename):
        with open(os.path.join('..', 'data', filename), 'rb') as f:
            return pickle.load(f, encoding='bytes')

    def adjust_window(imgs, shift=1, scale=127.5):
        return (imgs + shift) * scale

    train_imgs_ = load_pk(TRAIN_IMAGES_FILENAME)
    train_labels_ = load_pk(TRAIN_LABELS_FILENAME)
    test_imgs_ = load_pk(TEST_IMAGES_FILENAME)

    if adjust:
        train_imgs_ = adjust_window(train_imgs_)
        test_imgs_ = adjust_window(test_imgs_)

    if save_images:
        if not os.path.exists(os.path.join('..', 'images')):
            os.makedirs(os.path.join('..', 'images'))
        for i, (img, label) in enumerate(zip(train_imgs_, train_labels_)):
            filename = os.path.join('..', 'images', f'train_{i:02d}_{int(label):d}.png')
            if not os.path.exists(filename):
                save_image(img, filename)
        for i, img in enumerate(test_imgs_):
            filename = os.path.join('..', 'images', f'test_{i:02d}_{"?":s}.png')
            if not os.path.exists(filename):
                save_image(img, filename)

    return train_imgs_, train_labels_, test_imgs_



# Turns single image into torch tensor given file path
def load_image(image_path):

    img = Image.open(image_path)
    width, height = img.size
    
    # Resize by keeping the aspect ratio, but changing the dimension
    # so the shortest size is 255px
    img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
    width, height = img.size
    
    # Set the coordinates to do a center crop of 224 x 224
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    
    # Turn image into numpy array
    img = np.array(img)
    
    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))
    
    # Make all values between 0 and 1
    img = img/255
    
    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    
    # Add a fourth dimension to the beginning to indicate batch size
    img = img[np.newaxis,:]
    
    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    return image


# Using our model to predict the label
def predict(image, model):
    # Pass the image through our model
    output = model.forward(image)
    
    # Reverse the log function in our output
    output = torch.exp(output)
    
    # Get the top predicted class, and the output percentage for
    # that class
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()


def predictImages(images, model):
    for path in images:
        image = load_image(os.path.join('..', 'testimages', path))
        top_prob, top_class = predict(image, model)
        print(path)
        print("The model is ", top_prob*100, "% certain that the image has a predicted class of ", top_class)


def configureTransferLearningModel():
    # Turn off training for the parameters, don't want to train model
    for param in model.parameters():
        param.requires_grad = False

    # Create new classifier for model *TODO find better spot to be more clear*
    # copied default from https://towardsdatascience.com/a-beginners-tutorial
    #-on-building-an-ai-image-classifier-using-pytorch-6f85cb69cba7
    classifier_input = model.classifier.in_features
    num_labels = 2
    classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1)) # change validation step in main if changed

    # Replace default classifier with new classifier
    model.classifier = classifier


if __name__ == '__main__':

    # Load the training and testing data (and validation)
    test_imgs = os.listdir(test_dir)

    train_imgs = datasets.ImageFolder(train_dir, transform = transformations)
    val_imgs = datasets.ImageFolder(val_dir, transform = transformations)
    train_loader = torch.utils.data.DataLoader(train_imgs, batch_size=20, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_imgs, batch_size=20, shuffle=True)

    # Load or create model
    if load_path:
        model = torch.load(load_path)
        model.eval()

    else:
        configureTransferLearningModel()

        # Move to GPU or CPU if unavilable
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Set the error function using torch.nn as nn library
        criterion = nn.NLLLoss()
        # Set the optimizer function using torch.optim as optim library
        optimizer = optim.Adam(model.classifier.parameters())
        # optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)

        for epoch in range(epochs):
            train_loss = 0
            val_loss = 0
            accuracy = 0
        
            # Training the model
            model.train()
            counter = 0
            for inputs, labels in train_loader:
                # Move to device
                inputs, labels = inputs.to(device), labels.to(device)
                # Clear optimizers
                optimizer.zero_grad()
                # Forward pass
                output = model.forward(inputs)
                # Loss
                loss = criterion(output, labels)
                # Calculate gradients (backpropogation)
                loss.backward()
                # Adjust parameters based on gradients
                optimizer.step()
                # Add the loss to the training set's running loss
                train_loss += loss.item()*inputs.size(0)
                
                # Print the progress of our training
                counter += 1
                print(counter, "/", len(train_loader))

            model.eval()
            counter = 0
            #Tell torch not to calculate gradients
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # Move to device
                    inputs, labels = inputs.to(device), labels.to(device)
                    # Forward pass
                    output = model.forward(inputs)
                    # Calculate Loss
                    valloss = criterion(output, labels)
                    # Add loss to the validation set's running loss
                    val_loss += valloss.item()*inputs.size(0)
                    
                    # TODO would have to change if output changed
                    # Since our model outputs a LogSoftmax, find the real 
                    # percentages by reversing the log function
                    output = torch.exp(output)
                    # Get the top class of the output
                    top_p, top_class = output.topk(1, dim=1)
                    # See how many of the classes were correct?
                    equals = top_class == labels.view(*top_class.shape)
                    # Calculate the mean (get the accuracy for this batch)
                    # and add it to the running accuracy for this epoch
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    # Print the progress of our evaluation
                    counter += 1
                    print(counter, "/", len(val_loader))
    
            # Get the average loss for the entire epoch
            train_loss = train_loss/len(train_loader.dataset)
            valid_loss = val_loss/len(val_loader.dataset)

            # Print out the information
            print('Accuracy: ', accuracy/len(val_loader))
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

        if save_path:
            torch.save(model, save_path)

    # Evaluating the model
    model.eval()
    predictImages(test_imgs, model)




