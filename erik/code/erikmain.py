import os
import pickle

# Import Libraries
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

TRAIN_IMAGES_FILENAME = 'train_images_512.pk'
TRAIN_LABELS_FILENAME = 'train_labels_512.pk'
TEST_IMAGES_FILENAME = 'test_images_512.pk'

# Process our image
def process_image(image_path):
    # Load Image
    img = Image.open(image_path)
    
    # Get the dimensions of the image
    width, height = img.size
    
    # Resize by keeping the aspect ratio, but changing the dimension
    # so the shortest size is 255px
    img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
    
    # Get the dimensions of the new image size
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

# Show Image
def show_image(image):
    # Convert image to numpy
    image = image.numpy()
    
    # Un-normalize the image
    image[0] = image[0] * 0.226 + 0.445
    
    # Print the image
    fig = plt.figure(figsize=(25, 4))
    plt.imshow(np.transpose(image[0], (1, 2, 0)))

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


if __name__ == '__main__':
    train_imgs, train_labels, test_imgs = load_data()

    # Specify transforms using torchvision.transforms as transforms
    librarytransformations = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_imgs[slice(67)]
    train_labels[slice(67)]

    # Put into a Dataloader using torch library
    train_loader = torch.utils.data.DataLoader(train_imgs, batch_size=20, shuffle=False)
    train_loader2 = torch.utils.data.DataLoader(train_labels, batch_size=20, shuffle=False)
    # val_loader = torch.utils.data.DataLoader(test_imgs, batch_size =20, shuffle=True)

    vision.datasets.MNIST('../test', train=True, transform=None, target_transform=None, download=True)

    # Get pretrained model using torchvision.models as models 
    model = models.densenet161(pretrained=True)
    # Turn off training for their parameters
    for param in model.parameters():
        param.requires_grad = False

    # Create new classifier for model using torch.nn as nn library
    classifier_input = model.classifier.in_features
    num_labels = 2
    classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))

    # Replace default classifier with new classifier
    model.classifier = classifier

    # Find the device available to use using torch library
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model to the device specified above
    model.to(device)

    # Set the error function using torch.nn as nn library
    criterion = nn.NLLLoss()
    # Set the optimizer function using torch.optim as optim library
    optimizer = optim.Adam(model.classifier.parameters())
    # optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)

    epochs = 1
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        accuracy = 0
    
        # Training the model
        model.train()
        counter = 0
        for inputs, labels in zip(train_loader, train_loader2):
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)        # Clear optimizers
            optimizer.zero_grad()        # Forward pass
            output = model.forward(inputs)        # Loss
            loss = criterion(output, labels)        # Calculate gradients (backpropogation)
            loss.backward()        # Adjust parameters based on gradients
            optimizer.step()        # Add the loss to the training set's rnning loss
            train_loss += loss.item()*inputs.size(0)
            
            # Print the progress of our training
            counter += 1
            print(counter, "/", len(train_loader))
        
    # Evaluating the model
    model.eval()
    counter = 0
    # Tell torch not to calculate gradients
    # with torch.no_grad():
    #     for inputs, labels in val_loader:
    #         # Move to device
    #         inputs, labels = inputs.to(device), labels.to(device)            # Forward pass
    #         output = model.forward(inputs)            # Calculate Loss
    #         valloss = criterion(output, labels)            # Add loss to the validation set's running loss
    #         val_loss += valloss.item()*inputs.size(0)
            
    #         # Since our model outputs a LogSoftmax, find the real 
    #         # percentages by reversing the log function
    #         output = torch.exp(output)            # Get the top class of the output
    #         top_p, top_class = output.topk(1, dim=1)            # See how many of the classes were correct?
    #         equals = top_class == labels.view(*top_class.shape)            # Calculate the mean (get the accuracy for this batch)
    #         # and add it to the running accuracy for this epoch
    #         accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    #         # Print the progress of our evaluation
    #         counter += 1
    #         print(counter, "/", len(val_loader))
    
    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader.dataset)
    # valid_loss = val_loss/len(val_loader.dataset)    # Print out the information
    # print('Accuracy: ', accuracy/len(val_loader))
    # print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

model.eval()

# Process Image
image = process_image("/Users/Zoidberg/Desktop/CPSC-340-Final/images/train_69_1.png")# Give image to model to predict output
top_prob, top_class = predict(image, model)# Show the image
show_image(image)# Print the results
print("cat")
print("The model is ", top_prob*100, "% certain that the image has a predicted class of ", top_class  )




