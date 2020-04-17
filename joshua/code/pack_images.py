import os
import torchvision
import torch
import pickle

def pack_data(transforms, deadjust=True):

    def deadjust_window(imgs, shift=1, scale=1.0/127.5):
        return (imgs * scale) - 1.0

    path = os.path.join('..', 'in_images')
    train_dataset = torchvision.datasets.ImageFolder(
        root=path,
        transform=transforms
    )
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                 shuffle=False,
                                                 num_workers=0)
    imagearray = []



    dataiter = iter(train_loader)
    for image, label in dataiter:
        imagearray.append(image)

    images = torch.stack(imagearray)
    images = images.mean(1)
    

    if deadjust:
        images = deadjust_window(images)

    
    ds_filename = os.path.join('..', 'data', 'saved_dataset.pk')
    with open(ds_filename, 'wb') as f:
        pickle.dump(images, f)


