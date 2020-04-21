import os
import torchvision
import torch
import pickle

def augment_dataset(in_imgs_filename, in_labels_filename, out_sample_count, transforms):

    def load_pk(filename):
        with open(os.path.join('..', 'data', filename), 'rb') as f:
            return pickle.load(f, encoding='bytes')

    in_imgs_ = load_pk(in_imgs_filename)
    in_labels_ = load_pk(in_labels_filename)

    t1 = in_imgs_.clone()
    transforms(t1)
    t2 = in_imgs_.clone()
    transforms(t2)
    t = torch.cat([t1, t2], dim=0)
    print(t.shape)