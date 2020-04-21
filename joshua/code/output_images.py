import os
import pickle

from torchvision.utils import save_image

def output_images(adjust=True, save_images=True):
    
    #TRAIN_IMAGES_FILENAME = 'train_images_512.pk'
    TRAIN_IMAGES_FILENAME = 'saved_dataset.pk'
    TRAIN_LABELS_FILENAME = 'train_labels_512.pk'

    def load_pk(filename):
        with open(os.path.join('..', 'data', filename), 'rb') as f:
            return pickle.load(f, encoding='bytes')

    def adjust_window(imgs, shift=1, scale=127.5):
        return (imgs + shift) * scale

    train_imgs_ = load_pk(TRAIN_IMAGES_FILENAME)
    train_labels_ = load_pk(TRAIN_LABELS_FILENAME)

    print(train_imgs_.shape)
    if adjust:
        train_imgs_ = adjust_window(train_imgs_)

    if save_images:
        if not os.path.exists(os.path.join('..', 'out_images')):
            os.makedirs(os.path.join('..', 'out_images'))
        for i, (img, label) in enumerate(zip(train_imgs_, train_labels_)):
            filename = os.path.join('..', 'out_images', f'train_{i:02d}_{int(label):d}.png')
            if not os.path.exists(filename):
                save_image(img, filename)

    return train_imgs_, train_labels_