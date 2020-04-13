import os
import pickle

from torchvision.utils import save_image

TRAIN_IMAGES_FILENAME = 'train_images_512.pk'
TRAIN_LABELS_FILENAME = 'train_labels_512.pk'
TEST_IMAGES_FILENAME = 'test_images_512.pk'


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
    
