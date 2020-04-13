import os
import pickle
import numpy as np
from torchvision.utils import save_image
from sklearn import preprocessing


class DataSet:
    TRAIN_IMAGES_FILENAME = 'train_images_512.pk'
    TRAIN_LABELS_FILENAME = 'train_labels_512.pk'
    TEST_IMAGES_FILENAME = 'test_images_512.pk'

    DATA_DIR = os.path.join('', 'data')
    IMAGES_DIR = os.path.join('', 'images')

    def __init__(self, shift=1, scale=127.5, downsample=4):
        self.shift = shift
        self.scale = scale
        self.downsample = downsample
        self.X_train = None
        self.X_test = None
        self.y_train = None

    def load_data(self, save_images=True, overwrite=False):
        def load_pk(filename):
            with open(os.path.join(DataSet.DATA_DIR, filename), 'rb') as f:
                return pickle.load(f, encoding='bytes')

        def adjust_image(imgs, shift, scale):
            return (imgs + shift) * scale

        def resize(X_in, downsample):
            n, x, y = X_in.shape
            X_out = np.empty((n, x // downsample, y // downsample))
            for i in range(X_out.shape[1]):
                for j in range(X_out.shape[2]):
                    i_start, i_end = i * downsample, (i + 1) * downsample
                    j_start, j_end = j * downsample, (j + 1) * downsample
                    Xij = X_in[:, i_start:i_end, j_start:j_end]
                    X_out[:, i, j] = Xij.reshape((n, -1)).mean(1)
            return X_out

        raw_train_imgs = load_pk(self.TRAIN_IMAGES_FILENAME)
        raw_train_labels = load_pk(self.TRAIN_LABELS_FILENAME)
        raw_test_imgs = load_pk(self.TEST_IMAGES_FILENAME)

        raw_train_imgs = adjust_image(raw_train_imgs, self.shift, self.scale)
        raw_test_imgs = adjust_image(raw_test_imgs, self.shift, self.scale)

        if save_images:
            if not os.path.exists(os.path.join('', 'images')):
                os.makedirs(os.path.join('', 'images'))
            for idx, (img, label) in enumerate(zip(raw_train_imgs, raw_train_labels)):
                filename = f'train_{idx:02d}_{int(label):d}.png'
                full_filename = os.path.join(DataSet.IMAGES_DIR, filename)
                if not os.path.exists(full_filename) or overwrite:
                    save_image(img, full_filename)
            for idx, img in enumerate(raw_test_imgs):
                filename = f'test_{idx:02d}_{"?":s}.png'
                full_filename = os.path.join(DataSet.IMAGES_DIR, filename)
                if not os.path.exists(full_filename) or overwrite:
                    save_image(img, full_filename)

        raw_train_imgs = raw_train_imgs.mean(1)
        raw_test_imgs = raw_test_imgs.mean(1)

        self.X_train = resize(raw_train_imgs, self.downsample)
        self.X_test = resize(raw_test_imgs, self.downsample)

        self.y_train = np.array(raw_train_labels)

        return raw_train_imgs, raw_train_labels, raw_test_imgs

    def preprocess_data(self):
        pass


if __name__ == '__main__':
    D = DataSet()
    D.load_data()
