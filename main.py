import os
import pickle

from DataSet import DataSet
from torchvision.utils import save_image

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


if __name__ == '__main__':
    D = DataSet()
    D.load_data()
    D.preprocess_data()

    X_train = D.X_train[:, :, :, None]
    X_test = D.X_test[:, :, :, None]

    y_train = to_categorical(D.y_train)

    model = Sequential()

    model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10)
