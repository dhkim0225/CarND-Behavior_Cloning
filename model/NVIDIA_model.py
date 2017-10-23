'''
This is NVIDIA end-to-end deeplearning model.
https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
'''

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Lambda, Cropping2D, Dropout
from keras.regularizers import l2

def nvidia():
    model = Sequential()

    model.add(Cropping2D(cropping=((70, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), strides=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), strides=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(10, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1))

    return model