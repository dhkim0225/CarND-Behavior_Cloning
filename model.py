import csv, os, sys, math
import cv2
import numpy as np
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/model")

from NVIDIA_model import nvidia
from sklearn.utils import shuffle
'''
||| DATA |||
Csv file include below informations
center_img left_img right_img steering throttle brake speed

We need 3 image files to get steering
X : center_img, left_img, right_img
Y : steering

||| MODEL |||
See /model/NVIDIA_model.py
'''
# Read lines of log file.
lines = []
with open(os.path.join(dir_path, 'data/driving_log.csv'), 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

# shuffle data
lines = np.array(lines)
np.random.shuffle(lines)

# train, val = 4 : 1
index = int(len(lines) * 4 / 5)
val_lines = lines[index:]
lines = lines[:index]

train_size = len(lines)
val_size = len(val_lines)

# Data generator ==> read 32 lines of log in one time.
def data_generator():
    while 1:
        count = 0
        images = []  # 3 image datas - center, left, right
        measurements = []  # steering value
        for line in lines:
            count += 1
            for i in range(3):
                img_path = os.path.join(dir_path, 'data/IMG/') + line[i].split('/')[-1]

                image = cv2.imread(img_path)
                images.append(image)
                images.append(cv2.flip(image, 1))

                measurement = float(line[3])
                measurements.append(measurement)
                measurements.append(measurement * -1.0)

            if count % 32 == 0 and len(images) != 0:
                imgs, meases = shuffle(np.array(images), np.array(measurements))
                yield imgs, meases
                images.clear()
                measurements.clear()

            elif train_size == count and len(images) != 0:
                imgs, meases = shuffle(np.array(images), np.array(measurements))
                yield imgs, meases
                images.clear()
                measurements.clear()

# Val_data generator
def val_generator():
    while 1:
        count = 0
        images = []  # 3 image datas - center, left, right
        measurements = []  # steering value
        for line in val_lines:
            count += 1
            for i in range(3):
                img_path = './data/IMG/' + line[i].split('/')[-1]

                image = cv2.imread(img_path)
                images.append(image)
                images.append(cv2.flip(image, 1))

                measurement = float(line[3])
                measurements.append(measurement)
                measurements.append(measurement * -1.0)

            if count % 8 == 0 and len(images) != 0:
                yield np.array(images), np.array(measurements)
                images.clear()
                measurements.clear()

            elif val_size == count and len(images) != 0:
                yield np.array(images), np.array(measurements)
                images.clear()
                measurements.clear()


# struct model ==> NVIDIA Model
model = nvidia()
model.compile(loss='mse', optimizer='adam')
model.summary()

# training with model.fit_generator
t_gen = data_generator()
v_gen = val_generator()
history_object = model.fit_generator(t_gen,
                                     steps_per_epoch=math.ceil(train_size/32),
                                     validation_data=v_gen,
                                     validation_steps=math.ceil(val_size/8),
                                     verbose=1,
                                     epochs=10)

# save model
model.save(os.path.join(dir_path, 'model_data/model.h5'))

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('./model_data/data.png')
