#housecat vs lion classifier

import tensorflow as tf
import os
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = 'data'

image_extensions = ['jpg', 'jpeg', 'png', 'bmp']

#remove non-image files
for image_file in os.listdir(data_dir):
    file_path = os.path.join(data_dir, image_file)
    # Skip non-image files or corrupted images
    if image_file.split('.')[-1].lower() not in image_extensions:
        os.remove(file_path)
    else:
        try:
            img = image.load_img(file_path)  # Try loading the image
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            os.remove(file_path)  # Remove corrupted files if an

#load images
#builds data pipeline
data = tf.keras.utils.image_dataset_from_directory('data')
#lets us go through pipeline
data_iterator = data.as_numpy_iterator()
#going through pipeline
batch = data_iterator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

#tranformation in pipeline, x = images y = target variable
data = data.map(lambda x, y: (x/255, y))
scaled = data.as_numpy_iterator()

train_size = int(len(data)*0.7)
val_size = int(len(data)*0.2)+1
test_size = int(len(data)*0.1)+1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

model = Sequential([])

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

model.summary()

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()

pre = Precision()
rec = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    x, y = batch
    ywhat = model.predict(x)
    pre.update_state(y, ywhat)
    rec.update_state(y, ywhat)
    acc.update_state(y, ywhat)  

print('Precision:', pre.result().numpy())
print('Recall:', rec.result().numpy())
print('Accuracy:', acc.result().numpy())


