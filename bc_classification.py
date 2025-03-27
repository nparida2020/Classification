"""
# Breast Cancer Image Classification Using CNN

"""
import os
os.environ['TF_USE_LEGALCY_KERAS'] = "1"

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.python.keras.layers import Input,Dense
from tensorflow.python.keras.utils import image_dataset_from_directory
from tensorflow.python.keras import Sequential

import matplotlib.pyplot as plt
from PIL import Image
import os 
import pathlib 
import random

#Root Data directory path

path = '~/cnn_bc_detection/CNN/data'
data_dir_path = pathlib.Path(path)
print('Success')
#BC cancer class names - normal,benign,malignant

classNames = np.array([class_type.name for class_type in data_dir_path.glob("*")])
classNames

# data path for the benign,normal and malignant

benign_data_path = pathlib.Path(os.path.join(data_dir_path,'benign'))
normal_data_Path = pathlib.Path(os.path.join(data_dir_path,'normal'))
malignant_data_Path = pathlib.Path(os.path.join(data_dir_path,'malignant'))

# Images - count for benign,normal and malignant

countImgBenign = len(list(benign_data_path.glob('*.png')))
countImgNormal = len(list(normal_data_Path.glob('*.png')))
countImgMalignant = len(list(malignant_data_Path.glob('*.png')))
totalImageCount = countImgBenign + countImgNormal + countImgMalignant

print("No of total Images: ", totalImageCount)
print("No. of Benign (non-dangerous) Images: {}({})".format(countImgBenign, round(countImgBenign*100/totalImageCount, 2)))
print("No. of Malignant (dangerous) Images: {}({})".format(countImgMalignant, round(countImgMalignant*100/totalImageCount, 2)))
print("No. of Normal (No Traces) Images: {}({})".format(countImgNormal, round(countImgNormal*100/totalImageCount, 2)))

#

batch_size = 32
img_height = 224
img_width = 224


#create train data set for the model 70%:30%
train_ds = image_dataset_from_directory(data_dir_path,validation_split=0.3,subset="training",seed=123,image_size=(img_height, img_width),batch_size=batch_size)

#create validation data set for the model
val_ds = image_dataset_from_directory(data_dir_path,validation_split=0.3,subset="validation",seed=123,image_size=(img_height,img_width),batch_size=batch_size)


def create_cnn_model():
#Create CNN Model , 3 CNN layers, each followed by max pooling layer, kernel size 3x3
# 3 BC classes- therefore softmax layer to generate the output

  model = tf.keras.Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
    
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
    
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
    
  layers.Dropout(0.3),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(3,activation="softmax")
])

#compile the model, optimizer =Adam, loss function = SparseCategoricalCrossentrop, metrics=Accuracy

  model.compile(optimizer="Adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
  return model

# create model instance
model = create_cnn_model()

# Model Summery and parameters
model.summary()

# Create epcoh for each batch
epochs = 15
history = model.fit(train_ds,
                    validation_data=val_ds, 
                    epochs=epochs,
                    batch_size=batch_size)


file_name = 'bc_cnn_model.keras'
file_path = '~/cnn_bc_detection/CNN/'
save_data_path = pathlib.Path(os.path.join(file_path,file_name))
# Save the entire model as a `.keras` zip archive.

model.save(save_data_path)


#Generate History keys
history.history.keys()

# derive accuracy vs validation loss

acc = history.history['accuracy']
val_acc =  history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range,acc,label='Accuracy')
plt.plot(epochs_range,val_acc,label="Validation Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range,loss,label='Loss')
plt.plot(epochs_range,val_loss,label="Validation Loss")
plt.legend()
plt.show()

# Evaluate the model
model.evaluate(val_ds)

#plot the 
plt.figure(figsize=(15, 15))
class_names = val_ds.class_names
result = ' | False'
for images, labels in val_ds.take(1):
    for i in range(25):
        
        ax = plt.subplot(5, 5, i + 1)
        img = images[i].numpy().astype("uint8")
        img = tf.expand_dims(img, axis=0)
        # predictions 
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)
        if class_names[predicted_class] == class_names[labels[i]]:
            result = ' | TRUE'
            
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[predicted_class]+result)
        plt.axis("off")

# load saved model
new_model = tf.keras.models.load_model(saved_model_path)

# Show the model architecture
new_model.summary()
loss, acc = new_model.evaluate(val_ds)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
#plot the 
plt.figure(figsize=(15, 15))
class_names = val_ds.class_names
result = ' | False'
for images, labels in val_ds.take(1):
    for i in range(25):
        
        ax = plt.subplot(5, 5, i + 1)
        img = images[i].numpy().astype("uint8")
        img = tf.expand_dims(img, axis=0)
        # predictions 
        predictions = new_model.predict(img)
        predicted_class = np.argmax(predictions)
        if class_names[predicted_class] == class_names[labels[i]]:
            result = ' | TRUE'
            
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[predicted_class]+result)
        plt.axis("off")



print(new_model.predict(val_ds).shape)
