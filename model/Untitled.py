#!/usr/bin/env python
# coding: utf-8

# In[67]:


import pandas as pd
import os
from pathlib import Path
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.layers import *
from keras.models import *
from keras.preprocessing import image


# In[68]:


FILE_PATH = Path('chestxray/metadata.csv').absolute()
IMAGE_PATH = Path('chestxray/images/').absolute()


# In[ ]:





# In[69]:


df = pd.read_csv(FILE_PATH)


# In[70]:


df.head()


# In[71]:


TARGET_DIR = Path('./Dateset/Covid').absolute()
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)
    print("Covid Folder Created")


# In[72]:


cnt = 0

for (i, row) in df.iterrows():
    if 'COVID-19' in row['finding'] and row['view'] == 'PA':
        filename = row["filename"]
        image_path = IMAGE_PATH / filename
        target_copy_path = TARGET_DIR / filename
        shutil.copy2(image_path, target_copy_path)
        cnt += 1

print(cnt)


# In[73]:


# Sampling of Images from Kaggle
KAGGLE_FILE_PATH = Path('chest_xray/train/NORMAL/').absolute()
TARGET_NORMAL_DIR = Path("Dateset/Normal/").absolute()


# In[74]:


image_names = os.listdir(KAGGLE_FILE_PATH)
random.shuffle(image_names)


# In[75]:


for i in range(142):
    image_name = image_names[i]
    image_path = KAGGLE_FILE_PATH / image_name
    target_path = TARGET_NORMAL_DIR / image_name
    shutil.copy2(image_path, target_path)


# In[76]:


TRAIN_PATH = Path('CovidDataset/Train').absolute()
VAL_PATH = Path('CovidDataset/Test').absolute()


# In[ ]:





# In[77]:


# CNN Based Model in Keras


# In[78]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
model.summary()


# In[79]:


# Train Model from scrach


# In[80]:


train_datagen = image.ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
)

test_dataset = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'CovidDataset/Train/',
    target_size=(224, 224),
    batch_size = 32,
    class_mode = 'binary'
)


# In[ ]:





# In[81]:


validation_generator = test_dataset.flow_from_directory(
    'CovidDataset/Val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)


# In[84]:


hist = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=2
)


# In[ ]:




