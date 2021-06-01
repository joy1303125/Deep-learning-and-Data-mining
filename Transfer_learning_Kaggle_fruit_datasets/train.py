#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten, Conv2D,GlobalAveragePooling2D
import os
from os import listdir, makedirs
from os.path import join, exists, expanduser
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.optimizers import SGD,Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization


# ### Define the folder paths

# In[2]:

batch_size = 16
train_dir=sys.argv[1]

# ## Build the Model

# #### Determine the number of classes

# In[6]:


num_classes = len(os.listdir(train_dir))
print(f"Number of classes is {num_classes}")


# #### Create the data generators

# In[7]:


IMAGE_SIZE=[224, 224]
BATCH_SIZE=32

validation_split = 0.2

# In[8]:

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,validation_split=validation_split)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    class_mode='categorical',subset='training')

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    class_mode='categorical',subset='validation')


# In[ ]:


inception_base = ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = inception_base.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(512, activation='relu')(x)
# and a fully connected output/classification layer
predictions = Dense(num_classes, activation='softmax')(x)
# create the full network so we can train on it
inception_transfer = Model(inputs=inception_base.input, outputs=predictions)


# In[ ]:

filepath=sys.argv[2]
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]


# In[ ]:


inception_transfer.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# In[ ]:


history_pretrained = inception_transfer.fit(
    train_generator,
    epochs=20, shuffle = True, verbose = 1, validation_data = validation_generator,callbacks=callbacks_list)


# In[ ]:

