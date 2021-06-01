#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten, Conv2D,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD,Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
import os 
# In[ ]:


K.set_image_data_format('channels_last')




test_dir=sys.argv[1]
num_classes = len(os.listdir(test_dir))

# In[ ]:



test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')


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
inception_transfer_test = Model(inputs=inception_base.input, outputs=predictions)

# In[ ]:


inception_transfer_test.load_weights(sys.argv[2])


# In[ ]:


inception_transfer_test.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# In[ ]:


scores = inception_transfer_test.evaluate(test_generator,verbose=1)

print('accuracy:', (scores[1]))

# In[ ]:


f = open("Output_resnet.txt", "w")
test_error = (1 - scores[1]) * 100
test_accuracy=scores[1]*100
s2 = "Test Accuracy:\t\t"+str(test_accuracy) + "%"
f.write(s2)

