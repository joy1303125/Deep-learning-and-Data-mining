#!/usr/bin/env python
# coding: utf-8

# In[42]:


import tensorflow 
import numpy as np
import os
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import itertools
from tensorflow import keras
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot
from keras.layers import  BatchNormalization,Activation


# In[43]:


import os
base_dir = os.path.join('/kaggle/input/pokemon-images-dataset/pokemon_jpg')
pokemon_pictures = os.listdir(os.path.join(base_dir, "pokemon_jpg"))
print(pokemon_pictures[:10])


# In[44]:


from PIL import ImageOps, Image
size = 64,64

for f in os.listdir(os.path.join(base_dir, "pokemon_jpg")):
    im = Image.open(os.path.join(base_dir, "pokemon_jpg", f)).resize(size, Image.ANTIALIAS)
    break

big_arr = np.array([np.array(im)]).reshape(1, 64, 64, 3)
for f in os.listdir(os.path.join(base_dir,"pokemon_jpg"))[1:]:
    big_arr = np.append(big_arr, [np.array(Image.open(os.path.join(base_dir, "pokemon_jpg", f)).resize(size, Image.ANTIALIAS)).reshape(64, 64, 3)], axis=0)
  


# In[45]:


def define_discriminator(in_shape=(64,64,3)):
    model = Sequential()
# normal
    model.add(Conv2D(64, (5,5), padding='same',kernel_initializer='glorot_uniform', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, (5,5), strides=(2,2),kernel_initializer='glorot_uniform', padding='same'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256, (5,5), strides=(2,2), kernel_initializer='glorot_uniform',padding='same'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(512, (5,5), strides=(2,2), kernel_initializer='glorot_uniform',padding='same'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(alpha=0.2))
    
# classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
# compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model


# In[46]:


# define the standalone generator model
def define_generator(latent_dim):
    model = Sequential()

    n_nodes = 512 * 4 * 4
    model.add(Dense(n_nodes,kernel_initializer='glorot_uniform', input_dim=latent_dim))
    model.add(Reshape((4, 4, 512)))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(256, (5,5), strides=(2,2), kernel_initializer='glorot_uniform',padding='same'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(128, (5,5), strides=(2,2),kernel_initializer='glorot_uniform', padding='same'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(64, (5,5), strides=(2,2),kernel_initializer='glorot_uniform', padding='same'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(3, (5,5), strides=(2,2),kernel_initializer='glorot_uniform', padding='same'))
    model.add(Activation('tanh'))

    optimizer = Adam(lr=0.00015, beta_1=0.5)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,
                  metrics=None)
    model.summary()
    return model


# In[47]:


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.00015, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# In[48]:


def load_real_samples():
    # load the face dataset
    # convert from unsigned ints to floats
    X = big_arr.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X

import random



# In[50]:


def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # select images
    X = dataset[ix]
    # generate class labels
    y = (np.ones(n_samples) -np.random.random_sample(n_samples) * 0.2)
    return X, y


# In[57]:


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
# generate points in the latent space
    x_input = randn(latent_dim * n_samples)
# reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# In[58]:


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
# generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
# predict outputs
    X = g_model.predict(x_input)
# create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X,y


# In[59]:


# create and save a plot of generated images
def save_plot(examples, epoch, n=10):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n * n):
    # define subplot
        pyplot.subplot(n, n, 1 + i)
    # turn off axis
        pyplot.axis('off')
    # plot raw pixel data
        pyplot.imshow(examples[i])
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()


# In[60]:


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save plot
    save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch+1)
    g_model.save(filename)
    


# In[ ]:


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=64):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = np.random.normal(0, 1,size=(n_batch,) +(1, 1, 100))
            # create inverted labels for the fake samples
            y_gan = (np.ones(n_batch ) -np.random.random_sample(n_batch) * 0.2
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
        # evaluate the model performance, sometimes
        if (i+1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


# In[ ]:


# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)

