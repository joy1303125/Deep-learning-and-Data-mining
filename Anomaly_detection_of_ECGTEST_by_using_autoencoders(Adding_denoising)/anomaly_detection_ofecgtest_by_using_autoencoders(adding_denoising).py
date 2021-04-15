# -*- coding: utf-8 -*-
"""Anomaly_detection_ofECGTEST_by using autoencoders(Adding_denoising).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1td5Xmcd5pH-3MqOKYlQEm126TVR9gmmx
"""

!pip install tensorflow-gpu

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.grid'] = False

!cat ECG5000_TRAIN.txt ECG5000_TEST.txt > ecg_final.txt

!wget http://www.timeseriesclassification.com/Downloads/ECG5000.zip
!unzip ECG5000.zip

!ls -alrt

!cat ECG5000_TRAIN.txt ECG5000_TEST.txt > ecg_final.txt

df=pd.read_csv('ecg_final.txt', sep='  ', header=None)

df

df = df.add_prefix('c')

train_data, test_data, train_labels, test_labels = train_test_split(df.values, df.values[:,0:1], test_size=0.2, random_state=111)

scaler = MinMaxScaler()
data_scaled = scaler.fit(train_data)

train_data_scaled = data_scaled.transform(train_data)
test_data_scaled = data_scaled.transform(test_data)

normal_train_data= pd.DataFrame(train_data_scaled).add_prefix('c').query('c0 == 0').values[:,1:]
anomaly_train_data = pd.DataFrame(train_data_scaled).add_prefix('c').query('c0 > 0').values[:,1:]

normal_test_data= pd.DataFrame(test_data_scaled).add_prefix('c').query('c0 == 0').values[:,1:]
anomaly_test_data = pd.DataFrame(test_data_scaled).add_prefix('c').query('c0 > 0').values[:,1:]

normal_train_dataNoise = np.random.normal(loc=0.5, scale=0.5, size=normal_train_data.shape)
normal_test_dataNoise = np.random.normal(loc=0.5, scale=0.5, size=normal_test_data.shape)
normal_trainX_dataNoise = np.clip(normal_train_data + normal_train_dataNoise, 0, 1)
normal_testX_dataNoise = np.clip(normal_test_data + normal_test_dataNoise, 0, 1)

anomaly_train_dataNoise = np.random.normal(loc=0.5, scale=0.5, size=anomaly_train_data.shape)
anomaly_test_dataNoise = np.random.normal(loc=0.5, scale=0.5, size=anomaly_test_data.shape)
anomaly_trainX_dataNoise =np.clip(anomaly_train_dataNoise + anomaly_train_data, 0, 1)
anomaly_testX_dataNoise = np.clip(anomaly_test_dataNoise + anomaly_test_data, 0, 1)

train_data_scaled_Noise = np.random.normal(loc=0.5, scale=0.5, size=train_data_scaled.shape)
test_data_scaled_Noise = np.random.normal(loc=0.5, scale=0.5, size=test_data_scaled.shape)
trainX_data_scaled_Noise = np.clip(train_data_scaled_Noise + train_data_scaled, 0, 1)
testX_data_scaled_Noise = np.clip(test_data_scaled_Noise + test_data_scaled, 0, 1)

class AutoEncoder(Model):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.Dense(64, activation="relu"),
      tf.keras.layers.Dense(32, activation="relu"),
      tf.keras.layers.Dense(16, activation="relu"),
      tf.keras.layers.Dense(8, activation="relu")])
    
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Dense(16, activation="relu"),
      tf.keras.layers.Dense(32, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      tf.keras.layers.Dense(140, activation="sigmoid")])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

model= AutoEncoder()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=10,
                                                    mode='min')

model.compile(optimizer='adam', loss='mae')

history = model.fit(normal_trainX_dataNoise, normal_train_data, 
          epochs=50, 
          batch_size=128,
          validation_data=(trainX_data_scaled_Noise[:,1:], train_data_scaled[:,1:]),
          shuffle=True,
          callbacks=[early_stopping])

encoder_out = model.encoder(normal_testX_dataNoise).numpy()
decoder_out = model.decoder(encoder_out).numpy()

plt.plot(normal_test_data[0],'b')
plt.plot(decoder_out[0],'r')

encoder_out_a = model.encoder(anomaly_testX_dataNoise).numpy()
decoder_out_a = model.decoder(encoder_out_a).numpy()

plt.plot(anomaly_test_data[0],'b')
plt.plot(decoder_out_a[0],'r')

reconstructions = model.predict(normal_testX_dataNoise)
train_loss = tf.keras.losses.mae(reconstructions, normal_test_data)

plt.hist(train_loss, bins=50)

np.mean(train_loss)

np.std(train_loss)

threshold = np.mean(train_loss) + 2*np.std(train_loss)

reconstructions_a = model.predict(anomaly_testX_dataNoise)
train_loss_a = tf.keras.losses.mae(reconstructions_a, anomaly_test_data)

plt.hist(train_loss_a, bins=50)

plt.hist(train_loss, bins=50, label='normal')
plt.hist(train_loss_a, bins=50, label='anomaly')
plt.axvline(threshold, color='r', linewidth=3, linestyle='dashed', label='{:0.3f}'.format(threshold))
plt.legend(loc='upper right')
plt.show()

np.mean(train_loss_a)

np.std(train_loss_a)

tf.math.less(train_loss, threshold)

preds = tf.math.less(train_loss, threshold)

tf.math.count_nonzero(preds)

preds.shape

preds_a = tf.math.greater(train_loss_a, threshold)

tf.math.count_nonzero(preds_a)

preds_a.shape