"""
A Simple SimpleRNN

참고문헌: Advanced Forecasting with Python - Joos Korstanje
"""

##================================================##
##====== Time Series Forecasting using RNNs ======##
##================================================##

# Importing the data
import keras
import tensorflow
import pandas as pd
from zipfile import ZipFile
import os
uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"

zip_path = tensorflow.keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
zip_file = ZipFile(zip_path)
zip_file.extractall()
csv_path = "jena_climate_2009_2016.csv"
df = pd.read_csv(csv_path)
del zip_file

# Keep only temperature data
df = df[['T (degC)']]

# Apply a MinMaxScaler
# apply a min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns = ['T'])

# Preparing the sequence data
ylist = list(df['T'])
n_future = 72
n_past = 3*72
total_period = 4*72

idx_end = len(ylist)
idx_start = idx_end - total_period

X_new = []
y_new = []
while idx_start > 0:
    x_line = ylist[idx_start:idx_start+n_past]
    y_line = ylist[idx_start+n_past:idx_start+total_period]
    X_new.append(x_line)
    y_new.append(y_line)
    idx_start = idx_start - 1

# converting list of lists to numpy array
import numpy as np
X_new = np.array(X_new)
y_new = np.array(y_new)

# Splitting into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.33, random_state=42)

# Reshape the data to be recognized by Keras
batch_size = 32

n_samples = X_train.shape[0]
n_timesteps = X_train.shape[1]
n_steps = y_train.shape[1]
n_features = 1

X_train_rs = X_train.reshape(n_samples, n_timesteps, n_features )
X_test_rs = X_test.reshape(X_test.shape[0], n_timesteps, n_features )

# Parameterize a small network with SimpleRNN
import random
random.seed(42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# A Simple SimpleRNN
simple_model = Sequential([
 SimpleRNN(8, activation='tanh',input_shape=(n_timesteps, n_features)),
 Dense(y_train.shape[1]),
])

"""
## SimpleRNN with Hidden Layers

random.seed(42)
simple_model = Sequential([
 SimpleRNN(32, activation='tanh',input_shape=(n_timesteps, n_features),
return_sequences=True),
 SimpleRNN(32, activation='tanh', return_sequences = True),
 SimpleRNN(32, activation='tanh'),
 Dense(y_train.shape[1]),
])
simple_model.summary()
"""

simple_model.summary()

simple_model.compile(
 optimizer=keras.optimizers.Adam(learning_rate=0.001),
 loss='mean_absolute_error',
 metrics=['mean_absolute_error'],
)


smod_history = simple_model.fit(X_train_rs, y_train,
                 validation_split=0.2,
                 epochs=5,
                 batch_size=batch_size,
                 shuffle = True)

preds = simple_model.predict(X_test_rs)

from sklearn.metrics import r2_score
print(r2_score(preds, y_test))
# 0.7136273198838317

import matplotlib.pyplot as plt
plt.plot(smod_history.history['loss'])
plt.plot(smod_history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()