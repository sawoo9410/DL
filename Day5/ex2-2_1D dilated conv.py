# -*- coding: utf-8 -*-
"""
Demonstration of 1-D Time Dilitated Convolutions

https://colab.research.google.com/github/tensorchiefs/dl_book/blob/master/chapter_02/nb_ch02_04.ipynb

WaveNet: A Generative Model for Raw Audio

Wavenet:  https://velog.io/@changdaeoh/Convolutionforsequence

Wavenet:  https://velog.io/@changdaeoh/Convolutionforsequence

"""

# Import required libraries:
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten , Activation, Lambda, Conv1D
from tensorflow.keras.utils import to_categorical 

##---- An other simple toy example ----##

# Creation of the training data
np.random.seed(1) # Fixing the seed, so that data is always the same
seq_length = 128  # Sequence length used for training
look_ahead =  10  # The number of data points the model should predict 

def gen_data(size=1000, noise=0.1): # We create 1000 data-points
  s = seq_length + look_ahead
  d = np.zeros((size, s,1))
  for i in range(size):
    start = np.random.uniform(0, 2*np.pi) # Random start point
    d[i,:,0] = np.sin(start + np.linspace(0, 20*np.pi, s)) * np.sin(start + np.linspace(0, np.pi, s)) + np.random.normal(0,noise,s)
  return d[:,0:seq_length], d[:,seq_length:s]

X,Y = gen_data()
for i in range(1):
  plt.plot(range(0, seq_length),X[i,:,0],'b-')
  plt.plot(range(seq_length, seq_length + look_ahead),Y[i,:,0],'bo',color='orange')
plt.show()

print('The training data X (solid) line and the next predictions Y (dotted), which should be forecasted.')
#The training data X (solid) line and the next predictions Y (dotted), which should be forecasted.

X.shape, Y.shape
#((1000, 128, 1), (1000, 10, 1))

def slice(x, slice_length):
    return x[:,-slice_length:,:]
  
model = Sequential()
ks = 5
model.add(Conv1D(filters=32, kernel_size=ks, padding='causal', batch_input_shape=(None, None, 1)))
model.add(Conv1D(filters=32, kernel_size=ks, padding='causal'))
model.add(Conv1D(filters=32, kernel_size=ks, padding='causal'))
model.add(Conv1D(filters=32, kernel_size=ks, padding='causal'))
model.add(Dense(1))
model.add(Lambda(slice, arguments={'slice_length':look_ahead}))

model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')

history1 = model.fit(X[0:800], Y[0:800],
                    epochs=100,
                    batch_size=128,
                    validation_data=(X[800:1000],Y[800:1000]),
                    verbose=1)

plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])

plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error Loss')
plt.title('Loss Over Time')
plt.legend(['Train','Valid'])

# Testing on new data
X,Y = gen_data(noise=0)
res = model.predict(X) 
print('Shapes X {} Y{} res{}'.format(X.shape, Y.shape, res.shape))
print('MSE for complete look-ahead ', np.average((res[:,:,0] - Y[:,:,0])**2)) 
print('MSE for one step look-ahead ', np.average((res[:,0,0] - Y[:,0,0])**2)) 
print('MSE baseline (same as last)', np.average((X[:,-1,0] - Y[:,0,0])**2)) 
#Shapes X (1000, 128, 1) Y(1000, 10, 1) res(1000, 10, 1)
#MSE for complete look-ahead  0.006175160843588977
#MSE for one step look-ahead  0.011793306071592236
#MSE baseline (same as last) 0.061507885229804334

x_test,y_test = gen_data(size=1,noise=0.0)
model.predict(x_test).reshape(-1),y_test.reshape(-1)

#(array([ 0.9797643 ,  0.9846904 ,  0.7957371 ,  0.4545659 ,  0.033907  ,
#        -0.37807393, -0.6964255 , -0.8570964 , -0.82994324, -0.62439007],
#       dtype=float32),
#array([ 0.85230934,  0.88453287,  0.73468149,  0.43794298,  0.05913728,
#        -0.32152731, -0.6255856 , -0.7926458 , -0.7924877 , -0.63048301]))
    
# Prediction one after another
def predict_sequence(input_sequence, model, pred_steps):

    history_sequence = input_sequence.copy()
    pred_sequence = np.zeros((1,pred_steps * look_ahead,1)) # initialize output (pred_steps time steps)  
    
    for i in range(pred_steps):
        
        # record next time step prediction (last time step of model output) 
        last_step_pred = model.predict(history_sequence)[0,-look_ahead:,0]
        pred_sequence[0,(i * look_ahead) : ((i+1) * look_ahead),0] = last_step_pred
        
        # add the next time step prediction to the history sequence
        history_sequence = np.concatenate([history_sequence, 
                                           last_step_pred.reshape(-1,look_ahead,1)], axis=1)

    return pred_sequence

plt.figure(num=None, figsize=(13,5))
pred_steps = 10
x_test,y_test = gen_data(size=1,noise=0.0)
preds = predict_sequence(x_test, model, pred_steps)
plt.plot(range(0,len(x_test[0])),x_test[0,:,0])
plt.plot(range(len(x_test[0]),len(x_test[0])+len(preds[0])),preds[0,:,0],color='orange')
plt.ylim((-1,1))

##---- The model with time dilated convolutions ----##

X,Y = gen_data(noise=0)

model_dil = Sequential()
#<------ Just replaced this block
model_dil.add(Conv1D(filters=32, kernel_size=ks, padding='causal', dilation_rate=1, 
                           batch_input_shape=(None, None, 1)))
model_dil.add(Conv1D(filters=32, kernel_size=ks, padding='causal', dilation_rate=2))
model_dil.add(Conv1D(filters=32, kernel_size=ks, padding='causal', dilation_rate=4))
model_dil.add(Conv1D(filters=32, kernel_size=ks, padding='causal', dilation_rate=8))
#<------ Just replaced this block

model_dil.add(Dense(1))
model_dil.add(Lambda(slice, arguments={'slice_length':look_ahead}))

model_dil.summary()

model_dil.compile(optimizer='adam',loss='mean_squared_error')

hist_dil = model_dil.fit(X[0:800], Y[0:800],
                    epochs=200,
                    batch_size=128,
                    validation_data=(X[800:1000],Y[800:1000]), verbose=1)

plt.plot(hist_dil.history['loss'])
plt.plot(hist_dil.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error Loss')
plt.title('Loss Over Time (Dilated)')
plt.legend(['Train','Valid'])

# Testing
X,Y = gen_data()
res = model_dil.predict(X) 
print('MSE for one step look-ahead ', np.average((res[:,0,0] - Y[:,0,0])**2)) #One step look-ahead prediction
print('MSE baseline (same as last)', np.average((X[:,-1,0] - Y[:,0,0])**2)) 
#MSE for one step look-ahead  0.014241653115197758
#MSE baseline (same as last) 0.08379639134464463

plt.figure(num=None, figsize=(13,5))
pred_steps = 10
x_test,y_test = gen_data(size=1,noise=0.0)
preds = predict_sequence(x_test, model_dil, pred_steps)
plt.plot(range(0,len(x_test[0])),x_test[0,:,0])
plt.plot(range(len(x_test[0]),len(x_test[0])+len(preds[0])),preds[0,:,0],color='orange')
plt.ylim((-1,1))
