'''
3-1: 7일간의 데이터를 사용하여 8일째의 주식 가격을 예측하는 LSTM 모형

https://jfun.tistory.com/194
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(777)  # reproducibility

# train Parameters
seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 500

xy1 = np.loadtxt('day8_stock price.csv', delimiter=',')
xy1 = xy1[::-1]  # reverse order (chronically ordered)
min1 = np.min(xy1, 0)
max1 = np.max(xy1, 0)

xy=(xy1-min1)/(max1-min1)

xy[1]
x = xy
y0 = xy[:, -1]
y = xy[:, [-1]]  # Close as label

# build a dataset
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:(i + seq_length)]
    _y = y[i + seq_length]  # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)
dataX[0]

# train/test split
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])


model1 = tf.keras.models.Sequential()
model1.add(tf.keras.layers.LSTM(10, input_shape = (7, 5)))
model1.add(tf.keras.layers.Dense(1))

optimizer = tf.keras.optimizers.Adam(0.01)
model1.compile(loss='mean_squared_error',
               # optimizer='adam',
               optimizer=optimizer,               
               metrics=['mean_absolute_error', 'mean_squared_error'])

model1.fit(trainX, trainY, epochs=100)
model1.evaluate(testX, testY)

pred=model1.predict(testX)

testY1=(testY*(max1[-1]-min1[-1]))+min1[-1]
pred1=(pred*(max1[-1]-min1[-1]))+min1[-1]

plt.figure(figsize=(10,10))
plt.plot(testY1, c="blue" )
plt.plot(pred1, c="red" )
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.show()


plt.figure(figsize=(10,10))
plt.plot(testY1[:100], c="blue" )
plt.plot(pred1[:100], c="red" )
plt.xlabel("Time Period")
plt.ylabel("predict")
plt.show() 


