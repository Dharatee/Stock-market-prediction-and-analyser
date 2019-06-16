

#1. Data  preprocessing

#importing dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as mplt

#importing the training set
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values 

#feature scaling
#scale down the values to the same range
#MinMax scaler to scale all the values in the range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

#implementing with LSTM; based on time stamps
#creating a data structure with 60 timesteps and 1 output
#observe the data from previous 60 timestamps/features (variable) to make future predictions
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping
#In RNN, have to reshape the data  based on the input shape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))


#2. Building the RNN

#importing Keras libraries and packages
import tensorflow as tf
from Keras.models import Sequential
from Keras.layers import Dense
from Keras.layers import LSTM
from Keras.layers import Dropout

#initializing RNN
regressor = Sequential()

#adding thr first LSTM layer and show dropout regularization
regressor.add(LSTM(units =50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#adding the second LSTM layer and show Dropout regularization
regressor.add(LSTM(units =50, return_sequences = True))
regressor.add(Dropout(0.2))

#adding the third LSTM layer and show Dropout regularization
regressor.add(LSTM(units =50, return_sequences = True))
regressor.add(Dropout(0.2))

#adding the fourth LSTM layer and show dropout regularization
regressor.add(LSTM(units =50, return_sequences = True))
regressor.add(Dropout(0.2))

#adding the output layer
regressor.add(Dense(units = 2))

#compiling RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_erroe')

#fitting RNN into the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


#3. 