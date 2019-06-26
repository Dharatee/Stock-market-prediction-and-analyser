
# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as mplt

#importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
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
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

# 1258 = len(training_set_scaled)
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping
#In RNN, have to reshape the data  based on the input shape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))



#2. Building the RNN

#importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense #to create neurons
from keras.layers import LSTM
from keras.layers import Dropout
#Dropout helps to deactivate the neurons that give '0' output in each iteration/epochs
#Dense: used only for output

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
#dropout ratio of 20 percent

# Adding a second LSTM layer and some Dropout regularisation
#hidden layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
#units=50-->neurons

# Adding a third LSTM layer and some Dropout regularisation
#hidden layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
#hidden layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
#Dense layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
#find the global minima point very quickly


# Fitting the RNN to the Training set
#fit-->run epochs
regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)




#3.Making the predictions and visualizing  the results

#getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#getting the predicated stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) -60:].values
#take previous 60 data
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test =[]
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#gets inversely transformed to the value that was initially defined

#visulaizing the results
mplt.plot(real_stock_price, color ='red', label = 'Real Google Stock Price')
mplt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
mplt.title('Google Stock Price Prediction')
mplt.xlabel('Time')
mplt.ylabel('Google Stock Price')
mplt.legend()
mplt.show()
