#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM

#Reading the dataset
data_set = pd.read_csv('Currency.csv', na_values='ND')

#Dataset Shape
data_set.shape

#Dataset Head
data_set.head()

#title
currency = ["ALB", "ARG", "AUS", "AUT", "BEL", "BGR", "BRA", "CAN", "CHE", "CHL", "CHN", "COL", "CRI", "CYP", "CZE", "DEU", "DNK", "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HKG", "HRV", "IDN", "IND", "IRL", "ILS", "ITA", "JPN", "KOR", "LTU", "LUX", "LVA", "MAR", "MDG", "MEX", "MKD", "MLT", "NLD", "NOR", "NZL", "POL", "PRT", "ROU", "RUS", "SAU", "SGP", "SRB", "SVK", "SVN", "SWE", "TUR", "USA", "ZAF", "ZMB"]

#Data Frame
currency = input(currency)
df = data_set[currency]
df

#Preprocessing the data set
df = np.array(df).reshape(-1,1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df = scaler.fit_transform(df)
print(df)

#Training and Testing Datasets
train = df[:11]
test = df[11:]

print(train.shape)
print(test.shape)

def get_data(data, look_back):
  data_x, data_y = [],[]
  for i in range(len(data) - look_back - 1):
    data_x.append(data[i: (i+look_back), 0])
    data_y.append(data[i+look_back,0])
  return np.array(data_x), np.array(data_y)

look_back = 1

x_train, y_train = get_data(train, look_back)

print(x_train.shape)
print(y_train.shape)

x_test, y_test = get_data(test, look_back)
print(x_test.shape)
print(y_test.shape)

#Processing train and test sets for LSTM Model
x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
x_test.reshape(x_test.shape[0],x_test.shape[1], 1)


print(x_train.shape)
print(x_test.shape)

#Defining the LSTM Model
n_features = x_train.shape[1]
model = Sequential()
model.add(LSTM(100, activation ='relu', input_shape=(1,1)))
model.add(Dense(n_features))

#Model Summary
model.summary()

#Compiling

model.compile(optimizer='adam', loss = 'mse')

#Training
model.fit(x_train,y_train, epochs = 5, batch_size = 1)

#Prediction using the trained model
scaler.scale_

y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)
print(y_pred[:10])

#Processing the test shape
y_test = np.array(y_test).reshape(-1,1)
y_test = scaler.inverse_transform(y_test)
print(y_test[:10])

#Visualising the results
plt.figure(figsize=(10,5))
plt.title("Foreign Exchange Rate of United Kingdom")
plt.plot(y_test, label='Actual', color='g')
plt.plot(y_pred, label='Predicted', color='r')
plt.legend()
