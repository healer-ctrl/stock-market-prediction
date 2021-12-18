# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2

# import training set
training_set=pd.read_csv('TSLA.csv')
training_set=training_set.iloc[:,4:5].values


# feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler()
training_set=sc.fit_transform(training_set)


# Geting the input and output
X_train= training_set[0:249]
y_train= training_set[1:250]

# Reshaping
X_train=np.reshape(X_train, (249 , 1 , 1))
y_train=np.reshape(y_train, (249 , 1 , 1))

# importing the Keras libraries and Packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# initialize the RNN
regressor = Sequential()

# adding the input layer and LSTM layer
regressor.add(LSTM(units=4, activation= 'tanh', input_shape= (None,1)))

# adding the output layer
regressor.add(Dense( units=1 ))

# compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# fitting the RNN to the training set
regressor.fit(X_train, y_train, batch_size=32, epochs=200)

# Geting the real stock price
test_set = pd.read_csv('Tesla_test.csv')
real_stock_price = test_set.iloc[:,4:5].values

# Geting the Predicted Stock Price
inputs = real_stock_price

# feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler()
inputs=sc.fit_transform(inputs)

inputs = np.reshape(inputs, (21, 1, 1))

predicted_stock_price = regressor.predict(inputs)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

test_loss = regressor.evaluate(predicted_stock_price, real_stock_price)
test_acc = regressor.evaluate(predicted_stock_price, real_stock_price)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# Visulising the Result 
plt2.plot( predicted_stock_price , color = 'blue' , label = 'Predicted Google Stock Price')

plt1.plot( real_stock_price , color = 'red' , label = 'Real Google Stock Price')
plt1.title('Google Stock Price Prediction')
plt2.title('Google Stock Price Prediction')
plt2.xlabel( 'time' )
plt2.ylabel( 'Google Stock Price' )
plt1.xlabel( 'time' )
plt1.ylabel( 'Google Stock Price' )
plt2.legend()
plt1.show()
plt2.show()

