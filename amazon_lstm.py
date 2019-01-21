# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 18:02:08 2019

@author: mahit

"""
#load libraries
import urllib
import json
import pandas as pd
import datetime as dt
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error


#get data using alphavantage
api_key = 'xxxxxxxxxxxxxxxxxxxxx'

ticker = "AMZN"

url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

#format json data to dataframe
with urllib.request.urlopen(url_string) as url:
    data = json.loads(url.read().decode())
    data = data['Time Series (Daily)']
    amazon_stock = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
    for key,value in data.items():
        date = dt.datetime.strptime(key, '%Y-%m-%d')
        data_row = [date.date(),float(value['3. low']),float(value['2. high']),
                    float(value['4. close']),float(value['1. open'])]
        amazon_stock.loc[-1,:] = data_row
        amazon_stock.index = amazon_stock.index + 1

#save data and load
with open('filename_pi.obj', 'wb') as file_pi:
    pickle.dump(amazon_stock, file_pi)

with open('filename_pi.obj', 'rb') as file_pi:
    amazon_stock = pickle.load(file_pi)

#sort data by Date and plot
amazon_stock = amazon_stock.sort_values('Date')
plt.plot(amazon_stock['Date'],(amazon_stock['Close']))
plt.xlabel('Date',fontsize=10)
plt.ylabel('Closing Price',fontsize=10)
plt.show()

amazon_stock = amazon_stock.set_index('Date')
data = amazon_stock[['Close']]

#split data into train and test
split1 = dt.date(2016,1,1)
train = data.loc[:split1]
test = data.loc[split1:]

#scale data
scaler = MinMaxScaler()
train_sc = scaler.fit_transform(train)
test_sc = scaler.transform(test)

X_train = train_sc[:-1]
y_train = train_sc[1:]
X_test = test_sc[:-1]
y_test = test_sc[1:]

#reshape data for input
X_tr_t = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_tst_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

#LSTM model using keras

model_lstm = Sequential()
model_lstm.add(LSTM(40, input_shape=(1, X_train.shape[1]), activation='relu', kernel_initializer='glorot_normal', return_sequences=False))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history_model_lstm = model_lstm.fit(X_tr_t, y_train, epochs=200,batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])

#After parameter tuning, 40 neurons with one hidden layer ha sgiven good results
#save model
model_lstm.save("model40.h5")

#load model
model_lstm = load_model('model40.h5')

#predict 
y_pred_test_lstm = model_lstm.predict(X_tst_t)
y_train_pred_lstm = model_lstm.predict(X_tr_t)

#inverse tranform predictions to calculate accuracy
y_pred = scaler.inverse_transform(y_pred_test_lstm)
#mse_train = mean_squared_error(train.Close, y_train_pred_lstm)
mse_test = mean_squared_error(test.Close[:-1], y_pred)
print(mse_test)

plt.plot(range(0,4529),train.Close, label='Train')
plt.plot(range(4528,5282),test.Close, label='Test')
plt.plot(range(4528,5281),y_pred, label='LSTM')
plt.legend(loc='best')
plt.show()