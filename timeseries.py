# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 18:02:08 2019

@author: mahit
"""
#load libraries
import pickle
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
from statsmodels.api import tsa

#load data
with open('filename_pi.obj', 'rb') as file_pi:
    amazon_stock = pickle.load(file_pi)

amazon_stock = amazon_stock.sort_values('Date')
plt.plot(amazon_stock['Date'],(amazon_stock['Close']))
plt.xlabel('Date',fontsize=10)
plt.ylabel('Closing Price',fontsize=10)
plt.show()

amazon_stock['Date'] = pd.to_datetime(amazon_stock['Date'])
amazon_stock = amazon_stock.set_index('Date')
data = amazon_stock[['Close']]

#split data into train and test
split1 = datetime.date(2016,1,1)
train = data.loc[:split1]
test = data.loc[split1:]

#naive approach
dd = np.asarray(train.Close)
pred = test.copy()
pred['naive'] = dd[len(dd) - 1]

plt.plot(train.index, train['Close'], label='Train')
plt.plot(test.index,test['Close'], label='Test')
plt.plot(pred.index,pred['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.show()

#accuracy
mse_naive = mean_squared_error(test.Close, pred.naive)
print(mse_naive)


#Simpleaverage
pred['simple_average'] = train['Close'].mean()
plt.plot(train['Close'], label='Train')
plt.plot(test['Close'], label='Test')
plt.plot(pred['simple_average'], label='Simple Average')
plt.legend(loc='best')
plt.show()

#accuracy
mse_average = mean_squared_error(test.Close, pred['simple_average'])
print(mse_average)


#MovingAverage
pred['moving_avg'] = train['Close'].rolling(7).mean().iloc[-1]
plt.plot(train['Close'], label='Train')
plt.plot(test['Close'], label='Test')
plt.plot(pred['moving_avg'], label='Moving Average')
plt.legend(loc='best')
plt.show()

#accuracy
mse_movavg = mean_squared_error(test.Close, pred['moving_avg'])
print(mse_movavg)

#simple exponential smoothing
fit_ses = SimpleExpSmoothing(np.asarray(train['Close'])).fit(smoothing_level=0.4, optimized=True)
pred['SES'] = fit_ses.forecast(len(test))
plt.plot(train['Close'], label='Train')
plt.plot(test['Close'], label='Test')
plt.plot(pred['SES'], label='SES')
plt.legend(loc='best')
plt.show()

#accuracy
mse_ses = mean_squared_error(test.Close, pred['SES'])
print(mse_ses)

#time series decomposition
sm.tsa.seasonal_decompose(train.Close.astype(np.float64), freq = 30).plot()
result = sm.tsa.stattools.adfuller(train.Close)
plt.show()

#Holt linear trend method
fit_holt = Holt(np.asarray(train['Close'])).fit(smoothing_level = 0.35,smoothing_slope = 0.1)
pred['Holt_linear'] = fit_holt.forecast(len(test))

plt.plot(train['Close'], label='Train')
plt.plot(test['Close'], label='Test')
plt.plot(pred['Holt_linear'], label='Holt_linear')
plt.legend(loc='best')
plt.show()

#accuracy
mse_holt = mean_squared_error(test.Close, pred['Holt_linear'])
print(mse_holt)

#Holt winter
fit_holtwinter = ExponentialSmoothing(np.asarray(train['Close']) ,seasonal_periods=120,trend='add', seasonal='add').fit(smoothing_slope = 0.01)
pred['Holt_Winter'] = fit_holtwinter.forecast(len(test))

plt.plot( train['Close'], label='Train')
plt.plot(test['Close'], label='Test')
plt.plot(pred['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show() 
#accuracy
mse_holtwinter = mean_squared_error(test.Close, pred['Holt_Winter'])
print(mse_holtwinter)

#check stationarity
from statsmodels.tsa.stattools import adfuller
adftest = adfuller(train.Close, autolag='AIC')

#arima
#using auto arima to find p,q values
#By differencing two times, data became stationary

from pmdarima.arima import auto_arima
stepwise_fit = auto_arima(train, start_p=4, start_q=4, max_p=10, max_q=10,
                          d=2,trace=True,seasonal = False, scoring = 'mse', solver = 'lbfgs',
                          error_action='ignore',  # don't want to know if an order does not work
                          suppress_warnings=True,  # don't want convergence warnings
                          stepwise=True)  # set to stepwise

stepwise_fit.summary()

#fit arima from obtained values

fit_arima = tsa.ARIMA(np.asarray(train.Close), order=(4,2,4)).fit(maxiter = 1000)

pred['ARIMA'] = fit_arima.forecast(steps = 754)[0]
plt.plot( train['Close'], label='Train')
plt.plot(test['Close'], label='Test')
plt.plot(pred['ARIMA'], label='ARIMA')
plt.legend(loc='best')
plt.show()

#accuracy
mse_arima = mean_squared_error(test.Close, pred['ARIMA'])
print(mse_arima)