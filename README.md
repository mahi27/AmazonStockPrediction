# TimeSeriesForecasting

Be it traders or banks or eCommerce giants, time-series forecasting is a common statistical task to predict future values and customize strategies.  There are several time-series forecasting techniques like auto-regression models, moving average models, Holt-winters, ARIMA etc., to name a few. In the recent years, LSTM Networks have shown to achieve state-of-the-art results to forecast time-series.

To understand the capabilities of LSTMs,  I got the Amazon stock price data using AlphaVantage API and predicted closing price using traditional time series methods and LSTM. The results obtained are shown below. 

| Method                      | Mean Squared Error  |
| ----------------------------|:-------------------:|
| Simple Average              | 1121933.05          | 
| Moving Average              | 359590.03           | 
| Naive Method                | 358679.00           | 
| Simple Exponential Smoothing| 355182.80           |
| ARIMA                       | 121021.70           |
|Holt's Linear Trend Method   | 37990.49            |
|Holt-Winters Method          | 35639.95            |
|LSTM                         | 3519.83550          |
