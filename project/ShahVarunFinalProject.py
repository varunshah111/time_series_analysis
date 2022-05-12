#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.tsa.seasonal import STL
import seaborn as sns
import datetime
from sklearn.model_selection import train_test_split
import statsmodels.tsa.holtwinters as ets
import statsmodels.api as sm
from numpy import linalg as LA
from scipy.stats import chi2

#%%
import sys
sys.path.append(r'/Users/varunshah/PycharmProjects/TimeSeries/toolbox.py')
from toolbox import cal_rolling_mean_var, errors, autocorr, ACF_PACF_Plot, \
    box_pierce, correlation_coefficient, ADF_Cal, kpss_test, \
    difference,calc_gpac, average_forecast, naive_method, drift_method, ses_method, lm_algorithm


#%%
# ================
# Load the data:
# ===============

data = 'https://raw.githubusercontent.com/varunshah111/time_series_analysis/main/project/data/train.csv'
bike = pd.read_csv(data)

#%%
# =========================
# Data Prepreprocessing
# =========================

# Printing the first 5 rows
bike.head()

#%%
bike.info()
#%%
# Converting the object type to datetime
bike['datetime'] = pd.to_datetime(bike['datetime'])

#%%
bike.info()

#%%
# Checking for Null values
bike.isna().sum()
#%%

# Plotting Count against time
x = bike['datetime']
y = bike['count']

f, ax = plt.subplots(figsize=(30,20))
plt.plot(x, y)

plt.xlabel('Time', fontsize = 30)
plt.ylabel('Number of Bikes rented', fontsize=30)
plt.xticks(rotation=45, horizontalalignment="center", fontsize=25)
plt.yticks(rotation=0, horizontalalignment="right", fontsize=25)
plt.title('Total Bike Rental over Time', fontsize=50)
plt.grid()
plt.show()

#%%

# Plotting Casual Renters, Registered bike renters and total bikes rented(combined)

x = bike['datetime']
y1 = bike['casual']
y2 = bike['registered']
y3 = bike['count']

fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3, figsize=(30, 20), sharex=True)

ax1.plot(x, y1)
ax1.set_title("Casual Rented Bikes over Time", fontsize=25)
ax1.set_xlabel('Time', fontsize = 20)
ax1.set_ylabel('Number of Bikes rented', fontsize=20)
ax1.grid()

ax2.plot(x, y2)
ax2.set_title("Registered Rented Bikes over Time", fontsize=25)
ax2.set_xlabel('Time', fontsize = 20)
ax2.set_ylabel('Number of Bikes rented', fontsize=20)
ax2.grid()

ax3.plot(x, y3)
ax3.set_title("Total Bike Rental over Time", fontsize=25)
ax3.set_xlabel('Time', fontsize = 20)
ax3.set_ylabel('Number of Bikes rented', fontsize=20)

ax3.grid()
plt.show()


#%%
# ACF Plot

# For simplicity, we can focus only on the Total count of Bike Rentals.

autocorr(bike['count'], 50)

#%%
# ACF and PACF

ACF_PACF_Plot(bike['count'], 50)



#%%
# Correlation Heatmap

plt.subplots(figsize=(10,10))
sns.heatmap(bike.drop(columns = 'datetime').corr(), annot=True, cmap='viridis' )
#plt.savefig('heatmap.png')
plt.yticks(rotation=0)
plt.show()


#%%
# Converting Season, holiday, workingday and weather to categorical


bike['season'] = bike['season'].astype('category')
bike['holiday'] = bike['holiday'].astype('category')
bike['workingday'] = bike['workingday'].astype('category')
bike['weather'] = bike['weather'].astype('category')
#%%
X = bike.drop(['count'], axis=1)
y = bike['count']
#%%
# Train-Test Split

y_train, y_test = train_test_split(y, shuffle= False, test_size=0.2)
X_train, X_test = train_test_split(X, shuffle= False, test_size=0.2)

h = len(y_test)

#%%
# Stationarity:
cal_rolling_mean_var(y_train, X_train['datetime'], "Bike Rental")

# ADF Test
ADF_Cal(y_train)
# we can reject the Null Hypothesis and adopt the alternative and state that the data is stationary

# KPSS Test
kpss_test(y_train)
# We are unable to reject the Null Hypothesis and can state that the data is stationary

#%%
# Sampling at seasonality of 24
# Lecture 11 slides
y_24 = y_train[:24:]
ACF_PACF_Plot(y_24, 10)

#%%

# Logarithmic First Differencing - seasonal: 24

Logarithmic = np.log(y_train)
# First Order Differencing for the logarithmic series:

FirstLogDiff = difference(Logarithmic, 24)

datetime_First = X_train['datetime'][24:]

cal_rolling_mean_var(FirstLogDiff, datetime_First, "Logarithmic First Differencing with 24")

#%%
ACF_PACF_Plot(FirstLogDiff, 75)

# ADF Test
ADF_Cal(FirstLogDiff)
# we can reject the Null Hypothesis and adopt the alternative and state that the data is stationary

# KPSS Test
kpss_test(FirstLogDiff)
# We are unable to reject the Null Hypothesis and can state that the data is stationary

#%%
FirstDiff = difference(y_train, 24)

datetime_First = X_train['datetime'][24:]

cal_rolling_mean_var(FirstDiff, datetime_First, "First Differencing with 24")

#%%
ACF_PACF_Plot(FirstDiff, 75)

#%%
SecondDiff = difference(FirstDiff, 24)

datetime_Second = datetime_First[24:]

cal_rolling_mean_var(SecondDiff, datetime_Second, "Second Differencing with 24")

#%%
ACF_PACF_Plot(SecondDiff, 75)

#%%

ThirdDiff = difference(SecondDiff, 1)

datetime_Third = datetime_Second[1:]

cal_rolling_mean_var(ThirdDiff, datetime_Third, "Third Differencing with 1")

#%%
ACF_PACF_Plot(ThirdDiff, 75)

#%%

# 8.
# STL Decomposition

transformed = pd.Series(FirstLogDiff,
                  index=pd.date_range('2011-01-01', periods=len(FirstLogDiff), freq='h'), name='Transformed Data')

STL = STL(transformed)
res = STL.fit()
fig = res.plot()
plt.show()

T = res.trend
S = res.seasonal
R = res.resid

adj_seasonal = transformed - S
adj_trend = transformed - T

plt.plot(transformed, label = "Transformed Data")
plt.plot(adj_seasonal, label = "Seasonally adjusted Data")

plt.title("Seasonally Adjusted Data - STL")
plt.xlabel("Time")
plt.ylabel("Scale")
plt.legend()
plt.show()

F_S = np.maximum(0,1 - np.var(np.array(R))/np.var(np.array(S)+np.array(R)))
print("Strength of Seasonality for this data set is: ", F_S)

# F_S is equal to 0 exhibits no seasonality.
plt.plot(transformed, label = "Transformed Data") ,
plt.plot(adj_trend, label = "Adjusted Trend Data")

plt.title("Adjusted Trend Data - STL")
plt.xlabel("Time")
plt.ylabel("Scale")
plt.legend()
plt.show()

F_T = np.maximum(0,1 - np.var(np.array(R))/np.var(np.array(T)+np.array(R)))
print("Strength of Trend for this data set is: ", F_T)

# F_T is equal to 0, hence its completely detrended



#%%
#Holt-Winter Seasonal Method

holtt_seasonal = ets.ExponentialSmoothing(y_train, trend='add', damped_trend=True, seasonal='add', seasonal_periods=24).fit()
holtf_seasonal = holtt_seasonal.forecast(steps=h)
seasonal_predict = holtt_seasonal.fittedvalues

plt.plot(X_train['datetime'], y_train, label = 'Train')
plt.plot(X_test['datetime'], y_test, label = 'Test')
plt.plot(X_test['datetime'], holtf_seasonal, label = 'Seasonal', alpha=0.5)
plt.title("Holt Seasonal Method")
plt.xlabel("Time")
plt.ylabel("Number of Bike Rental")
plt.legend()
plt.show()

#%%
# MSE of forecast error with Holt-Winter Method
mse_forecast_error_HoltS = errors(y_test, holtf_seasonal, h, 'mse')
print('MSE of Forecast Errors with Holt Seasonal Method: ',mse_forecast_error_HoltS)

# Variance of errors with Holt-Winter Seasonal Method
prediction_error_holtS = errors(y_train, seasonal_predict, 1, 'error')
forecast_error_holtS = errors(y_test, holtf_seasonal, 5, 'error')

var_pred_holtS = round((np.var(prediction_error_holtS)), 2)
var_forecast_holtS = round((np.var(forecast_error_holtS)), 2)

print('\nVariance of Prediction Error with Holt Seasonal Method', var_pred_holtS)
print('Variance of Forecast Error with Holt Seasonal Method', var_forecast_holtS)
#%%
var_ratio_holt = var_pred_holtS/var_forecast_holtS
print("The Ratio of Variance is: \n", var_ratio_holt)

#%%
# ACF Plot of the residuals
# Holt-Seasonal Method
autocorr(forecast_error_holtS, 50)

#%%
# Q-Value:
T = len(bike)
box_pierce_holtS = box_pierce(prediction_error_holtS, T, 48)
print("The Q value (Box-Pierce Test) with Holt Seasonal, for the prediction error is: ", round((box_pierce_holtS),2))
#%%

# Correlation-Coefficient:
print("The Correlation Coefficient with Holt Seasonal Method is ", correlation_coefficient(forecast_error_holtS, y_test))

#%%
# Feature Selection
X = bike.drop(['count', 'datetime', 'casual','registered'], axis=1)
y = bike['count']

from sklearn.preprocessing import MinMaxScaler

# The StandardScaler
mms = MinMaxScaler()

#%%
# Normalize the training data


normalized_temp = X['temp'].values.reshape((len(X['temp']), 1))
scaler = MinMaxScaler(feature_range=(0, 1))
X['temp'] = scaler.fit_transform(normalized_temp)

normalized_atemp = X['atemp'].values.reshape((len(X['atemp']), 1))
scaler = MinMaxScaler(feature_range=(0, 1))
X['atemp'] = scaler.fit_transform(normalized_atemp)

normalized_humidity = X['humidity'].values.reshape((len(X['humidity']), 1))
scaler = MinMaxScaler(feature_range=(0, 1))
X['humidity'] = scaler.fit_transform(normalized_humidity)

normalized_windspeed = X['windspeed'].values.reshape((len(X['windspeed']), 1))
scaler = MinMaxScaler(feature_range=(0, 1))
X['windspeed'] = scaler.fit_transform(normalized_windspeed)


#%%

# Train-Test Split

# Q1. Train-Test Split
X = sm.add_constant(X) # Adding the constant 1 at the beginning
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = False, test_size=0.2)

h = len(y_test)


#%%
# =================================
# Collinearity Detection
# ================================

# Constructing the Matrices
XMat = X_train.values
YMat = y_train.values


#%%

# Singular Value Decomposition
H = np.matmul(XMat.T, XMat)
_, d, _ = np.linalg.svd(H)
print("Singular Values of X are: \n",d)

# Based on the singular values, we can state that there is collinearity between at-least 4 features.
# This is based on the lowest values from SVD, which are close to zero.
# These can be eliminated from our model.
#%%

# Condition Number
print(f'The condition Number for X is {round(LA.cond(XMat), 2)}')

# The Condition Number is greater than 1000, showing severe degree of collinearity

#%%
# Model Summary using OLS Function:

# =======================================
# Finding Coefficients using OLS Method
# =========================================


model = sm.OLS(y_train, X_train).fit()
print(model.summary())

#%%
# ================================
# Backward Stepwise Regression
# ================================

# The "weather" feature has the highest p-value at 0.583, from the above summary, implying it is not significant.
# Removing the "weather" feature

X_train.drop(['weather'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

#%%

# Removing "workingday" which is at p-value: 0.394

X_train.drop(['workingday'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

#%%

# Removing "holiday" which is at p-value: 0.371
X_train.drop(['holiday'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

#%%

# Removing "temperature" as it is collinear with "atemp"

X_train.drop(['temp'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

#%%
# Removing "season" at p-value: 0.149
# Final Model

X_train.drop(['season'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

#%%
# ==========================
# Q8. Predictions and plots.
# ==========================

X_test = X_test[['const','atemp','humidity','windspeed']]

predictionOLS = model.predict(X_train)
forecastOLS = model.predict(X_test)

plt.plot(y_train, label = "Training Set") # X_train,
plt.plot(y_test, label = "Test Set") #X_test,
plt.plot(forecastOLS, label = "Prediction Values") # X_test,

plt.title("Forecast using OLS Regression")
plt.xlabel("Number of Samples")
plt.ylabel("Bike Rentals")
plt.legend()
plt.show()

#%%

# ===================================
# Prediction errors and ACF Plot
# ==================================
prediction_errorsOLS = errors(y_train, predictionOLS, 1, 'error')
autocorr(prediction_errorsOLS, 48, True)

#%%

# =================================
# Forecast Errors and ACF Plot
# =================================
forecast_errorsOLS = errors(y_test, forecastOLS, h, 'error')
autocorr(forecast_errorsOLS, 48, True)


#%%
# =====================
# Mean Squared Errors:
# =====================

# MSE of forecast error with Holt-Winter Method
mse_forecast_errorOLS = errors(y_test, forecastOLS, h, 'mse')
print('MSE of Forecast Errors with OLS Method: ',mse_forecast_errorOLS)

#%%
T = len(bike)
box_pierceOLS = box_pierce(prediction_errorsOLS, T, 48)
print("The Q value (Box-Pierce Test) with OLS Method, for the prediction error is: ", round((box_pierceOLS),2))
#%%

# Correlation-Coefficient:
print("The Correlation Coefficient with OLS Method is ", correlation_coefficient(forecast_errorsOLS, y_test))


#%%
# ==============================================================
# Estimated Variance of prediction errors and forecast errors
# ================================================================

SSE_predictionOLS = errors(y_train, predictionOLS, 1, 'sse')
SSE_forecastOLS = errors(y_test, forecastOLS, len(y_test), 'sse')

# k = Number of predictors
# T = Number of observations

T_trainSet = len(predictionOLS) # Number of samples of training set
T_testSet = len(forecastOLS) # Number of samples of test set

k = X_train.shape[1] # Number of predictors

var_predictOLS = ((SSE_predictionOLS/(T_trainSet-k-1))**0.5)
var_forecastOLS = ((SSE_forecastOLS/(T_testSet-k-1))**0.5)

print("The Estimated Variance for Prediction Errors is: ", round(var_predictOLS, 2))
print("The Estimated Variance for Forecast Errors is: ", round(var_forecastOLS, 2))

#%%
var_ratio_ols = var_predictOLS/var_forecastOLS
print("The Variance Ratio is: ", var_ratio_ols)

#%%
# T-Test and F-Test:

# T-Test:

print(model.params)
print(model.pvalues)
print(model.tvalues)

#%%
# F Test:
A = np.identity(len(model.params))
A = A[1:,:]
print(model.f_test(A))

print(model.fvalue)
print(model.f_pvalue)

#%%

# Base Models:
# Train-test Split:

X = bike.drop(['count'], axis=1)
y = bike['count']

y_train, y_test = train_test_split(y, shuffle= False, test_size=0.2)
X_train, X_test = train_test_split(X, shuffle= False, test_size=0.2)

h = len(y_test)
#%%

# Average Method
y_hat_train_avg, y_hat_test_avg = average_forecast(y_train, h)

# Plot

plt.plot(X_train['datetime'], y_train, label = "Training Set")
plt.plot(X_test['datetime'], y_test, label = "Test Set")
plt.plot(X_test['datetime'], y_hat_test_avg, label = "h-step prediction")

plt.title("Forecast using Average Method")
plt.xlabel("Months")
plt.ylabel("Number of Bike Rentals")
plt.legend()
plt.show()

#%%

# MSE for Forecast Error with Average Method
mse_forecast_error_avg = errors(y_test, y_hat_test_avg, h, 'mse')
print('MSE of Forecast Errors with Average Method: ', mse_forecast_error_avg)
#%%

# Variance of errors with Average Method:
prediction_error_avg = errors(y_train, y_hat_train_avg, 1, 'error')
forecast_error_avg = errors(y_test, y_hat_test_avg, h, 'error')

var_pred_avg = round((np.var(prediction_error_avg)), 2)
var_forecast_avg = round((np.var(forecast_error_avg)), 2)

print('Variance of Prediction Error with Average Method', var_pred_avg)
print('Variance of Forecast Error with Average Method', var_forecast_avg)

#%%
var_ratio_avg = var_pred_avg/var_forecast_avg
print("The Variance Ratio is: ", var_ratio_avg)

#%%
# Average Method - ACF
autocorr(prediction_error_avg, 48, plot = True)

#%%

T = len(bike)
box_pierce_avg = box_pierce(prediction_error_avg, T, 48)
print("The Q value (Box-Pierce Test) with Average Method, for the prediction error is: ", round((box_pierce_avg),2))

#%%

# Correlation Coefficients for Forecast Errors and Test Set
print("The Correlation Coefficient with Average Method is ",correlation_coefficient(y_test, forecast_error_avg))

#%%

# Naive Method:

y_hat_train_naive, y_hat_test_naive = naive_method(y_train, h)

plt.plot(X_train['datetime'], y_train, label = "Training Set")
plt.plot(X_test['datetime'], y_test , label = "Test Set")
plt.plot(X_test['datetime'], y_hat_test_naive, label = "h-step prediction")

plt.title("Forecast using Naive Method")
plt.xlabel("Months")
plt.ylabel("Number of Bike Rentals")
plt.legend()
plt.show()

#%%
# MSE of forecast error with Naive Method
mse_forecast_error_naive = errors(y_test, y_hat_test_naive, h, 'mse')
print('MSE of Forecast Errors with Naive Method: ',mse_forecast_error_naive)


# Variance of Errors with Naive Method:
prediction_error_naive = errors(y_train, y_hat_train_naive, 1, 'error')
forecast_error_naive = errors(y_test, y_hat_test_naive, h, 'error')

var_pred_naive = round((np.var(prediction_error_naive)), 2)
var_forecast_naive = round((np.var(forecast_error_naive)), 2)

print('\nVariance of Prediction Error with Naive method', var_pred_naive)
print('Variance of Forecast Error with Naive Method', var_forecast_naive)

#%%

var_ratio_naive = var_pred_naive/var_forecast_naive
print("The Variance Ratio is: ", var_ratio_naive)

#%%

# Naive Method - ACF
autocorr(prediction_error_naive, 48, plot = True)

box_pierce_naive = box_pierce(prediction_error_naive, T, 48)
print("The Q value (Box-Pierce Test) with Naive Method, for the prediction error is: ", round((box_pierce_naive),2))

print("The Correlation Coefficient with Naive Method is ", correlation_coefficient(y_test, forecast_error_naive))

#%%
# Drift Method

y_hat_train_drift, y_hat_test_drift = drift_method(y_train, h)

plt.plot(X_train['datetime'], y_train, label = "Training Set")
plt.plot(X_test['datetime'], y_test , label = "Test Set")
plt.plot(X_test['datetime'], y_hat_test_drift, label = "Forecast")

plt.title("Forecast using Drift Method")
plt.xlabel("Observations")
plt.ylabel("Number of Bike Rentals")
plt.legend()
plt.show()

#%%
# MSE of forecast error with Drift Method
mse_forecast_error_drift = errors(y_test, y_hat_test_drift, h, 'mse')
print('MSE of Forecast Errors with Drift Method: ',mse_forecast_error_drift)

#%%
# Variance of errors with Drift method:

prediction_error_drift = errors(y_train, y_hat_train_drift, 1, 'error')
forecast_error_drift = errors(y_test, y_hat_test_drift, 5, 'error')

var_pred_drift = round((np.var(prediction_error_drift)), 2)
var_forecast_drift = round((np.var(forecast_error_drift)), 2)

print('\nVariance of Prediction Error with Drift Method', var_pred_drift)
print('Variance of Forecast Error with Drift Method', var_forecast_drift)

#%%
var_ratio_drift = var_pred_drift/var_forecast_drift
print("The Variance Ratio is: ", var_ratio_drift)


# Drift Method
autocorr(prediction_error_drift, 48, plot = True)
#%%
box_pierce_drift = box_pierce(prediction_error_drift, T, 48)
print("The Q value (Box-Pierce Test) with Drift Method, for the prediction error is: ", round((box_pierce_drift),2))

#%%
print("The Correlation Coefficient with Drift Method is ", correlation_coefficient(y_test, forecast_error_drift))

#%%

# SES Method
y_hat_train_ses, y_hat_test_ses = ses_method(y_train, 0.10, y_train[0], h)
#%%
plt.plot(X_train['datetime'], y_train, label = "Training Set")
plt.plot(X_test['datetime'], y_test , label = "Test Set")
plt.plot(X_test['datetime'], y_hat_test_ses, label = "h-step prediction")
plt.title("Forecast using SES Method at alpha=0.10")
plt.xlabel("Year")
plt.ylabel("Number of Bike Rentals")
plt.legend()
plt.show()

#%%
# MSE of forecast error with SES Method at alpha 0.10
mse_forecast_error_ses = errors(y_test, y_hat_test_ses, h, 'mse')
print('MSE of Forecast Errors with SES Method: ',mse_forecast_error_ses)

#%%

# Variance of errors with SES Method with alpha at 0.10
prediction_error_ses = errors(y_train, y_hat_train_ses, 1, 'error')
forecast_error_ses = errors(y_test, y_hat_test_ses, h, 'error')

var_pred_ses = round((np.var(prediction_error_ses)), 2)
var_forecast_ses = round((np.var(forecast_error_ses)), 2)

print('\nVariance of Prediction Error with SES Method', var_pred_ses)
print('Variance of Forecast Error with SES Method', var_forecast_ses)

#%%
var_ratio_ses = var_pred_ses/var_forecast_ses
print("The Variance Ratio is: ", var_ratio_ses)


#%%

# SES Method with alpha = 0.10 - ACF
autocorr(prediction_error_ses, 48, plot = True)

#%%
box_pierce_ses = box_pierce(prediction_error_ses, T, 48)
print("The Q value (Box-Pierce Test) with SES Method (at alpha=0.10), for the prediction error is: ", round((box_pierce_ses),2))


#%%
# Correlation Coefficients for Forecast Errors and Test Set
print("The Correlation Coefficient with SES Method is ", correlation_coefficient(forecast_error_ses, y_test))

#%%

# ARIMA/SARIMA Process

X = bike.drop(['count'], axis=1)
y = bike['count']

# Train-Test Split

y_train, y_test = train_test_split(y, shuffle= False, test_size=0.2)
X_train, X_test = train_test_split(X, shuffle= False, test_size=0.2)

h = len(y_test)

#%%
ACF_PACF_Plot(FirstLogDiff, 75)


#%%
# First Log Diff

ry,_, _ = autocorr(FirstLogDiff, 100, plot=False)

# 7x7 GPAC
gpac = calc_gpac(ry, 7, 7)


#%%

# 30 x 30
gpac = calc_gpac(ry, 26, 26)

#%%
print(gpac.to_string())

#%%
# Fitting the model:

# Order From Logarithmic First Differencing with 24:
#non_seasonalOrder = (1,0,2)
#seasonalOrder = (0,1,2,24)

# Alternative: 2 seasonal differencing + 1 non-seasonal differencing
non_seasonalOrder = (2,1,2)
seasonalOrder = (0,2,1,24)

model = sm.tsa.statespace.SARIMAX(y_train, order=non_seasonalOrder, seasonal_order=seasonalOrder)

#%%
result = model.fit()

#%%
print(result.summary())


#%%
print("The estimated parameters are: \n", result.params)

#%%

# Confidence Interval:
print("The Confidence Interval is: \n",result.conf_int())


#%%
# One-Step Prediction

train_predict=result.get_prediction(start=1, end=len(y_train), dynamic=False)
sarima_pred=train_predict.predicted_mean

#%%
#
plt.plot(X_train['datetime'], y_train, label = "Training Set")
#plt.plot(X_test['datetime'], y_test , label = "Test Set")
plt.plot(X_train['datetime'], sarima_pred, label = "One-Step Prediction")

plt.title("One-Step Prediction using ARIMA Method")
plt.xlabel("Observations")
plt.ylabel("Scale")
plt.legend()
plt.show()

#%%
residuals = errors(y_train, sarima_pred, 1, 'error')

# ACF Plot:
autocorr(residuals, 75, plot = True)

ACF_PACF_Plot(residuals, 75)

#%%

# Q Values
T = len(y_train)
Qvalue = box_pierce(residuals, T, 48)
print("The Q Value is", Qvalue)

#%%

Q_value = sm.stats.acorr_ljungbox(residuals, lags=[48]) #, return_df=True)

print("The Q Value is", Q_value)

#%%
# Chi Test
alpha = 0.01
DOF = 1 + 48
chi_critical = chi2.ppf(1 - alpha, DOF)
chi_critical

print("The chi critical value is:", chi_critical)
if chi_critical > Qvalue:
    print("The residual passes the whiteness test")
else:
    print("The residual does not pass the whiteness test")


#%%
# Variance:

var_residual_arima = np.var(residuals)
print("The variance of residual error is: \n", var_residual_arima)


#%%

# h-step prediction

test_forecast=result.get_prediction(start=len(y_train)+1, end=len(bike), dynamic=False)
sarima_forecast=test_forecast.predicted_mean


#%%

plt.plot(X_test['datetime'], y_test , label = "Test Set")
plt.plot(X_test['datetime'], sarima_forecast, label = "h-Step Prediction")
plt.title("Forecast using ARIMA Method")
plt.xlabel("Observations")
plt.ylabel("Number of Bike Rentals")
plt.legend()
plt.show()

#%%

plt.plot(X_train['datetime'], y_train, label = 'Train')
plt.plot(X_test['datetime'], y_test , label = "Test Set")
plt.plot(X_test['datetime'], sarima_forecast, label = "h-Step Prediction", alpha=0.50)
plt.title("Forecast using ARIMA Method")
plt.xlabel("Observations")
plt.ylabel("Number of Bike Rentals")
plt.legend()
plt.show()

#%%
ACF_PACF_Plot(sarima_forecast, 75)

#%%
# MSE of forecast error with ARIMA method
mse_forecast_error_sarima = errors(y_test, sarima_forecast, len(y_test), 'mse')
print('MSE of Forecast Errors with ARIMA Method: ',mse_forecast_error_sarima)

#%%
forecast_errors_arima = errors(y_test, sarima_forecast, 1, 'error')


# ACF Plot:
autocorr(forecast_errors_arima, 75, plot = True)

#%%
ACF_PACF_Plot(forecast_errors_arima, 75)

#%%
# Variance:
var_forecast_arima = np.var(forecast_errors_arima)
print("The variance of forecast error is: \n", var_forecast_arima)

#%%
var_ratio_arima = var_residual_arima/var_forecast_arima
print("The Variance Ratio is: ", var_ratio_arima)

#%%

# Correlation Coefficients for Forecast Errors and Test Set
print("The Correlation Coefficient with ARIMA Method is ", correlation_coefficient(forecast_errors_arima, y_test))

#%%


