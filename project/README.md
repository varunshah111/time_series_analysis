## Predicting Bike Sharing Demand - Time Series Analysis 

##### This project consists of Time-Series Analysis of Bike Sharing Demand over time. In this project, we will first understand the data and achieve stationarity by using various transformation methods. We then use Time-Series Decomposition methods to test if the data is completely detrended and seasonality has been adjusted. This is done by using the STL-Decomposition Method. We then begin by using simple forecasting methods on our original data, including the base models such as Holt-Winter Method, Average Method, Naïve Method, Drift and Simple Exponential Smoothing Method. We then use the OLS Regression Method, here we first check the collinearity among the features and then use Backward Stepwise regression for Feature Selection and to get our best model. We then develop an ARIMA model on the transformed data, we first determine the order of the AR and MA processes using ACF/PACF Plots and GPAC, and then estimate the parameters for the model. We evaluate all these models using the Mean- Squared Error of Forecast error, Variance of errors, Q-Value and Correlation Coefficients. We also plot the ACF Plots for each residual errors to see if the residuals fall under white noise.


| Features |  Description                                                           | 
| :-------- | :-----------------------------------------------------------------------------|
| datetime | Hourly Data with timestamp  |
| season   | 1: spring, 2: summer, 3:fall, 4:winter         |
| holiday |   Whether the day is considered holiday                                                                      | 
| workingday| Whether the day is neither the weekend or holiday   |
| weather | 1: Clear/Partly Cloudy/Few Clouds |
| | 2: Mist + Cloudy, Mist+Few Clouds |
| | 3: Light Snow, Light rain + Thunderstorm, Light rain + scattered clouds  |
| | 4: Heavy rain + ice Pallets + Thunderstorm + Mist/fog, Snow + fog |
| temp | Temperature in Celsius. | 
| a_temp | “feels like” temperature in Celsius | 
| humidity | relative humidity | 
| windspeed | wind speed | 
| Stage | Stage of Rule (Eg: Proposed Rule, Final Rule) | 
| casual |  Number of non-registered user rentals initiated  |
| registered |   Number of registered user rentals initiated    |
| count | Number of Total rentals. |

Source: https://www.kaggle.com/competitions/bike-sharing-demand/overview

##### Note: the file toolbox.py includes all the functions defined necessary for this project. This code is not uploaded here on git to avoid any kind of plagiarism.
