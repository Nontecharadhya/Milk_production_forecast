#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os


# In[2]:


os.getcwd()


# In[9]:


os.chdir('C:\\Users\\Abhi\\downloads')


# In[11]:


df =pd.read_csv('monthly-milk-production.csv')


# In[12]:


df.head()


# In[22]:


df.dtypes


# In[17]:


df.rename(columns={'Month':'Time','Monthly milk production (pounds per cow)':'Production'},inplace=True)


# In[21]:


df['Time']=pd.to_datetime(df['Time'])


# In[25]:


# Add new columns for month and year
df['month'] = df['Time'].dt.month
df['year'] = df['Time'].dt.year

print(df.tail())


# In[29]:


month_names = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}


# In[30]:


# Convert month numbers to month names
df['month'] = df['month'].map(month_names)

print(df)


# In[31]:


df.index.freq='MS'
df.head()


# In[32]:


# TIME PLOT 
df['Production'].plot(figsize=(20,5))
plt.title('Production',size=24)
plt.show()


# In[35]:


df = df.set_index('Time')
df.head()


# In[37]:


from statsmodels.tsa.seasonal import seasonal_decompose
result =seasonal_decompose (df['Production'],model='add')
result.plot();


# # ARIMA MODEL

# In[42]:


from pmdarima import auto_arima
auto_arima(df['Production'],seasonal=True,m=12).summary()


# In[43]:


len(df)


# In[44]:


start_index=len(df)-12
start_index


# In[45]:


df_train =df.iloc[:start_index]
df_test =df.iloc[start_index:]
df_train.tail()


# In[46]:


len(df_train)


# In[47]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[49]:


model =SARIMAX(df_train['Production'],order=(2,0,0),seasonal_order=(1,1,0,12))
results =model.fit()
results.summary()


# In[56]:


import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


# In[57]:


# Obtain predicted values
start = len(df_train)
end = len(df_train) + len(df_test) - 1
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels')


# In[59]:


predictions
# predictions within the test set only


# In[60]:


# compare predicitons to expected values 
for i in range (len(predictions)):
    print(f'predicited=={predictions[i]:<11.10},expected ={df_test['Production'][i]}')


# In[61]:


# comparing bet predicted and expected values
title ='Monthly Milk Prodcuction'
ylabel ='Milk Prodcution'
xlabel =''
ax=df_test['Production'].plot(legend =True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel,ylabel=ylabel)


# In[64]:


# perform mse
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(df_test['Production'],predictions)
print(f'SARIMA(2,0,0)(1,1,0,12) MSE Error:{mse:11.10}')


# In[65]:


# RMSE is simply the square root of the MSE
import math
math.sqrt(mse)


# In[69]:


# Define and fit SARIMAX model
model = SARIMAX(df['Production'], order=(2, 0, 0), seasonal_order=(1, 1, 0, 12))
result = model.fit()

# Make forecasts
forecast = result.predict(start=len(df), end=len(df) + 11, typ='levels').rename('SARIMAX(2,0,0)(1,1,0,12) Forecast')
print(forecast)


# In[70]:


forecast.to_csv('Forecast values.csv')


# # Thank you
