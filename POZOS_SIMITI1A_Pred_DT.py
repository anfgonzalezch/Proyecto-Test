#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r"C:\Users\andre\Downloads\volve_wells.csv", usecols=['WELL', 'DEPTH', 'RHOB', 'GR', 'NPHI', 'PEF', 'DT'])


# In[3]:


df['WELL'].unique()


# In[4]:


# Training Wells
training_wells = ['15/9-F-11 B', '15/9-F-11 A', '15/9-F-1 A']

# Test Well
test_well = ['15/9-F-1 B']


# In[5]:


train_val_df = df[df['WELL'].isin(training_wells)].copy()
test_df = df[df['WELL'].isin(test_well)].copy()


# In[6]:


train_val_df.describe()


# In[7]:


test_df.describe()


# In[8]:


train_val_df.dropna(inplace=True)
test_df.dropna(inplace=True)
train_val_df.describe()


# In[9]:


#Modelo Random Forest

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


# In[10]:


#Determinar las curvas de entrenamiento y la curva que se quiere predecir

X = train_val_df[['RHOB', 'GR', 'NPHI', 'PEF']]
y = train_val_df['DT']


# In[11]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


# In[12]:


regr = RandomForestRegressor()


# In[13]:


regr.fit(X_train, y_train)


# In[14]:


y_pred = regr.predict(X_val)


# In[15]:


metrics.mean_absolute_error(y_val, y_pred)


# In[16]:


mse = metrics.mean_squared_error(y_val, y_pred)
rmse = mse**0.5


# In[17]:


rmse


# In[18]:


plt.scatter(y_val, y_pred)
plt.xlim(40, 140)
plt.ylim(40, 140)
plt.ylabel('Predicted DT')
plt.xlabel('Actual DT')
plt.plot([40,140], [40,140], 'black') #1 to 1 line


# In[19]:


#Predicción de las pruebas

test_well_x = test_df[['RHOB', 'GR', 'NPHI', 'PEF']]


# In[20]:


test_df['TEST_DT'] = regr.predict(test_well_x)


# In[21]:


plt.scatter(test_df['DT'], test_df['TEST_DT'])
plt.xlim(40, 140)
plt.ylim(40, 140)
plt.ylabel('Predicted DT')
plt.xlabel('Actual DT')
plt.plot([40,140], [40,140], 'black') #1 to 1 line


# In[22]:


plt.figure(figsize=(15, 5))
plt.plot(test_df['DEPTH'], test_df['DT'], label='Actual DT')
plt.plot(test_df['DEPTH'], test_df['TEST_DT'], label='Predicted DT')
plt.xlabel('Depth (m)')
plt.ylabel('DT')
plt.ylim(40, 140)
plt.legend()
plt.grid()

