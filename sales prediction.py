#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[2]:


data=pd.read_csv('advertising.csv')


# In[3]:


data


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


print(data.shape)


# In[7]:


data.describe()


# In[8]:


data.info()


# In[9]:


data.isnull().sum()


# In[10]:


#independent variable 
x=data[['Radio']]
y=data[['TV']]
z=data[['Newspaper']]


# In[11]:


#dependent variable
a=data['Sales']


# In[12]:


#train test split of x

X_train, X_test, a_train, a_test = train_test_split(x, a, test_size=0.2, random_state=42)


# In[13]:


regression=LinearRegression()


# In[14]:


regression.fit(X_train, a_train)


# In[15]:


a_pred = regression.predict(X_test)


# In[16]:


mse = mean_squared_error(a_test, a_pred)


# In[17]:


r2 = r2_score(a_test, a_pred)


# In[18]:


print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[19]:


#train test split of y

y_train, y_test, a_train, a_test = train_test_split(y, a, test_size=0.2, random_state=42)


# In[20]:


regression.fit(y_train, a_train)


# In[21]:


a_pred = regression.predict(y_test)


# In[22]:


mse = mean_squared_error(a_test, a_pred)


# In[23]:


r2 = r2_score(a_test, a_pred)


# In[24]:


print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[25]:


#train test split of z

z_train, z_test, a_train, a_test = train_test_split(z, a, test_size=0.2, random_state=42)


# In[26]:


regression.fit(z_train, a_train)


# In[27]:


a_pred = regression.predict(z_test)


# In[28]:


mse = mean_squared_error(a_test, a_pred)


# In[29]:


r2 = r2_score(a_test, a_pred)


# In[30]:


print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[ ]:




