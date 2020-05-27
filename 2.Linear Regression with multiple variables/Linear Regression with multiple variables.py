#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[4]:


df = pd.read_csv('multiprice.csv')
df


# In[5]:


df.bedrooms.median()


# In[6]:


df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())
df


# In[7]:


reg = linear_model.LinearRegression()
reg.fit(df.drop('price',axis='columns'),df.price)


# In[8]:


reg.coef_


# In[9]:


reg.intercept_


# Claculate the price of home with 3200 sqr ft area, 4 bedrooms, 25 year old

# In[10]:


reg.predict([[3200, 4, 25]])


# In[11]:


130.26781094*3200+1657.53838192*4+ -5482.93657639*25


# In[ ]:




