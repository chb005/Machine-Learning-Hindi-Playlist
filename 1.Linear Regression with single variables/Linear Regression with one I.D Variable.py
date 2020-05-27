#!/usr/bin/env python
# coding: utf-8

# # Linear Regression With One Variable

# We are going to build a machine learning model that can predict home prices based on square feet area.

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('prices.csv')
df


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df['area'],df['price'],color='blue',marker='*')


# In[6]:


new_price = df.drop('price',axis='columns')
new_price


# In[7]:


price = df.price
price


# In[8]:


reg = linear_model.LinearRegression()
reg.fit(new_price,price)


# In[14]:


reg.predict([[3400]])


# In[15]:


reg.coef_


# In[16]:


reg.intercept_


# Y = m * x + c (m is coefficient and c is intercept)

# In[17]:


135.78767123*3400+180616.43835616432


# In[13]:


reg.predict([[5200]])


# In[21]:


narea_df = pd.read_csv("areas.csv")
narea_df.head()


# In[22]:


p11 = reg.predict(area_df)
p11


# In[23]:


narea_df['prices']=p11
narea_df


# In[24]:


narea_df.to_csv("narea_pred.csv")


# In[ ]:




