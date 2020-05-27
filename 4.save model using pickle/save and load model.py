#!/usr/bin/env python
# coding: utf-8

# # Save and Load Trained Model using Pickle

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[2]:


df = pd.read_csv("newh.csv")
df.head()


# In[3]:


newm= linear_model.LinearRegression()
newm.fit(df[['area']],df.price)


# In[4]:


newm.coef_


# In[5]:


newm.intercept_


# In[7]:


newm.predict([[4750]])


# using pickle we can save the model

# In[8]:


import pickle


# In[9]:


with open('new_train_model','wb') as file:
    pickle.dump(newm,file)


# use pickle model

# In[12]:


with open('new_train_model','rb') as linear:
    nm = pickle.load(linear)


# In[13]:


nm.coef_


# In[15]:


nm.intercept_


# In[16]:


nm.predict([[4750]])


# Using joblib we can also save the model

# In[17]:


from sklearn.externals import joblib


# In[19]:


joblib.dump(newm, 'new_model_joblib')


# load joblib model

# In[20]:


nejob = joblib.load('new_model_joblib')


# In[21]:


nejob.coef_


# In[22]:


nejob.intercept_


# In[23]:


nejob.predict([[4750]])


# In[ ]:




