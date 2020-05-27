#!/usr/bin/env python
# coding: utf-8

# # One Hot Encoding

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("gujhome.csv")
df


# Create dummy variabl- pandas  

# In[3]:


dummies = pd.get_dummies(df.city)
dummies


# In[4]:


df_dummies= pd.concat([df,dummies],axis='columns')
df_dummies


# In[5]:


df_dummies.drop('city',axis='columns',inplace=True)
df_dummies


# In[6]:


df_dummies.drop('Mehsana',axis='columns',inplace=True)
df_dummies


# In[7]:


X = df_dummies.drop('price',axis='columns')
y = df_dummies.price

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)


# In[8]:


X


# In[9]:


model.predict(X)


# In[10]:


model.score(X,y)


# In[11]:


model.predict([[3400,0,0]])


# In[12]:


model.predict([[2800,0,1]])


# # onehotencoding using sklearn

# we may use label encoder to convert city into numbers

# In[13]:


from sklearn.preprocessing import LabelEncoder
leb = LabelEncoder()


# In[14]:


dfle = df
dfle.city= leb.fit_transform(dfle.city)
dfle


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




