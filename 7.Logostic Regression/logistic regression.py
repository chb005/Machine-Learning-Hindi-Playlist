#!/usr/bin/env python
# coding: utf-8

# We are Predicting if a person will buy LIC insurnace based on his age using logistic regression.(Two outcomes:YES/NO)

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("ins_data.csv")
df.head()


# In[3]:


plt.scatter(df['age'],df['bought_insurance'],marker='*',color='blue')


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(df[['age']],df['bought_insurance'],train_size=0.85)


# In[9]:


X_test


# In[10]:


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()


# In[11]:


lg.fit(X_train,y_train)


# In[12]:


X_test


# In[13]:


ypred=lg.predict(X_test)


# In[15]:


lg.predict(X_test)


# In[14]:


lg.predict_proba(X_test)


# In[16]:


lg.score(X_test,y_test)


# In[17]:


X_test


# In[21]:


lg.predict(X_test)


# In[23]:


lg.predict([[12]])


# In[24]:


lg.predict([[71]])


# In[25]:


lg.predict([[39]])


# In[26]:


lg.predict([[32]])


# In[ ]:




