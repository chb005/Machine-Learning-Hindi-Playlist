#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df = pd.read_csv("carguj.csv")
df.head()


# In[6]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


#car average vs Selling price


# In[8]:


plt.scatter(df['Average'],df['Sell Price($)'])


# In[9]:


plt.scatter(df['Old(yrs)'],df['Sell Price($)'])


# In[11]:


X = df[['Average','Old(yrs)']]


# In[12]:


y = df['Sell Price($)']


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3) 


# In[14]:


X_train


# In[15]:


X_test


# In[16]:


y_train


# In[17]:


y_test


# In[18]:


from sklearn.linear_model import LinearRegression
tst = LinearRegression()
tst.fit(X_train, y_train)


# In[19]:


X_test


# In[20]:


tst.predict(X_test)


# In[21]:


y_test


# In[24]:


tst.score(X_test,y_test)


# In[ ]:





# In[ ]:




