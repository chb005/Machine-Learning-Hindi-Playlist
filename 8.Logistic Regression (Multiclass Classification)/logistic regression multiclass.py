#!/usr/bin/env python
# coding: utf-8

# # Multiclass classification using logistic regression|
# 

# In[1]:


from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
digits = load_digits()


# In[2]:


plt.gray() 


# In[3]:


plt.matshow(digits.images[1]) 


# In[4]:


for i in range(6):
    plt.matshow(digits.images[i])


# In[5]:


dir(digits)


# In[7]:


digits.data[2]  #for 2 array representation


# In[9]:


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()


# In[11]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size=0.25)


# In[14]:


lg.fit(x_train,y_train)


# In[15]:


lg.score(x_test,y_test)


# In[16]:


lg.predict(digits.data[0:4])


# In[17]:


ypred=lg.predict(x_test)


# In[20]:


from sklearn.metrics import confusion_matrix
comt = confusion_matrix(y_test, ypred)
comt


# In[22]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(comt, annot=True)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')


# In[ ]:




