#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_digits
digits = load_digits()


# In[2]:


dir(digits)


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[5]:


plt.gray() 
for i in range(5):
    plt.matshow(digits.images[i])


# In[6]:


df = pd.DataFrame(digits.data)
df.head()


# In[7]:


df['target'] = digits.target


# In[8]:


df[0:12]


# In[9]:


X = df.drop('target',axis='columns')
y = df.target


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15)


# In[11]:


from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators=20)
rf.fit(X_train, y_train)


# In[12]:


rf.score(X_test, y_test)


# In[14]:


ypred = rf.predict(X_test)


# In[15]:


from sklearn.metrics import confusion_matrix
com = confusion_matrix(y_test, ypred)
com


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(com, annot=True)
plt.xlabel('Predicted Values')
plt.ylabel('Truth Values')


# In[ ]:




