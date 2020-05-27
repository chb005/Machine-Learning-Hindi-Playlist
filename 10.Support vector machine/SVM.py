#!/usr/bin/env python
# coding: utf-8

# # SVM

# In[3]:


import pandas as pd
from sklearn.datasets import load_iris
flw = load_iris()


# In[4]:


flw.feature_names


# In[5]:


flw.target_names


# In[6]:


df = pd.DataFrame(flw.data,columns=iris.feature_names)
df.head()


# In[7]:


df['target'] =flw.target
df.head()


# In[8]:


df[df.target==1].head()


# In[9]:


df[df.target==2].head()


# In[10]:


df['flower_name'] =df.target.apply(lambda x: flw.target_names[x])
df.head()


# In[12]:


df[95:105]


# In[13]:


df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]


# In[14]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="blue",marker='*')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="red",marker='+')


# In[16]:


plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='*')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="red",marker='+')


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X = df.drop(['target','flower_name'], axis='columns')
y = df.target


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


# In[21]:


len(X_train)


# In[22]:


len(X_test)


# In[24]:


from sklearn.svm import SVC
svc = SVC()


# In[25]:


svc.fit(X_train, y_train)


# In[26]:


svc.score(X_test, y_test)


# In[27]:


svc.predict([[4.6,3.4,1.8,0.7]])


# In[28]:


model_C = SVC(C=1)
model_C.fit(X_train, y_train)
model_C.score(X_test, y_test)


# In[29]:


model_C = SVC(C=10)
model_C.fit(X_train, y_train)
model_C.score(X_test, y_test)


# In[30]:


model_g = SVC(gamma=10)
model_g.fit(X_train, y_train)
model_g.score(X_test, y_test)


# In[31]:


model_linear_kernal = SVC(kernel='linear')
model_linear_kernal.fit(X_train, y_train)


# In[32]:


model_linear_kernal.score(X_test, y_test)


# In[ ]:




