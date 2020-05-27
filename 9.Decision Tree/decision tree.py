#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Classification

# In[7]:


import pandas as pd


# In[8]:


df = pd.read_csv("ind_sal.csv")
df.head()


# In[9]:


raw = df.drop('salary_more_then_100k',axis='columns')


# In[10]:


target = df['salary_more_then_100k']


# In[11]:


from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()


# In[12]:


raw['company_ind'] = le_company.fit_transform(inputs['company'])
raw['job_ind'] = le_job.fit_transform(inputs['job'])
raw['degree_ind'] = le_degree.fit_transform(inputs['degree'])


# In[13]:


raw


# In[14]:


raw_n = raw.drop(['company','job','degree'],axis='columns')


# In[15]:


raw_n


# In[16]:


target


# In[17]:


from sklearn import tree
model = tree.DecisionTreeClassifier()


# In[19]:


model.fit(raw_n, target)


# In[20]:


model.score(raw_n,target)


# In[23]:


model.predict([[2,1,0]]) #TCS,Computer prog,B,E   salary >100K


# In[22]:


model.predict([[2,1,1]])


# In[24]:


model.predict([[0,0,1]])


# In[ ]:




