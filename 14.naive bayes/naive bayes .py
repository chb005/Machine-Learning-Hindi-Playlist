#!/usr/bin/env python
# coding: utf-8

# Predicting survival from titanic crash

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("titanic.csv")
df.head()


# In[3]:


df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head()


# In[4]:


ints = df.drop('Survived',axis='columns')
target = df.Survived


# In[6]:


dummies = pd.get_dummies(ints.Sex)
dummies.head(3)


# In[9]:


ints = pd.concat([ints,dummies],axis='columns')
ints.head(3)


# In[10]:


ints.drop(['Sex','male'],axis='columns',inplace=True)
ints.head(3)


# In[12]:


ints.columns[ints.isna().any()]


# In[15]:


ints.Age[:10]


# In[38]:


ints.Age = ints.Age.fillna(ints.Age.mean())
ints.head(9)


# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(ints,target,test_size=0.25)


# In[40]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


# In[41]:


model.fit(X_train,y_train)


# In[42]:


model.score(X_test,y_test)


# In[43]:


X_test[0:10]


# In[44]:


y_test[0:10]


# In[45]:


model.predict(X_test[0:10])


# In[46]:


model.predict_proba(X_test[:10])


# In[47]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(GaussianNB(),X_train, y_train, cv=5)
np.average(scores)


# In[ ]:




