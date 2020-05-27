#!/usr/bin/env python
# coding: utf-8

# # KFold Cross Validation

# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# We are checking above models accuracy using cross validation

# In[23]:


from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
digits = load_digits()


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.25)


# In[25]:


lg = LogisticRegression(solver='liblinear',multi_class='ovr')
lg.fit(X_train, y_train)
lg.score(X_test, y_test)


# In[26]:


svm = SVC(gamma='auto')
svm.fit(X_train, y_train)
svm.score(X_test, y_test)


# In[27]:


rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)


# In[49]:


from sklearn.model_selection import KFold
kfc = KFold(n_splits=3)
kfc


# In[50]:


for train_index, test_index in kfc.split([11,22,33,44,55,66,77,88,99]):
    print(train_index, test_index)


# In[8]:


from sklearn.model_selection import cross_val_score


# In[35]:


scores1=cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), digits.data, digits.target,cv=3)
scores1


# In[36]:


np.average(scores1)


# In[37]:


score2=cross_val_score(SVC(gamma='auto'), digits.data, digits.target,cv=3)
score2


# In[40]:


np.average(score2)


# In[42]:


score3=cross_val_score(RandomForestClassifier(n_estimators=40),digits.data, digits.target,cv=3)
score3


# In[43]:


np.average(score3)


# In[44]:


scores1 = cross_val_score(RandomForestClassifier(n_estimators=5),digits.data, digits.target, cv=10)#change number of tress
np.average(scores1)


# In[51]:


scores1 = cross_val_score(RandomForestClassifier(n_estimators=10),digits.data, digits.target, cv=10)#change number of tress
np.average(scores1)


# In[52]:


scores1 = cross_val_score(RandomForestClassifier(n_estimators=15),digits.data, digits.target, cv=10)#change number of tress
np.average(scores1)


# In[53]:


scores1 = cross_val_score(RandomForestClassifier(n_estimators=30),digits.data, digits.target, cv=10)#change number of tress
np.average(scores1)


# In[54]:


scores1 = cross_val_score(RandomForestClassifier(n_estimators=40),digits.data, digits.target, cv=10)#change number of tress
np.average(scores1)


# In[55]:


scores1 = cross_val_score(RandomForestClassifier(n_estimators=50),digits.data, digits.target, cv=10)#change number of tress
np.average(scores1)


# In[ ]:




