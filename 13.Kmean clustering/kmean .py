#!/usr/bin/env python
# coding: utf-8

# #  K Means Clustering

# In[1]:


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("guj_salary.csv")
df.head()


# In[4]:


plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age_Years')
plt.ylabel('Salary($)')


# In[9]:


km = KMeans(n_clusters=3)
ypred = km.fit_predict(df[['Age','Income($)']])
ypred


# In[10]:


df['cluster']=ypred
df.head()


# In[11]:


km.cluster_centers_


# In[12]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age_Years')
plt.ylabel('Salary ($)')
plt.legend()


# In[13]:


scaler = MinMaxScaler()

scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])


# In[14]:


df.head()


# In[16]:


plt.scatter(df.Age,df['Income($)'],color='green')


# In[17]:


km = KMeans(n_clusters=3)
y_pred = km.fit_predict(df[['Age','Income($)']])
y_pred


# In[19]:


df['cluster']=y_pred
df.head()


# In[20]:


km.cluster_centers_


# In[21]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()


# In[22]:


sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)


# In[23]:


plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)


# In[ ]:




