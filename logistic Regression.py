#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#done by Rohit Roy Chowdhury
#MTECH 2years DS


# In[1]:


#make imports
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd


# In[2]:


#load data
data = load_iris()


# In[3]:


#get the data and features
X = data.data
Y = data.target


# In[4]:


#convert into a dataframe
features = ['sepal-length','sepal-with','petal-length','petal-width']
df = pd.DataFrame(data=X, columns =features)
df['target'] = Y
print(df.head())


# In[5]:


#spliting the data into features and classes using splitting
X = df.iloc[:, 0:4].values
X = StandardScaler().fit_transform(X)
Y = df['target'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=32)


# In[8]:


#creating a logistic regression model and fit and predict
lr = LogisticRegression(multi_class='ovr')
lr.fit(X_train,Y_train)
y_predict = lr.predict(X_test)


# In[9]:


#report
print(classification_report(Y_test, y_predict))


# In[ ]:




