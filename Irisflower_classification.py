#!/usr/bin/env python
# coding: utf-8

# # TASK 1: 
# 
# Iris Flower Classification

# # Import modules

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Loading the dataset

# In[4]:


df = pd.read_csv('E:\\Machine Learning\\IRIS.csv')
df.head()


# In[6]:


# to display stats about data
df.describe()


# In[7]:


# to basic info about datatype
df.info()


# In[9]:


# to display no. of samples on each class
df['pecies'].value_counts()


# # Preprocessing the dataset

# In[10]:


# check for null values
df.isnull().sum()


# # Exploratory Data Analysis

# In[13]:


# histograms
df['sepal_length'].hist()


# In[14]:


df['sepal_width'].hist()


# In[15]:


df['petal_length'].hist()


# In[16]:


df['petal_width'].hist()


# In[17]:


# scatterplot
colors = ['red', 'orange', 'blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']


# In[18]:


for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['sepal_length'], x['sepal_width'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[19]:


for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['petal_length'], x['petal_width'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[20]:


for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['sepal_length'], x['petal_length'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[21]:


for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['sepal_width'], x['petal_width'], c = colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


# # correlation matrix

# In[22]:


df.corr()


# In[23]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')


# # Label Encoder

# In[24]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[25]:


df['species'] = le.fit_transform(df['species'])
df.head()


# # Model Training

# In[26]:


from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['species'])
Y = df['species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# In[27]:


# logistic regression 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[28]:


# model training
model.fit(x_train, y_train)


# In[29]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[30]:


# knn - k-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[31]:


model.fit(x_train, y_train)


# In[32]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[33]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[37]:


model.fit(x_train, y_train)


# In[40]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[41]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','KNeighborsClassifier','Decision Tree'],
    'Score': [97.777,95.555,97.777]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


# In[ ]:




