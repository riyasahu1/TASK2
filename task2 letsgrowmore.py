#!/usr/bin/env python
# coding: utf-8

# #  TASK2:- Prediction using Decision Tree  Algorithm

# Create the Decision Tree classifier and visualize it graphically. 
# 
# The purpose is if we feed any new data to this classifier, it would be able to  predict the right class accordingly.  

# AUTHOR:- RIYA SAHU,
# INTERN AT LETSGROWMORE,
# DOMAIN:- DATA ANALYST
# 

# In[8]:


#IMPORTING ALL THE REQUIRED LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn import tree


# In[9]:


#import dataset
data = load_iris()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['target'] = data.target


# In[12]:


# Showing first 5 values
df.head()


# In[27]:


# Showing last 5 values
df.tail()


# In[13]:


#checking for null values
df.isnull().sum()


# In[14]:


# No. of rows and columns
df.shape


# In[15]:


# Showing only target data (Dependent Variable)
print(df['target'])


# In[16]:


# splitting data

fc = [x for x in df.columns if x!="target"]
x= df[fc]
y= df["target"]
X_train, X_test, Y_train, Y_test = train_test_split(x,y, random_state = 100, test_size = 0.30)


# In[17]:


# Display of data
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[18]:


#building Desicion tree model

model1 = DecisionTreeClassifier()


# In[19]:


model1.fit(X_train,Y_train)


# In[21]:


Y_pred = model1.predict(X_test)


# In[22]:


data2 = pd.DataFrame({"Actual":Y_test,"Predicted":Y_pred})
data2.head()


# In[23]:


# Testing the accuracy of model prediction
accuracy_score(Y_test,Y_pred)


# In[24]:


# Plotting
f_n = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
c_n = ["Setosa", "Versicolor", "Virginica"]
plot_tree(model1,feature_names = f_n, class_names = c_n , filled = True)


# In[25]:


modelx= DecisionTreeClassifier().fit(x,y)


# In[26]:


plt.figure(figsize = (20,15))
tree = tree.plot_tree(modelx, feature_names = f_n, class_names = c_n, filled = True)


# In[ ]:




