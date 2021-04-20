#!/usr/bin/env python
# coding: utf-8

# In[5]:


# For data manipulation
import pandas as pd

# for scientific computation
import numpy as np

# for data analysis
from sklearn.preprocessing import StandardScaler 
from sklearn import linear_model
import sklearn.metrics as sm

# for diagramming 
import matplotlib.pyplot as plt
import seaborn as sns

# For serialization and deserialization of data from/to file
import pickle


# ## Step 2: Data Preparation

# ### 2.1 Read data

# In[28]:


# read the data
df = pd.read_csv("student_scores.csv", sep = ',')


# In[29]:


df.shape


# In[30]:


# get idea of the look
df.head()


# In[31]:


# see which are the attribute labels
list(df)


# In[32]:


# get idea of columns and types
df.info()


# ### 2.2 Get Asquainted with Descriptive Statistics

# In[33]:


# get idea of basic statistical parameters for each column
df.describe()


# In[34]:


# if you want to change the format, for example to avoid scientific notation, e.g. e+04
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[35]:


# plot all
plt.ylabel('Scores')
plt.xlabel('Hours')
plt.scatter(df.Scores, df.Hours, color='green')
plt.show()


# In[39]:


# sns.histplot(df['age'],  label='age')  
sns.distplot(df['Scores'],  label='Scores', norm_hist=True)  


# In[40]:


sns.distplot(df['Hours'],  label='Hours', norm_hist=True) 


# ### 2.4 Investigate the Inter-Dependencies of the Features
# Create a correlation matrix to see which features determine the output at most, as well as whether there are some correlated features. <br>
# If two features are correlated, only one of them can represent both.

# In[43]:


corr_matrix = df.corr()
corr_matrix


# In[44]:


# plot the matrix as a heat map
plt.subplots(figsize = (10, 8))
sns.heatmap(corr_matrix, annot=True)


# ## Step 3: Train a Model

# ### 3.1 Split the Data in Dependent y and Independent Data Sets

# In[46]:


# Split X and y
X, y = df.Hours, df.Scores


# In[47]:


X


# In[48]:


y


# In[49]:


# plot all
plt.ylabel('y')
plt.xlabel('X')
plt.scatter(X, y, color='blue')
plt.show()


# ### 3.2 Split the Data in Training and Testing Sets

# In[50]:


# split the set into subsets for training and testing
from sklearn.model_selection import train_test_split

# default proportion is 75:25
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.4) 


# In[51]:


# the shape of the subsets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[52]:


# randomly selected
y_train


# ### 3.3 Select a Method, Create a Model

# In[53]:


# build a model with method 'polyfit'
model = np.polyfit(X_train, y_train, 1)


# In[54]:


# get the result of fitting the regression line on the train data
model


# In[55]:


# apply the model to the test data
test = np.polyfit(X_test, y_test, 1)
test


# ## Step 4: Test the Model

# ### 4.1 Test with Known Data

# In[56]:


predict = np.poly1d(model)


# In[57]:


y_predicted = predict(X_test)


# In[71]:


plt.scatter(X, y, color='blue') 
plt.plot(X_test, y_predicted, c = 'r')


# In[72]:


y_predicted


# In[73]:


y_test


# In[74]:


# MAE
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_predicted)
print(mae)


# In[75]:


# MSE
mse = metrics.mean_squared_error(y_test, y_predicted)
print(mse)


# In[76]:


# RMSE
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_predicted))
print(rmse)


# Comparing these metrics:
# 
# - **MAE** is the easiest to understand, because it's the average error
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units

# ### Calculate R-squared

# In[77]:


# Explained variance score: 1 is perfect prediction
eV = round(sm.explained_variance_score(y_test, y_predicted), 2)
print('Explained variance score ',eV )


# In[78]:


# R-squared
from sklearn.metrics import r2_score
r2_score(y, predict(X))
# r2_score(y_test, y_predicted)


# The model tested the algorith to a 0.95~ out of 1, which means it should be around 95% accurate at this point. Which is a good result. Depending on what data you're working with and what type of model and algorithm used, you can improve the R2 Squared by training the model further. One of the methods being, running multiple if not thousands of epochs. 

# Exercise 7-4 & 7-5. - Question: How does the three different methods differ and where should each method be utilized?

# We use Simple Linear Regression in this exercise cause we are only dealing with one explanatory variable. For 7-4 where Multiple Linear Regression is used, makes sense since we're dealing with more than one explanatory variable. In that case multiple correlated dependent variables are predicted, rather than a single scalar variable. For 7-5 aka. Polynomial Model is used when the latter simply fails to make a optimal fit line (mean) through our data points, which is the case where we have a curvilinear relationship between the dependent and independent variables. 
