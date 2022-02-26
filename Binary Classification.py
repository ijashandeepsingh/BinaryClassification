#!/usr/bin/env python
# coding: utf-8

# ## Binary Classification
# 
# #### Logistic Regression following Sigmoid Function 
# #### Libraries used: Pandas, MatplotLib
# #### Author: Jashandeep Singh 

# In[3]:


import pandas as pd
from matplotlib import pyplot as ppt


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


df = pd.read_csv("insurance_data.csv")


# In[26]:


del df["Unnamed: 2"] #deleting redundant column 


# In[28]:


df = df.drop([26]) # deleting the last redundant row 


# In[29]:


df


# In[30]:


ppt.scatter(df.age, df.bought_insurance, marker= "*", color = "blue")


# Moving with the prediction using scikit learn library, once we have the data frame ready. 

# In[13]:


# !pip install -U scikit-learn


# In[14]:


from sklearn.model_selection import train_test_split


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.8)


# In[33]:


X_test


# In[34]:


X_train


# In[36]:


from sklearn.linear_model import LogisticRegression


# In[37]:


model = LogisticRegression()


# In[38]:


model.fit(X_train, y_train)


# In[39]:


X_test


# In[44]:


predicted = model.predict(X_test)


# In[46]:


model.predict_proba(X_test)


# In[47]:


model.score(X_test, y_test)


# In[48]:


predicted


# In[49]:


X_test


# In[50]:


model.coef_ # model.coef_ indicates value of m in y=m*x + b equation


# In[51]:


model.intercept_ # model.intercept_ indicates value of b in y=m*x + b equation


# In[56]:


# Lets defined sigmoid function now and do the math with hand
import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# In[57]:


def prediction_function(age):
    z = 0.042 * age - 1.53 # 0.04150133 ~ 0.042 and -1.52726963 ~ -1.53
    y = sigmoid(z)
    return y


# In[59]:


age = 35

prediction_function(age)


# 0.485 is less than 0.5 which means person with 35 age will not buy insurance

# In[ ]:




