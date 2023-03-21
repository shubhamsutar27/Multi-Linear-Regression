#!/usr/bin/env python
# coding: utf-8

# age: age of primary beneficiary
# 
# sex: insurance contractor gender, female, male
# 
# bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
# objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
# 
# children: Number of children covered by health insurance / Number of dependents
# 
# smoker: Smoking
# 
# region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
# 
# charges: Individual medical costs billed by health insurance

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('insurance.csv')
df


# In[3]:


df.shape


# In[4]:


df.isna().sum()


# In[5]:


df.duplicated().sum()


# In[6]:


df.drop_duplicates(inplace=True)


# In[7]:


df


# In[8]:


df.describe()


# In[9]:


df['sex'].value_counts()


# In[10]:


df['region'].value_counts()


# In[11]:


df['smoker'].value_counts()


# In[12]:


from matplotlib import cm
color = cm.inferno_r(np.linspace(.4, .8, 30))


# In[13]:


plt.figure(figsize=(10,4))
plt.title("Distribution of age Variable.")
sns.distplot(df['age'],color='#8B1A1A');


# In[14]:


plt.figure(figsize=(10,4))
plt.title("Distribution of bmi Variable.")
sns.distplot(df['bmi'],color='#8B1A1A');


# In[15]:


plt.figure(figsize=(10,4))
plt.title("Distribution of charges Variable.")
sns.distplot(df['charges'],color='#8B1A1A');


# In[16]:


# apply log transformation to Price
df['charges'] = np.log(df['charges'])


# In[17]:


df


# In[18]:


plt.figure(figsize=(10,4))
plt.title("Distribution of charges Variable.")
sns.distplot(df['charges'],color='#8B1A1A');


# In[19]:


f, ax = plt.subplots(1, 1, figsize=(12, 8))
ax = sns.barplot(x='region', y='charges', hue='sex', data=df, palette='cool')


# In[20]:


f, ax = plt.subplots(1, 1, figsize=(10, 6))
ax = sns.barplot(x='region', y='charges', hue='smoker', data=df, palette='cool')


# In[21]:


f, ax = plt.subplots(1, 1, figsize=(12, 8))
ax = sns.barplot(x='region', y='charges', hue='children', data=df, palette='Set1')


# In[22]:


# Count the  number of people with dependents
dependent_count  = df['children'].value_counts()

sns.countplot(df['children'])
plt.xlabel('Number of dependents')
plt.ylabel('Number of people')
plt.title('Number of people with dependents')
plt.show()


# In[23]:


# Create a scatter plot 
sns.scatterplot(data=df,x='children',y='charges')
plt.xlabel('Number of Depedents')
plt.ylabel('Insurance Charges')
plt.title('Insurance charges vs Number of Dependents')
plt.show()


# In[24]:


df.info()


# In[25]:


# Filter the data to only include individuals who have childrens
parents = df[df['children']>0]

# Display the number of people who have insurance
print('Number of people who have insurance: ',len(parents))


# In[26]:


# Display the age of each person who has childrens
print('Age of people who have childrens: ')
print(parents[['age','children']])


# In[27]:


# Create a histogram of the age distribution of people who have children
plt.hist(parents['age'],bins=20)
plt.xlabel('age')
plt.ylabel('Frequency')
plt.title('Age distribution of people with children')
plt.show()


# In[28]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Convert categorical variables to numerical variables using one-hot encoding
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])
X = df.drop(['charges'], axis=1)
Y = df['charges']


# In[29]:


df


# In[34]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


# In[40]:


# Create a linear regression object
model = LinearRegression()

# Train the model on the training data
model.fit(X_train,Y_train)

# Evaluate the model on the testing data
score = model.score(X_test,Y_test)

#Print the coefficient of determination
print('R^2 score: ',score)


# In[36]:


from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error,explained_variance_score


# In[37]:


Y_pred = model.predict(X_test)

# Compute metrics
mse = mean_squared_error(Y_test,Y_pred)
rmse = mean_squared_error(Y_test,Y_pred,squared=False)
mae = mean_absolute_error(Y_test,Y_pred)
evs = explained_variance_score(Y_test,Y_pred)


# Print the metrics
print('MSE: ',mse)
print('RMSE: ',rmse)
print('MAE: ',mae)
print('Explained variance score: ',evs)


# In[ ]:




