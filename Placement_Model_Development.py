#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Import necessary library for analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[3]:


## Load data in pandas df
df = pd.read_csv(r"C:\Users\lenovo\Downloads\DS_Practice\Campus_Recuritment_Model\Placement_Data.csv", index_col=0)
df.head(3)


# ### What column name refers to 
# 
# ssc_p:- Secondary Education(%) 10th Grade
# 
# ssc_b:- 10th Board of Education
# 
# hsc_p:- Higher Secondary Education(%) 12th Grade
# 
# hsc_b:- 12th Board of Education
# 
# hsc_s:- Specialization in Higher Secondary Education
# 
# degree_p:- Undergraduate (%)
# 
# degree_t:- Undergraduate degree type
# 
# workex:- Work experience
# 
# etest_p:- Placement test (%)
# 
# specialisation:- MBA specialisation
# 
# mba_p:- MBA (%)
# 
# status:- Hiring status    

# In[4]:


## Check data type and shape of every column 
df.info()


# ### Find marginal probability through two way table
# 
# Marginal Probability is the probability of the occurence of the single event

# In[5]:


## Marginal Probability beteween categorical variables status & specialisation
pd.crosstab(index = df['specialisation'], columns=df['status'], margins =True, normalize = True)


# The marginal probability shows, out of 56% of Finance candidate 44% gets placed and in HR out of 44% only 24% gets placed that means the success ratio of MBA in Finance specialisation is more than the HR. Now find does work experience is a one of the factor of success for Finance candidate.

# In[6]:


## Marginal Probability beteween categorical variables status & specialisation
pd.crosstab(index = df['degree_t'], columns=df['status'], margins =True, normalize =True)


# ### Which degree specialization is much demanded by corporate?

# In[7]:


sns.countplot(x="status", data=df, hue='specialisation')
plt.title("Degree Specialization vs Candidate Placement")
plt.xlabel("Status of Placement")
plt.ylabel("Number of candidate")
plt.show()


# #### Above plot shows Mkt&Fin specialization is dominating in campus, for placement.

# ### Does work experience affects placement of a candidate?

# In[8]:


sns.countplot(x="status", data=df, hue='workex')
plt.title("Candidate Work Experience in Placement")
plt.xlabel("Status of Placement")
plt.ylabel("Number of candidate")
plt.show()


# #### Acc. to data, most of candidates who have work experience are placed in a company. That means work experience is an important factor of getting placement.

# ### Find which degree technology studied by candidate is placed more 

# In[9]:


sns.countplot(x="degree_t", data=df, hue='status')
plt.title("Candidate degree Technology in Placement")
plt.xlabel("Degree Technology")
plt.ylabel("Number of candidate")
plt.show()


# ### Relationship between placement of a student to their percentage.
# 
# Pairwise Plot :- Used to plot relationship in a dataset
# 
# Creates scatter plots for join relationship and histogram for univariate distributions 
# 

# In[10]:


sns.pairplot(df.drop(["salary"], axis=1), kind="scatter", hue="status")
plt.show()


# ### Check average or mean of numeric variable for unique values of status 

# In[11]:


df.groupby(["status"]).mean()


# #### Check how many students have been placed or not

# In[12]:


df.status.value_counts()


# In[13]:


## Variance Infilation Factor 
# X = numeric_data.drop(columns=['salary','mba_p',], axis=1)
# vif = pd.DataFrame()
# vif["features"] = X.columns
# vif["vif_factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# print(vif)


# # Classification Model

# In[14]:


df_status = df.copy()
df.head(4)


# In[15]:


df_status.info()


# In[16]:


df_status.drop('salary', axis=1, inplace = True)


# In[17]:


X_features = ['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation']
encoded_df = pd.get_dummies(df_status[X_features], drop_first = True)


# In[18]:


df_final = pd.concat([encoded_df, df_status], axis=1,sort=False)


# In[19]:


df_final.columns


# In[20]:


X = df_final.drop('status', axis=1)


# In[21]:


X.drop(df_final[X_features], axis=1, inplace=True)


# In[22]:


y = df_final['status']


# ## Supervised learning
# #### Model validation via cross-validation :- Two Fold

# In[23]:


## Gaussian naive Bayes is often a good model to use as a baseline classification

from sklearn.model_selection import train_test_split


# In[25]:


X1, X2, y1, y2 = train_test_split(X, y, test_size=0.3, random_state=5)  ## DIvide data with 50%
print(X1.shape, X2.shape, y1.shape, y2.shape)


# In[26]:


from sklearn.naive_bayes import GaussianNB       # 1. choose model class
model = GaussianNB()                             # 2. instantiate model


# In[27]:


## We do two validation trials, alternately using each half of the data as a holdout set.
y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)


# In[28]:


##We can use the accuracy_score utility to see the fraction of predicted labels that match their true value:
from sklearn.metrics import accuracy_score

accuracy_score(y1, y1_model), accuracy_score(y2, y2_model)


# ### Visualization of five-fold cross-validation
# #### Demonstrates how to estimate the accuracy of a linear kernel support vector machine on the iris dataset by splitting the data, fitting a model and computing the score 5 consecutive times (with different splits each time):

# In[29]:


from sklearn.model_selection import cross_val_score
score1 = cross_val_score(model, X, y, cv=5)
score1


# In[30]:


print('The mean score and standard deviation of model prediction is', (score1.mean(), score1.std() * 2))


# ## Unsupervised learning example: 

# In[31]:


## Use principal component analysis

from sklearn.decomposition import PCA                      # 1. Choose the model class
model = PCA(n_components=2)                                # 2. Instantiate the model with hyperparameters
model.fit(X)                                            # 3. Fit to data. Notice y is not specified!
X_2D = model.transform(X)                               # 4. Transform the data to two dimensions


# In[32]:


df['PCA1'] = X_2D[:, 0]
df['PCA2'] = X_2D[:, 1]


# In[33]:


sns.lmplot("PCA1", "PCA2", hue='status', data=df, fit_reg=False)

