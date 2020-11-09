#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Import necessary library for analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[2]:


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

# In[3]:


## Check detail description about numeric variables 
df.describe()


# In[4]:


## Check data type and shape of every column 
df.info()


# #### Check how many students have been placed or not

# In[5]:


df.status.value_counts()


# #### Create another dataframe of only placed students.

# In[6]:


## Create another dataframe of only placed students and check its shape 
df_placed = df.dropna(how="any")
df_placed.shape


# #### Check average or mean of numeric variable for unique values of status 

# In[7]:


df_placed.groupby(["status"]).mean()


# ### Salary distribution through box-whiskers plot & Histogram

# In[8]:


## Box whiskers plot & histogram on the same window 
## Split the plotting window into 2 parts

f, (ax_box, ax_hist)= plt.subplots(2, gridspec_kw={"height_ratios": (.15, .85)})
## Add and create  box plot
sns.boxplot(df_placed["salary"], ax=ax_box)

sns.distplot(df_placed["salary"], ax=ax_hist, kde=False)
plt.show()


# In[26]:


## Remove outliers from salary column
df_placed = df_placed[df_placed['salary'] < 650000]


# ### Find correlation between numeric variables in dataframe
# 
# Correlation :- Calculate relationship between two numerical variables.
# 
# Excluding null valuees & excluding the categorical variables to find the Pearson's correlation
# 
# • Positive correlation – the other variable has a tendency to also increase 
# 
# • Negative correlation – the other variable has a tendency to decrease
# 
# • No correlation – the other variable does not tend to either increase or decrease.

# In[28]:


numeric_data = df_placed.select_dtypes(exclude = [object])
numeric_data.shape


# In[29]:


corr_matrix = numeric_data.corr()
corr_matrix


# Now, show the correlation matrix into heatmap for better understanding and visualization

# In[32]:


plt.figure(figsize=(15,10))
sns.heatmap(corr_matrix,annot=True,cmap='YlGnBu')
plt.show()


# ###### According to correlation matrix and its visualization in heatmaps, there is almost -ve correlation between undergraduate(%) and salary. The salary variable only shows (+ve) correlation with MBA(%) & placement test(%) . And maximum positive coorelation will be shown between undergraduate(%) and MBA(%).  
# ##### In the heatmap of correlation matrix the darker the color of tile the correlation between the variables is highly positive.And lighter the color of tile the correlation between the variables is highly negative.

# In[23]:


## Variance Infilation Factor 
# X = numeric_data.drop(columns=['salary','mba_p',], axis=1)
# vif = pd.DataFrame()
# vif["features"] = X.columns
# vif["vif_factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# print(vif)


# In[35]:


## Box and whiskers plot is very useful to find relationship between numerical & categorical variable.

sns.boxplot(x=df_placed['degree_t'], y = df_placed['salary'], hue='specialisation', data=df_placed)
plt.title("Salary vs Gender")
plt.xlabel("Gender of an candidate")
plt.xlabel("Salary of an candidate")
plt.show()


# ### Find does gender of an student affects the salary or any other variable

# In[31]:


## Box and whiskers plot is very useful to find relationship between numerical & categorical variable.

sns.boxplot(x=df_placed['gender'], y = df_placed['salary'], hue='specialisation', data=df_placed)
plt.title("Salary vs Gender")
plt.xlabel("Gender of an candidate")
plt.xlabel("Salary of an candidate")
plt.show()


# ###### According to above box-whiskers plot the outliners or extreme value in salary column lie more in male category than female. One of assumptions is may be some of the male candidates getting hire for higher post in a company thats why they are getting more salary.

# # Model Development :-  Salary Prediction
# 
# #### Now, build a Linear Regression and Random Forest Model on placed dataframe only.

# In[25]:


df_predict = df_placed.copy()
df_predict.shape


# In[33]:


df_predict.drop(columns=['status','degree_p'], axis=1, inplace=True)
df_predict.shape


# In[36]:


## Converting categorical variables to dummy variables
##df_predict=pd.get_dummies(df_placed, drop_first=True)
df_predict = pd.get_dummies(df_predict, columns=['gender', 'ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation'])
df_predict.shape


# In[37]:


df_predict.head(3)


# In[38]:


## Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import metrics


# ## MODEL BUILDING

# In[39]:


## Separating input and output features
x1 = df_predict.drop(['salary'], axis='columns', inplace=False)
y1 = df_predict['salary']


# ### FInd skewness of salary column. 

# In[65]:


#SalePrice
sns.distplot(df_predict['salary'])


# In[67]:


#skewness
print("The skewness of SalePrice is {}".format(df_placed['salary'].skew()))


# In[66]:


#now transforming the target variable
tar = np.log(df_placed['salary'])
print('Skewness is', tar.skew())
sns.distplot(tar)


# In[40]:


prices =pd.DataFrame({"1. Before":y1, "2. After":np.log(y1)})
prices.hist()
plt.show()


# In[41]:


## Transform price as a logarithmic value
y1 =np.log(y1)


# #### Splitting data into test and train to fit model & predict. Train set contains 70% data because test_size =0.3 and random state is a predefined algorithm its called pseudo random number generato

# In[44]:


x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.40, random_state = 5)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# ## BASELINE MODEL FOR OMITTED DATA
# 
# WE are making a base model by using test data mean value. This is to set a benchmark and to compare with our regression model.

# In[45]:


## Finding the mean for test data value
base_pred =np.mean(y_test)
print(base_pred)


# In[46]:


## Representing some value till length of test data
base_pred = np.repeat(base_pred, len(y_test))


# In[47]:


base_pred


# In[48]:


## Finding the RMSE(Root Mean Squared Error)
## RMSE computes the difference between the test value and the predicted value and squared them and divides them by number of samples.

base_root_mean_square_error = np.sqrt(mean_squared_error(y_test, base_pred))
print(base_root_mean_square_error)


# ## LINEAR REGRESSION WITH PLACED DATA

# In[49]:


## Setting intercept as true
lgr = LinearRegression(fit_intercept =True)


# In[50]:


## MODEL
model_lin1 = lgr.fit(x_train, y_train)


# In[51]:


## Predicting model on test set
salary_predictions_lin1 = lgr.predict(x_test)


# In[52]:


## Computing MSE and RMSE
lin_mse1 = mean_squared_error(y_test, salary_predictions_lin1)
lin_rmse1 = np.sqrt(lin_mse1)
print(lin_rmse1)


# In[53]:


## R squared value
r2_lin_test1 = model_lin1.score(x_test, y_test)
r2_lin_train1 = model_lin1.score(x_train, y_train)
print(r2_lin_test1, r2_lin_train1)


# In[54]:


## Regression diagnostics :- Resident plot analysis
## It is differnce test data and your prediction. It is just difference between actual & predicted value.
residuals1 = y_test - salary_predictions_lin1
sns.regplot(x = salary_predictions_lin1, y=residuals1, scatter=True, fit_reg=False, data=df_placed)
residuals1.describe()


# In[55]:


#To retrieve the intercept:
print(model_lin1.intercept_)

#For retrieving the slope:
coeff_df = pd.DataFrame(model_lin1.coef_, x1.columns,columns=['Coefficient'])  
coeff_df


# ### Check the difference between the actual value and predicted value.

# In[56]:


df1 = pd.DataFrame({'Actual': y_test, 'Predicted':salary_predictions_lin1})
df1.head(10)


# ### Now let's plot the comparison of Actual and Predicted values

# In[57]:


df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.show()


# ## RANDOM FOREST REGRESSOR

# In[58]:



## MODEL PARAMETERS
rf = RandomForestRegressor(n_estimators = 100, max_features='auto', max_depth=100, min_samples_split=10, min_samples_leaf=4, random_state=1)


# In[59]:


## MODEL
model_rf1 =rf.fit(x_train, y_train)


# In[60]:


## Predicting model on test set
salary_predictions_rf1 = rf.predict(x_test)


# In[61]:


## Computing MSE and RSME
rf_mse1 = mean_squared_error(y_test, salary_predictions_rf1)
rf_rmse1 = np.sqrt(rf_mse1)
print(rf_rmse1)


# In[62]:


## R Squared value
r2_rf_test1 = model_rf1.score(x_test, y_test)
r2_rf_train1 = model_rf1.score(x_train, y_train)
print(r2_rf_test1, r2_rf_train1)


# ### Check the difference between the actual value and predicted value.

# In[63]:


df2 = pd.DataFrame({'Actual': y_test, 'Predicted':salary_predictions_rf1})
df2.head(10)


# ### Check the difference between the actual value and predicted value.

# In[64]:


df2.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.show()

