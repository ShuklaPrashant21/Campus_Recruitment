## Import necessary library for analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

## Load data in pandas df
df = pd.read_csv(r"C:\Users\lenovo\Downloads\DS_Practice\Campus_Recuritment_Model\Placement_Data.csv", index_col=0)
df.head(3)

## Check detail description about numeric variables 
df.describe()

## Check data type and shape of every column 
df.info()

df.status.value_counts()

## Create another dataframe of only placed students and check its shape 
df_placed = df.dropna(how="any")
df_placed.shape

df_placed.groupby(["status"]).mean()

# ### Salary distribution through box-whiskers plot & Histogram

## Box whiskers plot & histogram on the same window 
## Split the plotting window into 2 parts

f, (ax_box, ax_hist)= plt.subplots(2, gridspec_kw={"height_ratios": (.15, .85)})
## Add and create  box plot
sns.boxplot(df_placed["salary"], ax=ax_box)

sns.distplot(df_placed["salary"], ax=ax_hist, kde=False)
plt.show()

## Remove outliers from salary column
df_placed = df_placed[df_placed['salary'] < 650000]

## Find correlation between numeric variables in dataframe

numeric_data = df_placed.select_dtypes(exclude = [object])
numeric_data.shape

corr_matrix = numeric_data.corr()
corr_matrix

# Now, show the correlation matrix into heatmap for better understanding and visualization
plt.figure(figsize=(15,10))
sns.heatmap(corr_matrix,annot=True,cmap='YlGnBu')
plt.show()

## Box and whiskers plot is very useful to find relationship between numerical & categorical variable.

sns.boxplot(x=df_placed['degree_t'], y = df_placed['salary'], hue='specialisation', data=df_placed)
plt.title("Salary vs Gender")
plt.xlabel("Gender of an candidate")
plt.xlabel("Salary of an candidate")
plt.show()

## Box and whiskers plot is very useful to find relationship between numerical & categorical variable.

sns.boxplot(x=df_placed['gender'], y = df_placed['salary'], hue='specialisation', data=df_placed)
plt.title("Salary vs Gender")
plt.xlabel("Gender of an candidate")
plt.xlabel("Salary of an candidate")
plt.show()

## Model Development :-  Salary Prediction
## Now, build a Linear Regression and Random Forest Model on placed dataframe only.

df_predict = df_placed.copy()
df_predict.drop(columns=['status','degree_p'], axis=1, inplace=True)

## Converting categorical variables to dummy variables
##df_predict=pd.get_dummies(df_placed, drop_first=True)
df_predict = pd.get_dummies(df_predict, columns=['gender', 'ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation'])
df_predict.head(3)

## Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import metrics

## Separating input and output features
x1 = df_predict.drop(['salary'], axis='columns', inplace=False)
y1 = df_predict['salary']

## FInd skewness of salary column. 

#SalePrice
sns.distplot(df_predict['salary'])

#skewness
print("The skewness of SalePrice is {}".format(df_placed['salary'].skew()))

#now transforming the target variable
tar = np.log(df_placed['salary'])
print('Skewness is', tar.skew())
sns.distplot(tar)

## Transform price as a logarithmic value
y1 =np.log(y1)

## Splitting data into test and train to fit model & predict. Train set contains 60% data because test_size =0.4 and random state is a predefined algorithm its called pseudo random number generato

x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.40, random_state = 5)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

## LINEAR REGRESSION WITH PLACED DATA
## Setting intercept as true
lgr = LinearRegression(fit_intercept =True)

## MODEL
model_lin1 = lgr.fit(x_train, y_train)

## Predicting model on test set
salary_predictions_lin1 = lgr.predict(x_test)

## Computing MSE and RMSE
lin_mse1 = mean_squared_error(y_test, salary_predictions_lin1)
lin_rmse1 = np.sqrt(lin_mse1)
print(lin_rmse1)

## R squared value
r2_lin_test1 = model_lin1.score(x_test, y_test)
r2_lin_train1 = model_lin1.score(x_train, y_train)
print(r2_lin_test1, r2_lin_train1)

## Regression diagnostics :- Resident plot analysis
## It is differnce test data and your prediction. It is just difference between actual & predicted value.
residuals1 = y_test - salary_predictions_lin1
sns.regplot(x = salary_predictions_lin1, y=residuals1, scatter=True, fit_reg=False, data=df_placed)
residuals1.describe()

#To retrieve the intercept:
print(model_lin1.intercept_)

#For retrieving the slope:
coeff_df = pd.DataFrame(model_lin1.coef_, x1.columns,columns=['Coefficient'])  
coeff_df

## Check the difference between the actual value and predicted value.

df1 = pd.DataFrame({'Actual': y_test, 'Predicted':salary_predictions_lin1})
df1.head(10)

## Now let's plot the comparison of Actual and Predicted values

df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.show()

## RANDOM FOREST REGRESSOR

## MODEL PARAMETERS
rf = RandomForestRegressor(n_estimators = 100, max_features='auto', max_depth=100, min_samples_split=10, min_samples_leaf=4, random_state=1)

## MODEL
model_rf1 =rf.fit(x_train, y_train)

## Predicting model on test set
salary_predictions_rf1 = rf.predict(x_test)

## Computing MSE and RSME
rf_mse1 = mean_squared_error(y_test, salary_predictions_rf1)
rf_rmse1 = np.sqrt(rf_mse1)
print(rf_rmse1)

## R Squared value
r2_rf_test1 = model_rf1.score(x_test, y_test)
r2_rf_train1 = model_rf1.score(x_train, y_train)
print(r2_rf_test1, r2_rf_train1)

## Check the difference between the actual value and predicted value.
df2 = pd.DataFrame({'Actual': y_test, 'Predicted':salary_predictions_rf1})
df2.head(10)

## Check the difference between the actual value and predicted value.
df2.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.show()
