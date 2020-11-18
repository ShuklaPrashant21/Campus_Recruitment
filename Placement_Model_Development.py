## Import necessary library for analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

## Load data in pandas df
df = pd.read_csv(r"C:\Users\lenovo\Downloads\DS_Practice\Campus_Recuritment_Model\Placement_Data.csv", index_col=0)
df.head(3)

## Check data type and shape of every column 
df.info()


## Find marginal probability through two way table
# Marginal Probability is the probability of the occurence of the single event

pd.crosstab(index = df['specialisation'], columns=df['status'], margins =True, normalize = True)

pd.crosstab(index = df['degree_t'], columns=df['status'], margins =True, normalize =True)


## Which degree specialization is much demanded by corporate?

sns.countplot(x="status", data=df, hue='specialisation')
plt.title("Degree Specialization vs Candidate Placement")
plt.xlabel("Status of Placement")
plt.ylabel("Number of candidate")
plt.show()


## Above plot shows Mkt&Fin specialization is dominating in campus, for placement.

## Does work experience affects placement of a candidate?

sns.countplot(x="status", data=df, hue='workex')
plt.title("Candidate Work Experience in Placement")
plt.xlabel("Status of Placement")
plt.ylabel("Number of candidate")
plt.show()

## Find which degree technology studied by candidate is placed more 
sns.countplot(x="degree_t", data=df, hue='status')
plt.title("Candidate degree Technology in Placement")
plt.xlabel("Degree Technology")
plt.ylabel("Number of candidate")
plt.show()

# Creates scatter plots for join relationship and histogram for univariate distributions 

sns.pairplot(df.drop(["salary"], axis=1), kind="scatter", hue="status")
plt.show()


## Check average or mean of numeric variable for unique values of status 
df.groupby(["status"]).mean()

## Check how many students have been placed or not
df.status.value_counts()

## Variance Infilation Factor 
# X = numeric_data.drop(columns=['salary','mba_p',], axis=1)
# vif = pd.DataFrame()
# vif["features"] = X.columns
# vif["vif_factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# print(vif)


## Classification Model
df_status = df.copy()
df.head(4)

df_status.drop('salary', axis=1, inplace = True)

X_features = ['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation']
encoded_df = pd.get_dummies(df_status[X_features], drop_first = True)

df_final = pd.concat([encoded_df, df_status], axis=1,sort=False)
df_final.columns

X = df_final.drop('status', axis=1)

X.drop(df_final[X_features], axis=1, inplace=True)

y = df_final['status']

## Supervised learning :- Gaussian Naive Bayes classifier
### Model validation via cross-validation :- Two Fold

from sklearn.model_selection import train_test_split

X1, X2, y1, y2 = train_test_split(X, y, test_size=0.3, random_state=5)  ## Divide data 
print(X1.shape, X2.shape, y1.shape, y2.shape)

from sklearn.naive_bayes import GaussianNB       # 1. choose model class
model = GaussianNB()                             # 2. instantiate model

## We do two validation trials, alternately using each half of the data as a holdout set.
y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)

## We can use the accuracy_score utility to see the fraction of predicted labels that match their true value:
from sklearn.metrics import accuracy_score

accuracy_score(y1, y1_model), accuracy_score(y2, y2_model)


## Visualization of five-fold cross-validation

from sklearn.model_selection import cross_val_score
score1 = cross_val_score(model, X, y, cv=5)
score1

print('The mean score and standard deviation of model prediction is', (score1.mean(), score1.std() * 2))

## Logistic Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)
logmodel = LogisticRegression()

## Fit training data into model to make prediction further.
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

## Create confusion matrix to visualize accuracy of model
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
