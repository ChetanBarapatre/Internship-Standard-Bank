# -*- coding: utf-8 -*-
"""Task_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/122tH_KEu7wp8gpBaLkTJ2Y0KtCtXxvsV

# Task 2 

## Credit / Home Loans - AutoML vs Bespoke ML

Standard Bank is embracing the digital transformation wave and intends to use new and exciting technologies to give their customers a complete set of services from the convenience of their mobile devices.
As Africa’s biggest lender by assets, the bank aims to improve the current process in which potential borrowers apply for a home loan. The current process involves loan officers having to manually process home loan applications. This process takes 2 to 3 days to process upon which the applicant will receive communication on whether or not they have been granted the loan for the requested amount.
To improve the process Standard Bank wants to make use of machine learning to assess the credit worthiness of an applicant by implementing a model that will predict if the potential borrower will default on his/her loan or not, and do this such that the applicant receives a response immediately after completing their application. 

You will be required to follow the data science lifecycle to fulfill the objective. The data science lifecycle (https://www.datascience-pm.com/crisp-dm-2/) includes:

- Business Understanding
- Data Understanding
- Data Preparation
- Modelling
- Evaluation
- Deployment.

You now know the CRoss Industry Standard Process for Data Mining (CRISP-DM), have an idea of the business needs and objectivess, and understand the data. Next is the tedious task of preparing the data for modeling, modeling and evaluating the model. Luckily, just like EDA the first of the two phases can be automated. But also, just like EDA this is not always best. 


In this task you will be get a taste of AutoML and Bespoke ML. In the notebook we make use of the library auto-sklearn/autosklearn (https://www.automl.org/automl/auto-sklearn/) for AutoML and sklearn for ML. We will use train one machine for the traditional approach and you will be required to change this model to any of the models that exist in sklearn. The model we will train will be a Logistic Regression. Parts of the data preparation will be omitted for you to do, but we will provide hints to lead you in the right direction.

The data provided can be found in the Resources folder as well as (https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset).

- train will serve as the historical dataset that the model will be trained on and,
- test will serve as unseen data we will predict on, i.e. new ('future') applicants.

### Part One

There are many AutoEDA Python libraries out there which include:

- dtale (https://dtale.readthedocs.io/en/latest/)
- pandas profiling (https://pandas-profiling.ydata.ai/docs/master/index.html)
- autoviz (https://readthedocs.org/projects/autoviz/)
- sweetviz (https://pypi.org/project/sweetviz/)

and many more. In this task we will use Sweetviz.. You may be required to use bespoke EDA methods.

The Home Loans Department manager wants to know the following:

1. An overview of the data. (HINT: Provide the number of records, fields and their data types. Do for both).

2. What data quality issues exist in both train and test? (HINT: Comment any missing values and duplicates)

3. How do the the loan statuses compare? i.e. what is the distrubition of each?

4. How do women and men compare when it comes to defaulting on loans in the historical dataset?

5. How many of the loan applicants have dependents based on the historical dataset?

6. How do the incomes of those who are employed compare to those who are self employed based on the historical dataset? 

7. Are applicants with a credit history more likely to default than those who do not have one?

8. Is there a correlation between the applicant's income and the loan amount they applied for? 

### Part Two

Run the AutoML section and then fill in code for the traditional ML section for the the omitted cells.

Please note that the notebook you submit must include the analysis you did in Task 2.

## Import Libraries
"""

!pip install sweetviz 
#uncomment the above if you need to install the library 
!pip install auto-sklearn
#uncomment the above if you need to install the library

!pip install --upgrade scipy

!pip install --upgrade scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz 
import autosklearn.classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

"""## Import Datasets"""

train = pd.read_csv('/content/train (1).csv')
test = pd.read_csv('/content/test.csv')



"""# Part One

## EDA
"""

train.head()

test.head()

# we concat for easy analysis
n = train.shape[0] # we set this to be able to separate the
df = pd.concat([train, test], axis=0)
df.head()

"""### Sweetviz"""

autoEDA = sweetviz.analyze(train)
autoEDA.show_notebook()

"""### Your Own EDA 

"""

train.info()

test.info()

train.shape,test.shape

train.sample(5)

train

test.sample(5)

train.isnull().sum()

test.isnull().sum()

train.duplicated().sum()

test.duplicated().sum()

#univariant analysis
train['Gender'].value_counts().plot(kind='bar')

train['Married'].value_counts().plot(kind='bar')

#5. How many of the loan applicants have dependents based on the historical dataset?
train['Dependents'].value_counts().plot(kind='barh')

train['Education'].value_counts().plot(kind='bar')

train['Self_Employed'].value_counts().plot(kind='bar')

sns.distplot(train['ApplicantIncome'])

sns.distplot(train['CoapplicantIncome'])

sns.distplot(train['LoanAmount'])

sns.distplot(train['Loan_Amount_Term'])

#7. Are applicants with a credit history more likely to default than those who do not have one?
sns.distplot(train['Credit_History'])

train['Property_Area'].value_counts().plot(kind='bar')

train['Loan_Status'].value_counts().plot(kind='bar')

#byvariant analysis
sns.scatterplot(train['ApplicantIncome'],train['LoanAmount'])

#How do women and men compare when it comes to defaulting on loans in the historical dataset?
train.groupby('Gender')['Loan_Status'].value_counts().plot(kind='bar')

#6. How do the incomes of those who are employed compare to those who are self employed based on the historical dataset? 
sns.barplot(train['ApplicantIncome'],train['Self_Employed'])

"""## Your anwers:

1.There are 8 categorical and 5 numerical column in our dataset---I used info() fuction to understanding the data type of of the columns.

2.There are some missing values in training and test dataset
for eg in training dataset(gender columns has 13 missing values)and no duplicated data available in training and test dataset

3.all the graphs for univariant analysis given above

4.According to graph we observe that man getting more loan than women

5.about 345 loan applicants has zero dependends and 102 has one,101 has two and 51 has three dependents

6.people who are self-employed are mor income than employed

7.yes the people with credit history mor likely to default

8.loan amount as applicant income increase

9.The acuuracy little increase in Bespoke ML sklearn than autoML are shown below

10.
"""



"""# Part Two

## Auto ML wth autosklearn
"""

# Matrix of features

X = train[['Gender',
'Married',
'Dependents',
'Education',
'Self_Employed',
'ApplicantIncome',
'CoapplicantIncome',
'LoanAmount',
'Loan_Amount_Term',
'Credit_History',
'Property_Area']]

# convert string(text) to categorical
X['Gender'] = X['Gender'].astype('category')
X['Married'] = X['Married'].astype('category')
X['Education'] = X['Education'].astype('category')
X['Dependents'] = X['Dependents'].astype('category')
X['Self_Employed'] = X['Self_Employed'].astype('category')
X['Property_Area'] = X['Property_Area'].astype('category')


# label encode target
y = train['Loan_Status'].map({'N':0,'Y':1}).astype(int)


# # train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train
autoML = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=2*30, per_run_time_limit=30, n_jobs=8) # imposing a 1 minute time limit on this
autoML.fit(X_train, y_train)

# predict
predictions_autoML = autoML.predict(X_test)

print('Model Accuracy:', accuracy_score(predictions_autoML, y_test))

print(confusion_matrix(predictions_autoML, y_test))

"""## Bespoke ML sklearn

### Data Preparation
"""

# Matrix of features

df= train[['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome',
           'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']]

### Include Numerical Features Here ###
df_num=train[['ApplicantIncome','CoapplicantIncome','LoanAmount',
              'Loan_Amount_Term','Credit_History']]
df_cat =train[['Gender','Married','Dependents','Education','Self_Employed','Property_Area']]              
### Handle Missing Values Here ###
imputer_num=SimpleImputer(strategy='mean',missing_values=np.nan)
imputer_num=imputer_num.fit(df[['LoanAmount','Loan_Amount_Term','Credit_History']])
df[['LoanAmount','Loan_Amount_Term','Credit_History']]=imputer_num.transform(df[['LoanAmount','Loan_Amount_Term','Credit_History']])
                 
imputer_cat=SimpleImputer(strategy='most_frequent',missing_values=np.nan)
imputer_cat=imputer_cat.fit(df[['Gender','Married','Dependents','Education','Self_Employed','Property_Area']]) 
df[['Gender','Married','Dependents','Education','Self_Employed','Property_Area']]= imputer_cat.transform(df[['Gender',
                                                                                                             'Married',
                                                                                                             'Dependents',
                                                                                                             'Education',
                                                                                                             'Self_Employed',
                                                                                                             'Property_Area']])               

# convert string(text) to categorical
df['Gender'] = df['Gender'].astype('category')
df['Married'] = df['Married'].astype('category')
df['Education'] = df['Education'].astype('category')
df['Dependents'] = df['Dependents'].astype('category')
df['Self_Employed'] = df['Self_Employed'].astype('category')
df['Property_Area'] = df['Property_Area'].astype('category')



# label encode target
y = train['Loan_Status'].map({'N':0,'Y':1}).astype(int)

# # encode with get dummies
X = pd.DataFrame(df, columns=df.columns)
X = pd.get_dummies(X, drop_first=True)

# # train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape,X_test.shape,y_train.shape,y_test.shape

### Scale Here ###
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)

# some classifiers you can pick from (remember to import)
import sklearn
classifiers = sklearn.utils.all_estimators(type_filter=None)
for name, class_ in classifiers:
    if hasattr(class_, 'predict_proba'):
        print(name)

# train
clf = LogisticRegression() #change model here
clf.fit(X_train, y_train)

# predict
predictions_clf = clf.predict(X_test)

print('Model Accuracy:', accuracy_score(predictions_clf, y_test))

print(confusion_matrix(predictions_clf, y_test))

"""##DecisionTreeClassifier"""

from sklearn.tree import DecisionTreeClassifier
# train
dtc = DecisionTreeClassifier() #change model here
dtc.fit(X_train, y_train)

# predict
predictions_dtc = dtc.predict(X_test)

print('Model Accuracy:', accuracy_score(predictions_dtc, y_test))

print(confusion_matrix(predictions_dtc, y_test))

"""#SGDClassifier

---


"""

from sklearn.linear_model import SGDClassifier
# train
SGD = DecisionTreeClassifier() #change model here
SGD.fit(X_train, y_train)

# predict
predictions_SGD = SGD.predict(X_test)

print('Model Accuracy:', accuracy_score(predictions_SGD, y_test))

print(confusion_matrix(predictions_SGD, y_test))