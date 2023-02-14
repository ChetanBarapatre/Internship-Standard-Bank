# internship-standard-bank

Standard Bank is embracing the digital transformation wave and intends to use new and exciting technologies to give their customers a complete set of services from the convenience of their mobile devices. As Africa’s biggest lender by assets, the bank aims to improve the current process in which potential borrowers apply for a home loan. The current process involves loan officers having to manually process home loan applications. This process takes 2 to 3 days to process upon which the applicant will receive communication on whether or not they have been granted the loan for the requested amount. To improve the process Standard Bank wants to make use of machine learning to assess the credit worthiness of an applicant by implementing a model that will predict if the potential borrower will default on his/her loan or not, and do this such that the applicant receives a response immediately after completing their application.

The Home Loans Department manager wants to know the following:

1. An overview of the data.

The datasets consist of a total of 614 entries and 12 columns in a training dataset and 367 entries and 11 columns in testing datasets.(8-Categorical and 5-Numerical)

train.info()

2. What data quality issues exist in both train and test? 

The data looks clean because very less percentage values are missing and no duplicate values in the data.

train.isnull().sum()

train.duplicated().sum()

3 . How do the the loan statuses compare? i.e. what is the distrubition of each?

train['Loan_Status'].value_counts()

Y    422

N    192

train['Loan_Status'].value_counts().plot(kind='bar')


![image](https://user-images.githubusercontent.com/117656346/218703704-78a13320-efc1-46fc-bdfb-18b00cbd86d2.png)

4. How do women and men compare when it comes to defaulting on loans in the historical dataset?

train.groupby('Gender')['Loan_Status'].value_counts()

Gender  Loan_Status

Female   

           Y               75

           N               37
        
Male     

           Y              339

           N              150
           
 train.groupby('Gender')['Loan_Status'].value_counts().plot(kind='bar')
 
 ![image](https://user-images.githubusercontent.com/117656346/218705921-90afec18-767c-4a1b-a3a4-9df7703099eb.png)

5. How many of the loan applicants have dependents based on the historical dataset?

train[train['Dependents'] != '0'].shape[0]/train.shape[0]

0.4381107491856677(about 44% people are dependents)

train['Dependents'].value_counts().plot(kind='barh')

![image](https://user-images.githubusercontent.com/117656346/218708350-f19cf7a0-fb3d-4fb0-b42c-902c1b6b0649.png)

6. How do the incomes of those who are employed compare to those who are self employed based on the historical dataset?

train.groupby('Self_Employed')['ApplicantIncome'].describe()

![Screenshot 2023-02-14 160527](https://user-images.githubusercontent.com/117656346/218711064-a4c6082c-bfa7-4244-a42b-ef4ad9843f66.png)

![image](https://user-images.githubusercontent.com/117656346/218724323-798f4eae-6314-4cf7-ba7e-e90c3b758896.png)

7. Are applicants with a credit history more likely to default than those who do not have one?

train.groupby('Credit_History')['Loan_Status'].value_counts(normalize=True)

![Screenshot 2023-02-14 170857](https://user-images.githubusercontent.com/117656346/218727026-48652d7c-4c2f-4dc7-b48a-e2462abfd08b.png)


8 . Is there a correlation between the applicant's income and the loan amount they applied for?

train.corr()

![Screenshot 2023-02-14 171555](https://user-images.githubusercontent.com/117656346/218729157-e75ab7a6-6f3c-446f-9c3f-b069f7586e44.png)

sns.scatterplot(train['ApplicantIncome'],train['LoanAmount'])

![image](https://user-images.githubusercontent.com/117656346/218729480-7077d5ba-c1e4-4e86-a907-a90e3c0b4129.png)

#  Model Building

## 1. Auto ML wth autosklearn

###  data Preparation

![Screenshot 2023-02-14 173908](https://user-images.githubusercontent.com/117656346/218734538-7c94a286-236a-4d21-a391-22caff9c7e0a.png)

### Training a Model 

autoML = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=2*30, per_run_time_limit=30, n_jobs=8)

autoML.fit(X_train, y_train)

### Predict

predictions_autoML = autoML.predict(X_test)

### Model Evaluation

print('Model Accuracy:', accuracy_score(predictions_autoML, y_test))

Model Accuracy: 0.7804878048780488

![image](https://user-images.githubusercontent.com/117656346/218754463-dd3feefb-ee21-472d-aa73-9fbca86a5cd6.png)


print(confusion_matrix(predictions_autoML, y_test))

[[18  2]
 
 [25 78]]
 
![image](https://user-images.githubusercontent.com/117656346/218749951-beb20dd8-85b2-403d-b525-f52212e7c022.png)

 
True Negative: Model has given prediction No, and the real or actual value was also No(Model predicted that 18 people not getting  loan and
actually they are not get loan)

True Positive: The model has predicted yes, and the actual value was also true.(Model predicted that 78 people getting loan and actually they get loan)

False Negative: The model has predicted no, but the actual value was Yes, it is also called as Type-II error.(Model predicted that 25 people not getting loan but
actually they get loan)

False Positive: The model has predicted Yes, but the actual value was No. It is also called a Type-I error.(Model predicted that 2 people getting a loan but
actually they not get loan)

## 2.Bespoke ML sklearn

### Data Preparation 

![Screenshot 2023-02-14 191912](https://user-images.githubusercontent.com/117656346/218757368-8c4d900c-7dd4-4631-b79e-e8b8ae4ddebf.png)

![Screenshot 2023-02-14 192119](https://user-images.githubusercontent.com/117656346/218757716-56272f52-9ddd-434b-a385-f621995d0720.png)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform (X_test)

### Model building and Evaluation

###  DecisionTreeClassifier--Accuracy score -69.10%

![Screenshot 2023-02-14 193657](https://user-images.githubusercontent.com/117656346/218762403-ae520405-f605-4a08-b562-2b02fe43dc12.png)

### Logistic Regression--Accuracy score -78.86%

![Screenshot 2023-02-14 194253](https://user-images.githubusercontent.com/117656346/218763294-3ae8d533-d1ad-419a-8232-0510ac6b1915.png)

###SGDClassifier--Accuracy score -69.91%

![Screenshot 2023-02-14 194432](https://user-images.githubusercontent.com/117656346/218763738-05d05810-14b4-493a-b576-c7ae854f5204.png)






























video link for ppt https://drive.google.com/file/d/1zRi2M6g6jvfQdErwIQpdkgiipgAnabJU/view?usp=share_link
