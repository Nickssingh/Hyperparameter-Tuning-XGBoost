# XGBoost Hyperparamter Tuning - Churn Prediction

## A. Goal
XGBoost is an effective machine learning algorithm; it outperforms many other algorithms in terms of both speed and efficiency. The implementation of XGBoost requires inputs for a number of different parameters. To completely harness the model, we need to tune its parameters. We will list some of the important parameters and tune our model by finding their optimal values. 

*Note*: I have used an Amazon EC2 instance (t2.micro on Ubuntu) for running Jupyter. This is a free tier instance, and I used Powershell to SSH to it.

## B. Dataset
The focus of the dataset is on customer attrition within the telecommunication industry. The data is of a telecom service provider. Such type of data is often used by organizations to build prediction models, and companies use the models to identify customers who can discontinue services. Following are the variables in the dataset.

*Target Variable*  
Churn - Customers who discontinued services with last month

*Input Variables*  
Demographic Information - age, gender, partner, and dependents
Service Details - phone, multiple lines, internet, online backup, online security, tech support, device protection, and streaming
Account Information - subscription tenure, contract, monthly charges, total charges, paperless billing, payment method

Data Source: https://www.kaggle.com/blastchar/telco-customer-churn
