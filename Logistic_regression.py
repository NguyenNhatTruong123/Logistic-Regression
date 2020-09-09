import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data_test=pd.read_csv("test.csv")
data_train=pd.read_csv("train.csv")

#Remove inconsequential columns
data_train = data_train.drop(['PassengerId','Name','Ticket','Cabin','Parch'], axis=1)
data_test = data_test.drop(['PassengerId','Name','Ticket','Cabin','Parch'], axis=1)

#Digitizing non-number coulumns and drop them
sex_train=pd.get_dummies(data_train["Sex"])
sex_test=pd.get_dummies(data_test["Sex"])

data_train=data_train.drop(["Sex"],axis=1)
data_test=data_test.drop(["Sex"],axis=1)

data_train=pd.concat([data_train,sex_train],axis=1)
data_test=pd.concat([data_test,sex_test],axis=1)

data_train["Age"].fillna(data_train["Age"].min(),inplace=True)
data_test["Age"].fillna(data_test["Age"].min(),inplace=True)

embarked_train=pd.get_dummies(data_train["Embarked"])
embarked_test=pd.get_dummies(data_test["Embarked"])

data_train=data_train.drop(["Embarked"],axis=1)
data_test=data_test.drop(["Embarked"],axis=1)

data_train=pd.concat([data_train,embarked_train],axis=1)
data_test=pd.concat([data_test,embarked_test],axis=1)


#Fill in the blank on Fare column, can use min, max or mean of column 
data_test["Fare"].fillna(data_test["Fare"].min(),inplace=True)

#Training data using Logistic Regression in sklearn library
y_train=data_train["Survived"].values
x_train=data_train.drop(["Survived"],axis=1).values

x_test=data_test.values
y_abs=pd.read_csv("genderclassfare.csv")["Survived"].values

logre=LogisticRegression()
logre.fit(x_test,y_abs)
y_pre=logre.predict(x_test)
print(np.mean(y_pre==y_abs))

count = 0
for i in range(x_test.shape[0]):
    y_pre=logre.predict(x_test[i].reshape(1,-1))
    if(np.mean(y_pre==y_abs[i])>0.7):
        count=count+1
print(count)


