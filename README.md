# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

Program to implement the simple linear regression model for predicting the marks scored.

Developed by: BAUDHIGAN D

RegisterNumber: 212223230028

```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('student_scores.csv')
print(df)
print()
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:

DATASET:

<img width="192" height="568" alt="image" src="https://github.com/user-attachments/assets/6136bfc5-ce26-404b-8ce1-390e1192c100" />


HEAD VALUES:

<img width="183" height="137" alt="image" src="https://github.com/user-attachments/assets/102ee73f-d85a-4490-b0fd-27c1d966f051" />

TAIL VALUES:

<img width="194" height="127" alt="image" src="https://github.com/user-attachments/assets/7b332c01-2256-468f-b63b-77b8bb8218d4" />
   
X AND Y VALUES:

<img width="739" height="577" alt="image" src="https://github.com/user-attachments/assets/6f06e4c0-1555-46cf-8c30-a740abc0eadf" />

PREDICTION VALUES OF X AND Y:

<img width="701" height="73" alt="image" src="https://github.com/user-attachments/assets/34651b98-bfcd-4f20-8d68-11077cd9c9dd" />

MSE,MAE AND RMSE:

<img width="277" height="87" alt="image" src="https://github.com/user-attachments/assets/321b147f-7fbe-4dc5-8721-faeac3678e34" />

TRAINING SET:

<img width="563" height="453" alt="image" src="https://github.com/user-attachments/assets/aba2158e-6b95-48cd-bc18-02699f09dd11" />

TESTING SET:

<img width="563" height="453" alt="cf1d6ee1c2b28212d35c7394a88df2e7_T8KRoFxukl50gAAAABJRU5ErkJggg==" src="https://github.com/user-attachments/assets/e431140b-dede-4e33-bf06-59010f812969" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
