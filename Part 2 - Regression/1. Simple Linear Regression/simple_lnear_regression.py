#SimpleLinearRegressions

#Import the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing DataSets
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

#Splitting Training and Test Set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)

'''#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#Fitting Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the Test Set Results
Y_pred = regressor.predict(X_test)

#Visualizing the Training Set Results
plt.scatter(X_train,Y_train,color ='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('SALARY VS EXPERIENCE(Traing Set)')
plt.xlabel('Years of Experience')
plt.xlabel('Salary')
plt.show()
plt.scatter(X_test,Y_test,color ='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('SALARY VS EXPERIENCE(Test Set)')
plt.xlabel('Years of Experience')
plt.xlabel('Salary')
plt.show()