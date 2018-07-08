#Polynomial regressions

#Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing DataSets
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

'''#Splitting Training and Test Set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)'''

'''#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#Fitting Linear Regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Fitting Polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,Y)

#Visualizing the Linear Regression Results
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Linear Regression Prediction(Truth or Bluff)')
plt.xlabel('Position level')
plt.ylabel('Salary')

#Visualizing the Polynomial Regression Results
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Polynomial Regression Prediction(Truth or Bluff)')
plt.xlabel('Position level')
plt.ylabel('Salary')

#Visualizing the Polynomial Regression Results
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Polynomial Regression Prediction(Truth or Bluff)')
plt.xlabel('Position level')
plt.ylabel('Salary')

#Predicting a new result using Linear Regression
lin_pred = lin_reg.predict(6.5)
#Predicting a new result using Polynomial Regression
poly_pred = lin_reg2.predict(poly_reg.fit_transform(6.5))