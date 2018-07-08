#Support vector regression
#Regression Template

#SVR regressions

#Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing DataSets
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:3].values

'''#Splitting Training and Test Set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)'''

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

#Fitting the SVR Regression to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)

#Predicting a new result using SVR Regression
Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
#Visualizing the SVR Regression Results
plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('SVR Regression Prediction')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
#Visualzing XYZ regression at a higher resolution
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('SVR Regression Prediction')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()