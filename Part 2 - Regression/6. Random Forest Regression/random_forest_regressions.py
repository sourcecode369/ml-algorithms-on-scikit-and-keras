#Random_Forest regressions

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

#Fitting the Random_Forest Regression to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=1000,random_state=0)
regressor.fit(X,Y)

#Predicting a new result using Random_Forest Regression
Y_pred = regressor.predict(6.5)


#Visualzing Random_Forest regression at a higher resolution
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Random_Forest Regression Prediction')
plt.xlabel('Psoition')
plt.ylabel('Salaries')
plt.show()