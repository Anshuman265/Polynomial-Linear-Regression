#Importing the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Reading in the csv file
dataset = pd.read_csv('Position_Salaries.csv')
X  = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Training the linear regression model just to check its accuracy
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
linear_regressor = lin_reg.fit(X,y)

#Training the polynomial Regression model on the whole dataset
#Creating the power matrix
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
#Creating new transformed matrix here
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression().fit(X_poly,y)

#Visualising the linear Regression results
plt.scatter(X,y,color = 'red',label = 'Original Data')
plt.plot(X,linear_regressor.predict(X),color = 'blue',label = 'Linear Regression')
plt.xlabel("Positions")
plt.ylabel("Income")
plt.title("Position level v/s Income (Linear Regression)")
plt.legend()
plt.show()

#Visualising the polynomial Regression Results
plt.scatter(X,y,color = 'red',label = 'Original Data')
plt.plot(X,lin_reg_2.predict(X_poly),color = 'blue',label = 'Polynomial Linear Regression')
plt.xlabel("Positions")
plt.ylabel("Income")
plt.title("Position level v/s Income (Polynomial Linear Regression)")
plt.legend()
plt.show()

#Increasing the resolution and drawing a smoother curve
X_poly_smooth = np.arange(min(X),max(X),0.1)
X_poly_smooth = X_poly_smooth.reshape(len(X_poly_smooth),1)
plt.scatter(X,y,color = 'red')
plt.plot(X_poly_smooth,lin_reg_2.predict(poly_reg.fit_transform(X_poly_smooth)),color = 'blue')
plt.xlabel("Positions")
plt.ylabel("Income")
plt.title("Position level v/s Income (Polynomial Linear Regression)")
plt.show()

#Predicting with linear Regressor
linear_predict = lin_reg.predict([[6.5]])
print(f'The salary predicted by the simple linear regressor is {linear_predict}')

#Predicting with polynomial Linear Regressor 
poly_linear_predict = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(f'The salary predicted by the polynomial linear regressor is {poly_linear_predict}')
