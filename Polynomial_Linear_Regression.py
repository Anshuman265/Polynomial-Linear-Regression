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
poly_reg = PolynomialFeatures(degree = 2)
#Creating new transformed matrix here
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression().fit(X_poly,y)
