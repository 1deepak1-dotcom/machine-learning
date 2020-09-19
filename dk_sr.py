# Importing the libraries:-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the datasets:-

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Splitting the datasets into training set and test set:-

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

# Training the simple regression model in the training set:-

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the test sets:-

y_pred = regressor.predict(X_test)

# Viualizing the training set results:-

plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('no of Experience')
plt.ylabel('Salary')


# Visualizing the test set:-  

plt.scatter(X_test,y_test,color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('no of Experience')
plt.ylabel('Salary')