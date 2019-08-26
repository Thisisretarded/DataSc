#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Created on Mon Aug 26 12:29:59 2019

@author: osboxes
"""
#%%
#importing libraries
import pandas as pd
import numpy as np
#%%
#importing tha dadtaset

dataset = pd.read_csv("Advertising.csv")
dataset['Advt_Budget'] = dataset['TV']+dataset['radio']+dataset['newspaper']
df = dataset[['Advt_Budget', 'sales']]
print(df.head())
X=df.iloc[:, 0:1]
y=df.iloc[:, 1:]
#%%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0, )

#%%
# Fitting Simple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression


regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_train)

B0 = regressor.coef_
B1 = regressor.intercept_

xtrainScore = regressor.score(X_test, y_test)*100
print()
print()
print("Eqn is" + " Sales = TV X " + str(B1) +" + " + str(*B0))
print()
print("Score of prediction: " +  str(xtrainScore))
#%%
# importing pyplot module from matplotlib package
import matplotlib.pyplot as plt

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
#plt.plot(X_train, y_pred, color = 'blue')
#plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.grid()
plt.title('Advt vs. Sales (Training set)')
plt.xlabel('Advt Budget in $')
plt.ylabel('Sales in $')
plt.show()


# Visualising the Test set results
y_pred = regressor.predict(X_test)
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
#plt.plot(X_train, y_predz, color = 'green')
plt.grid()
plt.title('Advt vs. Sales (Test set)')
plt.xlabel('Advt Budget in $')
plt.ylabel('Sales in $')
plt.show()

#%%
residual_sum = y_test - y_pred
plt.scatter(residual_sum**2, y_pred)
plt.show()

#%%
y_pred_df = pd.DataFrame(data = np.array([x for a in y_pred for x in a ]))
print(y_pred_df[0].corr(residual_sum['sales']))