# Author: Debasis Pradhan
# Date: 15 April 2023
# Project: Regression analysis of startup companies data

# Description: This program performs a regression analysis on a dataset of 50 startup companies,
# using three predictor variables (R&D Spend, Administration, and Marketing Spend) to predict
# the outcome variable (Profit).


#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data from a CSV file
data = pd.read_csv('50_Startups.csv')

# Explore the data
print(data.head())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Visualize the data
sns.pairplot(data)

# Define the independent and dependent variables
X = data[['R&D Spend', 'Administration', 'Marketing Spend']]
y = data['Profit']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the list of regression algorithms to compare
models = [
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', Ridge()),
    ('Lasso Regression', Lasso()),
    ('ElasticNet Regression', ElasticNet()),
    ('Random Forest Regression', RandomForestRegressor()),
    ('Gradient Boosting Regression', GradientBoostingRegressor()),
    ('Support Vector Regression', SVR())
]

# Evaluate each model using 10-fold cross-validation
for name, model in models:
    mse_scores = -cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
    mae_scores = -cross_val_score(model, X, y, cv=10, scoring='neg_mean_absolute_error')
    r2_scores = cross_val_score(model, X, y, cv=10, scoring='r2')
    print(name, 'MSE:', mse_scores.mean())
    print(name, 'MAE:', mae_scores.mean())
    print(name, 'R-squared:', r2_scores.mean())

# Train the best model on the entire training set
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('MSE:', mse)
print('MAE:', mae)
print('R-squared:', r2)

# Visualize the model predictions
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()
