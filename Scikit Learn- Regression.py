"""
Purpose:
This script utilizes Scikit Learn to perform regression on the diabetes dataset using three models:
Linear Regression, Decision Tree Regression, and Random Forest Regression.
The script demonstrates data manipulation, model training, and performance evaluation using common regression metrics.
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class RegressionModel:
    """
    RegressionModel (Latin: regressio; Greek: μοντέλο)
    
    A class for training and evaluating a regression model.
    
    Attributes:
    -----------
    name : str
        Name of the regression model.
    model : object
        The regression model instance from Scikit Learn.
    metrics : dict
        Dictionary containing performance metrics.
    """
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.metrics = {}
    
    def train(self, X_train, y_train):
        """Trains the regression model using the training data."""
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluates the model using the test data.
        
        Parameters:
        -----------
        X_test : array-like
            Test features.
        y_test : array-like
            True target values for the test set.
        """
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        self.metrics = {"MSE": mse, "R2": r2}
        return self.metrics

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Split data into training and test sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Instantiate the three regression models
linear_reg = RegressionModel("Linear Regression", LinearRegression())
tree_reg = RegressionModel("Decision Tree Regression", DecisionTreeRegressor(random_state=42))
forest_reg = RegressionModel("Random Forest Regression", RandomForestRegressor(random_state=42))

# List of models for iteration
models = [linear_reg, tree_reg, forest_reg]

# Train and evaluate each model
evaluation_results = {}
for model in models:
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    evaluation_results[model.name] = metrics
    print(f"Model: {model.name}")
    print(f"Mean Squared Error: {metrics['MSE']:.2f}")
    print(f"R2 Score: {metrics['R2']:.2f}\n")

# Determine best performing model based on R2 Score (higher is better)
best_model = max(models, key=lambda m: m.metrics["R2"])

# Brief explanation of the best model's performance
explanation = f"""Best Performing Model: {best_model.name}
The {best_model.name} achieved an R2 score of {best_model.metrics['R2']:.2f} and a Mean Squared Error of {best_model.metrics['MSE']:.2f}.
This indicates that it explains {best_model.metrics['R2']*100:.1f}% of the variance in the target variable, making it the most effective model among those tested.
"""
print(explanation)
