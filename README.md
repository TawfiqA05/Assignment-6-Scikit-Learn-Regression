# Assignment-6-Scikit-Learn-Regression

## Purpose
This project applies machine learning regression using Scikit Learn on the built-in diabetes dataset. The goal is to build and evaluate three regression models to predict disease progression.

## Project Structure
- **diabetes_regression.py**: Contains the implementation of three regression models (Linear Regression, Decision Tree Regression, Random Forest Regression) using a custom `RegressionModel` class.
- **README.md**: Provides an overview of the project, the class design, and explanations.
- **chat_log.txt**: Contains the generative AI chat log used during the development of this project.

## Class Design and Implementation
### RegressionModel Class
- **Purpose**: Encapsulates functionalities for training and evaluating a regression model.
- **Attributes**:
  - `name`: The name of the regression model.
  - `model`: The Scikit Learn model instance.
  - `metrics`: A dictionary to store evaluation metrics (MSE and R2 Score).
- **Methods**:
  - `train(X_train, y_train)`: Trains the model with the provided training data.
  - `evaluate(X_test, y_test)`: Evaluates the model on test data, calculating Mean Squared Error and R2 Score.

## Limitations
- Evaluation is based on a single train-test split.
- Hyperparameter tuning and cross-validation are not performed.
- The project focuses solely on the three chosen regression models; other models might yield different performance outcomes.

