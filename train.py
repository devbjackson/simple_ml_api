import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

# 1. Generate Synthetic Data
# For a real project, this would be loaded from a database, CSV, etc.
print("Generating synthetic data...")
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]) # Features
y = np.array([2, 4, 5, 4, 5, 8, 9, 10, 11, 12]) # Target variable (y = 1.0 * X + 1.0 + noise)

# 2. Train a Simple Linear Regression Model
print("Training Linear Regression model...")
model = LinearRegression()
model.fit(X, y)

# 3. Print Model Coefficients (optional, for verification)
print(f"Model trained. Coefficients: {model.coef_}, Intercept: {model.intercept_}")

# 4. Save the Trained Model
# We'll save it to a file using joblib.
model_filename = 'model.joblib'
joblib.dump(model, model_filename)
print(f"Model saved as '{model_filename}'")

# Verify file existence (optional)
if os.path.exists(model_filename):
    print("Model file created successfully.")
else:
    print("Error: Model file was not created.")