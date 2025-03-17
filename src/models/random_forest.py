from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

'''NOTE: This code only works for a snapshot of the vitals at a certain time.
Therefore, we could use three different approaches.
1) We could just use the data sequentially, then poll the results of each inference.
2) Before running the model we could find the mean, median, etc... of the data then run inference on those.
3) We could use an LSTM to extract the time series data then input the final layer in the RF.
'''

# Example dataset TODO: replace this with real, cleaned, and preprocessed data.
# 0 -> Low
# 1 -> Normal
# 2 -> High
X = [[8, 70, 12, 90, 110, 60], [10, 75, 8, 80, 120, 70]]
y = [0, 1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid to perform the grid search with
# These will be used to find the best hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [None, 5, 10, 20],  # Maximum depth of trees
    'max_features': ['sqrt', 'log2'],  # Number of features to consider at each split
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]  # Minimum samples required at each leaf node
}

# Create the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='accuracy',  # Metric to optimize
    cv=5,  # 5-fold cross-validation
)

# Perform grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the model with the best hyperparameters
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
print("Test Set Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature importance
print("Feature Importances:", best_model.feature_importances_)