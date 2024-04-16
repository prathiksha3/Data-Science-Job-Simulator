import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.inspection import _partial_dependence

# Load the data
data = pd.read_csv('customer_booking_data.csv')

# Data cleaning and preprocessing (replace this with your data preprocessing steps)
# For simplicity, let's assume the data is clean and doesn't require preprocessing

# Split data into features (X) and target variable (y)
X = data.drop(columns=['flight_hour'])
y = data['flight_hour']

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Hyperparameter Tuning (Example with GridSearchCV)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Model evaluation using best estimator from grid search
best_model = grid_search.best_estimator_
y_test_pred_best = best_model.predict(X_test)
test_accuracy_best = accuracy_score(y_test, y_test_pred_best)
print("Testing Accuracy (Best Model):", test_accuracy_best)
print("Classification Report on Testing Set (Best Model):")
print(classification_report(y_test, y_test_pred_best))

# Interpretation of results
# Analyze feature importances
feature_importances = pd.DataFrame(best_model.feature_importances_,
                                   index=X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print("Feature Importances:")
print(feature_importances)

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances.index, feature_importances['importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importances')
plt.show()

# Partial dependence plots
fig, ax = plt.subplots(figsize=(12, 8))
_partial_dependence.plot_partial_dependence(best_model, X_train, features=[0, 1, (0, 1)], ax=ax)
plt.suptitle('Partial Dependence Plots')
plt.show()
