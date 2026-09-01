import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from features import X_train, X_test, y_train, y_test

# Adding scaling features so all values are at comparable ranges before training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)     

# Simple baseline model. Note that this is now adjusted so the model doesn't guess only up/down
log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train_scaled, y_train)
log_predictions = log_model.predict(X_test_scaled)
log_coefficients = pd.Series(log_model.coef_[0], index=X_train.columns)

print("Logistic Regression")
print(f"Accuracy: {accuracy_score(y_test, log_predictions):.4f}")
print(classification_report(y_test, log_predictions))

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators = 200,
    max_depth = 5,          
    class_weight = "balanced",
    random_state = 42 
)

rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)

print("\n=== Random Forest ===")
print(f"Accuracy: {accuracy_score(y_test, rf_predictions):.4f}")
print(classification_report(y_test, rf_predictions))

naive_accuracy = (y_test == 1).mean()
print(f"\nNaive baseline (always predict 'up'): {naive_accuracy:.4f}")

# What the model is relying on
importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
print("\nFeature importance (Random Forest):")
print(importances.sort_values(ascending=False))