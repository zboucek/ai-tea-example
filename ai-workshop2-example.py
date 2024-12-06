import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
file_path = 'Workshop02/data/DATA_1.csv'
data = pd.read_csv(file_path, delimiter=';', decimal=',')

# Feature selection
features = ['lpow', 'lspeed', 'dist', 'vol']
target = 'Ra'

X = data[features]
y = data[target]

# Handle missing values if any
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Scaling numerical features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
    "XGBoost": XGBRegressor(random_state=42, n_estimators=100)
}

# Dictionary to store results
results = {}

# Train and evaluate original models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MAE": mae, "MSE": mse, "R²": r2}

# Add Polynomial Regression with Ridge Regularization
poly_pipeline = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
    ("scaler", StandardScaler()),
    ("ridge_regression", Ridge(alpha=1.0))
])

# Train polynomial regression with Ridge
poly_pipeline.fit(X_train, y_train)
y_pred_poly = poly_pipeline.predict(X_test)

# Evaluate Polynomial Regression with Ridge
mae_poly = mean_absolute_error(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

results["Polynomial Regression (Ridge)"] = {"MAE": mae_poly, "MSE": mse_poly, "R²": r2_poly}

# Updated Model Comparison
results_df = pd.DataFrame(results).T
print("Updated Model Comparison:")
print(results_df)

# Plot residuals for each model
plt.figure(figsize=(12, 10))
for i, (name, model) in enumerate(models.items(), start=1):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    plt.subplot(2, 3, i)
    plt.scatter(y_test, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"{name} Residuals")
    plt.xlabel("Actual Values")
    plt.ylabel("Residuals")

# Add residuals for Polynomial Regression (Ridge)
residuals_poly = y_test - y_pred_poly
plt.subplot(2, 3, len(models) + 1)
plt.scatter(y_test, residuals_poly)
plt.axhline(0, color='red', linestyle='--')
plt.title("Polynomial Regression (Ridge) Residuals")
plt.xlabel("Actual Values")
plt.ylabel("Residuals")

plt.tight_layout()
plt.show()
