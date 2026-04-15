# Task 1: Create Dataset & Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


np.random.seed(42)
n = 60

area_sqft = np.random.randint(500, 3000, n)
num_bedrooms = np.random.randint(1, 5, n)
age_years = np.random.randint(0, 30, n)


price_lakhs = (
    area_sqft * 0.05 +
    num_bedrooms * 10 -
    age_years * 0.3 +
    np.random.normal(0, 5, n)
)

data = pd.DataFrame({
    'area_sqft': area_sqft,
    'num_bedrooms': num_bedrooms,
    'age_years': age_years,
    'price_lakhs': price_lakhs
})

X = data[['area_sqft', 'num_bedrooms', 'age_years']]
y = data['price_lakhs']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

print("\nFirst 5 Actual vs Predicted:")
print(comparison.head())

# Task 2: Evaluation Metrics

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
# Task 3: Residual Analysis
residuals = y_test - y_pred

plt.figure()
plt.hist(residuals, bins=10)
plt.title("Residuals Distribution")
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.show()
