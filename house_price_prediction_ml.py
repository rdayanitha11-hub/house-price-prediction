import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Housing.csv")

print("Dataset Loaded Successfully")
print(df.head())

# =========================
# CONVERT TEXT TO NUMBERS
# =========================
df = pd.get_dummies(df, drop_first=True)

# Split data
X = df.drop("price", axis=1)
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Training Completed")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nR2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Sample prediction
sample = X_test.iloc[0].values.reshape(1, -1)
print("\nSample Prediction:", model.predict(sample)[0])