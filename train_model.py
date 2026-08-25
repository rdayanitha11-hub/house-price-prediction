import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Load dataset
df = pd.read_csv("Housing.csv")

# Convert categorical data into numbers
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop("price", axis=1)
y = df["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("Model trained successfully!")
print("R2 Score:", round(r2, 4))

# Save model + feature columns
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(
        {
            "model": model,
            "features": X.columns.tolist(),
            "r2_score": r2
        },
        file
    )

print("Model saved as house_price_model.pkl")