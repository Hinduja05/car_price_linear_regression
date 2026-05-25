import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = pd.read_csv(
    "../data/processed/cleaned_no_outliers.csv"
)

# ---------------------------------------------------
# FEATURES
# ---------------------------------------------------
features = [
    "Year",
    "Engine_Size",
    "Mileage",
    "Doors",
    "Owner_Count"
]
x = df[features]
y = df["Price"]

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# SCALING
# ---------------------------------------------------
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------

joblib.dump(
    model,
    "../models/linear_regression_model.pkl"
)

joblib.dump(
    scaler,
    "../models/scaler.pkl"
)
print("Model saved successfully")