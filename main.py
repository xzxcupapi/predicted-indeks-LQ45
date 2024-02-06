from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from pyswarm import pso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

app = Flask(__name__)

# Load data
data = pd.read_csv("dataset_fix.csv")  # Ganti dengan nama file CSV Anda
data['Date'] = pd.to_datetime(data['Date'])  # Konversi kolom tanggal ke tipe data datetime
data.set_index('Date', inplace=True)  # Set kolom tanggal sebagai indeks
data = data.loc['2021-01-04':'2023-12-29']  # Filter data untuk 3 tahun terakhir

# Preprocessing Data
features = ['Open', 'High', 'Low', 'Close']
X = data[features]
y = data['Close']

# Data Imputation (jika diperlukan)

# Data Normalization (Min Max Scaler)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter Initialization
# Inisialisasi parameter-parameter XGBoost
params = {
    'max_depth': 3,
    'gamma': 0,
    'reg_lambda': 1,
    'base_score': 0.5,
    # Tambahkan parameter lain di sini
}

# Define the XGBoost model
model = xgb.XGBRegressor(**params)

# Train the model
model.fit(X_train, y_train)

# PSO Hyperparameter Tuning
def tune_xgboost(params):
    max_depth = int(params[0])
    # Define XGBoost model with given parameters
    model = xgb.XGBRegressor(
        max_depth=int(max_depth),
        gamma=params[1],
        reg_lambda=params[2],
        base_score=params[3],
        # Tambahkan parameter lain di sini
    )
    # Train the model
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

lb = [3, 0, 0, 0]  # Lower bounds for parameters
ub = [10, 1, 10, 1]  # Upper bounds for parameters

# Perform PSO optimization
best_params, _ = pso(tune_xgboost, lb, ub)

# Update the model with tuned parameters
model = xgb.XGBRegressor(
    max_depth=int(best_params[0]),
    gamma=best_params[1],
    reg_lambda=best_params[2],
    base_score=best_params[3],
    # Tambahkan parameter lain di sini
)
model.fit(X_train, y_train)

# Evaluasi Model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
vaf = 1 - (np.var(y_test - y_pred) / np.var(y_test))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("RMSE:", rmse)
print("R^2:", r2)
print("MAE:", mae)
print("Variance Accounted For (VAF):", vaf)
print("MAPE:", mape)

# Simpan model untuk digunakan di GUI
import joblib
joblib.dump(model, 'xgboost_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from form
        open_price = float(request.form['open'])
        high_price = float(request.form['high'])
        low_price = float(request.form['low'])
        # Scale input values
        input_data = scaler.transform([[open_price, high_price, low_price, 0]])
        # Predict close price
        predicted_close = model.predict(input_data)[0]
        return render_template('index.html', prediction=f'Predicted Close Price: {predicted_close:.2f}')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
