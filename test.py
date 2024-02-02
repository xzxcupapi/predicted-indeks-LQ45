import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from pyswarm import pso
import joblib
from flask import Flask, render_template, request

# Persiapan Data
data = pd.read_csv('dataset_fix.csv')  
data['Date'] = pd.to_datetime(data['Date'])

# Feature Engineering 
data['Return'] = (data['Close'] - data['Open']) / data['Open']

# Pembagian Data
X = data[['High', 'Low', 'Open', 'Volume']]  
y = data['Close']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Penskalaan Fitur
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fungsi objektif untuk dioptimalkan
def objective_function(params):
    learning_rate, max_depth, n_estimators = params
    model = xgb.XGBRegressor(learning_rate=learning_rate, max_depth=int(max_depth), n_estimators=int(n_estimators))
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Tentukan batasan (ruang pencarian) untuk setiap parameter
lb = [0.01, 1, 10]  # Batas bawah untuk learning rate, max_depth, n_estimators
ub = [0.3, 10, 100]  # Batas atas untuk learning rate, max_depth, n_estimators

# Lakukan optimasi PSO
best_params, best_mse = pso(objective_function, lb, ub)

# Simpan Model
learning_rate, max_depth, n_estimators = best_params
best_model = xgb.XGBRegressor(learning_rate=learning_rate, max_depth=int(max_depth), n_estimators=int(n_estimators))
best_model.fit(X_train_scaled, y_train)
joblib.dump(best_model, 'xgboost_model.pkl')

# Inisialisasi Aplikasi Flask
app = Flask(__name__)

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
        input_data = scaler.transform([[high_price, low_price, open_price, 0]])  # Sesuaikan urutan fitur dengan urutan saat training
        # Predict close price
        predicted_close = best_model.predict(input_data)[0]
        return render_template('index.html', prediction=f'Predicted Close Price: {predicted_close:.2f}')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
