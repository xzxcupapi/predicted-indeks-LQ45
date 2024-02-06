import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import joblib
from flask import Flask, render_template, request

# Persiapan Data
data = pd.read_csv('dataset_fix.csv')  
data['Date'] = pd.to_datetime(data['Date'])

# Feature Engineering 
data['Return'] = (data['Close'] - data['Open']) / data['Open']

# Imputasi nilai yang hilang
imputer = SimpleImputer(strategy='mean')  
data[['High', 'Low', 'Open', 'Volume']] = imputer.fit_transform(data[['High', 'Low', 'Open', 'Volume']])


# Pembagian Data
X = data[['High', 'Low', 'Open', 'Volume']]  
y = data['Close']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Penskalaan Fitur
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Pelatihan Model
model = xgb.XGBRegressor()
model.fit(X_train_scaled, y_train)

# Evaluasi Model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
r2 = r2_score(y_test, y_pred)
print("R^2:", r2)
mae = mean_absolute_percentage_error(y_test, y_pred)
print("MAE:", mae)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print("MAPE:", mape)

# Simpan Model
joblib.dump(model, 'xgboost_model.pkl')

# Inisialisasi Aplikasi Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


def denormalize(scaled_value, scaler):
    min_val = scaler.data_min_[3]
    max_val = scaler.data_max_[3]
    return scaled_value * (max_val - min_val) + min_val

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
        predicted_close = model.predict(input_data)[0]
        return render_template('index.html', prediction=f'Predicted Close Price: {predicted_close:.2f}')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
