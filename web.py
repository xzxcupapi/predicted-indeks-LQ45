from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

app = Flask(__name__)

# Load data
data = pd.read_csv("dataset_fix.csv")  # Ganti dengan nama file CSV Anda

# Imputasi nilai yang hilang hanya untuk kolom numerik
numeric_cols = data.select_dtypes(include=np.number).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Preprocessing Data
features = ['Open', 'High', 'Low', 'Close']
X = data[features]
y = data['Close']

# Data Normalization (Min Max Scaler)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter Initialization
params = {
    'max_depth': 6,       # Mengubah kedalaman maksimum menjadi 6
    'gamma': 0.1,         # Mengubah parameter gamma
    'reg_lambda': 0.5,    # Mengubah parameter reg_lambda
    'learning_rate': 0.1, # Menambahkan learning rate
    'n_estimators': 100   # Menambahkan jumlah estimators
}

# Define the XGBoost model
model = xgb.XGBRegressor(**params)

# Train the model
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

# Menyimpan hasil aktual untuk digunakan di Flask
y_test_actual = y_test.values.tolist()

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

        # Denormalize the predicted close price
        predicted_close_denormalized = scaler.inverse_transform([[0, 0, 0, predicted_close]])[0][3]
        
        # Get the actual close price (denormalized)
        actual_close_denormalized = y_test.mean() # Just using the mean as a placeholder for actual value
        
        # Render predicted and actual close prices in the web interface
        return render_template('index.html', prediction=f'Predicted Close Price: {predicted_close_denormalized:.2f}', actual=f'Actual Close Price: {actual_close_denormalized:.2f}')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
