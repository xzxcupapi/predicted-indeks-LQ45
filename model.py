import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb  # Ubah nama modul dari "model" menjadi "xgboost"
from sklearn.metrics import mean_squared_error
import joblib

# Persiapan Data
data = pd.read_csv('dataset_fix.csv')  
data['Date'] = pd.to_datetime(data['Date'])

# Feature Engineering 
data['Return'] = (data['Close'] - data['Open']) / data['Open']

# Pembagian Data
X = data[['High', 'Low', 'Open', 'Volume']]  
y = data['Close']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pelatihan Model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Evaluasi Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Simpan Model
joblib.dump(model, 'xgboost_model.pkl')


