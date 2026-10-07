import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import numpy as np

# 1. Generate Synthetic Logistics Data
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    'distance_km': np.random.uniform(5, 500, n_samples),
    'traffic_index': np.random.randint(1, 10, n_samples),
    'weather_condition': np.random.choice([0, 1, 2], n_samples), # 0:Clear, 1:Rain, 2:Snow
    'truck_type': np.random.choice([0, 1], n_samples) # 0:Standard, 1:Refrigerated
})
# Target: ETA = distance/60 + traffic*5 + weather*10 + noise
data['actual_eta_minutes'] = (data['distance_km'] / 60 * 60) + (data['traffic_index'] * 5) + (data['weather_condition'] * 10) + np.random.normal(0, 5, n_samples)

# 2. Train/Test Split
X = data[['distance_km', 'traffic_index', 'weather_condition', 'truck_type']]
y = data['actual_eta_minutes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train LightGBM Model
print("Training LightGBM Regressor...")
model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"Model trained! Mean Absolute Error: {mae:.2f} minutes")

# 5. Save Model
joblib.dump(model, 'lgbm_eta_model.pkl')
print("Model saved to lgbm_eta_model.pkl")
