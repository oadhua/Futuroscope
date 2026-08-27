import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Đọc file
df = pd.read_csv("D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H03_predict_pure_thermal.csv")

# Kiểm tra tất cả các cột
print("Tất cả các cột:", df.columns.tolist())

# Danh sách các cột nguyên bản (chưa qua feature engineering)
# Loại bỏ các đặc trưng biến đổi/lag/rolling/sin-cos/interaction:
# - heure_sin, heure_cos, mois_sin, mois_cos
# - hour_x_freq_HF, hour_x_freq_MF
# - temp_decalage_1h, temp_decalage_2h, temp_decalage_3h, temp_roll_mean_3h, temp_roll_mean_6h
# - date (cột thời gian chuỗi)

raw_features = [
    'year', 'month', 'day', 'hour', 'week', 'is_weekend', 'is_open', 'jf',
    'frequentation_park_daily', 'surface', 'duree', 'ouvert', 'interrompu',
    'operation', 'visitor_count', 'temperature', 'humidite', 'rayonnement_solaire',
    'day_degree_cold', 'day_degree_hot', 'temp_max', 'temp_min', 'temp_moy',
    'humidite_max', 'humidite_min', 'humidite_moy', 'freq_HF', 'freq_MF', 'freq_THF'
]

print("\nSố lượng đặc trưng nguyên bản:", len(raw_features))
print("Danh sách:", raw_features)

# Split Train (< 2026) & Test (>= 2026)
train_mask = df["year"] < 2026
test_mask = df["year"] >= 2026

# Target 1: elec_1
X_train = df.loc[train_mask, raw_features]
y_train_1 = df.loc[train_mask, "ec_value"]
X_test = df.loc[test_mask, raw_features]
y_test_1 = df.loc[test_mask, "ec_value"]

# Train XGBoost cho elec_1
xgb_1 = XGBRegressor(n_estimators=300, learning_rate=0.04, max_depth=6, random_state=42)
xgb_1.fit(X_train, y_train_1)
pred_1 = xgb_1.predict(X_test)


# Đánh giá
def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

r1, m1, r2_1 = get_metrics(y_test_1, pred_1)

print(f"\n--- KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST 2026 (RAW FEATURES) ---")
print(f"[ec_value] RMSE: {r1:.4f} kWh | MAE: {m1:.4f} kWh | R²: {r2_1:.4f}")

# Top 10 feature importance cho elec_1
imp_1 = pd.Series(xgb_1.feature_importances_, index=raw_features).sort_values(ascending=False)
print("\nTop 10 đặc trưng quan trọng nhất (elec_1):")
print(imp_1.head(10))