import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# ==============================================================================
# 1. ĐỌC DỮ LIỆU FILE MASTER BRUT
# ==============================================================================
# ---> BẠN CHỈNH LẠI ĐƯỜNG DẪN TỚI FILE CSV BÊN ĐƯỚI CHO ĐÚNG BỘ DỮ LIỆU CỦA BẠN <---
master_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master\master_H03_elec.csv"

if not os.path.exists(master_path):
  raise FileNotFoundError(f"Không tìm thấy file tại đường dẫn: {master_path}")

print("-> Đang tải dữ liệu...")
df_master = pd.read_csv(master_path)

# ==============================================================================
# 2. ĐIỀU CHỈNH & CHUẨN BỊ MẤU ĐÁNH GIÁ (EVALUATION SET)
# ==============================================================================
# Đảm bảo các biến thời gian cơ bản được cập nhật
df_master['date'] = pd.to_datetime(df_master['date'])
df_master['month'] = df_master['date'].dt.month
df_master['hour'] = df_master['date'].dt.hour

# Lọc chỉ lấy các khung giờ công viên MỞ và ĐÃ CÓ dữ liệu visitor_count thực tế (2024-2025)
mask_eval = (df_master['is_open'] == 1) & (df_master['visitor_count'].notnull())
df_eval = df_master[mask_eval].copy()

# Chọn các biến đặc trưng (Features) không chứa điện năng
features_eval = [
    'hour',
    'month',
    'frequentation_park_daily',
    'is_weekend',
    'jf',
    'operation',
    "ouvert",
    "interrompu"
]
for opt in ['temperature', 'rayonnement_solaire']:
  if opt in df_eval.columns and df_eval[opt].isnull().sum() == 0:
    features_eval.append(opt)

print(f"-> Danh sách đặc trưng đưa vào đánh giá: {features_eval}")

# Mã hóa One-Hot Encoding cho biến type_frequentation
X = pd.get_dummies(
    df_eval[features_eval + ['type_frequentation']],
    columns=['type_frequentation'],
    drop_first=True,
)
y = df_eval['visitor_count']

# ==============================================================================
# 3. ĐÁNH GIÁ MÔ HÌNH BẰNG 5-FOLD CROSS VALIDATION
# ==============================================================================
print("-> Đang tiến hành kiểm tra chéo (Cross-Validation)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores, mae_scores, rmse_scores = [], [], []

for train_idx, val_idx in kf.split(X):
  X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
  y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

  model = RandomForestRegressor(
      n_estimators=150, max_depth=14, random_state=42, n_jobs=-1
  )
  model.fit(X_tr, y_tr)

  y_pred = model.predict(X_val)

  r2_scores.append(r2_score(y_val, y_pred))
  mae_scores.append(mean_absolute_error(y_val, y_pred))
  rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

# ==============================================================================
# 4. IN BÁO CÁO KẾT QUẢ VÀ ĐỘ QUAN TRỌNG CỦA CÁC BIẾN (FEATURE IMPORTANCES)
# ==============================================================================
print("\n" + "=" * 65)
print("BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH DỰ BÁO VISITOR_COUNT")
print("=" * 65)
print(
    f"1. Độ phù hợp R² Score : {np.mean(r2_scores):.4f} (+/-"
    f" {np.std(r2_scores):.4f})"
)
print(f"2. Lỗi trung bình MAE   : {np.mean(mae_scores):.2f} lượt khách/giờ")
print(f"3. Lỗi căn RMSE        : {np.mean(rmse_scores):.2f} lượt khách/giờ")
print("=" * 65)

# Huấn luyện trên toàn bộ tập eval để xem Feature Importances
model.fit(X, y)
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(
    ascending=False
)
print("\nTOP CÁC YẾU TỐ ẢNH HƯỜNG NHẤT TỚI DỰ BÁO LƯỢT KHÁCH H03:")
print(importances.head(8).to_string())