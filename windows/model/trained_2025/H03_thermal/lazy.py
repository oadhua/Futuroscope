import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lazypredict.Supervised import LazyRegressor
import sklearn.utils._testing
# Import trực tiếp các mô hình chạy nhanh và hiệu quả nhất từ thư viện gốc
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ==========================================================
# 1. Chargement des données préparées (Đọc dữ liệu Nhiệt)
# ==========================================================
donnees = pd.read_csv(
    r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H03_predict_pure_thermal.csv"
)
donnees["date"] = pd.to_datetime(donnees["date"])
donnees = donnees.sort_values("date").reset_index(drop=True)

# ==========================================================
# 2. Sélection automatique des variables d'entrée (X) et cible (y)
# ==========================================================
variables_exclues = ['date', 'ec_value', 'min', 'year', 'month', 'day', 'hour', 'week']
variables_entree = [col for col in donnees.columns if col not in variables_exclues]
variable_cible = 'ec_value'

X = donnees[variables_entree]
y = donnees[variable_cible]

# ==========================================================
# 3. Division temporelle (Time-Based Split 80/20)
# ==========================================================
# Giữ nguyên trình tự thời gian để tránh rò rỉ dữ liệu (Data Leakage)
taille_entrainement = int(len(X) * 0.8)

X_train, X_test = X.iloc[:taille_entrainement], X.iloc[taille_entrainement:]
y_train, y_test = y.iloc[:taille_entrainement], y.iloc[taille_entrainement:]

print(f"Tổng số dòng dữ liệu: {len(donnees)}")
print(f"Kích thước tập Train: {X_train.shape[0]} dòng")
print(f"Kích thước tập Test:  {X_test.shape[0]} dòng")
print("\n--- BẮT ĐẦU CHẠY THỬ NGHIỆM ĐA MÔ HÌNH VỚI LAZYPREDICT ---")
print("*(Quá trình này có thể mất vài phút tùy thuộc vào cấu hình máy)*\n")

# Khởi tạo bộ kiểm thử LazyRegressor gốc
reg_models = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

# Định nghĩa danh sách các CLASS mô hình an toàn thuần túy (Không dùng chuỗi string)
# Đây là các thuật toán tối ưu, chạy siêu tốc và rất phù hợp cho chuỗi dữ liệu năng lượng
nhom_mo_hinh_sieu_toc = [
    XGBRegressor,
    LGBMRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
]

# Gán thẳng danh sách class sạch này vào thuộc tính regressors của LazyPredict
reg_models.regressors = nhom_mo_hinh_sieu_toc

print(f"-> Đã gán cứng cấu hình thành công với {len(nhom_mo_hinh_sieu_toc)} mô hình tối ưu.")
print("-> Bắt đầu huấn luyện một lèo (Chắc chắn vọt mượt đến 100% không lo lỗi)...")

# Chạy lệnh fit
models, predictions = reg_models.fit(X_train, X_test, y_train, y_test)

# ==========================================================
# 5. Xuất bảng xếp hạng hiệu năng mô hình (Metrics)
# ==========================================================
print("\n==========================================================")
print("BẢNG XẾP HẠNG HIỆU NĂNG CÁC MÔ HÌNH (SẮP XẾP THEO R²):")
print("==========================================================")
# Hiển thị toàn bộ bảng kết quả thay vì bị ẩn bớt cột/dòng
pd.set_option('display.max_rows', None)
print(models)
print("==========================================================")

# Xuất kết quả ra file CSV để bạn có thể mở bằng Excel hoặc vẽ đồ thị làm báo cáo
output_csv_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\model\H03_thermal\lazy_predict_results.csv"
models.to_csv(output_csv_path)
print(f"\n-> Đã lưu bảng kết quả chi tiết vào: {output_csv_path}")