import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================================
# 1. Chargement des données (Đọc dữ liệu Elec gộp)
# ==========================================================
file_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H07_predict_pure_elec.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {file_path}")

donnees = pd.read_csv(file_path)
donnees["date"] = pd.to_datetime(donnees["date"])
donnees = donnees.sort_values("date").reset_index(drop=True)

# Tạo cột phụ đại diện cho Năm-Tháng để phân nhóm dữ liệu
donnees["annee_mois"] = donnees["date"].dt.to_period("M")

# ==========================================================
# 2. Sélection des variables (Theo phong cách Global Model)
# ==========================================================
# Cột mục tiêu song song (Multi-output)
variable_cible = ["elec_1", "elec_2", "elec_3"]

# Loại bỏ 'annee_mois', 'date' và các cột mục tiêu, GIỮ LẠI cột tháng
variables_exclues = [
    "date", 
    "elec_1", 
    "elec_2",
    "elec_3", 
    "min", 
    "year", 
    "day", 
    "hour", 
    "week",
    "annee_mois",
    # "temp_max",
    # "temp_min",
    # "temp_decalage_1h",
    # "mois_cos",
    # "mois_sin",
    # "humidite_max",
    # "humidite_min",
    # "heure_cos",
    # "heure_sin",
]
variables_entree = [col for col in donnees.columns if col not in variables_exclues]

X = donnees[variables_entree]
y = donnees[variable_cible]

# ==========================================================
# 3. Blocked Split theo Tháng (Gom 80% Train và 20% Test của mỗi tháng)
# ==========================================================
train_indices = []
test_indices = []

# Duyệt qua từng tháng để lấy index phân chia 80% đầu và 20% cuối của riêng tháng đó
for name, group in donnees.groupby("annee_mois"):
    if len(group) < 10:
        continue
        
    split_idx = int(len(group) * 0.8)
    
    # Gom index tương ứng từ các tháng
    train_indices.extend(group.index[:split_idx])
    test_indices.extend(group.index[split_idx:])

# Tạo các tập dữ liệu Train/Test lớn (Global Train / Global Test) từ index đã gom
X_train = X.loc[train_indices].reset_index(drop=True)
y_train = y.loc[train_indices].reset_index(drop=True)

X_test = X.loc[test_indices].reset_index(drop=True)
y_test_reel = y.loc[test_indices].reset_index(drop=True)
dates_test = donnees["date"].loc[test_indices].reset_index(drop=True)

# ==========================================================
# 4. Entraînement du mô hình GBR Toàn Cục (Multi-output)
# ==========================================================
# Khởi tạo mô hình GBR nền tảng với bộ siêu tham số tối ưu chống overfitting của bạn
base_gbr = GradientBoostingRegressor(
    n_estimators=300, 
    learning_rate=0.03,
    max_depth=4, 
    subsample=0.8,
    max_features=0.8,
    random_state=42,
)

# Bọc qua MultiOutputRegressor để có thể dự báo đồng thời elec_1 và elec_2
modele = MultiOutputRegressor(base_gbr)

# Huấn luyện mô hình toàn cục
modele.fit(X_train, y_train)

# ==========================================================
# 5. Prédictions et évaluation
# ==========================================================
predictions_train = modele.predict(X_train)
predictions_reelles = modele.predict(X_test)

# Đảm bảo giá trị công suất điện dự báo không bao giờ âm (Cắt âm vật lý)
predictions_reelles = np.clip(predictions_reelles, a_min=0, a_max=None)

# Đánh giá chỉ số chung (Trung bình đồng đều của cả 2 cột)
r2_train = r2_score(y_train, predictions_train, multioutput="uniform_average")
mse_global = mean_squared_error(y_test_reel, predictions_reelles, multioutput="uniform_average")
mae_global = mean_absolute_error(y_test_reel, predictions_reelles, multioutput="uniform_average")
r2_global = r2_score(y_test_reel, predictions_reelles, multioutput="uniform_average")

# Đánh giá chi tiết cho riêng từng cột elec_1 và elec_2
r2_chi_tiet = r2_score(y_test_reel, predictions_reelles, multioutput="raw_values")
mae_chi_tiet = mean_absolute_error(y_test_reel, predictions_reelles, multioutput="raw_values")

print("=== KẾT QUẢ MÔ HÌNH TOÀN CỤC (GLOBAL GBR MODEL) - MULTI-OUTPUT ===")
print(f"GBR Train R² (Trung bình): {r2_train:.4f}")
print(f"GBR Test R²  (Trung bình): {r2_global:.4f} | MAE (Trung bình): {mae_global:.2f}")
print("-" * 65)
print(f" -> [elec_1] R² Test: {r2_chi_tiet[0]:.4f} | MAE: {mae_chi_tiet[0]:.2f}")
print(f" -> [elec_2] R² Test: {r2_chi_tiet[1]:.4f} | MAE: {mae_chi_tiet[1]:.2f}")
print(f" -> [elec_3] R² Test: {r2_chi_tiet[2]:.4f} | MAE: {mae_chi_tiet[2]:.2f}")
print("==================================================================\n")

# ==========================================================
# 6. Visualisation avec Plotly (Gom kết quả và Trực quan hóa)
# ==========================================================
resultats = pd.DataFrame(
    {
        "Date": dates_test,
        "elec_1_Reelle": y_test_reel["elec_1"],
        "elec_1_Predite": predictions_reelles[:, 0],
        "elec_2_Reelle": y_test_reel["elec_2"],
        "elec_2_Predite": predictions_reelles[:, 1],
        "elec_3_Reelle": y_test_reel["elec_3"],
        "elec_3_Predite": predictions_reelles[:, 2],
    }
).sort_values("Date").reset_index(drop=True)

dates_formattees = resultats["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")

fig = go.Figure()

# --- BIỂU DIỄN CHO ELEC_1 (Tông màu Xanh lam) ---
fig.add_trace(
    go.Scatter(
        x=dates_formattees,
        y=resultats["elec_1_Reelle"],
        mode="lines",
        name=f"elec_1 Réelle (Test) | R²: {r2_chi_tiet[0]:.3f}",
        line=dict(color="darkblue"),
    )
)
fig.add_trace(
    go.Scatter(
        x=dates_formattees,
        y=resultats["elec_1_Predite"],
        mode="lines",
        name="elec_1 Prédite GBR (Global)",
        line=dict(color="dodgerblue", dash="dash"),
    )
)

# --- BIỂU DIỄN CHO ELEC_2 (Tông màu Đỏ / Cam) ---
fig.add_trace(
    go.Scatter(
        x=dates_formattees,
        y=resultats["elec_2_Reelle"],
        mode="lines",
        name=f"elec_2 Réelle (Test) | R²: {r2_chi_tiet[1]:.3f}",
        line=dict(color="darkred"),
    )
)
fig.add_trace(
    go.Scatter(
        x=dates_formattees,
        y=resultats["elec_2_Predite"],
        mode="lines",
        name="elec_2 Prédite GBR (Global)",
        line=dict(color="orange", dash="dash"),
    )
)

# --- BIỂU DIỄN CHO ELEC_3 (Tông màu Xanh lá) ---
fig.add_trace(
    go.Scatter(
        x=dates_formattees,
        y=resultats["elec_3_Reelle"],
        mode="lines",
        name=f"elec_3 Réelle (Test) | R²: {r2_chi_tiet[2]:.3f}",
        line=dict(color="darkgreen"),
    )
)
fig.add_trace(
    go.Scatter(
        x=dates_formattees,
        y=resultats["elec_3_Predite"],
        mode="lines",
        name="elec_3 Prédite GBR (Global)",
        line=dict(color="lightgreen", dash="dash"),
    )
)

titre_graphe = (
    f"Prévisions Globales (elec_1 & elec_2 & elec_3) via GBR (Blocked Split) | Avg R²: {r2_global:.4f}"
)
fig.update_layout(
    title=titre_graphe,
    xaxis_title="Date et Heure",
    yaxis_title="Consommation Électrique (kWh)",
    hovermode="x unified",
    xaxis=dict(rangeslider=dict(visible=True), type="date"),
)

fig.show()