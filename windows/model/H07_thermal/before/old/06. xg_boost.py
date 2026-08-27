import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap

# ==========================================================
# 1. Chargement des données préparées
# ==========================================================
donnees = pd.read_csv(
    r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H07_predict_pure_thermal.csv"
)
donnees["date"] = pd.to_datetime(donnees["date"])
donnees = donnees.sort_values("date").reset_index(drop=True)

# Tạo cột phụ đại diện cho Năm-Tháng để phân nhóm dữ liệu
donnees["annee_mois"] = donnees["date"].dt.to_period("M")

# ==========================================================
# 2. Sélection automatique des variables d'entrée (X) et cible (y)
# ==========================================================
# CHÚ Ý: Loại bỏ 'annee_mois' ra khỏi các biến đầu vào, nhưng GIỮ LẠI 'month' 
# để mô hình toàn cục (Global Model) có thể học được đặc trưng mùa vụ của các tháng cuối năm.
variables_exclues = [
    "date",
    "ec_value",
    "min",
    "year",
    "month",
    "day",
    # "hour",
    "week",
    "id_attraction",
    "annee_mois",
    "heure_sin",
    "heure_cos",
    "mois_sin",
    "mois_cos",
    "hour_x_freq_HF",
    "hour_x_freq_MF",
    "temp_decalage_1h"
]
variables_entree = [col for col in donnees.columns if col not in variables_exclues]
variable_cible = "ec_value"

X = donnees[variables_entree]
y = donnees[variable_cible]

# ==========================================================
# 3. Division temporelle (Blocked Split 80/20 theo từng Tháng)
# ==========================================================
train_indices = []
test_indices = []

# Duyệt qua từng tháng để chia 80/20 liên tục theo thời gian của riêng tháng đó
for name, group in donnees.groupby("annee_mois"):
    # Bỏ qua nếu tháng đó quá ít dữ liệu không đủ chia
    if len(group) < 10:
        continue
        
    split_idx = int(len(group) * 0.8)
    
    # Lấy index tương ứng
    train_indices.extend(group.index[:split_idx])
    test_indices.extend(group.index[split_idx:])

# Tạo các tập dữ liệu Train/Test từ index đã chia
X_train = X.loc[train_indices].reset_index(drop=True)
y_train = y.loc[train_indices].reset_index(drop=True)

X_test = X.loc[test_indices].reset_index(drop=True)
y_test_reel = y.loc[test_indices].reset_index(drop=True)
dates_test = donnees["date"].loc[test_indices].reset_index(drop=True)

# ==========================================================
# 4. Entraînement du mô hình XGBoost Toàn Cục (Global Model)
# ==========================================================
# Giữ nguyên cấu hình tham số chống overfitting tối ưu bạn đã thiết lập
modele = XGBRegressor(
    n_estimators=750,        # Số cây lớn đi kèm với learning rate nhỏ
    learning_rate=0.03,      
    max_depth=3,             # Cây nông giúp giảm overfitting
    subsample=0.75,          
    colsample_bytree=0.8,    
    reg_alpha=2,             # L1 Regularization
    reg_lambda=5,            # L2 Regularization
    min_child_weight=3,      
    random_state=42,
)
modele.fit(X_train, y_train)

# ==========================================================
# 5. Prédictions et évaluation
# ==========================================================
predictions_train = modele.predict(X_train)
predictions_reelles = modele.predict(X_test)
predictions_reelles = np.clip(predictions_reelles, a_min=0, a_max=None)  # Đảm bảo công suất không âm

r2_train = r2_score(y_train, predictions_train)
mse = mean_squared_error(y_test_reel, predictions_reelles)
mae = mean_absolute_error(y_test_reel, predictions_reelles)
r2 = r2_score(y_test_reel, predictions_reelles)

print("\n==========================================================")
print(f"XGBoost Train R²: {r2_train:.4f}")
print(f"XGBoost Test R²: {r2:.4f} | MAE: {mae:.2f} | MSE: {mse:.2f}")
print("==========================================================")

# ==========================================================
# 6. Visualisation avec Plotly
# ==========================================================
# Gom dữ liệu kiểm thử lại và sắp xếp theo trình tự thời gian tăng dần
resultats = pd.DataFrame(
    {
        "Date": dates_test,
        "Valeur_Reelle": y_test_reel,
        "Valeur_Predite": predictions_reelles,
    }
).sort_values("Date").reset_index(drop=True)

dates_formattees = resultats["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=dates_formattees,
        y=resultats["Valeur_Reelle"],
        mode="lines",
        name="Valeur Réelle",
        line=dict(color="darkblue"),
    )
)

fig.add_trace(
    go.Scatter(
        x=dates_formattees,
        y=resultats["Valeur_Predite"],
        mode="lines",
        name="Valeur Prédite",
        line=dict(color="darkorange", dash="dash"),
    )
)

titre_graphe = (
    f"Prévisions ec_value via XGBoost | R²: {r2:.4f} | MAE: {mae:.2f} | MSE: {mse:.2f}"
)
fig.update_layout(
    title=titre_graphe,
    xaxis_title="Date et Heure",
    yaxis_title="ec_value",
    hovermode="x unified",
    xaxis=dict(rangeslider=dict(visible=True), type="date"),
)
fig.write_html("prévisions_xgb.html")
fig.show()

# ==========================================
# BƯỚC 5.5: GIẢI THÍCH MÔ HÌNH BẰNG PHƯƠNG PHÁP SHAP
# ==========================================
# (Mở comment đoạn dưới nếu bạn muốn chạy phân tích SHAP)
# explainer = shap.TreeExplainer(modele)
# shap_values = explainer(X_test)
# print("\n--- Đang hiển thị biểu đồ SHAP Bar Plot ---")
# shap.plots.bar(shap_values, max_display=15)
# print("\n--- Đang hiển thị biểu đồ SHAP Beeswarm Plot ---")
# shap.plots.beeswarm(shap_values, max_display=15)