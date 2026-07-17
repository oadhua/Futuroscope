import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
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

# Tạo cột phụ để phân nhóm theo Tháng-Năm
donnees["annee_mois"] = donnees["date"].dt.to_period("M")

# ==========================================================
# 2. Sélection des variables d'entrée (X) et cible (y)
# ==========================================================
# Giữ lại 'month' (hoặc các biến chu kỳ tháng) để mô hình phân biệt mùa vụ
variables_exclues = [
    "date", 
    "ec_value", 
    "min", 
    "year", 
    "day", 
    "hour", 
    "week",
    "annee_mois", # Cột phụ dùng để group dữ liệu
    # "temp_max",
    # "temp_min",
    # "temperature",
    # "temp_decalage_1h",
    # # "mois_cos",
    # # "mois_sin",
    # "humidite_max",
    # # "humidite_min",
    # # "heure_cos",
    # # "heure_sin",
    # "hour_x_freq_HF",
    # "hour_x_freq_MF"
]
variables_tiem_nang = [col for col in donnees.columns if col not in variables_exclues]
variable_cible = "ec_value"

X_temp = donnees[variables_tiem_nang]
y_temp = donnees[variable_cible]

# ----------------------------------------------------------
# BƯỚC 2.5: FEATURE SELECTION BẰNG KENDALL CORRELATION (THÊM MỚI VÀO ĐÂY)
# ----------------------------------------------------------
print("\n--- BẮT ĐẦU LỌC BIẾN BẰNG KENDALL CORRELATION ---")

# Tạo dataframe tạm thời ghép X và y để tính tương quan
df_temp = X_temp.copy()
df_temp['target'] = y_temp

# Tính toán ma trận tương quan Kendall
print("Đang tính toán ma trận tương quan Kendall (vui lòng đợi vài giây)...")
corr_matrix = df_temp.corr(method='kendall')

# 1. Loại bỏ các biến đầu vào quá trùng lặp thông tin với nhau (Multicollinearity)
nguong_trung_lap = 0.80  # Loại bỏ bớt nếu 2 biến X tương quan với nhau > 0.8
bien_loai_bo = set()

for i in range(len(corr_matrix.columns) - 1):
    for j in range(i):
        col_i = corr_matrix.columns[i]
        col_j = corr_matrix.columns[j]
        if col_i != 'target' and col_j != 'target':
            if abs(corr_matrix.iloc[i, j]) > nguong_trung_lap:
                # So sánh tương quan của từng biến với target để loại bỏ biến yếu hơn
                target_corr_i = abs(corr_matrix.loc[col_i, 'target'])
                target_corr_j = abs(corr_matrix.loc[col_j, 'target'])
                if target_corr_i < target_corr_j:
                    bien_loai_bo.add(col_i)
                else:
                    bien_loai_bo.add(col_j)

print(f"  -> Đã loại bỏ {len(bien_loai_bo)} biến bị trùng lặp thông tin chéo: {list(bien_loai_bo)}")

# 2. Giữ lại các biến có tương quan thực sự ý nghĩa với biến mục tiêu (Relevance)
tuong_quan_target = corr_matrix['target'].drop('target').drop(labels=list(bien_loai_bo), errors='ignore')
tuong_quan_target_sap_xep = tuong_quan_target.abs().sort_values(ascending=False)

# Đặt ngưỡng tương quan tối thiểu với biến mục tiêu (ví dụ: |tau| >= 0.05)
nguong_tuong_quan_target = 0.05
variables_entree = tuong_quan_target_sap_xep[tuong_quan_target_sap_xep >= nguong_tuong_quan_target].index.tolist()

print(f"  -> Giữ lại {len(variables_entree)} biến có tương quan Kendall với target >= {nguong_tuong_quan_target}:")
for feat in variables_entree:
    print(f"     [+] {feat:<25} | Tau = {corr_matrix.loc[feat, 'target']:.4f}")
print("----------------------------------------------------------\n")

X = donnees[variables_entree]
y = donnees[variable_cible]

# ==========================================================
# 3. Blocked Split theo Tháng (Tránh leak dữ liệu & Giữ tính đại diện)
# ==========================================================
train_indices = []
test_indices = []

# Duyệt qua từng tháng để chia 80/20 theo trục thời gian liên tục của riêng tháng đó
for name, group in donnees.groupby("annee_mois"):
    # Bỏ qua nếu tháng đó quá ít dữ liệu không đủ chia
    if len(group) < 10:
        continue
        
    split_idx = int(len(group) * 0.8)
    
    # 80% thời gian đầu tháng làm Train, 20% cuối tháng làm Test
    train_indices.extend(group.index[:split_idx])
    test_indices.extend(group.index[split_idx:])

# Tạo các tập dữ liệu Train/Test từ index đã chia
X_train = X.loc[train_indices].reset_index(drop=True)
y_train = y.loc[train_indices].reset_index(drop=True)

X_test = X.loc[test_indices].reset_index(drop=True)
y_test_reel = y.loc[test_indices].reset_index(drop=True)
dates_test = donnees["date"].loc[test_indices].reset_index(drop=True)

# ==========================================================
# 4. Entraînement du modèle Gradient Boosting (XGBoost)
# ==========================================================
modele = GradientBoostingRegressor(
    n_estimators=300, 
    learning_rate=0.03,
    max_depth=4, 
    subsample=0.8,
    max_features=0.8,
    random_state=42,
)
modele.fit(X_train, y_train)

# ==========================================================
# 5. Prédictions et évaluation
# ==========================================================
predictions_train = modele.predict(X_train)
predictions_reelles = modele.predict(X_test)
predictions_reelles = np.clip(predictions_reelles, a_min=0, a_max=None)

r2_train = r2_score(y_train, predictions_train)
mse = mean_squared_error(y_test_reel, predictions_reelles)
mae = mean_absolute_error(y_test_reel, predictions_reelles)
r2 = r2_score(y_test_reel, predictions_reelles)

print(f"Gradient Boosting Train R²: {r2_train:.4f}")
print(f"Gradient Boosting Test R² (Blocked Split): {r2:.4f} | MAE: {mae:.2f} | MSE: {mse:.2f}")

# ==========================================================
# 6. Visualisation avec Plotly
# ==========================================================
# Do các mốc thời gian cuối tháng tự động được gom lại theo thứ tự thời gian tăng dần,
# chúng ta chỉ cần tạo dataframe và vẽ đồ thị mà không sợ bị nhảy nét đứt gãy.
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
        name="Valeur Réelle (Test cuối mỗi tháng)",
        line=dict(color="darkblue"),
    )
)

fig.add_trace(
    go.Scatter(
        x=dates_formattees,
        y=resultats["Valeur_Predite"],
        mode="lines",
        name="Valeur Prédite Gradient Boosting (Blocked Split)",
        line=dict(color="darkorange", dash="dash"),
    )
)

titre_graphe = (
    f"Prévisions ec_value via Gradient Boosting (Blocked Split) | R²: {r2:.4f} | MAE: {mae:.2f}"
)
fig.update_layout(
    title=titre_graphe,
    xaxis_title="Date et Heure",
    yaxis_title="ec_value",
    hovermode="x unified",
    xaxis=dict(rangeslider=dict(visible=True), type="date"),
)

fig.show()

# # ==========================================
# # BƯỚC 5.5: GIẢI THÍCH MÔ HÌNH BẰNG PHƯƠNG PHÁP SHAP
# # ==========================================
# # (Mở comment đoạn dưới nếu bạn muốn chạy phân tích SHAP)
# explainer = shap.TreeExplainer(modele)
# shap_values = explainer(X_test)
# print("\n--- Đang hiển thị biểu đồ SHAP Bar Plot ---")
# shap.plots.bar(shap_values, max_display=15)
# print("\n--- Đang hiển thị biểu đồ SHAP Beeswarm Plot ---")
# shap.plots.beeswarm(shap_values, max_display=15)