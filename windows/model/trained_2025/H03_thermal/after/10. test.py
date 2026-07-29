import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==========================================================
# 1. Chargement des données préparées
# ==========================================================
# Thay đường dẫn bên dưới tới file của bạn
path_file = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H07_predict_pure_thermal.csv"
donnees = pd.read_csv(path_file)

donnees["date"] = pd.to_datetime(donnees["date"])
donnees = donnees.sort_values("date").reset_index(drop=True)

# Tạo cột phụ đại diện cho Năm-Tháng để phân nhóm dữ liệu
donnees["annee_mois"] = donnees["date"].dt.to_period("M")

# ==========================================================
# 2. Sélection automatique des variables d'entrée (X) et cible (y)
# ==========================================================
# variables_exclues = [
#     "date",
#     "ec_value",
#     "min",
#     "year",
#     "month",
#     "day",
#     "hour",
#     "week",
#     "id_attraction",
#     "annee_mois",
#     # "visitor_count"
# ]

# H03_thermal
# variable_in = [
#     "temp_moy",
#     "temp_min",
#     "is_open",
#     "heure_cos",
#     "heure_sin",
#     "is_weekend",
#     "hour_x_freq_MF",
#     "mois_sin",
#     "hour_x_freq_HF",
#     "mois_cos"
# ]

# H07_thermal
variable_in = [
    "temp_moy",
    "visitor_count",
    "temp_min",
    "temp_decalage_1h",
    "mois_sin",
    "heure_cos",
    "temp_max",
    "mois_cos",
    "rayonnement_solaire",
    "is_open"
]

# variables_entree = [col for col in donnees.columns if col not in variables_exclues]
variables_entree = [col for col in donnees.columns if col in variable_in]
variable_cible = "ec_value"

X = donnees[variables_entree]
y = donnees[variable_cible]

# ==========================================================
# 3. Division temporelle (Blocked Split 80/20 theo từng Tháng)
# ==========================================================
train_indices = []
test_indices = []

for name, group in donnees.groupby("annee_mois"):
    if len(group) < 10:
        continue
    split_idx = int(len(group) * 0.8)
    train_indices.extend(group.index[:split_idx])
    test_indices.extend(group.index[split_idx:])

X_train = X.loc[train_indices].reset_index(drop=True)
y_train = y.loc[train_indices].reset_index(drop=True)

X_test = X.loc[test_indices].reset_index(drop=True)
y_test_reel = y.loc[test_indices].reset_index(drop=True)
dates_test = donnees["date"].loc[test_indices].reset_index(drop=True)

# ==========================================================
# 4. Entraînement du modèle XGBoost Global
# ==========================================================
modele = XGBRegressor(
    n_estimators=750,
    learning_rate=0.03,      
    max_depth=3,             
    subsample=0.75,          
    colsample_bytree=0.8,    
    reg_alpha=2,             
    reg_lambda=5,            
    min_child_weight=3,      
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

print("\n==========================================================")
print(f"XGBoost Train R²: {r2_train:.4f}")
print(f"XGBoost Test R²: {r2:.4f} | MAE: {mae:.2f} | MSE: {mse:.2f}")
print("==========================================================")

# ==========================================================
# 6. Visualisation avec Plotly
# ==========================================================
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

titre_graphe = f"Prévisions ec_value via XGBoost | R²: {r2:.4f} | MAE: {mae:.2f} | MSE: {mse:.2f}"
fig.update_layout(
    title=titre_graphe,
    xaxis_title="Date et Heure",
    yaxis_title="ec_value",
    hovermode="x unified",
    xaxis=dict(rangeslider=dict(visible=True), type="date"),
)
fig.write_html("prévisions_xgb.html")
print("-> Đã xuất đồ thị tương tác ra file 'prévisions_xgb.html'")

# ==========================================================
# 7. PHÂN TÍCH FEATURE IMPORTANCE VÀ SHAP VALUES
# ==========================================================
print("\n--- ĐANG PHÂN TÍCH FEATURE IMPORTANCE & SHAP ---")

# 7.1. Feature Importance mặc định của XGBoost
importance_df = pd.DataFrame({
    'Feature': variables_entree,
    'Importance': modele.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 đặc trưng quan trọng nhất (XGBoost Native):")
print(importance_df.head(10).to_string(index=False))

# 7.2. Giải thích mô hình bằng SHAP
try:
    explainer = shap.TreeExplainer(modele)
    shap_values = explainer(X_test)
    
    # Biểu đồ Bar Plot SHAP
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("SHAP Feature Importance (Bar Plot)")
    plt.tight_layout()
    plt.savefig("shap_bar_plot.png")
    plt.close()

    # Biểu đồ Beeswarm SHAP
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP Beeswarm Plot")
    plt.tight_layout()
    plt.savefig("shap_beeswarm_plot.png")
    plt.close()
    
    print("-> Đã lưu biểu đồ SHAP vào 'shap_bar_plot.png' và 'shap_beeswarm_plot.png'")
except Exception as e:
    print(f"Không thể tính SHAP: {e}")

# ==========================================================
# 8. TỐI ƯU HÓA TẦN SỐ THIẾT BỊ (WEIGHTED COST GRID SEARCH)
# ==========================================================
print("\n--- ĐANG CHẠY KỊCH BẢN TỐI ƯU HÓA KẾT HỢP TRỌNG SỐ VẬN HÀNH ---")

# 1. Tìm các mốc thời gian khi công viên ĐANG MỞ CỬA và thiết bị đang chạy
sample_peak_indices = X_test[
    (X_test['is_open'] == 1) & 
    ((X_test.get('freq_HF', 0) > 0) | (X_test.get('freq_MF', 0) > 0) | (X_test.get('freq_THF', 0) > 0))
].index

if len(sample_peak_indices) > 0:
    sample_index = sample_peak_indices[0]
else:
    sample_index = X_test[X_test['is_open'] == 1].index[0]

sample_context = X_test.loc[sample_index].copy()
sample_date = dates_test.loc[sample_index]

possible_levels = [0.0, 1.0, 2.0, 3.0]  

# Trọng số chi phí/tải thiết bị: THF tốn tải nhất (2.0), MF (1.5), HF (1.0)
W_HF, W_MF, W_THF = 1.0, 1.5, 2.0 
ALPHA = 0.01  # Trọng số phạt để giải quyết các trường hợp ec_value bằng nhau

scenarios = []
for hf in possible_levels:
    for mf in possible_levels:
        for thf in possible_levels:
            # RÀNG BUỘC: Bắt buộc ít nhất 1 hệ thống chạy để đảm bảo làm mát/thông gió khi mở cửa
            if (hf + mf + thf) < 1.0:
                continue
                
            test_row = sample_context.copy()
            
            if 'freq_HF' in variables_entree: test_row['freq_HF'] = float(hf)
            if 'freq_MF' in variables_entree: test_row['freq_MF'] = float(mf)
            if 'freq_THF' in variables_entree: test_row['freq_THF'] = float(thf)
            
            if 'hour_x_freq_HF' in variables_entree and 'hour' in test_row:
                test_row['hour_x_freq_HF'] = float(test_row['hour']) * float(hf)
            if 'hour_x_freq_MF' in variables_entree and 'hour' in test_row:
                test_row['hour_x_freq_MF'] = float(test_row['hour']) * float(mf)
                
            pred_ec = modele.predict(pd.DataFrame([test_row]))[0]
            pred_ec = max(0, float(pred_ec))
            
            # Tính tổng chi phí (Năng lượng dự báo + Phạt tải thiết bị)
            equipment_cost = (hf * W_HF) + (mf * W_MF) + (thf * W_THF)
            total_score = pred_ec + (ALPHA * equipment_cost)
            
            scenarios.append({
                'freq_HF': hf,
                'freq_MF': mf,
                'freq_THF': thf,
                'predicted_ec': pred_ec,
                'equipment_cost': equipment_cost,
                'total_score': total_score
            })

df_scenarios = pd.DataFrame(scenarios)

initial_hf = float(sample_context.get('freq_HF', 0))
initial_mf = float(sample_context.get('freq_MF', 0))
initial_thf = float(sample_context.get('freq_THF', 0))

initial_row = df_scenarios[
    (df_scenarios['freq_HF'] == initial_hf) & 
    (df_scenarios['freq_MF'] == initial_mf) & 
    (df_scenarios['freq_THF'] == initial_thf)
]

initial_ec_val = initial_row['predicted_ec'].values[0] if len(initial_row) > 0 else modele.predict(pd.DataFrame([sample_context]))[0]

# Chọn kịch bản có tổng điểm (Total Score) nhỏ nhất
best_scenario = df_scenarios.loc[df_scenarios['total_score'].idxmin()]

print(f"Mốc thời gian thử nghiệm: {sample_date}")
print(f"Cấu hình thực tế ban đầu [HF, MF, THF] : [{initial_hf}, {initial_mf}, {initial_thf}]")
print(f"Mức ec_value ban đầu                  : {initial_ec_val:.2f} kWh")
print("-" * 55)
print(f"Cấu hình TỐI ƯU đề xuất [HF, MF, THF]   : [{best_scenario['freq_HF']}, {best_scenario['freq_MF']}, {best_scenario['freq_THF']}]")
print(f"Mức ec_value TỐI ƯU                    : {best_scenario['predicted_ec']:.2f} kWh")
print(f"Tải vận hành thiết bị giảm             : {((initial_hf*W_HF + initial_mf*W_MF + initial_thf*W_THF) - best_scenario['equipment_cost']):.1f} điểm")