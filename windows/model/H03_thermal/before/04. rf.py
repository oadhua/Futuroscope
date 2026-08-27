import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import shap

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES ET CONFIGURATION DES FEATURES
# ---------------------------------------------------------
file_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H03_predict_pure_thermal.csv"
df = pd.read_csv(file_path)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

features_ecvalue = [
    "year",
    "month",
    "day",
    "hour",
    "week",
    "is_weekend",
    "is_open",
    "jf",
    "frequentation_park_daily",
    "surface",
    "duree",
    "ouvert",
    "interrompu",
    "operation",
    "visitor_count",
    "temperature",
    "humidite",
    "rayonnement_solaire",
    "day_degree_cold",
    "day_degree_hot",
    "temp_max",
    "temp_min",
    "temp_moy",
    "humidite_max",
    "humidite_min",
    "humidite_moy",
    "freq_HF",
    "freq_MF",
    "freq_THF",
]

# Séparation des ensembles d'entraînement (< 2026) et de test (>= 2026)
train_mask = df["year"] < 2026
test_mask = df["year"] >= 2026

X1_train = df.loc[train_mask, features_ecvalue]
X1_test = df.loc[test_mask, features_ecvalue]
Y1_train = df.loc[train_mask, "ec_value"]
Y1_test = df.loc[test_mask, "ec_value"]

test_dates = (
    df.loc[test_mask, "date"] if "date" in df.columns else df.loc[test_mask].index
)

# ---------------------------------------------------------
# 2. ENTRAÎNEMENT DU MODÈLE RANDOM FOREST
# ---------------------------------------------------------
model_ecvalue = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_ecvalue.fit(X1_train, Y1_train)

pred_ecvalue = model_ecvalue.predict(X1_test)


# ---------------------------------------------------------
# 3. ÉVALUATION DES PERFORMANCES ET IMPORTANCE DES FEATURES
# ---------------------------------------------------------
def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


r1, m1, r2_1 = get_metrics(Y1_test, pred_ecvalue)

print("\n=========================================================")
print("--- RÉSULTATS : RANDOM FOREST REGRESSION (ec_value) ---")
print(f"[ec_value] RMSE: {r1:.4f} | MAE: {m1:.4f} | R²: {r2_1:.4f}")

# Extraction du Top 5 Feature Importance de Random Forest
feature_importances = pd.DataFrame(
    {"Feature": features_ecvalue, "Importance": model_ecvalue.feature_importances_}
).sort_values(by="Importance", ascending=False)

print("\n--- TOP 5 DES CARACTÉRISTIQUES LES PLUS INFLUENTES ---")
print(feature_importances.head(5).to_string(index=False))
print("=========================================================\n")

# # ---------------------------------------------------------
# # 4. EXPLICABILITÉ XAI AVEC SHAP (OPTIMISÉ POUR RANDOM FOREST)
# # ---------------------------------------------------------
# print("\n---------------------------------------------------------")
# print("--- ANALYSE D'EXPLICABILITÉ APPROFONDIE (SHAP VALUES) ---")

# # Lấy mẫu ngẫu nhiên 500-1000 dòng để tính SHAP nhanh và tránh tràn RAM
# sample_size = min(500, len(X1_test))
# X1_test_sample = X1_test.sample(n=sample_size, random_state=42)

# # Khởi tạo TreeExplainer
# explainer_1 = shap.TreeExplainer(model_ecvalue)
# shap_values_obj = explainer_1(X1_test_sample)

# # Trích xuất mảng numpy shap_values an toàn cho cả mảng 2D và 3D
# if isinstance(shap_values_obj, np.ndarray):
#     shap_vals = shap_values_obj
# elif hasattr(shap_values_obj, "values"):
#     shap_vals = shap_values_obj.values
# else:
#     shap_vals = np.array(shap_values_obj)

# # Nếu shap_vals có 3 chiều (đối với một số phiên bản RF/SHAP), lấy slice phù hợp
# if shap_vals.ndim == 3:
#     shap_vals = shap_vals[:, :, 0]

# # Vẽ và lưu biểu đồ SHAP Summary Plot
# plt.figure(figsize=(10, 6))
# shap.summary_plot(shap_vals, X1_test_sample, show=False)
# plt.title("SHAP Summary Plot - ec_value (Random Forest)", fontsize=13, pad=15)
# plt.tight_layout()
# shap_img_1 = "shap_summary_ecvalue.png"
# plt.savefig(shap_img_1, dpi=300)
# plt.close()

# # Tính trung bình tầm quan trọng SHAP (|SHAP value|)
# mean_shap_1 = np.abs(shap_vals).mean(axis=0)
# df_shap_1 = pd.DataFrame(
#     {"Feature": features_ecvalue, "SHAP_Importance": mean_shap_1}
# ).sort_values(by="SHAP_Importance", ascending=False)

# print(
#     f"\n[SHAP Value Importance] Top 5 đặc trưng quan trọng nhất từ SHAP [ec_value] (trên {sample_size} mẫu):"
# )
# print(df_shap_1.head(5).to_string(index=False))
# print(f"[OK] Biểu đồ SHAP Beeswarm Plot đã lưu thành công: {shap_img_1}")
# print("=========================================================\n")

# ---------------------------------------------------------
# 4. VISUALISATION INTERACTIVE AVEC PLOTLY & EXPORT HTML
# ---------------------------------------------------------
subtitle_1 = f"<b>ec_value : Réel vs Random Forest Regression</b><br><span style='font-size: 11px; color: #555;'>RMSE: {r1:.2f} | MAE: {m1:.2f} | R²: {r2_1:.4f}</span>"

fig = make_subplots(rows=1, cols=1, subplot_titles=(subtitle_1,))

# Courbe Réelle
fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=Y1_test,
        mode="lines",
        name="ec_value Réel",
        line=dict(color="#1f77b4", width=1.5),
        hovertemplate="<b>Horodatage:</b> %{x}<br><b>Valeur réelle:</b> %{y:.2f}<extra></extra>",
    ),
    row=1,
    col=1,
)

# Courbe Prédite (Random Forest)
fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=pred_ecvalue,
        mode="lines",
        name="ec_value Prédiction (Random Forest)",
        line=dict(color="#ff7f0e", width=1.5, dash="dot"),
        customdata=np.stack((Y1_test, np.abs(Y1_test - pred_ecvalue)), axis=-1),
        hovertemplate="<b>Horodatage:</b> %{x}<br><b>Prédiction:</b> %{y:.2f}<br><b>Écart:</b> %{customdata[1]:.2f}<extra></extra>",
    ),
    row=1,
    col=1,
)

fig.update_layout(
    height=550,
    margin=dict(t=100, b=60, l=60, r=40),
    title_text="<b>PRÉVISION THERMIQUE (EC_VALUE) PAR RANDOM FOREST (TEST 2026)</b>",
    title_x=0.5,
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
)

fig.for_each_annotation(lambda a: a.update(font=dict(size=13)))

fig.update_yaxes(title_text="Valeur ec_value", row=1, col=1)
fig.update_xaxes(rangeslider_visible=True, row=1, col=1)

# Exportation en HTML
output_html = "04. rf.html"
fig.write_html(output_html, include_plotlyjs="cdn")
print(f"[OK] Graphique interactif exporté avec succès : {os.path.abspath(output_html)}")

fig.show()

# ---------------------------------------------------------
# 6. XUẤT TỆP KẾT QUẢ DỰ BÁO VÀ FEATURES ĐẦU VÀO
# ---------------------------------------------------------
# Tạo DataFrame kết quả trên tập Test (>= 2026)
df_results = df.loc[test_mask, ["date"] + features_ecvalue].copy()

# Thêm các cột thực tế, dự báo và sai số tuyệt đối
df_results["ec_value_real"] = Y1_test.values
df_results["ec_value_pred"] = pred_ecvalue
df_results["ec_value_error_abs"] = np.abs(Y1_test.values - pred_ecvalue)

# Export ra CSV
output_csv = "predict_2026.csv"
df_results.to_csv(output_csv, index=False, encoding="utf-8-sig")

# Export ra Excel
output_excel = "predict_2026.xlsx"
df_results.to_excel(output_excel, index=False, sheet_name="Forecast_Results")

print(f"[OK] File kết quả đã xuất thành công: {output_excel} và {output_csv}")
