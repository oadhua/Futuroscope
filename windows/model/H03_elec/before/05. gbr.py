import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES ET CONFIGURATION DES FEATURES
# ---------------------------------------------------------
file_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H03_predict_pure_elec.csv"
df = pd.read_csv(file_path)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

# Features d'entrée pour chaque cible
features_elec_1 = [
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

features_elec_2 = [
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

# Séparation des jeux d'entraînement (< 2026) et de test (>= 2026)
train_mask = df["year"] < 2026
test_mask = df["year"] >= 2026

# Données pour elec_1
X1_train = df.loc[train_mask, features_elec_1]
X1_test = df.loc[test_mask, features_elec_1]
Y1_train = df.loc[train_mask, "elec_1"]
Y1_test = df.loc[test_mask, "elec_1"]

# Données pour elec_2
X2_train = df.loc[train_mask, features_elec_2]
X2_test = df.loc[test_mask, features_elec_2]
Y2_train = df.loc[train_mask, "elec_2"]
Y2_test = df.loc[test_mask, "elec_2"]

test_dates = (
    df.loc[test_mask, "date"] if "date" in df.columns else df.loc[test_mask].index
)

# ---------------------------------------------------------
# 2. ENTRAÎNEMENT DES MODÈLES GRADIENT BOOSTING INDÉPENDANTS
# ---------------------------------------------------------
model_elec_1 = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
)
model_elec_1.fit(X1_train, Y1_train)
pred_elec_1 = model_elec_1.predict(X1_test)

model_elec_2 = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
)
model_elec_2.fit(X2_train, Y2_train)
pred_elec_2 = model_elec_2.predict(X2_test)


# ---------------------------------------------------------
# 3. ÉVALUATION DES PERFORMANCES & FEATURE IMPORTANCE MÔ HÌNH CÂY
# ---------------------------------------------------------
def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


r1, m1, r2_1 = get_metrics(Y1_test, pred_elec_1)
r2, m2, r2_2 = get_metrics(Y2_test, pred_elec_2)

print("\n=========================================================")
print("--- KẾT QUẢ: INDEPENDENT GRADIENT BOOSTING REGRESSOR ---")
print(f"[elec_1] RMSE: {r1:.4f} kWh | MAE: {m1:.4f} kWh | R²: {r2_1:.4f}")
print(f"[elec_2] RMSE: {r2:.4f} kWh | MAE: {m2:.4f} kWh | R²: {r2_2:.4f}")

# Hiển thị Top features quan trọng mặc định từ GBR (Feature Importance gốc)
importances_1 = pd.DataFrame(
    {"Feature": features_elec_1, "Importance": model_elec_1.feature_importances_}
).sort_values(by="Importance", ascending=False)
print("\n[GBR Feature Importance] Top 5 đặc trưng ảnh hưởng lớn nhất tới [elec_1]:")
print(importances_1.head(5).to_string(index=False))

importances_2 = pd.DataFrame(
    {"Feature": features_elec_2, "Importance": model_elec_2.feature_importances_}
).sort_values(by="Importance", ascending=False)
print("\n[GBR Feature Importance] Top 5 đặc trưng ảnh hưởng lớn nhất tới [elec_2]:")
print(importances_2.head(5).to_string(index=False))

# # ---------------------------------------------------------
# # 4. EXPLICABILITÉ XAI AVEC SHAP (TREE EXPLAINER)
# # ---------------------------------------------------------
# print("\n---------------------------------------------------------")
# print("--- ANALYSE D'EXPLICABILITÉ APPROFONDIE (SHAP VALUES) ---")

# # --- SHAP POUR ELEC_1 ---
# explainer_1 = shap.TreeExplainer(model_elec_1)
# shap_values_1 = explainer_1(X1_test)

# plt.figure(figsize=(10, 6))
# shap.summary_plot(shap_values_1, X1_test, show=False)
# plt.title("SHAP Summary Plot - elec_1 (Gradient Boosting)", fontsize=13, pad=15)
# plt.tight_layout()
# shap_img_1 = "shap_summary_elec_1.png"
# plt.savefig(shap_img_1, dpi=300)
# plt.close()

# mean_shap_1 = np.abs(shap_values_1.values).mean(axis=0)
# df_shap_1 = pd.DataFrame(
#     {"Feature": features_elec_1, "SHAP_Importance": mean_shap_1}
# ).sort_values(by="SHAP_Importance", ascending=False)

# print("\n[SHAP Value Importance] Top 5 đặc trưng quan trọng nhất từ SHAP [elec_1]:")
# print(df_shap_1.head(5).to_string(index=False))

# # --- SHAP POUR ELEC_2 ---
# explainer_2 = shap.TreeExplainer(model_elec_2)
# shap_values_2 = explainer_2(X2_test)

# plt.figure(figsize=(10, 6))
# shap.summary_plot(shap_values_2, X2_test, show=False)
# plt.title("SHAP Summary Plot - elec_2 (Gradient Boosting)", fontsize=13, pad=15)
# plt.tight_layout()
# shap_img_2 = "shap_summary_elec_2.png"
# plt.savefig(shap_img_2, dpi=300)
# plt.close()

# mean_shap_2 = np.abs(shap_values_2.values).mean(axis=0)
# df_shap_2 = pd.DataFrame(
#     {"Feature": features_elec_2, "SHAP_Importance": mean_shap_2}
# ).sort_values(by="SHAP_Importance", ascending=False)

# print("\n[SHAP Value Importance] Top 5 đặc trưng quan trọng nhất từ SHAP [elec_2]:")
# print(df_shap_2.head(5).to_string(index=False))
# print(
#     f"\n[OK] Biểu đồ SHAP Beeswarm Plot đã lưu thành công: {shap_img_1} | {shap_img_2}"
# )
# print("=========================================================\n")

# ---------------------------------------------------------
# 5. VISUALISATION INTERACTIVE AVEC PLOTLY (KÍCH THƯỚC CHUẨN)
# ---------------------------------------------------------
subtitle_1 = f"<b>elec_1 : Réel vs Gradient Boosting</b><br><span style='font-size: 11px; color: #555;'>RMSE: {r1:.2f} kWh | MAE: {m1:.2f} kWh | R²: {r2_1:.4f}</span>"
subtitle_2 = f"<b>elec_2 : Réel vs Gradient Boosting</b><br><span style='font-size: 11px; color: #555;'>RMSE: {r2:.2f} kWh | MAE: {m2:.2f} kWh | R²: {r2_2:.4f}</span>"

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.18,
    subplot_titles=(subtitle_1, subtitle_2),
)

# Graphique elec_1
fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=Y1_test,
        mode="lines",
        name="elec_1 Réel",
        line=dict(color="#1f77b4", width=1.5),
        hovertemplate="<b>Horodatage:</b> %{x}<br><b>Valeur réelle:</b> %{y:.2f} kWh<extra></extra>",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=pred_elec_1,
        mode="lines",
        name="elec_1 Prédiction (GBR)",
        line=dict(color="#ff7f0e", width=1.5, dash="dot"),
        customdata=np.stack((Y1_test, np.abs(Y1_test - pred_elec_1)), axis=-1),
        hovertemplate="<b>Horodatage:</b> %{x}<br><b>Prédiction:</b> %{y:.2f} kWh<br><b>Écart:</b> %{customdata[1]:.2f} kWh<extra></extra>",
    ),
    row=1,
    col=1,
)

# Graphique elec_2
fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=Y2_test,
        mode="lines",
        name="elec_2 Réel",
        line=dict(color="#2ca02c", width=1.5),
        hovertemplate="<b>Horodatage:</b> %{x}<br><b>Valeur réelle:</b> %{y:.2f} kWh<extra></extra>",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=pred_elec_2,
        mode="lines",
        name="elec_2 Prédiction (GBR)",
        line=dict(color="#d62728", width=1.5, dash="dot"),
        customdata=np.stack((Y2_test, np.abs(Y2_test - pred_elec_2)), axis=-1),
        hovertemplate="<b>Horodatage:</b> %{x}<br><b>Prédiction:</b> %{y:.2f} kWh<br><b>Écart:</b> %{customdata[1]:.2f} kWh<extra></extra>",
    ),
    row=2,
    col=1,
)

fig.update_layout(
    height=850,
    margin=dict(t=100, b=60, l=60, r=40),
    title_text="<b>PRÉVISION PAR GRADIENT BOOSTING INDÉPENDANTS (TEST 2026)</b>",
    title_x=0.5,
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
)

fig.for_each_annotation(lambda a: a.update(font=dict(size=13)))

fig.update_yaxes(title_text="Consommation (kWh)", row=1, col=1)
fig.update_yaxes(title_text="Consommation (kWh)", row=2, col=1)
fig.update_xaxes(rangeslider_visible=True, row=2, col=1)

# Export en HTML
output_html = "05. gbr.html"
fig.write_html(output_html, include_plotlyjs="cdn")
print(f"[OK] Graphique interactif exporté avec succès : {os.path.abspath(output_html)}")

fig.show()

# ---------------------------------------------------------
# 6. XUẤT TỆP KẾT QUẢ DỰ BÁO VÀ FEATURES ĐẦU VÀO
# ---------------------------------------------------------
# Tạo DataFrame kết quả trên tập Test (>= 2026)
df_results = df.loc[test_mask, ["date"] + features_elec_1].copy()

# Thêm các cột thực tế, dự báo và sai số tuyệt đối
df_results["elec_1_real"] = Y1_test.values
df_results["elec_1_pred"] = pred_elec_1
df_results["elec_1_error_abs"] = np.abs(Y1_test.values - pred_elec_1)

df_results["elec_2_real"] = Y2_test.values
df_results["elec_2_pred"] = pred_elec_2
df_results["elec_2_error_abs"] = np.abs(Y2_test.values - pred_elec_2)

# Export ra CSV
output_csv = "predict_2026.csv"
df_results.to_csv(output_csv, index=False, encoding="utf-8-sig")

# Export ra Excel
output_excel = "predict_2026.xlsx"
df_results.to_excel(output_excel, index=False, sheet_name="Forecast_Results")

print(f"[OK] File kết quả đã xuất thành công: {output_excel} và {output_csv}")