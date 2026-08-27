import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import plotly.graph_objects as go
import plotly.io as pio
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
# train_mask = df["year"] < 2026
# test_mask = df["year"] >= 2026
train_mask = (df["year"] < 2025) | (df["year"] >= 2026)
test_mask = df["year"] == 2025

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
# 4. VISUALISATION INTERACTIVE ET DASHBOARD PLOTLY MULTI-TABS
# ---------------------------------------------------------
# Préparation des données de test (2026)
df_results = df.loc[test_mask, ["date"] + features_elec_1].copy()
df_results = df_results.sort_values("date")

df_results["elec_1_real"], df_results["elec_1_pred"] = Y1_test.values, pred_elec_1
df_results["elec_2_real"], df_results["elec_2_pred"] = Y2_test.values, pred_elec_2

# Agrégation mensuelle (Bổ sung trung bình/tổng các trường phụ trợ nếu cần)
df_results["month_num"] = df_results["date"].dt.month
monthly_summary = df_results.groupby("month_num").agg({
    "elec_1_real": "sum",
    "elec_1_pred": "sum",
    "elec_2_real": "sum",
    "elec_2_pred": "sum",
    "visitor_count": "sum",
    "is_open": "max",
    "operation": "max"
}).reset_index()

monthly_summary["month_label"] = monthly_summary["month_num"].apply(lambda x: f"Mois {x}")

# ---------------------------------------------------------
# 4.1 CALCUL DES INDICATEURS DE PERFORMANCE (KPIs)
# ---------------------------------------------------------
sum_e1_real = df_results["elec_1_real"].sum()
sum_e1_pred = df_results["elec_1_pred"].sum()
diff_e1 = sum_e1_pred - sum_e1_real
pct_e1 = (diff_e1 / sum_e1_real) * 100 if sum_e1_real != 0 else 0

sum_e2_real = df_results["elec_2_real"].sum()
sum_e2_pred = df_results["elec_2_pred"].sum()
diff_e2 = sum_e2_pred - sum_e2_real
pct_e2 = (diff_e2 / sum_e2_real) * 100 if sum_e2_real != 0 else 0

sum_tot_real = sum_e1_real + sum_e2_real
sum_tot_pred = sum_e1_pred + sum_e2_pred
diff_tot = sum_tot_pred - sum_tot_real
pct_tot = (diff_tot / sum_tot_real) * 100 if sum_tot_real != 0 else 0

monthly_summary["diff_e1"] = monthly_summary["elec_1_pred"] - monthly_summary["elec_1_real"]
monthly_summary["pct_e1"] = (monthly_summary["diff_e1"] / monthly_summary["elec_1_real"]) * 100

monthly_summary["diff_e2"] = monthly_summary["elec_2_pred"] - monthly_summary["elec_2_real"]
monthly_summary["pct_e2"] = (monthly_summary["diff_e2"] / monthly_summary["elec_2_real"]) * 100

months_list = [f"Mois {m}" for m in monthly_summary["month_num"]] + ["<b>CUMUL ANNUEL</b>"]

# ---------------------------------------------------------
# 4.2 COMPOSANT 1 : GRAPHIQUE INTERACTIF (TỐI ƯU HOVERTOOLTIP)
# ---------------------------------------------------------
FONT_GRAPH = "Segoe UI, Arial, sans-serif"

subtitles = (
    f"<b>elec_1 : Consommation Réelle vs Modèle GB</b><br><span style='font-size: 11px; color: #555;'>RMSE: {r1:.2f} kWh | MAE: {m1:.2f} kWh | R²: {r2_1:.4f}</span>",
    f"<b>elec_2 : Consommation Réelle vs Modèle GB</b><br><span style='font-size: 11px; color: #555;'>RMSE: {r2:.2f} kWh | MAE: {m2:.2f} kWh | R²: {r2_2:.4f}</span>"
)

fig_graph = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.18,
    subplot_titles=subtitles
)

colors = {
    "elec_1": {"real": "#1f77b4", "pred": "#ff7f0e"},
    "elec_2": {"real": "#2ca02c", "pred": "#d62728"}
}

# --- 1. MẢNG DỮ LIỆU CUSTOMDATA DÀNH CHO XEM THEO GIỜ (HOURLY) ---
customdata_e1_hourly = df_results[["elec_1_real", "elec_1_pred", "visitor_count", "is_open", "operation"]].values
customdata_e2_hourly = df_results[["elec_2_real", "elec_2_pred", "visitor_count", "is_open", "operation"]].values

# --- 2. MẢNG DỮ LIỆU CUSTOMDATA DÀNH CHO XEM THEO THÁNG (MONTHLY) ---
customdata_e1_monthly = monthly_summary[["elec_1_real", "elec_1_pred", "visitor_count", "is_open", "operation"]].values
customdata_e2_monthly = monthly_summary[["elec_2_real", "elec_2_pred", "visitor_count", "is_open", "operation"]].values

# Template hiển thị tổng hợp đầy đủ thông tin
hover_template_custom = (
    "<b>Date / Période:</b> %{x}<br>"
    "<b>Réel:</b> %{customdata[0]:,.2f} kWh<br>"
    "<b>Estimé:</b> %{customdata[1]:,.2f} kWh<br>"
    "<b>Visitors:</b> %{customdata[2]:,}<br>"
    "<b>Is Open:</b> %{customdata[3]}<br>"
    "<b>Operation:</b> %{customdata[4]}"
    "<extra></extra>"
)

# --- Traces Bar (Vue Mensuelle) ---
for row, target in enumerate(["elec_1", "elec_2"], start=1):
    cdata = customdata_e1_monthly if target == "elec_1" else customdata_e2_monthly
    
    fig_graph.add_trace(go.Bar(
        x=monthly_summary["month_label"], y=monthly_summary[f"{target}_real"],
        name=f"{target} Réel", marker_color=colors[target]["real"], visible=True,
        customdata=cdata,
        hovertemplate=hover_template_custom,
        hoverlabel=dict(namelength=0)
    ), row=row, col=1)
    
    fig_graph.add_trace(go.Bar(
        x=monthly_summary["month_label"], y=monthly_summary[f"{target}_pred"],
        name=f"{target} Estimé", marker_color=colors[target]["pred"], visible=True,
        hoverinfo="skip"
    ), row=row, col=1)

# --- Traces Scatter (Vue Horaire) ---
for row, target in enumerate(["elec_1", "elec_2"], start=1):
    cdata = customdata_e1_hourly if target == "elec_1" else customdata_e2_hourly
    
    fig_graph.add_trace(go.Scatter(
        x=df_results["date"], y=df_results[f"{target}_real"],
        mode="lines", name=f"{target} Réel", line=dict(color=colors[target]["real"], width=1.2), visible=False,
        customdata=cdata,
        hovertemplate=hover_template_custom,
        hoverlabel=dict(namelength=0)
    ), row=row, col=1)
    
    fig_graph.add_trace(go.Scatter(
        x=df_results["date"], y=df_results[f"{target}_pred"],
        mode="lines", name=f"{target} Estimé", line=dict(color=colors[target]["pred"], width=1.2, dash="dash"), visible=False,
        hoverinfo="skip"
    ), row=row, col=1)

# --- CẤU HÌNH TỔNG THỂ PHÔNG CHỮ & LAYOUT ---
fig_graph.update_layout(
    font=dict(family=FONT_GRAPH, size=12, color="#2c3e50"),
    
    title=dict(
        text="<b>ÉVALUATION GRAPHIQUE DES PRÉDICTIONS (2025)</b>",
        x=0.5,
        font=dict(family=FONT_GRAPH, size=15, color="#2c3e50")
    ),
    
    # Kích hoạt chế độ hover hợp nhất (x unified) hoặc dóng cột mốc thời gian
    hovermode="x unified",
    
    updatemenus=[
        dict(
            buttons=[
                dict(
                    label="Vue Mensuelle (Bâtons)",
                    method="update",
                    args=[
                        {"visible": [True, True, True, True, False, False, False, False]},
                        {"barmode": "group", "xaxis.type": "category", "xaxis2.type": "category", "xaxis2.rangeslider.visible": False}
                    ]
                ),
                dict(
                    label="Vue Horaire (Série Temporelle)",
                    method="update",
                    args=[
                        {"visible": [False, False, False, False, True, True, True, True]},
                        {"xaxis.type": "date", "xaxis2.type": "date", "xaxis2.rangeslider.visible": True}
                    ]
                )
            ],
            direction="down", showactive=True, x=0.0, xanchor="left", y=1.18, yanchor="top",
            font=dict(family=FONT_GRAPH, size=12, color="#2c3e50")
        )
    ],
    
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(family=FONT_GRAPH, size=11, color="#2c3e50")
    ),
    
    hoverlabel=dict(
        font=dict(family=FONT_GRAPH, size=12),
        bgcolor="#ffffff"
    ),
    
    margin=dict(l=60, r=40, t=100, b=50),
    height=750,
    template="plotly_white"
)

# Cập nhật các trục X và Y giữ nguyên
fig_graph.for_each_annotation(lambda a: a.update(font=dict(family=FONT_GRAPH, size=12, color="#2c3e50")))

fig_graph.update_xaxes(
    type="category",
    title_font=dict(family=FONT_GRAPH, size=12),
    tickfont=dict(family=FONT_GRAPH, size=11),
    row=1, col=1
)
fig_graph.update_xaxes(
    type="category",
    rangeslider_visible=False,
    title_font=dict(family=FONT_GRAPH, size=12),
    tickfont=dict(family=FONT_GRAPH, size=11),
    row=2, col=1
)

fig_graph.update_yaxes(
    title_text="Énergie Consommée (kWh)",
    title_font=dict(family=FONT_GRAPH, size=12),
    tickfont=dict(family=FONT_GRAPH, size=11),
    row=1, col=1
)
fig_graph.update_yaxes(
    title_text="Énergie Consommée (kWh)",
    title_font=dict(family=FONT_GRAPH, size=12),
    tickfont=dict(family=FONT_GRAPH, size=11),
    row=2, col=1
)

# ---------------------------------------------------------
# 4.3 COMPOSANTS TABLEAUX (TỐI ƯU PHÔNG CHỮ VÀ PADDING)
# ---------------------------------------------------------
# Cấu hình Style chung chống tràn chữ
header_style = dict(
    fill_color="#2c3e50",
    font=dict(color="white", size=12, family="Segoe UI, Arial, sans-serif"),
    align="center",
    height=38,
    line=dict(width=1, color="#1a252f")
)

cell_style_base = dict(
    align="center",
    font=dict(size=11, family="Segoe UI, Arial, sans-serif"),
    height=32,
    line=dict(width=1, color="#e0e0e0")
)

# 1. Tableau Bilan Global Annuel
fig_kpi = go.Figure(data=[go.Table(
    header=dict(
        values=["<b>Compteur</b>", "<b>Réel Total (kWh)</b>", "<b>Estimé Total (kWh)</b>", "<b>Écart (kWh)</b>", "<b>Erreur Relative (%)</b>"],
        **header_style
    ),
    cells=dict(
        values=[
            ["elec_1", "elec_2", "<b>TOTAL BÂTIMENT</b>"],
            [f"{sum_e1_real:,.2f}", f"{sum_e2_real:,.2f}", f"<b>{sum_tot_real:,.2f}</b>"],
            [f"{sum_e1_pred:,.2f}", f"{sum_e2_pred:,.2f}", f"<b>{sum_tot_pred:,.2f}</b>"],
            [f"{diff_e1:+,.2f}", f"{diff_e2:+,.2f}", f"<b>{diff_tot:+,.2f}</b>"],
            [f"{pct_e1:+.2f}%", f"{pct_e2:+.2f}%", f"<b>{pct_tot:+.2f}%</b>"]
        ],
        fill_color=[["#f8f9fa", "#ffffff", "#e9ecef"] * 3],
        **cell_style_base
    )
)])
fig_kpi.update_layout(
    title_text="<b>BILAN GLOBAL ANNUEL DE CONSOMMATION (2025)</b>",
    title_x=0.5,
    height=350,
    margin=dict(l=20, r=20, t=60, b=20),
    template="plotly_white"
)

# 2. Tableau Mensuel elec_1
e1_m_real = [f"{v:,.1f}" for v in monthly_summary["elec_1_real"]] + [f"<b>{sum_e1_real:,.1f}</b>"]
e1_m_pred = [f"{v:,.1f}" for v in monthly_summary["elec_1_pred"]] + [f"<b>{sum_e1_pred:,.1f}</b>"]
e1_m_diff = [f"{v:+,.1f}" for v in monthly_summary["diff_e1"]] + [f"<b>{diff_e1:+,.1f}</b>"]
e1_m_pct = [f"{v:+.2f}%" for v in monthly_summary["pct_e1"]] + [f"<b>{pct_e1:+.2f}%</b>"]

header_e1 = header_style.copy()
header_e1["fill_color"] = "#1f77b4"

fig_tab_e1 = go.Figure(data=[go.Table(
    header=dict(
        values=["<b>Mois</b>", "<b>elec_1 Réel (kWh)</b>", "<b>elec_1 Estimé (kWh)</b>", "<b>Écart (kWh)</b>", "<b>Erreur Relative (%)</b>"],
        **header_e1
    ),
    cells=dict(
        values=[months_list, e1_m_real, e1_m_pred, e1_m_diff, e1_m_pct],
        fill_color=[["#ffffff", "#f8f9fa"] * 6 + ["#e9ecef"]],
        **cell_style_base
    )
)])
fig_tab_e1.update_layout(
    title_text="<b>TABLEAU MENSUEL DÉTAILLÉ : ELEC_1</b>",
    title_x=0.5,
    height=620,
    margin=dict(l=20, r=20, t=60, b=20),
    template="plotly_white"
)

# 3. Tableau Mensuel elec_2
e2_m_real = [f"{v:,.1f}" for v in monthly_summary["elec_2_real"]] + [f"<b>{sum_e2_real:,.1f}</b>"]
e2_m_pred = [f"{v:,.1f}" for v in monthly_summary["elec_2_pred"]] + [f"<b>{sum_e2_pred:,.1f}</b>"]
e2_m_diff = [f"{v:+,.1f}" for v in monthly_summary["diff_e2"]] + [f"<b>{diff_e2:+,.1f}</b>"]
e2_m_pct = [f"{v:+.2f}%" for v in monthly_summary["pct_e2"]] + [f"<b>{pct_e2:+.2f}%</b>"]

header_e2 = header_style.copy()
header_e2["fill_color"] = "#2ca02c"

fig_tab_e2 = go.Figure(data=[go.Table(
    header=dict(
        values=["<b>Mois</b>", "<b>elec_2 Réel (kWh)</b>", "<b>elec_2 Estimé (kWh)</b>", "<b>Écart (kWh)</b>", "<b>Erreur Relative (%)</b>"],
        **header_e2
    ),
    cells=dict(
        values=[months_list, e2_m_real, e2_m_pred, e2_m_diff, e2_m_pct],
        fill_color=[["#ffffff", "#f8f9fa"] * 6 + ["#e9ecef"]],
        **cell_style_base
    )
)])
fig_tab_e2.update_layout(
    title_text="<b>TABLEAU MENSUEL DÉTAILLÉ : ELEC_2</b>",
    title_x=0.5,
    height=620,
    margin=dict(l=20, r=20, t=60, b=20),
    template="plotly_white"
)

# ---------------------------------------------------------
# 4.4 ASSEMBLAGE EN UN DASHBOARD HTML MULTI-TABS (TỐI ƯU RESPONSIVE)
# ---------------------------------------------------------
html_graph = pio.to_html(fig_graph, full_html=False, include_plotlyjs='cdn')
html_kpi = pio.to_html(fig_kpi, full_html=False, include_plotlyjs=False)
html_e1 = pio.to_html(fig_tab_e1, full_html=False, include_plotlyjs=False)
html_e2 = pio.to_html(fig_tab_e2, full_html=False, include_plotlyjs=False)

full_html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport Comparatif Énergétique 2025</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 15px;
            background-color: #f4f7f6;
            color: #333;
        }}
        .dashboard-container {{
            max-width: 1350px;
            margin: 0 auto;
            background: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        }}
        .tab-buttons {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 12px;
            margin-bottom: 20px;
        }}
        .tab-btn {{
            padding: 10px 18px;
            font-size: 13px;
            font-weight: 600;
            border: none;
            background-color: #edf2f7;
            color: #4a5568;
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.2s ease;
        }}
        .tab-btn:hover {{
            background-color: #cbd5e0;
            color: #2d3748;
        }}
        .tab-btn.active {{
            background-color: #2c3e50;
            color: #ffffff;
            box-shadow: 0 3px 8px rgba(44, 62, 80, 0.3);
        }}
        .tab-content {{
            display: none;
            width: 100%;
            overflow-x: auto; /* Cho phép cuộn ngang nếu màn hình quá nhỏ */
        }}
        .tab-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="tab-buttons">
            <button class="tab-btn active" onclick="switchTab('tab-graph', this)">Graphiques Comparatifs</button>
            <button class="tab-btn" onclick="switchTab('tab-kpi', this)">Bilan Global Annuel (KPIs)</button>
            <button class="tab-btn" onclick="switchTab('tab-e1', this)">Tableau Mensuel : elec_1</button>
            <button class="tab-btn" onclick="switchTab('tab-e2', this)">Tableau Mensuel : elec_2</button>
        </div>

        <div id="tab-graph" class="tab-content active">{html_graph}</div>
        <div id="tab-kpi" class="tab-content">{html_kpi}</div>
        <div id="tab-e1" class="tab-content">{html_e1}</div>
        <div id="tab-e2" class="tab-content">{html_e2}</div>
    </div>

    <script>
        function switchTab(tabId, btnElement) {{
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(el => el.classList.remove('active'));

            const buttons = document.querySelectorAll('.tab-btn');
            buttons.forEach(btn => btn.classList.remove('active'));

            document.getElementById(tabId).classList.add('active');
            btnElement.classList.add('active');

            // Trigger Plotly redraw để đảm bảo bảng vừa khít khung khi bật tab
            window.dispatchEvent(new Event('resize'));
        }}
    </script>
</body>
</html>
"""

output_combined_html = "dashboard_elec_2025_gb.html"
with open(output_combined_html, "w", encoding="utf-8") as f:
    f.write(full_html_content)

print(f"\n[OK] Dashboard interactif multi-onglets xuất thành công: {os.path.abspath(output_combined_html)}")