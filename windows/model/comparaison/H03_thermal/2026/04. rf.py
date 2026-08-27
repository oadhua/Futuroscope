import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import shap

import plotly.graph_objects as go
import plotly.io as pio
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

feature_importances = pd.DataFrame(
    {"Feature": features_ecvalue, "Importance": model_ecvalue.feature_importances_}
).sort_values(by="Importance", ascending=False)

print("\n--- TOP 5 DES CARACTÉRISTIQUES LES PLUS INFLUENTES ---")
print(feature_importances.head(5).to_string(index=False))
print("=========================================================\n")


# ---------------------------------------------------------
# 4. VISUALISATION INTERACTIVE ET DASHBOARD PLOTLY MULTI-TABS
# ---------------------------------------------------------
df_results = df.loc[test_mask, ["date"] + features_ecvalue].copy()
df_results = df_results.sort_values("date")

df_results["ec_value_real"] = Y1_test.values
df_results["ec_value_pred"] = pred_ecvalue

# Agrégation mensuelle
df_results["month_num"] = df_results["date"].dt.month
monthly_summary = df_results.groupby("month_num").agg({
    "ec_value_real": "sum",
    "ec_value_pred": "sum",
    "visitor_count": "sum",
    "is_open": "max",
    "operation": "max"
}).reset_index()

monthly_summary["month_label"] = monthly_summary["month_num"].apply(lambda x: f"Mois {x}")

# 4.1 CALCUL DES INDICATEURS DE PERFORMANCE (KPIs)
sum_ec_real = df_results["ec_value_real"].sum()
sum_ec_pred = df_results["ec_value_pred"].sum()
diff_ec = sum_ec_pred - sum_ec_real
pct_ec = (diff_ec / sum_ec_real) * 100 if sum_ec_real != 0 else 0

monthly_summary["diff_ec"] = monthly_summary["ec_value_pred"] - monthly_summary["ec_value_real"]
monthly_summary["pct_ec"] = (monthly_summary["diff_ec"] / monthly_summary["ec_value_real"]) * 100

months_list = [f"Mois {m}" for m in monthly_summary["month_num"]] + ["<b>CUMUL ANNUEL</b>"]

# 4.2 COMPOSANT 1 : GRAPHIQUE INTERACTIF MULTI-VUES
FONT_GRAPH = "Segoe UI, Arial, sans-serif"

subtitle_1 = f"<b>ec_value : Consommation Thermique Réelle vs Modèle RF</b><br><span style='font-size: 11px; color: #555;'>RMSE: {r1:.2f} | MAE: {m1:.2f} | R²: {r2_1:.4f}</span>"

fig_graph = make_subplots(
    rows=1, cols=1,
    subplot_titles=(subtitle_1,)
)

colors = {"real": "#1f77b4", "pred": "#ff7f0e"}

# Mảng dữ liệu customdata cho giờ và tháng
customdata_hourly = df_results[["ec_value_real", "ec_value_pred", "visitor_count", "is_open", "operation"]].values
customdata_monthly = monthly_summary[["ec_value_real", "ec_value_pred", "visitor_count", "is_open", "operation"]].values

hover_template_custom = (
    "<b>Date / Période:</b> %{x}<br>"
    "<b>Réel:</b> %{customdata[0]:,.2f}<br>"
    "<b>Estimé (RF):</b> %{customdata[1]:,.2f}<br>"
    "<b>Visitors:</b> %{customdata[2]:,}<br>"
    "<b>Is Open:</b> %{customdata[3]}<br>"
    "<b>Operation:</b> %{customdata[4]}"
    "<extra></extra>"
)

# Traces Bar (Vue Mensuelle)
fig_graph.add_trace(go.Bar(
    x=monthly_summary["month_label"], y=monthly_summary["ec_value_real"],
    name="ec_value Réel", marker_color=colors["real"], visible=True,
    customdata=customdata_monthly,
    hovertemplate=hover_template_custom,
    hoverlabel=dict(namelength=0)
), row=1, col=1)

fig_graph.add_trace(go.Bar(
    x=monthly_summary["month_label"], y=monthly_summary["ec_value_pred"],
    name="ec_value Estimé (RF)", marker_color=colors["pred"], visible=True,
    hoverinfo="skip"
), row=1, col=1)

# Traces Scatter (Vue Horaire)
fig_graph.add_trace(go.Scatter(
    x=df_results["date"], y=df_results["ec_value_real"],
    mode="lines", name="ec_value Réel", line=dict(color=colors["real"], width=1.2), visible=False,
    customdata=customdata_hourly,
    hovertemplate=hover_template_custom,
    hoverlabel=dict(namelength=0)
), row=1, col=1)

fig_graph.add_trace(go.Scatter(
    x=df_results["date"], y=df_results["ec_value_pred"],
    mode="lines", name="ec_value Estimé (RF)", line=dict(color=colors["pred"], width=1.2, dash="dash"), visible=False,
    hoverinfo="skip"
), row=1, col=1)

fig_graph.update_layout(
    font=dict(family=FONT_GRAPH, size=12, color="#2c3e50"),
    title=dict(
        text="<b>ÉVALUATION GRAPHIQUE DE LA PRÉVISION THERMIQUE (2026)</b>",
        x=0.5,
        font=dict(family=FONT_GRAPH, size=15, color="#2c3e50")
    ),
    hovermode="x",
    updatemenus=[
        dict(
            buttons=[
                dict(
                    label="Vue Mensuelle (Bâtons)",
                    method="update",
                    args=[
                        {"visible": [True, True, False, False]},
                        {"barmode": "group", "xaxis.type": "category", "xaxis.rangeslider.visible": False}
                    ]
                ),
                dict(
                    label="Vue Horaire (Série Temporelle)",
                    method="update",
                    args=[
                        {"visible": [False, False, True, True]},
                        {"xaxis.type": "date", "xaxis.rangeslider.visible": True}
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
    height=650,
    template="plotly_white"
)

fig_graph.for_each_annotation(lambda a: a.update(font=dict(family=FONT_GRAPH, size=12, color="#2c3e50")))
fig_graph.update_xaxes(showspikes=False, type="category", title_font=dict(family=FONT_GRAPH, size=12), tickfont=dict(family=FONT_GRAPH, size=11), row=1, col=1)
fig_graph.update_yaxes(showspikes=False, title_text="Valeur ec_value", title_font=dict(family=FONT_GRAPH, size=12), tickfont=dict(family=FONT_GRAPH, size=11), row=1, col=1)


# ---------------------------------------------------------
# 4.3 COMPOSANTS TABLEAUX (KPI & MENSUEL)
# ---------------------------------------------------------
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

# 1. Bilan Global Annuel
fig_kpi = go.Figure(data=[go.Table(
    header=dict(
        values=["<b>Indicateur / Cible</b>", "<b>Réel Total</b>", "<b>Estimé Total (RF)</b>", "<b>Écart</b>", "<b>Erreur Relative (%)</b>"],
        **header_style
    ),
    cells=dict(
        values=[
            ["<b>ec_value (Thermique)</b>"],
            [f"{sum_ec_real:,.2f}"],
            [f"{sum_ec_pred:,.2f}"],
            [f"{diff_ec:+,.2f}"],
            [f"{pct_ec:+.2f}%"]
        ],
        fill_color=[["#f8f9fa"]],
        **cell_style_base
    )
)])
fig_kpi.update_layout(
    title_text="<b>BILAN GLOBAL ANNUEL : EC_VALUE (2026)</b>",
    title_x=0.5,
    height=300,
    margin=dict(l=20, r=20, t=60, b=20),
    template="plotly_white"
)

# 2. Tableau Mensuel ec_value
ec_m_real = [f"{v:,.1f}" for v in monthly_summary["ec_value_real"]] + [f"<b>{sum_ec_real:,.1f}</b>"]
ec_m_pred = [f"{v:,.1f}" for v in monthly_summary["ec_value_pred"]] + [f"<b>{sum_ec_pred:,.1f}</b>"]
ec_m_diff = [f"{v:+,.1f}" for v in monthly_summary["diff_ec"]] + [f"<b>{diff_ec:+,.1f}</b>"]
ec_m_pct = [f"{v:+.2f}%" for v in monthly_summary["pct_ec"]] + [f"<b>{pct_ec:+.2f}%</b>"]

header_ec = header_style.copy()
header_ec["fill_color"] = "#1f77b4"

fig_tab_ec = go.Figure(data=[go.Table(
    header=dict(
        values=["<b>Mois</b>", "<b>Réel</b>", "<b>Estimé (RF)</b>", "<b>Écart</b>", "<b>Erreur Relative (%)</b>"],
        **header_ec
    ),
    cells=dict(
        values=[months_list, ec_m_real, ec_m_pred, ec_m_diff, ec_m_pct],
        fill_color=[["#ffffff", "#f8f9fa"] * 6 + ["#e9ecef"]],
        **cell_style_base
    )
)])
fig_tab_ec.update_layout(
    title_text="<b>TABLEAU MENSUEL DÉTAILLÉ : EC_VALUE</b>",
    title_x=0.5,
    height=620,
    margin=dict(l=20, r=20, t=60, b=20),
    template="plotly_white"
)


# ---------------------------------------------------------
# 4.4 ASSEMBLAGE EN UN DASHBOARD HTML MULTI-TABS
# ---------------------------------------------------------
html_graph = pio.to_html(fig_graph, full_html=False, include_plotlyjs='cdn')
html_kpi = pio.to_html(fig_kpi, full_html=False, include_plotlyjs=False)
html_ec = pio.to_html(fig_tab_ec, full_html=False, include_plotlyjs=False)

full_html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport Prédiction Thermique 2026 - Random Forest</title>
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
            overflow-x: auto;
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
            <button class="tab-btn" onclick="switchTab('tab-ec', this)">Tableau Mensuel : ec_value</button>
        </div>

        <div id="tab-graph" class="tab-content active">{html_graph}</div>
        <div id="tab-kpi" class="tab-content">{html_kpi}</div>
        <div id="tab-ec" class="tab-content">{html_ec}</div>
    </div>

    <script>
        function switchTab(tabId, btnElement) {{
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(el => el.classList.remove('active'));

            const buttons = document.querySelectorAll('.tab-btn');
            buttons.forEach(btn => btn.classList.remove('active'));

            document.getElementById(tabId).classList.add('active');
            btnElement.classList.add('active');

            window.dispatchEvent(new Event('resize'));
        }}
    </script>
</body>
</html>
"""

output_combined_html = "dashboard_thermal_2026_rf.html"
with open(output_combined_html, "w", encoding="utf-8") as f:
    f.write(full_html_content)

print(f"\n[OK] Dashboard interactif multi-onglets exporté avec succès : {os.path.abspath(output_combined_html)}")

# # ---------------------------------------------------------
# # 5. XUẤT TỆP KẾT QUẢ DỰ BÁO VÀ FEATURES ĐẦU VÀO
# # ---------------------------------------------------------
# df_results_export = df.loc[test_mask, ["date"] + features_ecvalue].copy()
# df_results_export["ec_value_real"] = Y1_test.values
# df_results_export["ec_value_pred"] = pred_ecvalue
# df_results_export["ec_value_error_abs"] = np.abs(Y1_test.values - pred_ecvalue)

# output_csv = "predict_thermal_2026.csv"
# df_results_export.to_csv(output_csv, index=False, encoding="utf-8-sig")

# output_excel = "predict_thermal_2026.xlsx"
# df_results_export.to_excel(output_excel, index=False, sheet_name="Forecast_Results")

# print(f"[OK] File kết quả đã xuất thành công: {output_excel} và {output_csv}")