import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES ET CONFIGURATION DES FEATURES
# ---------------------------------------------------------
df = pd.read_csv(
    r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H03_predict_pure_elec.csv"
)

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
# 2. ENTRAÎNEMENT DES MODÈLES INDÉPENDANTS
# ---------------------------------------------------------
# Modèle 1: elec_1
model_elec_1 = LinearRegression()
model_elec_1.fit(X1_train, Y1_train)
pred_elec_1 = model_elec_1.predict(X1_test)

# Modèle 2: elec_2
model_elec_2 = LinearRegression()
model_elec_2.fit(X2_train, Y2_train)
pred_elec_2 = model_elec_2.predict(X2_test)


# ---------------------------------------------------------
# 3. ÉVALUATION DES PERFORMANCES ET IMPORTANCE DES FEATURES
# ---------------------------------------------------------
def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


r1, m1, r2_1 = get_metrics(Y1_test, pred_elec_1)
r2, m2, r2_2 = get_metrics(Y2_test, pred_elec_2)

print("\n--- RÉSULTATS : MODÈLES DE RÉGRESSION LINÉAIRE INDÉPENDANTS ---")
print(f"[elec_1] RMSE: {r1:.4f} kWh | MAE: {m1:.4f} kWh | R²: {r2_1:.4f}")
print(f"[elec_2] RMSE: {r2:.4f} kWh | MAE: {m2:.4f} kWh | R²: {r2_2:.4f}")

# Top features pour elec_1
coef_df1 = pd.DataFrame(
    {"Feature": features_elec_1, "Coefficient": model_elec_1.coef_}
).sort_values(by="Coefficient", key=abs, ascending=False)
print("\nTop 5 des caractéristiques les plus influentes pour [elec_1] :")
print(coef_df1.head(5).to_string(index=False))

# Top features pour elec_2
coef_df2 = pd.DataFrame(
    {"Feature": features_elec_2, "Coefficient": model_elec_2.coef_}
).sort_values(by="Coefficient", key=abs, ascending=False)
print("\nTop 5 des caractéristiques les plus influentes pour [elec_2] :")
print(coef_df2.head(5).to_string(index=False))

# ---------------------------------------------------------
# 4. VISUALISATION INTERACTIVE AVEC PLOTLY
# ---------------------------------------------------------
# Sous-titres incluant les métriques d'évaluation (RMSE, MAE, R²)
subtitle_1 = f"<b>elec_1 : Réel vs Régression</b><br><sup>RMSE: {r1:.2f} kWh | MAE: {m1:.2f} kWh | R²: {r2_1:.4f}</sup>"
subtitle_2 = f"<b>elec_2 : Réel vs Régression</b><br><sup>RMSE: {r2:.2f} kWh | MAE: {m2:.2f} kWh | R²: {r2_2:.4f}</sup>"

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.12,
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
        name="elec_1 Prédiction",
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
        name="elec_2 Prédiction",
        line=dict(color="#d62728", width=1.5, dash="dot"),
        customdata=np.stack((Y2_test, np.abs(Y2_test - pred_elec_2)), axis=-1),
        hovertemplate="<b>Horodatage:</b> %{x}<br><b>Prédiction:</b> %{y:.2f} kWh<br><b>Écart:</b> %{customdata[1]:.2f} kWh<extra></extra>",
    ),
    row=2,
    col=1,
)

fig.update_layout(
    height=800,
    title_text="<b>PRÉVISION AVEC LINEARREGRESSION (TEST 2026)</b>",
    title_x=0.5,
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

fig.update_yaxes(title_text="Consommation (kWh)", row=1, col=1)
fig.update_yaxes(title_text="Consommation (kWh)", row=2, col=1)
fig.update_xaxes(rangeslider_visible=True, row=2, col=1)

output_html = "01. linear.html"
fig.write_html(output_html, include_plotlyjs="cdn")
print(f"\n[OK] Graphique interactif exporté avec succès : {os.path.abspath(output_html)}")

fig.show()