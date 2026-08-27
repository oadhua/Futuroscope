import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    "year", "month", "day", "hour", "week", "is_weekend", "is_open", "jf",
    "frequentation_park_daily", "surface", "duree", "ouvert", "interrompu",
    "operation", "visitor_count", "temperature", "humidite", "rayonnement_solaire",
    "day_degree_cold", "day_degree_hot", "temp_max", "temp_min", "temp_moy",
    "humidite_max", "humidite_min", "humidite_moy", "freq_HF", "freq_MF", "freq_THF"
]

# Séparation des ensembles d'entraînement (< 2026) et de test (>= 2026)
train_mask = df["year"] < 2026
test_mask = df["year"] >= 2026

X1_train = df.loc[train_mask, features_ecvalue]
X1_test = df.loc[test_mask, features_ecvalue]
Y1_train = df.loc[train_mask, "ec_value"]
Y1_test = df.loc[test_mask, "ec_value"]

test_dates = df.loc[test_mask, "date"] if "date" in df.columns else df.loc[test_mask].index

# ---------------------------------------------------------
# 2. ENTRAÎNEMENT DU MODÈLE XGBOOST
# ---------------------------------------------------------
model_ecvalue = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1,
    eval_metric="rmse"
)
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
print("--- RÉSULTATS : XGBOOST REGRESSION (ec_value) ---")
print(f"[ec_value] RMSE: {r1:.4f} | MAE: {m1:.4f} | R²: {r2_1:.4f}")

# Extraction du Top 5 Feature Importance de XGBoost
feature_importances = pd.DataFrame({
    "Feature": features_ecvalue,
    "Importance": model_ecvalue.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n--- TOP 5 DES CARACTÉRISTIQUES LES PLUS INFLUENTES ---")
print(feature_importances.head(5).to_string(index=False))
print("=========================================================\n")

# ---------------------------------------------------------
# 4. VISUALISATION INTERACTIVE AVEC PLOTLY & EXPORT HTML
# ---------------------------------------------------------
subtitle_1 = f"<b>ec_value : Réel vs XGBoost Regression</b><br><span style='font-size: 11px; color: #555;'>RMSE: {r1:.2f} | MAE: {m1:.2f} | R²: {r2_1:.4f}</span>"

fig = make_subplots(
    rows=1,
    cols=1,
    subplot_titles=(subtitle_1,)
)

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
    row=1, col=1
)

# Courbe Prédite (XGBoost)
fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=pred_ecvalue,
        mode="lines",
        name="ec_value Prédiction (XGBoost)",
        line=dict(color="#ff7f0e", width=1.5, dash="dot"),
        customdata=np.stack((Y1_test, np.abs(Y1_test - pred_ecvalue)), axis=-1),
        hovertemplate="<b>Horodatage:</b> %{x}<br><b>Prédiction:</b> %{y:.2f}<br><b>Écart:</b> %{customdata[1]:.2f}<extra></extra>",
    ),
    row=1, col=1
)

fig.update_layout(
    height=550,
    margin=dict(t=100, b=60, l=60, r=40),
    title_text="<b>PRÉVISION THERMIQUE (EC_VALUE) PAR XGBOOST (TEST 2026)</b>",
    title_x=0.5,
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1)
)

fig.for_each_annotation(lambda a: a.update(font=dict(size=13)))

fig.update_yaxes(title_text="Valeur ec_value", row=1, col=1)
fig.update_xaxes(rangeslider_visible=True, row=1, col=1)

# Exportation en HTML
output_html = "07. xgboost.html"
fig.write_html(output_html, include_plotlyjs="cdn")
print(f"[OK] Graphique interactif exporté avec succès : {os.path.abspath(output_html)}")

fig.show()