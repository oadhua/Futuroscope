import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES ET CONFIGURATION DES FEATURES
# ---------------------------------------------------------
file_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H07_predict_pure_thermal.csv"
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

X1_train_raw = df.loc[train_mask, features_ecvalue]
X1_test_raw = df.loc[test_mask, features_ecvalue]
Y1_train_raw = df.loc[train_mask, "ec_value"].values.reshape(-1, 1)
Y1_test = df.loc[test_mask, "ec_value"].values

test_dates = df.loc[test_mask, "date"] if "date" in df.columns else df.loc[test_mask].index

# ---------------------------------------------------------
# 2. STANDARDISATION ET ENTRAÎNEMENT DU MODÈLE SVR
# ---------------------------------------------------------
# SVR est très sensible à l'échelle : standardisation obligatoire de X et Y
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X1_train = scaler_X.fit_transform(X1_train_raw)
X1_test = scaler_X.transform(X1_test_raw)

Y1_train_scaled = scaler_y.fit_transform(Y1_train_raw).ravel()

# Initialisation et entraînement du SVM (noyau RBF)
model_ecvalue = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model_ecvalue.fit(X1_train, Y1_train_scaled)

# Prédiction et transformation inverse vers l'échelle d'origine
pred_scaled = model_ecvalue.predict(X1_test).reshape(-1, 1)
pred_ecvalue = scaler_y.inverse_transform(pred_scaled).flatten()

# ---------------------------------------------------------
# 3. PERMUTATION IMPORTANCE (POUR SVM RBF)
# ---------------------------------------------------------
def calculate_permutation_importance_svr(model, X_test_scaled, y_test_true, scaler_y, feature_names, top_n=5):
    """
    Calcul de la Permutation Importance pour le modèle SVM.
    """
    pred_base_scaled = model.predict(X_test_scaled).reshape(-1, 1)
    pred_base_actual = scaler_y.inverse_transform(pred_base_scaled).flatten()
    baseline_r2 = r2_score(y_test_true, pred_base_actual)
    
    importances = []
    for i in range(X_test_scaled.shape[1]):
        X_test_permuted = X_test_scaled.copy()
        np.random.seed(42)
        np.random.shuffle(X_test_permuted[:, i])
        
        pred_perm_scaled = model.predict(X_test_permuted).reshape(-1, 1)
        pred_perm_actual = scaler_y.inverse_transform(pred_perm_scaled).flatten()
        perm_r2 = r2_score(y_test_true, pred_perm_actual)
        
        importances.append(baseline_r2 - perm_r2)
    
    importances = np.array(importances)
    importances_pct = np.maximum(0, importances)
    if importances_pct.sum() > 0:
        importances_pct = (importances_pct / importances_pct.sum()) * 100

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance_Score": importances,
        "Importance_Pct": importances_pct
    }).sort_values(by="Importance_Pct", ascending=False)
    
    return df_imp.head(top_n)

top5_ecvalue = calculate_permutation_importance_svr(
    model_ecvalue, X1_test, Y1_test, scaler_y, features_ecvalue, top_n=5
)

# ---------------------------------------------------------
# 4. ÉVALUATION DES PERFORMANCES
# ---------------------------------------------------------
def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

r1, m1, r2_1 = get_metrics(Y1_test, pred_ecvalue)

print("\n=========================================================")
print("--- RÉSULTATS : SUPPORT VECTOR MACHINE (SVM - ec_value) ---")
print(f"[ec_value] RMSE: {r1:.4f} | MAE: {m1:.4f} | R²: {r2_1:.4f}")

print("\n--- TOP 5 DES CARACTÉRISTIQUES (PERMUTATION IMPORTANCE) ---")
print(top5_ecvalue.to_string(index=False))
print("=========================================================\n")

# ---------------------------------------------------------
# 5. VISUALISATION INTERACTIVE AVEC PLOTLY & EXPORT HTML
# ---------------------------------------------------------
subtitle_1 = f"<b>ec_value : Réel vs Support Vector Machine (SVM)</b><br><span style='font-size: 11px; color: #555;'>RMSE: {r1:.2f} | MAE: {m1:.2f} | R²: {r2_1:.4f}</span>"

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

# Courbe Prédite (SVR)
fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=pred_ecvalue,
        mode="lines",
        name="ec_value Prédiction (SVM)",
        line=dict(color="#ff7f0e", width=1.5, dash="dot"),
        customdata=np.stack((Y1_test, np.abs(Y1_test - pred_ecvalue)), axis=-1),
        hovertemplate="<b>Horodatage:</b> %{x}<br><b>Prédiction:</b> %{y:.2f}<br><b>Écart:</b> %{customdata[1]:.2f}<extra></extra>",
    ),
    row=1, col=1
)

fig.update_layout(
    height=550,
    margin=dict(t=100, b=60, l=60, r=40),
    title_text="<b>PRÉVISION THERMIQUE (EC_VALUE) PAR SVM (TEST 2026)</b>",
    title_x=0.5,
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1)
)

fig.for_each_annotation(lambda a: a.update(font=dict(size=13)))

fig.update_yaxes(title_text="Valeur ec_value", row=1, col=1)
fig.update_xaxes(rangeslider_visible=True, row=1, col=1)

# Exportation en HTML
output_html = "03. svm.html"
fig.write_html(output_html, include_plotlyjs="cdn")
print(f"[OK] Graphique interactif exporté avec succès : {os.path.abspath(output_html)}")

fig.show()