import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES ET CONFIGURATION DES FEATURES
# ---------------------------------------------------------
file_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H03_predict_pure_elec.csv"
df = pd.read_csv(file_path)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

features_elec_1 = [
    "year", "month", "day", "hour", "week", "is_weekend", "is_open", "jf",
    "frequentation_park_daily", "surface", "duree", "ouvert", "interrompu",
    "operation", "visitor_count", "temperature", "humidite", "rayonnement_solaire",
    "day_degree_cold", "day_degree_hot", "temp_max", "temp_min", "temp_moy",
    "humidite_max", "humidite_min", "humidite_moy", "freq_HF", "freq_MF", "freq_THF"
]

features_elec_2 = [
    "year", "month", "day", "hour", "week", "is_weekend", "is_open", "jf",
    "frequentation_park_daily", "surface", "duree", "ouvert", "interrompu",
    "operation", "visitor_count", "temperature", "humidite", "rayonnement_solaire",
    "day_degree_cold", "day_degree_hot", "temp_max", "temp_min", "temp_moy",
    "humidite_max", "humidite_min", "humidite_moy", "freq_HF", "freq_MF", "freq_THF"
]

train_mask = df["year"] < 2026
test_mask = df["year"] >= 2026

# ---------------------------------------------------------
# 2. CRÉATION SÉQUENCES TEMPORELLES & PERMUTATION IMPORTANCE (3D)
# ---------------------------------------------------------
def create_sequences(X, y, seq_length=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i : i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

SEQ_LENGTH = 24

def calculate_permutation_importance_rnn(model, X_test_seq, y_test_true, scaler_y, feature_names, top_n=5):
    """
    Calcul de la Permutation Importance pour les données 3D (échantillons, pas de temps, caractéristiques).
    """
    pred_base_scaled = model.predict(X_test_seq, verbose=0)
    pred_base_actual = scaler_y.inverse_transform(pred_base_scaled).flatten()
    baseline_r2 = r2_score(y_test_true, pred_base_actual)
    
    num_features = X_test_seq.shape[2]
    importances = []
    
    for i in range(num_features):
        X_test_permuted = X_test_seq.copy()
        np.random.seed(42)
        # Permutation de la caractéristique i sur l'ensemble des échantillons
        feat_data = X_test_permuted[:, :, i]
        np.random.shuffle(feat_data)
        X_test_permuted[:, :, i] = feat_data
        
        pred_perm_scaled = model.predict(X_test_permuted, verbose=0)
        pred_perm_actual = scaler_y.inverse_transform(pred_perm_scaled).flatten()
        perm_r2 = r2_score(y_test_true, pred_perm_actual)
        
        # Baisse du R2 suite à la permutation
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

# ---------------------------------------------------------
# 3. ENTRAÎNEMENT ET DÉROULEMENT DU MODÈLE KERAS LSTM
# ---------------------------------------------------------
def train_and_predict_keras_lstm(features, target_name):
    # Standardisation
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_raw = df.loc[train_mask, features].values
    y_train_raw = df.loc[train_mask, target_name].values.reshape(-1, 1)

    X_test_raw = df.loc[test_mask, features].values
    y_test_raw = df.loc[test_mask, target_name].values.reshape(-1, 1)

    # Concaténation des 24 derniers pas du jeu d'entraînement au jeu de test
    X_full = np.vstack((X_train_raw[-SEQ_LENGTH:], X_test_raw))
    y_full = np.vstack((y_train_raw[-SEQ_LENGTH:], y_test_raw))

    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw)

    X_full_scaled = scaler_X.transform(X_full)
    y_full_scaled = scaler_y.transform(y_full)

    # Transformation en séquences 3D (échantillons, pas de temps, caractéristiques)
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQ_LENGTH)
    X_test_seq, _ = create_sequences(X_full_scaled, y_full_scaled, SEQ_LENGTH)

    # Architecture Keras Sequential
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, len(features))),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Arrêt précoce si la perte de validation ne s'améliore plus après 5 époques
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Entraînement
    print(f"\n--- Entraînement du modèle Keras LSTM pour [{target_name}] ---")
    model.fit(
        X_train_seq, y_train_seq,
        epochs=30,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # Prédiction
    pred_scaled = model.predict(X_test_seq)
    pred_actual = scaler_y.inverse_transform(pred_scaled).flatten()

    # Calcul de la Permutation Importance pour le Top 5
    top_5_imp = calculate_permutation_importance_rnn(
        model, X_test_seq, y_test_raw.flatten(), scaler_y, features, top_n=5
    )

    return pred_actual, top_5_imp

# ---------------------------------------------------------
# 4. EXÉCUTION POUR LES DEUX CIBLES
# ---------------------------------------------------------
pred_elec_1, top5_elec_1 = train_and_predict_keras_lstm(features_elec_1, "elec_1")
pred_elec_2, top5_elec_2 = train_and_predict_keras_lstm(features_elec_2, "elec_2")

Y1_test = df.loc[test_mask, "elec_1"].values
Y2_test = df.loc[test_mask, "elec_2"].values
test_dates = df.loc[test_mask, "date"] if "date" in df.columns else df.loc[test_mask].index

# ---------------------------------------------------------
# 5. ÉVALUATION ET PERFORMANCES
# ---------------------------------------------------------
def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

r1, m1, r2_1 = get_metrics(Y1_test, pred_elec_1)
r2, m2, r2_2 = get_metrics(Y2_test, pred_elec_2)

print("\n=========================================================")
print("--- RÉSULTATS : MODÈLES KERAS LSTM INDÉPENDANTS ---")
print(f"[elec_1] RMSE: {r1:.4f} kWh | MAE: {m1:.4f} kWh | R²: {r2_1:.4f}")
print(f"[elec_2] RMSE: {r2:.4f} kWh | MAE: {m2:.4f} kWh | R²: {r2_2:.4f}")

print("\n--- TOP 5 DES CARACTÉRISTIQUES (PERMUTATION IMPORTANCE) ---")
print("\n[Top 5 Caractéristiques pour elec_1] :")
print(top5_elec_1.to_string(index=False))

print("\n[Top 5 Caractéristiques pour elec_2] :")
print(top5_elec_2.to_string(index=False))
print("=========================================================\n")

# ---------------------------------------------------------
# 6. VISUALISATION INTERACTIVE AVEC PLOTLY & EXPORT HTML
# ---------------------------------------------------------
subtitle_1 = f"<b>elec_1 : Réel vs Keras LSTM</b><br><span style='font-size: 11px; color: #555;'>RMSE: {r1:.2f} kWh | MAE: {m1:.2f} kWh | R²: {r2_1:.4f}</span>"
subtitle_2 = f"<b>elec_2 : Réel vs Keras LSTM</b><br><span style='font-size: 11px; color: #555;'>RMSE: {r2:.2f} kWh | MAE: {m2:.2f} kWh | R²: {r2_2:.4f}</span>"

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
        name="elec_1 Prédiction (Keras LSTM)",
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
        name="elec_2 Prédiction (Keras LSTM)",
        line=dict(color="#d62728", width=1.5, dash="dot"),
        customdata=np.stack((Y2_test, np.abs(Y2_test - pred_elec_2)), axis=-1),
        hovertemplate="<b>Horodatage:</b> %{x}<br><b>Prédiction:</b> %{y:.2f} kWh<br><b>Écart:</b> %{customdata[1]:.2f} kWh<extra></extra>",
    ),
    row=2,
    col=1,
)

# Configuration du Layout
fig.update_layout(
    height=850,
    margin=dict(t=100, b=60, l=60, r=40),
    title_text="<b>PRÉVISION PAR KERAS LSTM INDÉPENDANTS (TEST 2026)</b>",
    title_x=0.5,
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
)

fig.for_each_annotation(lambda a: a.update(font=dict(size=13)))

fig.update_yaxes(title_text="Consommation (kWh)", row=1, col=1)
fig.update_yaxes(title_text="Consommation (kWh)", row=2, col=1)
fig.update_xaxes(rangeslider_visible=True, row=2, col=1)

# Exportation en HTML
output_html = "08. lstm_keras.html"
fig.write_html(output_html, include_plotlyjs="cdn")
print(f"\n[OK] Graphique interactif exporté avec succès : {os.path.abspath(output_html)}")

fig.show()