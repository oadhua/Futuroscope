import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Configuration des seeds pour la reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================================
# 1. Chargement et préparation des données
# ==========================================================
donnees = pd.read_csv(
    r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H07_predict_pure_thermal.csv"
)
donnees["date"] = pd.to_datetime(donnees["date"])
donnees = donnees.sort_values("date").reset_index(drop=True)

# Groupe mensuel pour le Blocked Split
donnees["annee_mois"] = donnees["date"].dt.to_period("M")

# ==========================================================
# 2. Sélection des variables (Sans filtre Kendall)
# ==========================================================
variables_exclues = [
    "date",
    "ec_value",
    "min",
    "year",
    "month",
    "day",
    "hour",
    "week",
    "id_attraction",
    "annee_mois",
    # "heure_sin",
    # "heure_cos",
    # "mois_sin",
    # "mois_cos",
    # "hour_x_freq_HF",
    # "hour_x_freq_MF",
    # "temp_decalage_1h"
]
variables_entree = [col for col in donnees.columns if col not in variables_exclues]
variable_cible = "ec_value"

# Scalers requis pour LSTM
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

scaler_X.fit(donnees[variables_entree])
scaler_y.fit(donnees[[variable_cible]])

# ==========================================================
# 3. Blocked Split temporel (80% Train / 20% Test par mois)
# ==========================================================
LOOKBACK = 24

X_train_global = []
y_train_global = []
X_test_global = []
y_test_global = []
dates_test_global = []

def create_sequences(X_data, y_data, dates_data, lookback):
    X_seq, y_seq, dates_seq = [], [], []
    for i in range(lookback, len(X_data)):
        X_seq.append(X_data[i - lookback : i])
        y_seq.append(y_data[i])
        dates_seq.append(dates_data[i])
    return np.array(X_seq), np.array(y_seq), np.array(dates_seq)

for name, group in donnees.groupby("annee_mois"):
    if len(group) < (LOOKBACK + 10):
        continue
    
    X_scaled = scaler_X.transform(group[variables_entree])
    y_scaled = scaler_y.transform(group[[variable_cible]])
    dates = group["date"].values
    
    split_idx = int(len(group) * 0.8)
    
    X_train_part, X_test_part = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train_part, y_test_part = y_scaled[:split_idx], y_scaled[split_idx:]
    dates_train_part, dates_test_part = dates[:split_idx], dates[split_idx:]
    
    X_tr_seq, y_tr_seq, _ = create_sequences(X_train_part, y_train_part, dates_train_part, LOOKBACK)
    X_ts_seq, y_ts_seq, dates_ts_seq = create_sequences(X_test_part, y_test_part, dates_test_part, LOOKBACK)
    
    if len(X_tr_seq) > 0:
        X_train_global.append(X_tr_seq)
        y_train_global.append(y_tr_seq)
    if len(X_ts_seq) > 0:
        X_test_global.append(X_ts_seq)
        y_test_global.append(y_ts_seq)
        dates_test_global.append(dates_ts_seq)

X_train_global = np.concatenate(X_train_global, axis=0)
y_train_global = np.concatenate(y_train_global, axis=0)
X_test_global = np.concatenate(X_test_global, axis=0)
y_test_global = np.concatenate(y_test_global, axis=0)
dates_test_global = np.concatenate(dates_test_global, axis=0)

# ==========================================================
# 4. Entraînement du modèle LSTM
# ==========================================================
model = Sequential()
model.add(LSTM(
    units=64, 
    return_sequences=True, 
    input_shape=(X_train_global.shape[1], X_train_global.shape[2])
))
model.add(Dropout(0.2))

model.add(LSTM(units=32, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1))  

model.compile(optimizer="adam", loss="mse")

early_stopping = EarlyStopping(
    monitor="val_loss", 
    patience=5, 
    restore_best_weights=True
)

model.fit(
    X_train_global, 
    y_train_global,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# ==========================================================
# 5. Prédictions et évaluation
# ==========================================================
predictions_train_scaled = model.predict(X_train_global)
predictions_scaled = model.predict(X_test_global)

predictions_train = scaler_y.inverse_transform(predictions_train_scaled).flatten()
predictions_reelles = scaler_y.inverse_transform(predictions_scaled).flatten()
y_train_original = scaler_y.inverse_transform(y_train_global).flatten()
y_test_reel = scaler_y.inverse_transform(y_test_global).flatten()

# Limitation des valeurs négatives
predictions_reelles = np.clip(predictions_reelles, a_min=0, a_max=None)

r2_train = r2_score(y_train_original, predictions_train)
mse = mean_squared_error(y_test_reel, predictions_reelles)
mae = mean_absolute_error(y_test_reel, predictions_reelles)
r2 = r2_score(y_test_reel, predictions_reelles)

print("\n==========================================================")
print(f"LSTM Train R²: {r2_train:.4f}")
print(f"LSTM Test R²: {r2:.4f} | MAE: {mae:.2f} | MSE: {mse:.2f}")
print("==========================================================")

# ==========================================================
# 6. Génération et sauvegarde du graphique HTML
# ==========================================================
resultats = (
    pd.DataFrame(
        {
            "Date": pd.to_datetime(dates_test_global),
            "Valeur_Reelle": y_test_reel,
            "Valeur_Predite": predictions_reelles,
        }
    )
    .sort_values("Date")
    .reset_index(drop=True)
)

dates_formattees = resultats["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=dates_formattees,
        y=resultats["Valeur_Reelle"],
        mode="lines",
        name="Réel",
        line=dict(color="darkblue"),
    )
)

fig.add_trace(
    go.Scatter(
        x=dates_formattees,
        y=resultats["Valeur_Predite"],
        mode="lines",
        name="Prédiction LSTM",
        line=dict(color="darkorange", dash="dash"),
    )
)

titre_graphe = (
    f"Prévisions ec_value LSTM | R²: {r2:.4f} | MAE: {mae:.2f}"
)
fig.update_layout(
    title=titre_graphe,
    xaxis_title="Date",
    yaxis_title="ec_value",
    hovermode="x unified",
    xaxis=dict(rangeslider=dict(visible=True), type="date"),
)

# Sauvegarde locale automatique en HTML
fig.write_html("prévisions_lstm.html")

# Affichage temporaire dans le navigateur
fig.show()