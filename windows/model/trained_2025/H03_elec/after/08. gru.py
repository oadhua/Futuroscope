import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Fixation des graines pour la reproductibilité des résultats
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================================
# 1. Chargement des données (Fichier Elec global)
# ==========================================================
file_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H03_predict_pure_elec.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Impossible de trouver le fichier : {file_path}")

donnees = pd.read_csv(file_path)
donnees["date"] = pd.to_datetime(donnees["date"])
donnees = donnees.sort_values("date").reset_index(drop=True)

# Groupe mensuel pour le Blocked Split
donnees["annee_mois"] = donnees["date"].dt.to_period("M")

# ==========================================================
# 2. Sélection des variables (Selon le modèle global)
# ==========================================================
# Variables cibles multiples (Multi-output)
variable_cible = ["elec_1", "elec_2"]

# Exclusion des variables de fuite (leak) et temporelles brutes
variables_exclues = [
    "date",
    "elec_1",
    "elec_2",
    "min",
    "year",
    "day",
    "hour",
    "week",
    "annee_mois",
    # "heure_sin",
    # "heure_cos",
    # "mois_sin",
    # "mois_cos",
    # "hour_x_freq_HF",
    # "hour_x_freq_MF",
    # "temp_decalage_1h",
]
variables_entree = [col for col in donnees.columns if col not in variables_exclues]

# ==========================================================
# 3. Normalisation des données (MinMax Scaling)
# ==========================================================
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

# Ajustement des scalers sur l'ensemble des données d'entrée/sortie
scaler_X.fit(donnees[variables_entree])
scaler_y.fit(donnees[variable_cible])

# ==========================================================
# 4. Blocked Split & Génération des séquences temporelles
# ==========================================================
# Fenêtre d'observation passée (Lookback)
LOOKBACK = 24 

X_train_global = []
y_train_global = []
X_test_global = []
y_test_global = []
dates_test_global = []

def create_sequences(X_data, y_data, dates_data, lookback):
    """
    Découpe les données en fenêtres temporelles glissantes
    pour éviter toute fuite de données entre les mois.
    """
    X_seq, y_seq, dates_seq = [], [], []
    for i in range(lookback, len(X_data)):
        X_seq.append(X_data[i - lookback : i])
        y_seq.append(y_data[i])
        dates_seq.append(dates_data[i])
    return np.array(X_seq), np.array(y_seq), np.array(dates_seq)

# Division 80/20 par groupe mensuel
for name, group in donnees.groupby("annee_mois"):
    if len(group) < (LOOKBACK + 10):
        continue
    
    # Transformation des données du groupe
    X_scaled = scaler_X.transform(group[variables_entree])
    y_scaled = scaler_y.transform(group[variable_cible])
    dates = group["date"].values
    
    split_idx = int(len(group) * 0.8)
    
    X_train_part, X_test_part = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train_part, y_test_part = y_scaled[:split_idx], y_scaled[split_idx:]
    dates_train_part, dates_test_part = dates[:split_idx], dates[split_idx:]
    
    # Création des séquences pour le mois en cours
    X_tr_seq, y_tr_seq, _ = create_sequences(X_train_part, y_train_part, dates_train_part, LOOKBACK)
    X_ts_seq, y_ts_seq, dates_ts_seq = create_sequences(X_test_part, y_test_part, dates_test_part, LOOKBACK)
    
    if len(X_tr_seq) > 0:
        X_train_global.append(X_tr_seq)
        y_train_global.append(y_tr_seq)
    if len(X_ts_seq) > 0:
        X_test_global.append(X_ts_seq)
        y_test_global.append(y_ts_seq)
        dates_test_global.append(dates_ts_seq)

# Reconstitution des jeux de données globaux
X_train_global = np.concatenate(X_train_global, axis=0)
y_train_global = np.concatenate(y_train_global, axis=0)
X_test_global = np.concatenate(X_test_global, axis=0)
y_test_global = np.concatenate(y_test_global, axis=0)
dates_test_global = np.concatenate(dates_test_global, axis=0)

print(f"Dimensions des données d'entrée GRU :")
print(f" -> X_train: {X_train_global.shape} (Echantillons, Pas de temps, Variables)")
print(f" -> X_test : {X_test_global.shape}\n")

# ==========================================================
# 5. Construction et entraînement du modèle GRU (Multi-output)
# ==========================================================
model = Sequential()

# Première couche GRU
model.add(GRU(
    units=64, 
    return_sequences=True, 
    input_shape=(X_train_global.shape[1], X_train_global.shape[2])
))
model.add(Dropout(0.2))

# Deuxième couche GRU
model.add(GRU(units=32, return_sequences=False))
model.add(Dropout(0.2))

# Couche de sortie pour prédire elec_1 et elec_2 simultanément
model.add(Dense(units=2))  

model.compile(optimizer="adam", loss="mse")

# Callback d'arrêt précoce pour limiter le surapprentissage
early_stopping = EarlyStopping(
    monitor="val_loss", 
    patience=5, 
    restore_best_weights=True
)

print("--- DEBUT DE L'ENTRAINEMENT DU GRU GLOBAL ---")
history = model.fit(
    X_train_global, 
    y_train_global,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# ==========================================================
# 6. Prédictions et inversion de la normalisation
# ==========================================================
predictions_train_scaled = model.predict(X_train_global)
predictions_reelles_scaled = model.predict(X_test_global)

# Retour aux unités d'origine (kWh)
predictions_train = scaler_y.inverse_transform(predictions_train_scaled)
y_train_reel = scaler_y.inverse_transform(y_train_global)

predictions_reelles = scaler_y.inverse_transform(predictions_reelles_scaled)
y_test_reel = scaler_y.inverse_transform(y_test_global)

# Seuil physique pour éviter les valeurs négatives
predictions_train = np.clip(predictions_train, a_min=0, a_max=None)
predictions_reelles = np.clip(predictions_reelles, a_min=0, a_max=None)

# ==========================================================
# 7. Évaluation des performances
# ==========================================================
# Évaluation globale (Moyenne uniforme des deux colonnes)
r2_train = r2_score(y_train_reel, predictions_train, multioutput="uniform_average")
mse_global = mean_squared_error(
    y_test_reel, predictions_reelles, multioutput="uniform_average"
)
mae_global = mean_absolute_error(
    y_test_reel, predictions_reelles, multioutput="uniform_average"
)
r2_global = r2_score(y_test_reel, predictions_reelles, multioutput="uniform_average")

# Évaluation détaillée par compteur
r2_chi_tiet = r2_score(y_test_reel, predictions_reelles, multioutput="raw_values")
mae_chi_tiet = mean_absolute_error(
    y_test_reel, predictions_reelles, multioutput="raw_values"
)

print("=== RÉSULTATS DU MODÈLE GLOBAL GRU - MULTI-OUTPUT ===")
print(f"GRU Train R² (Moyenne): {r2_train:.4f}")
print(f"GRU Test R²  (Moyenne): {r2_global:.4f} | MAE (Moyenne): {mae_global:.2f}")
print("-" * 65)
print(f" -> [elec_1] R² Test: {r2_chi_tiet[0]:.4f} | MAE: {mae_chi_tiet[0]:.2f}")
print(f" -> [elec_2] R² Test: {r2_chi_tiet[1]:.4f} | MAE: {mae_chi_tiet[1]:.2f}")
print("=====================================================\n")

# ==========================================================
# 8. Visualisation avec Plotly (Format HTML standardisé)
# ==========================================================
resultats = (
    pd.DataFrame(
        {
            "Date": pd.to_datetime(dates_test_global),
            "elec_1_Reelle": y_test_reel[:, 0],
            "elec_1_Predite": predictions_reelles[:, 0],
            "elec_2_Reelle": y_test_reel[:, 1],
            "elec_2_Predite": predictions_reelles[:, 1]
        }
    )
    .sort_values("Date")
    .reset_index(drop=True)
)

dates_formattees = resultats["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")

fig = go.Figure()

# --- ÉLECTRICITÉ 1 (Tons Bleus) ---
fig.add_trace(
    go.Scatter(
        x=dates_formattees,
        y=resultats["elec_1_Reelle"],
        mode="lines",
        name="elec_1 Réelle",
        line=dict(color="darkblue"),
    )
)
fig.add_trace(
    go.Scatter(
        x=dates_formattees,
        y=resultats["elec_1_Predite"],
        mode="lines",
        name="elec_1 Prédite",
        line=dict(color="dodgerblue", dash="dash"),
    )
)

# --- ÉLECTRICITÉ 2 (Tons Rouges/Oranges) ---
fig.add_trace(
    go.Scatter(
        x=dates_formattees,
        y=resultats["elec_2_Reelle"],
        mode="lines",
        name="elec_2 Réelle",
        line=dict(color="darkred"),
    )
)
fig.add_trace(
    go.Scatter(
        x=dates_formattees,
        y=resultats["elec_2_Predite"],
        mode="lines",
        name="elec_2 Prédite",
        line=dict(color="orange", dash="dash"),
    )
)

titre_graphe = f"Prévisions Globales (elec_1 & elec_2) via GRU | elec_1_R²: {r2_chi_tiet[0]:.4f} - MAE: {mae_chi_tiet[0]:.2f} | elec_2_R²: {r2_chi_tiet[1]:.4f} - MAE: {mae_chi_tiet[1]:.2f}"
fig.update_layout(
    title=titre_graphe,
    xaxis_title="Date",
    yaxis_title="Consommation Électrique (kWh)",
    hovermode="x unified",
    xaxis=dict(rangeslider=dict(visible=True), type="date"),
)

# Sauvegarde automatique du fichier HTML
fig.write_html("prévisions_gru_elec.html")

fig.show()