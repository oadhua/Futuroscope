import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Fixation des graines pour la reproductibilité des résultats
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================================
# 1. Chargement des données (Fichier Elec H07)
# ==========================================================
file_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H07_predict_pure_elec.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Impossible de trouver le fichier : {file_path}")

donnees = pd.read_csv(file_path)
donnees["date"] = pd.to_datetime(donnees["date"])
donnees = donnees.sort_values("date").reset_index(drop=True)

# Groupe mensuel pour le Blocked Split
donnees["annee_mois"] = donnees["date"].dt.to_period("M")

# ==========================================================
# 2. Sélection des variables (Spécifique H07 - 3 targets)
# ==========================================================
# Variables cibles multiples (3 targets)
variable_cible = ["elec_1", "elec_2", "elec_3"]

# Exclusion des variables de fuite (leak) et temporelles brutes
variables_exclues = [
    "date", 
    "elec_1", 
    "elec_2", 
    "elec_3", 
    "min", 
    "year", 
    "day", 
    # "hour", 
    "week",
    "id_attraction",
    "annee_mois",
    "heure_sin",
    "heure_cos",
    "mois_sin",
    "mois_cos",
    "hour_x_freq_HF",
    "hour_x_freq_MF",
    "temp_decalage_1h",
]
variables_entree = [col for col in donnees.columns if col not in variables_exclues]

X = donnees[variables_entree]
y = donnees[variable_cible]

# ==========================================================
# 3. Blocked Split par mois (80% Train / 20% Test par mois)
# ==========================================================
train_indices = []
test_indices = []

for name, group in donnees.groupby("annee_mois"):
    if len(group) < 10:
        continue

    split_idx = int(len(group) * 0.8)

    train_indices.extend(group.index[:split_idx])
    test_indices.extend(group.index[split_idx:])

# Création des ensembles Train/Test globaux
X_train = X.loc[train_indices].reset_index(drop=True)
y_train = y.loc[train_indices].reset_index(drop=True)

X_test = X.loc[test_indices].reset_index(drop=True)
y_test_reel = y.loc[test_indices].reset_index(drop=True)
dates_test = donnees["date"].loc[test_indices].reset_index(drop=True)

# ==========================================================
# 4. Normalisation des données (Standard Scaling cho ANN)
# ==========================================================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test_reel)

# ==========================================================
# 5. Construction et entraînement du modèle ANN (3 Outputs)
# ==========================================================
model = Sequential()

# Couche d'entrée et première couche cachée
model.add(Dense(units=128, activation="relu", input_dim=X_train_scaled.shape[1]))
model.add(Dropout(0.2))

# Deuxième couche cachée
model.add(Dense(units=64, activation="relu"))
model.add(Dropout(0.2))

# Troisième couche cachée
model.add(Dense(units=32, activation="relu"))

# Couche de sortie configurée pour prédire elec_1, elec_2 et elec_3 simultanément
model.add(Dense(units=3, activation="linear"))  

model.compile(optimizer="adam", loss="mse")

# Callback d'arrêt précoce
early_stopping = EarlyStopping(
    monitor="val_loss", 
    patience=5, 
    restore_best_weights=True
)

print("--- DEBUT DE L'ENTRAINEMENT DE L'ANN GLOBAL H07 ---")
history = model.fit(
    X_train_scaled, 
    y_train_scaled,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# ==========================================================
# 6. Prédictions et inversion de la normalisation
# ==========================================================
predictions_train_scaled = model.predict(X_train_scaled)
predictions_reelles_scaled = model.predict(X_test_scaled)

# Retour aux unités d'origine (kWh)
predictions_train = scaler_y.inverse_transform(predictions_train_scaled)
y_train_reel_orig = scaler_y.inverse_transform(y_train_scaled)

predictions_reelles = scaler_y.inverse_transform(predictions_reelles_scaled)
y_test_reel_orig = scaler_y.inverse_transform(y_test_scaled)

# Seuil physique pour éviter les valeurs négatives
predictions_train = np.clip(predictions_train, a_min=0, a_max=None)
predictions_reelles = np.clip(predictions_reelles, a_min=0, a_max=None)

# ==========================================================
# 7. Évaluation des performances
# ==========================================================
# Évaluation globale (Moyenne uniforme des trois colonnes)
r2_train = r2_score(y_train_reel_orig, predictions_train, multioutput="uniform_average")
mse_global = mean_squared_error(
    y_test_reel_orig, predictions_reelles, multioutput="uniform_average"
)
mae_global = mean_absolute_error(
    y_test_reel_orig, predictions_reelles, multioutput="uniform_average"
)
r2_global = r2_score(y_test_reel_orig, predictions_reelles, multioutput="uniform_average")

# Évaluation détaillée par compteur
r2_chi_tiet = r2_score(y_test_reel_orig, predictions_reelles, multioutput="raw_values")
mae_chi_tiet = mean_absolute_error(
    y_test_reel_orig, predictions_reelles, multioutput="raw_values"
)

print("\n=== RÉSULTATS DU MODÈLE GLOBAL ANN H07 - MULTI-OUTPUT ===")
print(f"ANN Train R² (Moyenne): {r2_train:.4f}")
print(f"ANN Test R²  (Moyenne): {r2_global:.4f} | MAE (Moyenne): {mae_global:.2f}")
print("-" * 65)
print(f" -> [elec_1] R² Test: {r2_chi_tiet[0]:.4f} | MAE: {mae_chi_tiet[0]:.2f}")
print(f" -> [elec_2] R² Test: {r2_chi_tiet[1]:.4f} | MAE: {mae_chi_tiet[1]:.2f}")
print(f" -> [elec_3] R² Test: {r2_chi_tiet[2]:.4f} | MAE: {mae_chi_tiet[2]:.2f}")
print("=========================================================\n")

# ==========================================================
# 8. Visualisation avec Plotly (Format HTML standardisé - 3 Targets)
# ==========================================================
resultats = (
    pd.DataFrame(
        {
            "Date": dates_test,
            "elec_1_Reelle": y_test_reel_orig[:, 0],
            "elec_1_Predite": predictions_reelles[:, 0],
            "elec_2_Reelle": y_test_reel_orig[:, 1],
            "elec_2_Predite": predictions_reelles[:, 1],
            "elec_3_Reelle": y_test_reel_orig[:, 2],
            "elec_3_Predite": predictions_reelles[:, 2]
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

# --- ÉLECTRICITÉ 3 (Tons Verts) ---
fig.add_trace(
    go.Scatter(
        x=dates_formattees,
        y=resultats["elec_3_Reelle"],
        mode="lines",
        name="elec_3 Réelle",
        line=dict(color="darkgreen"),
    )
)
fig.add_trace(
    go.Scatter(
        x=dates_formattees,
        y=resultats["elec_3_Predite"],
        mode="lines",
        name="elec_3 Prédite",
        line=dict(color="lightgreen", dash="dash"),
    )
)

titre_graphe = f"Prévisions Globales (elec_1 & elec_2 & elec_3) via ANN | elec_1_R²: {r2_chi_tiet[0]:.4f} - MAE: {mae_chi_tiet[0]:.2f} | elec_2_R²: {r2_chi_tiet[1]:.4f} - MAE: {mae_chi_tiet[1]:.2f} | elec_3_R²: {r2_chi_tiet[2]:.4f} - MAE: {mae_chi_tiet[2]:.2f}"
fig.update_layout(
    title=titre_graphe,
    xaxis_title="Date",
    yaxis_title="Consommation Électrique (kWh)",
    hovermode="x unified",
    xaxis=dict(rangeslider=dict(visible=True), type="date"),
)

# Sauvegarde automatique du fichier HTML
fig.write_html("prévisions_ann_elec_H07.html")

fig.show()