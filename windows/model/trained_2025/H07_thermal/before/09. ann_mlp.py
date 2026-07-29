import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
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
    "temp_decalage_1h"
]
variables_entree = [col for col in donnees.columns if col not in variables_exclues]
variable_cible = "ec_value"

# Scalers requis pour ANN
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

scaler_X.fit(donnees[variables_entree])
scaler_y.fit(donnees[[variable_cible]])

# ==========================================================
# 3. Blocked Split temporel (80% Train / 20% Test par mois)
# ==========================================================
train_indices = []
test_indices = []

for name, group in donnees.groupby("annee_mois"):
    if len(group) < 10:
        continue
    split_idx = int(len(group) * 0.8)
    train_indices.extend(group.index[:split_idx])
    test_indices.extend(group.index[split_idx:])

X_train_raw = donnees.loc[train_indices, variables_entree].reset_index(drop=True)
y_train_raw = donnees.loc[train_indices, [variable_cible]].reset_index(drop=True)

X_test_raw = donnees.loc[test_indices, variables_entree].reset_index(drop=True)
y_test_raw = donnees.loc[test_indices, [variable_cible]].reset_index(drop=True)
dates_test = donnees["date"].loc[test_indices].reset_index(drop=True)

# Normalisation
X_train = scaler_X.transform(X_train_raw)
y_train = scaler_y.transform(y_train_raw)

X_test = scaler_X.transform(X_test_raw)
y_test = scaler_y.transform(y_test_raw)

# Extraction d'un jeu de validation à partir du train
val_split_idx = int(len(X_train) * 0.85)
X_train_final = X_train[:val_split_idx]
y_train_final = y_train[:val_split_idx]

X_val = X_train[val_split_idx:]
y_val = y_train[val_split_idx:]

# ==========================================================
# 4. Entraînement du modèle ANN (Multi-Layer Perceptron)
# ==========================================================
model = Sequential()

# Première couche dense
model.add(Dense(units=64, activation="relu", input_shape=(X_train_final.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Deuxième couche dense
model.add(Dense(units=32, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Troisième couche dense
model.add(Dense(units=16, activation="relu"))

# Couche de sortie
model.add(Dense(units=1, activation="linear"))  

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss="mse")

early_stopping = EarlyStopping(
    monitor="val_loss", 
    patience=8, 
    restore_best_weights=True
)

model.fit(
    X_train_final, 
    y_train_final,
    epochs=60,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# ==========================================================
# 5. Prédictions et évaluation
# ==========================================================
predictions_train_scaled = model.predict(X_train_final)
predictions_scaled = model.predict(X_test)

predictions_train = scaler_y.inverse_transform(predictions_train_scaled).flatten()
predictions_reelles = scaler_y.inverse_transform(predictions_scaled).flatten()
y_train_original = scaler_y.inverse_transform(y_train_final).flatten()
y_test_reel = scaler_y.inverse_transform(y_test).flatten()

# Limitation des valeurs négatives
predictions_reelles = np.clip(predictions_reelles, a_min=0, a_max=None)

r2_train = r2_score(y_train_original, predictions_train)
mse = mean_squared_error(y_test_reel, predictions_reelles)
mae = mean_absolute_error(y_test_reel, predictions_reelles)
r2 = r2_score(y_test_reel, predictions_reelles)

print("\n==========================================================")
print(f"ANN Train R²: {r2_train:.4f}")
print(f"ANN Test R²: {r2:.4f} | MAE: {mae:.2f} | MSE: {mse:.2f}")
print("==========================================================")

# ==========================================================
# 6. Génération et sauvegarde du graphique HTML
# ==========================================================
resultats = (
    pd.DataFrame(
        {
            "Date": dates_test,
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
        name="Prédiction ANN",
        line=dict(color="darkorange", dash="dash"),
    )
)

titre_graphe = (
    f"Prévisions ec_value ANN | R²: {r2:.4f} | MAE: {mae:.2f}"
)
fig.update_layout(
    title=titre_graphe,
    xaxis_title="Date",
    yaxis_title="ec_value",
    hovermode="x unified",
    xaxis=dict(rangeslider=dict(visible=True), type="date"),
)

# Sauvegarde locale automatique en HTML
fig.write_html("prévisions_ann.html")

# Affichage temporaire dans le navigateur
fig.show()