import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================================
# 1. Chargement et préparation des données
# ==========================================================
donnees = pd.read_csv(
    r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H03_predict_pure_thermal.csv"
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

X = donnees[variables_entree]
y = donnees[variable_cible]

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

X_train_raw = X.loc[train_indices].reset_index(drop=True)
y_train = y.loc[train_indices].reset_index(drop=True)

X_test_raw = X.loc[test_indices].reset_index(drop=True)
y_test_reel = y.loc[test_indices].reset_index(drop=True)
dates_test = donnees["date"].loc[test_indices].reset_index(drop=True)

# ==========================================================
# 4. Normalisation des données (Crucial pour SVR)
# ==========================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# ==========================================================
# 5. Entraînement du modèle SVR
# ==========================================================
modele = SVR(kernel='rbf', C=100.0, epsilon=0.1)
modele.fit(X_train, y_train)

# ==========================================================
# 6. Prédictions et évaluation
# ==========================================================
predictions_train = modele.predict(X_train)
predictions_reelles = modele.predict(X_test)
predictions_reelles = np.clip(predictions_reelles, a_min=0, a_max=None)

r2_train = r2_score(y_train, predictions_train)
mse = mean_squared_error(y_test_reel, predictions_reelles)
mae = mean_absolute_error(y_test_reel, predictions_reelles)
r2 = r2_score(y_test_reel, predictions_reelles)

print("\n==========================================================")
print(f"SVR Train R²: {r2_train:.4f}")
print(f"SVR Test R²: {r2:.4f} | MAE: {mae:.2f} | MSE: {mse:.2f}")
print("==========================================================")

# ==========================================================
# 7. Génération et sauvegarde du graphique HTML
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
        name="Prédiction SVR",
        line=dict(color="forestgreen", dash="dash"),
    )
)

titre_graphe = (
    f"Prévisions ec_value SVR | R²: {r2:.4f} | MAE: {mae:.2f}"
)
fig.update_layout(
    title=titre_graphe,
    xaxis_title="Date",
    yaxis_title="ec_value",
    hovermode="x unified",
    xaxis=dict(rangeslider=dict(visible=True), type="date"),
)

# Sauvegarde locale automatique en HTML
fig.write_html("prévisions_svr.html")

# Affichage temporaire dans le navigateur
fig.show()