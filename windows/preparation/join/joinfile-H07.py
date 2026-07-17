import pandas as pd
import numpy as np

# ==========================================
# 1. BASE TEMPORELLE ET CALENDRIER (HORAIRE)
# ==========================================
df_horaire = pd.read_csv(
    "D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_clean\\cadence\\horaire2025_final_fr.csv"
)
df_horaire["date_pure"] = pd.to_datetime(df_horaire["date"], format="%m/%d/%Y").dt.date

# Base stricte de 8760 heures pour 2025 (Année non-bissextile)
full_time_range = pd.date_range(
    start="2025-01-01 00:00:00", end="2025-12-31 23:00:00", freq="h"
)
df_master = pd.DataFrame(index=full_time_range)
df_master.index.name = "date"
df_master = df_master.reset_index()
df_master["date_pure"] = df_master["date"].dt.date

df_master = df_master.merge(
    df_horaire[["date_pure", "is_weekend", "is_open", "type_frequentation", "jf"]],
    on="date_pure",
    how="left",
)

# ==========================================
# 2. INTÉGRATION ET LOGIQUE DE CADENCE (H07)
# ==========================================
df_cadence_all = pd.read_excel(
    "D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_clean\\cadence\\cadence.xlsx"
)
# Normalisation des espaces pour éviter les KeyErrors
df_cadence_all.columns = df_cadence_all.columns.str.replace(
    r"\s+", " ", regex=True
).str.strip()

df_cadence_h07 = df_cadence_all[df_cadence_all["id_attraction"] == "H07"].copy()

cols_cadence = [
    "id_attraction",
    "capacite salle",
    "capacite file d'attente",
    "capacite pre-salle",
    "duree_longue",
    "duree_courte",
    "cycle de fonc d'une duree max_vl",
    "duty_cycle_max_vl",
    "cycle de fonc d'une duree max_vc",
    "duty_cycle_max_vc",
]
df_cadence_h07 = df_cadence_h07[cols_cadence]

df_master["id_attraction"] = "H07"
df_master = df_master.merge(df_cadence_h07, on="id_attraction", how="left")

# Initialisation des colonnes cibles
df_master["duree"] = np.nan
df_master["cycle_attraction_max"] = np.nan
df_master["duty_cycle_max"] = np.nan

# Condition 1 : Version Longue (BF / MF)
mask_vl = df_master["type_frequentation"].isin(["BF", "MF"])
df_master.loc[mask_vl, "duree"] = df_master.loc[mask_vl, "duree_longue"]
df_master.loc[mask_vl, "cycle_attraction_max"] = df_master.loc[
    mask_vl, "cycle de fonc d'une duree max_vl"
]
df_master.loc[mask_vl, "duty_cycle_max"] = df_master.loc[mask_vl, "duty_cycle_max_vl"]

# Condition 2 : Version Courte (HF / THF) -> CORRECTION ICI : TF au lieu de THF
mask_vc = df_master["type_frequentation"].isin(["HF", "THF"])
df_master.loc[mask_vc, "duree"] = df_master.loc[mask_vc, "duree_courte"]
df_master.loc[mask_vc, "cycle_attraction_max"] = df_master.loc[
    mask_vc, "cycle de fonc d'une duree max_vc"
]
df_master.loc[mask_vc, "duty_cycle_max"] = df_master.loc[mask_vc, "duty_cycle_max_vc"]

# Nettoyage des colonnes temporaires
cols_to_drop = [
    "duree_longue",
    "duree_courte",
    "cycle de fonc d'une duree max_vl",
    "duty_cycle_max_vl",
    "cycle de fonc d'une duree max_vc",
    "duty_cycle_max_vc",
]
df_master.drop(columns=cols_to_drop, inplace=True)

# ==========================================
# 3. CHARGEMENT ET SÉCURISATION DES FLUX DYNAMIQUES
# ==========================================
# Source Visiteurs
df_visit = pd.read_csv(
    "D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_brut\\visitor\\visit_H07..csv"
)
df_visit["date"] = pd.to_datetime(df_visit["date"])
df_visit = df_visit.drop_duplicates(subset=["date"])

# # Source Thermique
# df_thermal = pd.read_csv(
#     "D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_clean\\thermal\\thermal_H07.csv"
# )
# df_thermal["date"] = pd.to_datetime(df_thermal["Date"])
# col_ec = "H07_EC_Pavillon de la Vienne [kWh] [H07 - Pavillon de la Vienne]"
# df_thermal.rename(columns={col_ec: "ec_value"}, inplace=True)
# df_thermal = df_thermal.drop_duplicates(subset=["date"])

# Source Electrique
df_elec = pd.read_csv(
    "D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_clean\\elec\\elec_H07.csv"
)
df_elec["date"] = pd.to_datetime(df_elec["Date"])
col_ec = ["H07_ELEC_General TGBT (H07+H10+H20) [kWh] [H07 - Pavillon de la Vienne]",
"H07 ELEC Pavillon de la Vienne + H20 - Hors H10 cosmos [kWh] [H07 - Pavillon de la Vienne]",
"H07_ELEC_Consommation du simu 4 [kWh] [H07 - Pavillon de la Vienne]"]
df_elec.rename(columns={col_ec[0]: "elec_1", col_ec[1]: "elec_2", col_ec[2]: "elec_3"}, inplace=True)
df_elec = df_elec.drop_duplicates(subset=["date"])

# Source Météo -> CORRECTION : drop_duplicates pour éviter l'impact des données hors 2025
df_weather = pd.read_csv(
    "D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_clean\\meteo\\weather_hourly.csv"
)
df_weather["date"] = pd.to_datetime(df_weather["date"])
df_weather = df_weather.drop_duplicates(subset=["date"])

# Fusions horizontales (Left Joins) sécurisées
df_master = df_master.merge(df_visit[["date", "visitor_count"]], on="date", how="left")
# df_master = df_master.merge(df_thermal[["date", "ec_value"]], on="date", how="left")
df_master = df_master.merge(
    df_elec[["date", "elec_1", "elec_2", "elec_3"]], on="date", how="left"
)
df_master = df_master.merge(
    df_weather[
        [
            "date",
            "temperature",
            "humidite",
            "rayonnement_solaire",
            "temp_max",
            "temp_min",
            "temp_moy",
            "humidite_max",
            "humidite_min",
            "humidite_moy",
        ]
    ],
    on="date",
    how="left",
)

# ==========================================
# 4. STRUCTURATION TEMPORELLE ET EXPORTATION
# ==========================================
df_master["hour"] = df_master["date"].dt.hour
df_master["min"] = df_master["date"].dt.minute
df_master["day"] = df_master["date"].dt.day
df_master["month"] = df_master["date"].dt.month
df_master["year"] = df_master["date"].dt.year
df_master["week"] = df_master["date"].dt.isocalendar().week

# Génération d'un ID incrémental propre (1 à 8760)
df_master["id"] = range(1, len(df_master) + 1)
df_master.drop(columns=["date_pure"], inplace=True)

# Exportation finale du Master File
output_path = "D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_clean\\master\\master_H07_elec.csv"
df_master.to_csv(output_path, index=False)

print(f"Master File généré avec succès ! Lignes : {len(df_master)} (Attendu: 8760)")
print(f"Nombre de colonnes : {len(df_master.columns)}")
