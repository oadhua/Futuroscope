import os
import numpy as np
import pandas as pd

# ==========================================
# 1. BASE TEMPORELLE ET CALENDRIER (HORAIRE)
# ==========================================
BASE_DIR = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean"

# Đọc file lịch mở cửa đã gộp
df_horaire = pd.read_csv(f"{BASE_DIR}\\cadence\\horaire_all_years.csv")
df_horaire["date_pure"] = pd.to_datetime(
    df_horaire["date"], format="%m/%d/%Y"
).dt.date

# Khung thời gian master (2022-07-21 đến 2026-07-26)
full_time_range = pd.date_range(
    start="2022-07-21 00:00:00", end="2026-07-26 23:00:00", freq="h"
)
df_base = pd.DataFrame(index=full_time_range)
df_base.index.name = "date"
df_base = df_base.reset_index()
df_base["date_pure"] = df_base["date"].dt.date

df_base = df_base.merge(
    df_horaire[
        ["date_pure", "is_weekend", "is_open", "type_frequentation", "jf"]
    ],
    on="date_pure",
    how="left",
)

# ==========================================
# 2. INTÉGRATION ET LOGIQUE DE CADENCE (H07)
# ==========================================
df_cadence_all = pd.read_excel(f"{BASE_DIR}\\cadence\\cadence.xlsx")
df_cadence_all.columns = df_cadence_all.columns.str.replace(
    r"\s+", " ", regex=True
).str.strip()

# Lọc thông tin nhịp độ hoạt động riêng cho H07
df_cadence_h07 = df_cadence_all[
    df_cadence_all["id_attraction"] == "H07"
].copy()

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

df_base["id_attraction"] = "H07"
df_base = df_base.merge(df_cadence_h07, on="id_attraction", how="left")

# Initialisation des colonnes cibles
df_base["duree"] = np.nan
df_base["cycle_attraction_max"] = np.nan
df_base["duty_cycle_max"] = np.nan

# Condition 1 : Version Longue (BF / MF)
mask_vl = df_base["type_frequentation"].isin(["BF", "MF"])
df_base.loc[mask_vl, "duree"] = df_base.loc[mask_vl, "duree_longue"]
df_base.loc[mask_vl, "cycle_attraction_max"] = df_base.loc[
    mask_vl, "cycle de fonc d'une duree max_vl"
]
df_base.loc[mask_vl, "duty_cycle_max"] = df_base.loc[mask_vl, "duty_cycle_max_vl"]

# Condition 2 : Version Courte (HF / THF)
mask_vc = df_base["type_frequentation"].isin(["HF", "THF"])
df_base.loc[mask_vc, "duree"] = df_base.loc[mask_vc, "duree_courte"]
df_base.loc[mask_vc, "cycle_attraction_max"] = df_base.loc[
    mask_vc, "cycle de fonc d'une duree max_vc"
]
df_base.loc[mask_vc, "duty_cycle_max"] = df_base.loc[mask_vc, "duty_cycle_max_vc"]

cols_to_drop = [
    "duree_longue",
    "duree_courte",
    "cycle de fonc d'une duree max_vl",
    "duty_cycle_max_vl",
    "cycle de fonc d'une duree max_vc",
    "duty_cycle_max_vc",
]
df_base.drop(columns=cols_to_drop, inplace=True)

# ==========================================
# 3. CHARGEMENT ET SÉCURISATION DES FLUX DYNAMIQUES
# ==========================================


def load_energy_data(file_path, col_mapping):
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    rename_dict = {
        df.columns[idx]: new_name for idx, new_name in col_mapping.items()
    }
    df.rename(columns=rename_dict, inplace=True)
    return df.drop_duplicates(subset=["date"])


# 1. Điện H07 (Đọc cột 2 -> elec_1, cột 3 -> elec_2, cột 4 -> elec_3)
df_elec_h07 = load_energy_data(
    f"{BASE_DIR}\\elec\\elec_H07.csv",
    {2: "elec_1"},
)

# 2. Nhiệt H07 (Đọc cột 2 -> ec_value)
df_thermal_h07 = load_energy_data(
    f"{BASE_DIR}\\thermal\\thermal_H07.csv", {2: "ec_value"}
)

# 3. Source Météo
df_weather = pd.read_csv(f"{BASE_DIR}\\meteo\\weather_hourly.csv")
df_weather["date"] = pd.to_datetime(df_weather["date"]).dt.tz_localize(None)
df_weather = df_weather.drop_duplicates(subset=["date"])

weather_cols = [
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
    "day_degree_cold",
    "day_degree_hot",
]
weather_cols = [c for c in weather_cols if c in df_weather.columns]

# Merge thông tin thời tiết chung vào khung base
df_base = df_base.merge(df_weather[weather_cols], on="date", how="left")

# Thêm các tính năng thời gian
df_base["hour"] = df_base["date"].dt.hour
df_base["min"] = df_base["date"].dt.minute
df_base["day"] = df_base["date"].dt.day
df_base["month"] = df_base["date"].dt.month
df_base["year"] = df_base["date"].dt.year
df_base["week"] = df_base["date"].dt.isocalendar().week
df_base["id"] = range(1, len(df_base) + 1)
df_base.drop(columns=["date_pure"], inplace=True)

# Thư mục lưu kết quả
output_dir = f"{BASE_DIR}\\master"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 4. TẠO VÀ XUẤT 2 FILE MASTER H07
# ==========================================

# --- 4.1 MASTER ELEC H07 ---
df_master_elec = df_base.merge(
    df_elec_h07[["date", "elec_1"]],
    on="date",
    how="left",
)
path_elec = os.path.join(output_dir, "master_H07_elec.csv")
df_master_elec.to_csv(path_elec, index=False)
print(
    f"-> Đã xuất Master Điện H07: {path_elec} (Số dòng:"
    f" {len(df_master_elec)})"
)

# --- 4.2 MASTER THERMAL H07 ---
df_master_thermal = df_base.merge(
    df_thermal_h07[["date", "ec_value"]], on="date", how="left"
)
path_thermal = os.path.join(output_dir, "master_H07_thermal.csv")
df_master_thermal.to_csv(path_thermal, index=False)
print(
    f"-> Đã xuất Master Nhiệt H07: {path_thermal} (Số dòng:"
    f" {len(df_master_thermal)})"
)

# import pandas as pd
# import numpy as np

# # ==========================================
# # 1. BASE TEMPORELLE ET CALENDRIER (HORAIRE)
# # ==========================================
# df_horaire = pd.read_csv(
#     "D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_clean\\cadence\\horaire2025_final_fr.csv"
# )
# df_horaire["date_pure"] = pd.to_datetime(df_horaire["date"], format="%m/%d/%Y").dt.date

# # Base stricte de 8760 heures pour 2025 (Année non-bissextile)
# full_time_range = pd.date_range(
#     start="2025-01-01 00:00:00", end="2025-12-31 23:00:00", freq="h"
# )
# df_master = pd.DataFrame(index=full_time_range)
# df_master.index.name = "date"
# df_master = df_master.reset_index()
# df_master["date_pure"] = df_master["date"].dt.date

# df_master = df_master.merge(
#     df_horaire[["date_pure", "is_weekend", "is_open", "type_frequentation", "jf"]],
#     on="date_pure",
#     how="left",
# )

# # ==========================================
# # 2. INTÉGRATION ET LOGIQUE DE CADENCE (H07)
# # ==========================================
# df_cadence_all = pd.read_excel(
#     "D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_clean\\cadence\\cadence.xlsx"
# )
# # Normalisation des espaces pour éviter les KeyErrors
# df_cadence_all.columns = df_cadence_all.columns.str.replace(
#     r"\s+", " ", regex=True
# ).str.strip()

# df_cadence_h07 = df_cadence_all[df_cadence_all["id_attraction"] == "H07"].copy()

# cols_cadence = [
#     "id_attraction",
#     "capacite salle",
#     "capacite file d'attente",
#     "capacite pre-salle",
#     "duree_longue",
#     "duree_courte",
#     "cycle de fonc d'une duree max_vl",
#     "duty_cycle_max_vl",
#     "cycle de fonc d'une duree max_vc",
#     "duty_cycle_max_vc",
# ]
# df_cadence_h07 = df_cadence_h07[cols_cadence]

# df_master["id_attraction"] = "H07"
# df_master = df_master.merge(df_cadence_h07, on="id_attraction", how="left")

# # Initialisation des colonnes cibles
# df_master["duree"] = np.nan
# df_master["cycle_attraction_max"] = np.nan
# df_master["duty_cycle_max"] = np.nan

# # Condition 1 : Version Longue (BF / MF)
# mask_vl = df_master["type_frequentation"].isin(["BF", "MF"])
# df_master.loc[mask_vl, "duree"] = df_master.loc[mask_vl, "duree_longue"]
# df_master.loc[mask_vl, "cycle_attraction_max"] = df_master.loc[
#     mask_vl, "cycle de fonc d'une duree max_vl"
# ]
# df_master.loc[mask_vl, "duty_cycle_max"] = df_master.loc[mask_vl, "duty_cycle_max_vl"]

# # Condition 2 : Version Courte (HF / THF) -> CORRECTION ICI : TF au lieu de THF
# mask_vc = df_master["type_frequentation"].isin(["HF", "THF"])
# df_master.loc[mask_vc, "duree"] = df_master.loc[mask_vc, "duree_courte"]
# df_master.loc[mask_vc, "cycle_attraction_max"] = df_master.loc[
#     mask_vc, "cycle de fonc d'une duree max_vc"
# ]
# df_master.loc[mask_vc, "duty_cycle_max"] = df_master.loc[mask_vc, "duty_cycle_max_vc"]

# # Nettoyage des colonnes temporaires
# cols_to_drop = [
#     "duree_longue",
#     "duree_courte",
#     "cycle de fonc d'une duree max_vl",
#     "duty_cycle_max_vl",
#     "cycle de fonc d'une duree max_vc",
#     "duty_cycle_max_vc",
# ]
# df_master.drop(columns=cols_to_drop, inplace=True)

# # ==========================================
# # 3. CHARGEMENT ET SÉCURISATION DES FLUX DYNAMIQUES
# # ==========================================
# # Source Visiteurs
# df_visit = pd.read_csv(
#     "D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_brut\\visitor\\visit_H07..csv"
# )
# df_visit["date"] = pd.to_datetime(df_visit["date"])
# df_visit = df_visit.drop_duplicates(subset=["date"])

# # # Source Thermique
# # df_thermal = pd.read_csv(
# #     "D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_clean\\thermal\\thermal_H07.csv"
# # )
# # df_thermal["date"] = pd.to_datetime(df_thermal["Date"])
# # col_ec = "H07_EC_Pavillon de la Vienne [kWh] [H07 - Pavillon de la Vienne]"
# # df_thermal.rename(columns={col_ec: "ec_value"}, inplace=True)
# # df_thermal = df_thermal.drop_duplicates(subset=["date"])

# # Source Electrique
# df_elec = pd.read_csv(
#     "D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_clean\\elec\\elec_H07.csv"
# )
# df_elec["date"] = pd.to_datetime(df_elec["Date"])
# col_ec = ["H07_ELEC_General TGBT (H07+H10+H20) [kWh] [H07 - Pavillon de la Vienne]",
# "H07 ELEC Pavillon de la Vienne + H20 - Hors H10 cosmos [kWh] [H07 - Pavillon de la Vienne]",
# "H07_ELEC_Consommation du simu 4 [kWh] [H07 - Pavillon de la Vienne]"]
# df_elec.rename(columns={col_ec[0]: "elec_1", col_ec[1]: "elec_2", col_ec[2]: "elec_3"}, inplace=True)
# df_elec = df_elec.drop_duplicates(subset=["date"])

# # Source Météo -> CORRECTION : drop_duplicates pour éviter l'impact des données hors 2025
# df_weather = pd.read_csv(
#     "D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_clean\\meteo\\weather_hourly.csv"
# )
# df_weather["date"] = pd.to_datetime(df_weather["date"])
# df_weather = df_weather.drop_duplicates(subset=["date"])

# # Fusions horizontales (Left Joins) sécurisées
# df_master = df_master.merge(df_visit[["date", "visitor_count"]], on="date", how="left")
# # df_master = df_master.merge(df_thermal[["date", "ec_value"]], on="date", how="left")
# df_master = df_master.merge(
#     df_elec[["date", "elec_1", "elec_2", "elec_3"]], on="date", how="left"
# )
# df_master = df_master.merge(
#     df_weather[
#         [
#             "date",
#             "temperature",
#             "humidite",
#             "rayonnement_solaire",
#             "temp_max",
#             "temp_min",
#             "temp_moy",
#             "humidite_max",
#             "humidite_min",
#             "humidite_moy",
#         ]
#     ],
#     on="date",
#     how="left",
# )

# # ==========================================
# # 4. STRUCTURATION TEMPORELLE ET EXPORTATION
# # ==========================================
# df_master["hour"] = df_master["date"].dt.hour
# df_master["min"] = df_master["date"].dt.minute
# df_master["day"] = df_master["date"].dt.day
# df_master["month"] = df_master["date"].dt.month
# df_master["year"] = df_master["date"].dt.year
# df_master["week"] = df_master["date"].dt.isocalendar().week

# # Génération d'un ID incrémental propre (1 à 8760)
# df_master["id"] = range(1, len(df_master) + 1)
# df_master.drop(columns=["date_pure"], inplace=True)

# # Exportation finale du Master File
# output_path = "D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_clean\\master\\master_H07_elec.csv"
# df_master.to_csv(output_path, index=False)

# print(f"Master File généré avec succès ! Lignes : {len(df_master)} (Attendu: 8760)")
# print(f"Nombre de colonnes : {len(df_master.columns)}")
