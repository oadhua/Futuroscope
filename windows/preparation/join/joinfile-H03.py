import os
import numpy as np
import pandas as pd

# ==========================================
# 1. BASE TEMPORELLE ET CALENDRIER (HORAIRE)
# ==========================================
BASE_DIR = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean"

df_horaire = pd.read_csv(f"{BASE_DIR}\\cadence\\horaire_all_years.csv")
df_horaire["date_pure"] = pd.to_datetime(
    df_horaire["date"], format="%m/%d/%Y"
).dt.date

# Base temporelle (2022 - 2026)
full_time_range = pd.date_range(
    start="2022-07-21 00:00:00", end="2026-07-26 23:00:00", freq="h"
)
df_base = pd.DataFrame(index=full_time_range)
df_base.index.name = "date"
df_base = df_base.reset_index()
df_base["date_pure"] = df_base["date"].dt.date

# Merge lịch hoạt động + frequentation + gán frequentation = 0 khi is_open == 0
df_base = df_base.merge(
    df_horaire[
        ["date_pure", "is_weekend", "is_open", "type_frequentation", "jf", "frequentation"]
    ],
    on="date_pure",
    how="left",
)
df_base.loc[df_base["is_open"] == 0, "frequentation"] = 0
df_base["frequentation"] = df_base["frequentation"].fillna(0)

# ==========================================
# 2. INTÉGRATION ET LOGIQUE DE CADENCE (H03)
# ==========================================
df_cadence_all = pd.read_excel(f"{BASE_DIR}\\cadence\\cadence.xlsx")
df_cadence_all.columns = df_cadence_all.columns.str.replace(
    r"\s+", " ", regex=True
).str.strip()

df_cadence_h03 = df_cadence_all[
    df_cadence_all["id_attraction"] == "H03"
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
df_cadence_h03 = df_cadence_h03[cols_cadence]

df_base["id_attraction"] = "H03"
df_base = df_base.merge(df_cadence_h03, on="id_attraction", how="left")

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
# 3. CHARGEMENT VỚI HÀM DÙNG CHUNG & ĐỌC ÉTAT H03
# ==========================================

def load_energy_data(file_path, col_mapping):
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    rename_dict = {
        df.columns[idx]: new_name for idx, new_name in col_mapping.items()
    }
    df.rename(columns=rename_dict, inplace=True)
    return df.drop_duplicates(subset=["date"])


# 1. Đọc dữ liệu năng lượng
df_elec_h03 = load_energy_data(
    f"{BASE_DIR}\\elec\\elec2_H03.csv", {2: "elec_1", 3: "elec_2"}
)
df_thermal_h03 = load_energy_data(
    f"{BASE_DIR}\\thermal\\thermal_H03.csv", {2: "ec_value"}
)

# 2. Đọc thời tiết
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

# Merge thời tiết vào df_base
df_base = df_base.merge(df_weather[weather_cols], on="date", how="left")

# ----------------------------------------------------
# 3. ĐỌC VÀ LẤY CHÍNH XÁC 3 CỘT TỪ ETAT_H03.CSV
# ----------------------------------------------------
df_etat_h03 = pd.read_csv(f"{BASE_DIR}\\etat\\etat_H03.csv")
df_etat_h03["date"] = pd.to_datetime(df_etat_h03["date"]).dt.tz_localize(None)
df_etat_h03 = df_etat_h03.drop_duplicates(subset=["date"])

# Chỉ lấy 3 cột: interrompu, ouvert, operation cùng với cột date để merge
cols_etat = ["date", "interrompu", "ouvert", "operation"]
df_base = df_base.merge(df_etat_h03[cols_etat], on="date", how="left")

# Điền 0 cho những khoảng thời gian trước/sau giai đoạn ghi nhận file etat_H03
df_base[["interrompu", "ouvert", "operation"]] = df_base[
    ["interrompu", "ouvert", "operation"]
].fillna(0)

# Thêm các cột thời gian
df_base["hour"] = df_base["date"].dt.hour
df_base["min"] = df_base["date"].dt.minute
df_base["day"] = df_base["date"].dt.day
df_base["month"] = df_base["date"].dt.month
df_base["year"] = df_base["date"].dt.year
df_base["week"] = df_base["date"].dt.isocalendar().week
df_base["id"] = range(1, len(df_base) + 1)
df_base.drop(columns=["date_pure"], inplace=True)

# Tạo thư mục lưu kết quả nếu chưa có
output_dir = f"{BASE_DIR}\\master"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 4. TẠO VÀ XUẤT 2 FILE MASTER H03
# ==========================================

# --- 4.1 MASTER ELEC H03 ---
df_master_elec = df_base.merge(
    df_elec_h03[["date", "elec_1", "elec_2"]], on="date", how="left"
)
path_elec = os.path.join(output_dir, "master_H03_elec.csv")
df_master_elec.to_csv(path_elec, index=False)
print(f"-> Đã xuất Master Điện H03: {path_elec} (Số dòng: {len(df_master_elec)})")

# --- 4.2 MASTER THERMAL H03 ---
df_master_thermal = df_base.merge(
    df_thermal_h03[["date", "ec_value"]], on="date", how="left"
)
path_thermal = os.path.join(output_dir, "master_H03_thermal.csv")
df_master_thermal.to_csv(path_thermal, index=False)
print(
    f"-> Đã xuất Master Nhiệt H03: {path_thermal} (Số dòng:"
    f" {len(df_master_thermal)})"
)