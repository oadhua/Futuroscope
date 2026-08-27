import os
import numpy as np
import pandas as pd

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
TARGET_ATTRACTION = "H07"
BASE_DIR = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean"
MODEL_VISITOR_DIR = r"D:\Stage SI\Machine Learning\Futuroscope\windows\model\Visitor"

# Chọn file backcast (XGBoost / Random Forest / Linear Regression)
VISITOR_BACKCAST_PATH = os.path.join(MODEL_VISITOR_DIR, "03. xgboost", "visitor_fully_backcasted_xgb.csv")

print(f"=== KHỞI TẠO VÀ GỘP DỮ LIỆU MASTER HOÀN CHỈNH CHO {TARGET_ATTRACTION} ===")

# ==========================================
# 1. KHỞI TẠO KHUNG THỜI GIAN (2022 - 2026)
# ==========================================
df_horaire = pd.read_csv(f"{BASE_DIR}\\cadence\\horaire_all_years.csv")
df_horaire["date_pure"] = pd.to_datetime(
    df_horaire["date"], format="%Y/%m/%d"
).dt.date

full_time_range = pd.date_range(
    start="2022-01-01 00:00:00", end="2026-07-30 23:00:00", freq="h"
)
df_base = pd.DataFrame(index=full_time_range)
df_base.index.name = "date"
df_base = df_base.reset_index()
df_base["date_pure"] = df_base["date"].dt.date

# Join lịch hoạt động công viên
df_base = df_base.merge(
    df_horaire[[
        "date_pure",
        "is_weekend",
        "is_open",
        "type_frequentation",
        "jf",
        "frequentation",
    ]],
    on="date_pure",
    how="left",
)
df_base.rename(
    columns={"frequentation": "frequentation_park_daily"}, inplace=True
)

# ==========================================
# 2. JOIN THÔNG TIN DIỆN TÍCH & CADENCE (H07)
# ==========================================
df_surface = pd.read_excel(f"{BASE_DIR}\\surface\\surface.xlsx")
surface_val = df_surface.loc[
    df_surface["id_attraction"] == TARGET_ATTRACTION, "surface"
].values[0]
df_base["surface"] = surface_val

# Cadence H07
df_cadence_all = pd.read_excel(f"{BASE_DIR}\\cadence\\cadence.xlsx")
df_cadence_all.columns = (
    df_cadence_all.columns.str.replace(r"\s+", " ", regex=True).str.strip()
)

df_cadence_target = df_cadence_all[
    df_cadence_all["id_attraction"] == TARGET_ATTRACTION
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
df_cadence_target = df_cadence_target[cols_cadence]

df_base["id_attraction"] = TARGET_ATTRACTION
df_base = df_base.merge(df_cadence_target, on="id_attraction", how="left")

# Logic phim (BF/MF vs HF/THF)
df_base["duree"] = np.nan
df_base["cycle_attraction_max"] = np.nan
df_base["duty_cycle_max"] = np.nan

mask_vl = df_base["type_frequentation"].isin(["BF", "MF"])
df_base.loc[mask_vl, "duree"] = df_base.loc[mask_vl, "duree_longue"]
df_base.loc[mask_vl, "cycle_attraction_max"] = df_base.loc[
    mask_vl, "cycle de fonc d'une duree max_vl"
]
df_base.loc[mask_vl, "duty_cycle_max"] = df_base.loc[
    mask_vl, "duty_cycle_max_vl"
]

mask_vc = df_base["type_frequentation"].isin(["HF", "THF"])
df_base.loc[mask_vc, "duree"] = df_base.loc[mask_vc, "duree_courte"]
df_base.loc[mask_vc, "cycle_attraction_max"] = df_base.loc[
    mask_vc, "cycle de fonc d'une duree max_vc"
]
df_base.loc[mask_vc, "duty_cycle_max"] = df_base.loc[
    mask_vc, "duty_cycle_max_vc"
]

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
# 3. JOIN TRỰC TIẾP VISITOR + ETAT + WEATHER TỪ FILE BACKCAST
# ==========================================
if not os.path.exists(VISITOR_BACKCAST_PATH):
    raise FileNotFoundError(f"Chưa tìm thấy file visitor backcast tại: {VISITOR_BACKCAST_PATH}")

df_visitor_backcast = pd.read_csv(VISITOR_BACKCAST_PATH)
df_visitor_backcast["datetime"] = pd.to_datetime(df_visitor_backcast["datetime"])

df_visitor_target = df_visitor_backcast[df_visitor_backcast["id_attraction"] == TARGET_ATTRACTION].copy()
df_visitor_target = df_visitor_target.drop_duplicates(subset=["datetime"])

cols_from_backcast = [
    "datetime",
    "visitor_count",
    "ouvert",
    "interrompu",
    "operation",
    "temperature",
    "humidite",
    "rayonnement_solaire",
    "day_degree_cold",
    "day_degree_hot",
    "temp_max",
    "temp_min",
    "temp_moy",
    "humidite_max",
    "humidite_min",
    "humidite_moy"
]
cols_from_backcast = [c for c in cols_from_backcast if c in df_visitor_target.columns]

df_base = df_base.merge(
    df_visitor_target[cols_from_backcast],
    left_on="date",
    right_on="datetime",
    how="left"
).drop(columns=["datetime"])

# Ép kiểu & xử lý NaN
if "visitor_count" in df_base.columns:
    df_base["visitor_count"] = df_base["visitor_count"].fillna(0)

for etat_col in ["ouvert", "interrompu", "operation"]:
    if etat_col in df_base.columns:
        df_base[etat_col] = df_base[etat_col].fillna(0)

# Chỉ số thời gian
df_base["hour"] = df_base["date"].dt.hour
df_base["day"] = df_base["date"].dt.day
df_base["month"] = df_base["date"].dt.month
df_base["year"] = df_base["date"].dt.year
df_base["week"] = df_base["date"].dt.isocalendar().week
df_base["id"] = range(1, len(df_base) + 1)
df_base.drop(columns=["date_pure"], inplace=True)

# ==========================================
# 4. JOIN NĂNG LƯỢNG & SẮP XẾP CỘT CHUẨN
# ==========================================
def load_energy_data(file_path, col_mapping):
    if not os.path.exists(file_path):
        print(f"⚠️ Cảnh báo: Không tìm thấy file {file_path}")
        return None
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    rename_dict = {
        df.columns[idx]: new_name for idx, new_name in col_mapping.items()
    }
    df.rename(columns=rename_dict, inplace=True)
    return df.drop_duplicates(subset=["date"])

df_elec_target = load_energy_data(
    f"{BASE_DIR}\\elec\\elec2_{TARGET_ATTRACTION}.csv", {2: "elec_1", 3: "elec_2"}
)
df_thermal_target = load_energy_data(
    f"{BASE_DIR}\\thermal\\thermal_{TARGET_ATTRACTION}.csv", {2: "ec_value"}
)

# THỨ TỰ CỘT NỀN CHUẨN (FEATURES)
FEATURE_COLUMNS_ORDER = [
    # 1. Định danh & Thời gian
    "id", "date", "id_attraction", "year", "month", "day", "hour", "week",
    # 2. Đặc trưng lịch hoạt động
    "is_weekend", "is_open", "jf", "type_frequentation", "frequentation_park_daily",
    # 3. Đặc trưng công trình & cadence
    "surface", "capacite salle", "capacite file d'attente", "capacite pre-salle",
    "duree", "cycle_attraction_max", "duty_cycle_max",
    # 4. Trạng thái vận hành
    "ouvert", "interrompu", "operation",
    # 5. Lượng khách (đã backcast)
    "visitor_count",
    # 6. Thời tiết
    "temperature", "humidite", "rayonnement_solaire", "day_degree_cold", "day_degree_hot",
    "temp_max", "temp_min", "temp_moy", "humidite_max", "humidite_min", "humidite_moy"
]

output_dir = f"{BASE_DIR}\\master"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------
# A. Master Điện H07
# ------------------------------------------
if df_elec_target is not None:
    # Lấy các cột điện thực tế có trong file gốc (thông thường elec_1, elec_2)
    elec_cols = [c for c in ["elec_1", "elec_2"] if c in df_elec_target.columns]
    df_master_elec = df_base.merge(
        df_elec_target[["date"] + elec_cols], on="date", how="left"
    )
    elec_final_cols = [c for c in FEATURE_COLUMNS_ORDER if c in df_master_elec.columns] + elec_cols
    df_master_elec = df_master_elec[elec_final_cols]

    path_elec = os.path.join(output_dir, f"master_{TARGET_ATTRACTION}_elec.csv")
    df_master_elec.to_csv(path_elec, index=False, encoding="utf-8-sig")
    print(f"-> ✅ Đã xuất Master Điện HOÀN CHỈNH {TARGET_ATTRACTION}: {path_elec}")

# ------------------------------------------
# B. Master Nhiệt H07
# ------------------------------------------
if df_thermal_target is not None:
    df_master_thermal = df_base.merge(
        df_thermal_target[["date", "ec_value"]], on="date", how="left"
    )
    thermal_final_cols = [c for c in FEATURE_COLUMNS_ORDER if c in df_master_thermal.columns] + ["ec_value"]
    df_master_thermal = df_master_thermal[thermal_final_cols]

    path_thermal = os.path.join(output_dir, f"master_{TARGET_ATTRACTION}_thermal.csv")
    df_master_thermal.to_csv(path_thermal, index=False, encoding="utf-8-sig")
    print(f"-> ✅ Đã xuất Master Nhiệt HOÀN CHỈNH {TARGET_ATTRACTION}: {path_thermal}")

# import os
# import numpy as np
# import pandas as pd

# # ==========================================
# # 1. BASE TEMPORELLE ET CALENDRIER (HORAIRE)
# # ==========================================
# BASE_DIR = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean"

# df_horaire = pd.read_csv(f"{BASE_DIR}\\cadence\\horaire_all_years.csv")
# df_horaire["date_pure"] = pd.to_datetime(
#     df_horaire["date"], format="%m/%d/%Y"
# ).dt.date

# # Base temporelle (2022 - 2026)
# full_time_range = pd.date_range(
#     start="2022-07-21 00:00:00", end="2026-07-26 23:00:00", freq="h"
# )
# df_base = pd.DataFrame(index=full_time_range)
# df_base.index.name = "date"
# df_base = df_base.reset_index()
# df_base["date_pure"] = df_base["date"].dt.date

# # Merge lịch hoạt động + frequentation + gán frequentation = 0 khi is_open == 0
# df_base = df_base.merge(
#     df_horaire[
#         [
#             "date_pure",
#             "is_weekend",
#             "is_open",
#             "type_frequentation",
#             "jf",
#             "frequentation",
#         ]
#     ],
#     on="date_pure",
#     how="left",
# )
# df_base.loc[df_base["is_open"] == 0, "frequentation"] = 0
# df_base["frequentation"] = df_base["frequentation"].fillna(0)

# # ==========================================
# # 2. JOIN THÔNG TIN DIỆN TÍCH & VISIT H03 (GIỮ NGUYÊN BẢN THÔ)
# # ==========================================
# # Read surface.xlsx
# df_surface = pd.read_excel(f"{BASE_DIR}\\surface\\surface.xlsx")
# surface_h07 = df_surface.loc[
#     df_surface["id_attraction"] == "H07", "surface"
# ].values[0]
# df_base["surface"] = surface_h07

# # Read visit_H03.csv (Dữ liệu gốc từ 10/02/2024 -> 31/12/2025)
# df_visit_h07 = pd.read_csv(f"{BASE_DIR}\\visitor\\visit_H07.csv")
# df_visit_h07["date"] = pd.to_datetime(df_visit_h07["date"])
# df_visit_h07 = df_visit_h07.drop_duplicates(subset=["date"])

# # Join trực tiếp biến visitor_count gốc (giai đoạn trước 10/02/2024 sẽ để NaN tự nhiên)
# df_base = df_base.merge(
#     df_visit_h07[["date", "visitor_count"]], on="date", how="left"
# )

# # ==========================================
# # 2. INTÉGRATION ET LOGIQUE DE CADENCE (H07)
# # ==========================================
# df_cadence_all = pd.read_excel(f"{BASE_DIR}\\cadence\\cadence.xlsx")
# df_cadence_all.columns = df_cadence_all.columns.str.replace(
#     r"\s+", " ", regex=True
# ).str.strip()

# # Thay đổi từ H03 sang H07
# df_cadence_h07 = df_cadence_all[
#     df_cadence_all["id_attraction"] == "H07"
# ].copy()

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

# df_base["id_attraction"] = "H07"
# df_base = df_base.merge(df_cadence_h07, on="id_attraction", how="left")

# # Initialisation des colonnes cibles
# df_base["duree"] = np.nan
# df_base["cycle_attraction_max"] = np.nan
# df_base["duty_cycle_max"] = np.nan

# # Condition 1 : Version Longue (BF / MF)
# mask_vl = df_base["type_frequentation"].isin(["BF", "MF"])
# df_base.loc[mask_vl, "duree"] = df_base.loc[mask_vl, "duree_longue"]
# df_base.loc[mask_vl, "cycle_attraction_max"] = df_base.loc[
#     mask_vl, "cycle de fonc d'une duree max_vl"
# ]
# df_base.loc[mask_vl, "duty_cycle_max"] = df_base.loc[mask_vl, "duty_cycle_max_vl"]

# # Condition 2 : Version Courte (HF / THF)
# mask_vc = df_base["type_frequentation"].isin(["HF", "THF"])
# df_base.loc[mask_vc, "duree"] = df_base.loc[mask_vc, "duree_courte"]
# df_base.loc[mask_vc, "cycle_attraction_max"] = df_base.loc[
#     mask_vc, "cycle de fonc d'une duree max_vc"
# ]
# df_base.loc[mask_vc, "duty_cycle_max"] = df_base.loc[mask_vc, "duty_cycle_max_vc"]

# cols_to_drop = [
#     "duree_longue",
#     "duree_courte",
#     "cycle de fonc d'une duree max_vl",
#     "duty_cycle_max_vl",
#     "cycle de fonc d'une duree max_vc",
#     "duty_cycle_max_vc",
# ]
# df_base.drop(columns=cols_to_drop, inplace=True)

# # ==========================================
# # 3. CHARGEMENT VỚI HÀM DÙNG CHUNG & ĐỌC ÉTAT H07
# # ==========================================


# def load_energy_data(file_path, col_mapping):
#   df = pd.read_csv(file_path)
#   df["date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
#   rename_dict = {
#       df.columns[idx]: new_name for idx, new_name in col_mapping.items()
#   }
#   df.rename(columns=rename_dict, inplace=True)
#   return df.drop_duplicates(subset=["date"])


# # 1. Đọc dữ liệu năng lượng H07 (Lưu ý đường dẫn file điện/nhiệt của H07)
# df_elec_h07 = load_energy_data(
#     f"{BASE_DIR}\\elec\\elec2_H07.csv", {2: "elec_1", 3: "elec_2"}
# )
# df_thermal_h07 = load_energy_data(
#     f"{BASE_DIR}\\thermal\\thermal_H07.csv", {2: "ec_value"}
# )

# # 2. Đọc thời tiết
# df_weather = pd.read_csv(f"{BASE_DIR}\\meteo\\weather_hourly.csv")
# df_weather["date"] = pd.to_datetime(df_weather["date"]).dt.tz_localize(None)
# df_weather = df_weather.drop_duplicates(subset=["date"])

# weather_cols = [
#     "date",
#     "temperature",
#     "humidite",
#     "rayonnement_solaire",
#     "temp_max",
#     "temp_min",
#     "temp_moy",
#     "humidite_max",
#     "humidite_min",
#     "humidite_moy",
#     "day_degree_cold",
#     "day_degree_hot",
# ]
# weather_cols = [c for c in weather_cols if c in df_weather.columns]

# # Merge thời tiết vào df_base
# df_base = df_base.merge(df_weather[weather_cols], on="date", how="left")

# # ----------------------------------------------------
# # 3. ĐỌC VÀ LẤY CHÍNH XÁC 3 CỘT TỪ ETAT_H07.CSV
# # ----------------------------------------------------
# df_etat_h07 = pd.read_csv(f"{BASE_DIR}\\etat\\etat_H07.csv")
# df_etat_h07["date"] = pd.to_datetime(df_etat_h07["date"]).dt.tz_localize(None)
# df_etat_h07 = df_etat_h07.drop_duplicates(subset=["date"])

# cols_etat = ["date", "interrompu", "ouvert", "operation"]
# df_base = df_base.merge(df_etat_h07[cols_etat], on="date", how="left")

# # Thêm các cột thời gian
# df_base["hour"] = df_base["date"].dt.hour
# df_base["min"] = df_base["date"].dt.minute
# df_base["day"] = df_base["date"].dt.day
# df_base["month"] = df_base["date"].dt.month
# df_base["year"] = df_base["date"].dt.year
# df_base["week"] = df_base["date"].dt.isocalendar().week
# df_base["id"] = range(1, len(df_base) + 1)
# df_base.drop(columns=["date_pure"], inplace=True)

# # Tạo thư mục lưu kết quả nếu chưa có
# output_dir = f"{BASE_DIR}\\master"
# os.makedirs(output_dir, exist_ok=True)

# # ==========================================
# # 4. TẠO VÀ XUẤT 2 FILE MASTER H07
# # ==========================================

# # --- 4.1 MASTER ELEC H07 ---
# df_master_elec = df_base.merge(
#     df_elec_h07[["date", "elec_1", "elec_2"]], on="date", how="left"
# )
# path_elec = os.path.join(output_dir, "master_H07_elec.csv")
# df_master_elec.to_csv(path_elec, index=False)
# print(f"-> Đã xuất Master Điện H07: {path_elec} (Số dòng: {len(df_master_elec)})")

# # --- 4.2 MASTER THERMAL H07 ---
# df_master_thermal = df_base.merge(
#     df_thermal_h07[["date", "ec_value"]], on="date", how="left"
# )
# path_thermal = os.path.join(output_dir, "master_H07_thermal.csv")
# df_master_thermal.to_csv(path_thermal, index=False)
# print(
#     f"-> Đã xuất Master Nhiệt H07: {path_thermal} (Số dòng:"
#     f" {len(df_master_thermal)})"
# )