import os
import numpy as np
import pandas as pd

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & ĐỌC DỮ LIỆU
# ==============================================================================
TARGET_ATTRACTION = "H07"
BASE_DIR = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean"

master_path = os.path.join(BASE_DIR, "master", f"master_{TARGET_ATTRACTION}_elec.csv")

if not os.path.exists(master_path):
    raise FileNotFoundError(f"Không tìm thấy file Master: {master_path}")

df_master = pd.read_csv(master_path)

print("=" * 70)
print(f"1. BÁO CÁO DỮ LIỆU CÒN THIẾU BAN ĐẦU - {TARGET_ATTRACTION} (Tổng số dòng: {len(df_master)})")
print("=" * 70)
null_before = df_master.isnull().sum()
percent_before = (null_before / len(df_master)) * 100
df_before_report = pd.DataFrame({
    "Cột": null_before.index,
    "Số dòng thiếu": null_before.values,
    "Tỷ lệ (%)": percent_before.round(2).values,
}).sort_values(by="Số dòng thiếu", ascending=False).reset_index(drop=True)

missing_before = df_before_report[df_before_report["Số dòng thiếu"] > 0]
if missing_before.empty:
    print("-> ✅ Dữ liệu hoàn toàn sạch, không có cột nào bị NaN!")
else:
    print(missing_before.to_string(index=False))
print("=" * 70 + "\n")


# ==============================================================================
# 2. XỬ LÝ NỘI SUY DỮ LIỆU NĂNG LƯỢNG (ELEC) & ĐIỀU KIỆN RÀNG BUỘC
# ==============================================================================
print("2. TIẾN HÀNH XỬ LÝ DỮ LIỆU NĂNG LƯỢNG & RÀNG BUỘC LOGIC...")

# 2.1. Nội suy tuyến tính cho dữ liệu điện (nếu có gap nhỏ)
for elec_col in ["elec_1", "elec_2"]:
    if elec_col in df_master.columns:
        df_master[elec_col] = df_master[elec_col].interpolate(method="linear", limit_direction="both").fillna(0.0)
        # Đảm bảo điện tiêu thụ không bị âm do nội suy
        df_master[elec_col] = df_master[elec_col].clip(lower=0.0)

# 2.2. Ép các chỉ số vận hành và lượng khách về 0 khi công viên đóng cửa (is_open == 0)
cond_parc_ferme = df_master["is_open"] == 0

if "visitor_count" in df_master.columns:
    df_master.loc[cond_parc_ferme, "visitor_count"] = 0

for etat_col in ["ouvert", "interrompu", "operation"]:
    if etat_col in df_master.columns:
        df_master.loc[cond_parc_ferme, etat_col] = 0.0


# ==============================================================================
# 3. BÁO CÁO KẾT QUẢ SAU XỬ LÝ MISSING VALUE
# ==============================================================================
print("\n" + "=" * 70)
print(f"3. BÁO CÁO DỮ LIỆU SAU KHI XỬ LÝ MISSING VALUE")
print("=" * 70)
null_after = df_master.isnull().sum()
percent_after = (null_after / len(df_master)) * 100
df_after_report = pd.DataFrame({
    "Cột": null_after.index,
    "Số dòng thiếu": null_after.values,
    "Tỷ lệ (%)": percent_after.round(2).values,
}).sort_values(by="Số dòng thiếu", ascending=False).reset_index(drop=True)

missing_after = df_after_report[df_after_report["Số dòng thiếu"] > 0]
if missing_after.empty:
    print("-> 🎉 TẤT CẢ CÁC CỘT ĐÃ ĐƯỢC LÀM SẠCH HOÀN TOÀN (0% MISSING)!")
else:
    print(missing_after.to_string(index=False))
print("=" * 70 + "\n")


# ==============================================================================
# 4. SẮP XẾP CỘT CHUẨN VÀ XUẤT FILE KẾT QUẢ
# ==============================================================================
FEATURE_COLUMNS_ORDER = [
    # 1. Định danh & Thời gian
    "id", "date", "id_attraction", "year", "month", "day", "hour", "week",
    # 2. Đặc trưng lịch công viên
    "is_weekend", "is_open", "jf", "type_frequentation", "frequentation_park_daily",
    # 3. Thông số kỹ thuật
    "surface", "capacite salle", "capacite file d'attente", "capacite pre-salle", "duree", "cycle_attraction_max", "duty_cycle_max",
    # 4. Trạng thái vận hành & Lượng khách đã Backcast
    "ouvert", "interrompu", "operation", "visitor_count",
    # 5. Thời tiết sạch
    "temperature", "humidite", "rayonnement_solaire", "day_degree_cold", "day_degree_hot",
    "temp_max", "temp_min", "temp_moy", "humidite_max", "humidite_min", "humidite_moy",
    # 6. Biến Target Năng lượng Điện
    "elec_1", "elec_2"
]

# Loại bỏ h_ouv, h_ferm nếu còn sót trong df_master
cols_to_remove = ["h_ouv", "h_ferm", "h_ouv_hour", "h_ferm_hour", "h_ouv_h03", "h_ferm_h03"]
df_master = df_master.drop(columns=[c for c in cols_to_remove if c in df_master.columns])

# Định hình lại thứ tự cột
cols_final = [c for c in FEATURE_COLUMNS_ORDER if c in df_master.columns]
df_master_clean = df_master[cols_final]

output_dir = os.path.join(BASE_DIR, "master_missing")
os.makedirs(output_dir, exist_ok=True)
output_clean_path = os.path.join(output_dir, f"{TARGET_ATTRACTION}_elec_missing.csv")

df_master_clean.to_csv(output_clean_path, index=False, encoding="utf-8-sig")
print(f"-> ✅ Đã xuất file Master Điện sạch hoàn chỉnh: {output_clean_path}")

# import numpy as np
# import pandas as pd

# # ==============================================================================
# # 1. CHARGEMENT DU MASTER FILE BRUT CONSOLIDÉ (H07)
# # ==============================================================================
# master_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master\master_H07_elec.csv"
# df_master = pd.read_csv(master_path)

# # ------------------------------------------------------------------------------
# # BƯỚC 1: KIỂM TRA TRƯỚC KHI DỌN DẸP (BEFORE CLEANING)
# # ------------------------------------------------------------------------------
# print("=" * 65)
# print(f"1. BÁO CÁO DỮ LIỆU CÒN THIẾU BAN ĐẦU (Tổng số dòng: {len(df_master)})")
# print("=" * 65)

# null_before = df_master.isnull().sum()
# percent_before = (null_before / len(df_master)) * 100

# df_before_report = pd.DataFrame({
#     "Cột": null_before.index,
#     "Số dòng thiếu": null_before.values,
#     "Tỷ lệ (%)": percent_before.round(2).values,
# }).sort_values(by="Số dòng thiếu", ascending=False).reset_index(drop=True)

# missing_only_before = df_before_report[df_before_report["Số dòng thiếu"] > 0]

# if missing_only_before.empty:
#   print("-> Tuyệt vời! File gốc không có cột nào bị thiếu dữ liệu.")
# else:
#   print(missing_only_before.to_string(index=False))

# print("=" * 65 + "\n")


# # # ==============================================================================
# # # 2. DÉFINITION DES HORAIRES D'OUVERTURE ET FERMETURE LOGIQUES (H07)
# # # ==============================================================================
# # df_master["h_ouv_h07"] = np.nan
# # df_master["h_ferm_h07"] = np.nan

# # # Règle BF : 10h - 19h
# # df_master.loc[
# #     df_master["type_frequentation"] == "BF", ["h_ouv_h07", "h_ferm_h07"]
# # ] = [10, 19]

# # # Règle MF : 10h - 20h
# # df_master.loc[
# #     df_master["type_frequentation"] == "MF", ["h_ouv_h07", "h_ferm_h07"]
# # ] = [10, 20]

# # # Règle HF et TF/THF : 10h - 21h
# # df_master.loc[
# #     df_master["type_frequentation"].isin(["HF", "THF"]),
# #     ["h_ouv_h07", "h_ferm_h07"],
# # ] = [10, 21]

# # # Fallback nếu type_frequentation bị khuyết
# # df_master["h_ouv_h07"] = df_master["h_ouv_h07"].fillna(10)
# # df_master["h_ferm_h07"] = df_master["h_ferm_h07"].fillna(18)

# # ==============================================================================
# # 3. NETTOYAGE ET IMPUTATION DES VALEURS MANQUANTES (NaN)
# # ==============================================================================

# # --- VARIABLE 2 : DONNÉES MÉTÉOROLOGIQUES ---
# colonnes_meteo = [
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
# for col in colonnes_meteo:
#   if col in df_master.columns:
#     df_master[col] = (
#         df_master[col].interpolate(method="linear").ffill().bfill()
#     )

# # --- VARIABLE 3 : ÉNERGIE ELEC ---
# if "elec_1" in df_master.columns:
#   df_master["elec_1"] = df_master["elec_1"].interpolate(method="linear").fillna(0)

# # --- VARIABLE 4 : ETAT ET FONCTIONNEMENT ---
# for col in ["ouvert", "operation"]:
#   if col in df_master.columns:
#     profil_moyen = df_master.groupby(["type_frequentation", "hour"])[
#         col
#     ].transform("mean")
#     df_master[col] = df_master[col].fillna((profil_moyen >= 0.5).astype(int))

# if "interrompu" in df_master.columns:
#   df_master["interrompu"] = df_master["interrompu"].fillna(0)


# # ==============================================================================
# # 4. RECONSTITUTION ET SÉCURISATION DES VARIABLES TEMPORELLES
# # ==============================================================================
# df_master["date"] = pd.to_datetime(df_master["date"])
# df_master["year"] = df_master["date"].dt.year
# df_master["month"] = df_master["date"].dt.month
# df_master["day"] = df_master["date"].dt.day
# df_master["hour"] = df_master["date"].dt.hour
# df_master["min"] = df_master["date"].dt.minute
# df_master["week"] = df_master["date"].dt.isocalendar().week
# df_master["id"] = range(1, len(df_master) + 1)


# # ==============================================================================
# # 5. STRUCTURATION DES COLONNES ET EXPORTATION FINALE
# # ==============================================================================
# ordre_colonnes = [
#     "date",
#     "visitor_count",
#     "frequentation",
#     "elec_1",
#     "interrompu",
#     "ouvert",
#     "operation",
#     "cycle_attraction_max",
#     "duty_cycle_max",
#     "capacite salle",
#     "capacite file d'attente",
#     "capacite pre-salle",
#     "is_weekend",
#     "jf",
#     "is_open",
#     "type_frequentation",
#     "temperature",
#     "temp_max",
#     "temp_min",
#     "temp_moy",
#     "humidite",
#     "humidite_max",
#     "humidite_min",
#     "humidite_moy",
#     "rayonnement_solaire",
#     "day_degree_cold",
#     "day_degree_hot",
#     "hour",
#     "min",
#     "day",
#     "month",
#     "year",
#     "week",
# ]

# colonnes_finales = [c for c in ordre_colonnes if c in df_master.columns]
# df_master_clean = df_master[colonnes_finales]

# # ------------------------------------------------------------------------------
# # BƯỚC 2: KIỂM TRA LẠI SAU KHI DỌN DẸP (AFTER CLEANING)
# # ------------------------------------------------------------------------------
# print("=" * 65)
# print("2. BÁO CÁO DỮ LIỆU CÒN THIẾU SAU KHI XỬ LÝ (AFTER CLEANING)")
# print("=" * 65)

# null_after = df_master_clean.isnull().sum()
# percent_after = (null_after / len(df_master_clean)) * 100

# df_after_report = pd.DataFrame({
#     "Cột": null_after.index,
#     "Số dòng thiếu": null_after.values,
#     "Tỷ lệ (%)": percent_after.round(2).values,
# }).sort_values(by="Số dòng thiếu", ascending=False).reset_index(drop=True)

# print(df_after_report.to_string(index=False))
# print("=" * 65 + "\n")

# # Xuất file kết quả H07
# output_clean_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_missing\master_H07_elec_missing.csv"
# df_master_clean.to_csv(output_clean_path, index=False)

# print(f"-> Xuất file thành công: {output_clean_path}")


# # import pandas as pd
# # import numpy as np

# # # ==============================================================================
# # # 1. CHARGEMENT DES FILES BRUTS
# # # ==============================================================================
# # # Đường dẫn file của bạn
# # master_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master\master_H07_elec.csv"
# # horaire_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\cadence\horaire_all_years.csv"  # Đổi lại đường dẫn tuyệt đối nếu cần

# # df_master = pd.read_csv(master_path)
# # df_horaire = pd.read_csv(horaire_path)

# # # Đảm bảo cột date ở cả 2 dataframe đều ở định dạng datetime để merge chính xác
# # df_master['date'] = pd.to_datetime(df_master['date'])
# # df_horaire['date'] = pd.to_datetime(df_horaire['date'], format='%m/%d/%Y') # format DD/MM/YYYY của file horaire

# # # ==============================================================================
# # # 2. XỬ LÝ GIỜ MỞ/ĐÓNG CỬA TỪ FILE LỊCH CÔNG VIÊN
# # # ==============================================================================
# # # Chuyển đổi h_ouv và h_ferm từ dạng "HH:MM:SS" sang dạng giờ nguyên (Integer)
# # df_horaire['h_ouv_parsed'] = pd.to_datetime(df_horaire['h_ouv'], format='%H:%M:%S', errors='coerce').dt.hour
# # df_horaire['h_ferm_parsed'] = pd.to_datetime(df_horaire['h_ferm'], format='%H:%M:%S', errors='coerce').dt.hour

# # # Chọn các cột cần thiết để gộp vào master file
# # df_horaire_sub = df_horaire[['date', 'h_ouv_parsed', 'h_ferm_parsed']].copy()

# # # Gộp thông tin giờ mở/đóng vào master file theo cột 'date' (chỉ lấy phần ngày để khớp)
# # # Tạo cột phụ chỉ chứa ngày (YYYY-MM-DD) để merge
# # df_master['date_only'] = df_master['date'].dt.normalize()
# # df_horaire_sub['date_only'] = df_horaire_sub['date'].dt.normalize()

# # # Merge
# # df_master = df_master.merge(
# #     df_horaire_sub[['date_only', 'h_ouv_parsed', 'h_ferm_parsed']], 
# #     on='date_only', 
# #     how='left'
# # )

# # # Gán giá trị giờ mở/đóng thực tế của công viên cho H07
# # df_master['h_ouv_h07'] = df_master['h_ouv_parsed']
# # df_master['h_ferm_h07'] = df_master['h_ferm_parsed']

# # # Valeurs de secours (Fallback) nếu ngày đó công viên đóng cửa hoặc bị thiếu dữ liệu
# # # Sử dụng rule BF mặc định (10h - 19h) hoặc giá trị an toàn
# # df_master['h_ouv_h07'] = df_master['h_ouv_h07'].fillna(10)
# # df_master['h_ferm_h07'] = df_master['h_ferm_h07'].fillna(18)

# # # Xóa các cột phụ sau khi xử lý xong
# # df_master = df_master.drop(columns=['date_only', 'h_ouv_parsed', 'h_ferm_parsed'])

# # # ==============================================================================
# # # 3. NETTOYAGE ET IMPUTATION DES VALEURS MANQUANTES (NaN)
# # # ==============================================================================

# # # Đảm bảo cột 'hour' đã tồn tại trước khi dùng làm điều kiện lọc
# # df_master['hour'] = df_master['date'].dt.hour

# # # --- VARIABLE 1 : LE FLUX DE VISITEURS (visitor_count) ---
# # # # Cas 1 : Journées de fermeture complète du parc (is_open == 0)
# # # cond_parc_ferme = (df_master['is_open'] == 0)

# # # # Cas 2 : En dehors des horaires d'ouverture réels du parc
# # # cond_hors_amplitude = (df_master['is_open'] == 1) & (
# # #     (df_master['hour'] < df_master['h_ouv_h07']) | (df_master['hour'] >= df_master['h_ferm_h07'])
# # # )

# # # # Forçage à 0 visiteur pour les périodes d'inactivité de l'attraction
# # # df_master.loc[cond_parc_ferme | cond_hors_amplitude, 'visitor_count'] = df_master.loc[cond_parc_ferme | cond_hors_amplitude, 'visitor_count'].fillna(0)

# # # # Cas 3 : Interpolation linéaire pour les NaN restants (pendant l'ouverture)
# # # df_master['visitor_count'] = df_master['visitor_count'].interpolate(method='linear')
# # # df_master['visitor_count'] = df_master['visitor_count'].clip(lower=0).round().astype(int)


# # # --- VARIABLE 2 : DONNÉES MÉTÉOROLOGIQUES ---
# # colonnes_meteo = [
# #     'temperature', 'humidite', 'rayonnement_solaire', 
# #     'temp_max', 'temp_min', 'temp_moy', 
# #     'humidite_max', 'humidite_min', 'humidite_moy',
# #     'day_degree_cold', 'day_degree_hot',
# # ]
# # for col in colonnes_meteo:
# #     if col in df_master.columns:
# #         df_master[col] = df_master[col].interpolate(method='linear').ffill().bfill()


# # # --- VARIABLE 3 : ÉNERGIE CALORIFIQUE (ec_value) ---
# # for col in ['elec_1', 'elec_2', 'elec_3']:
# #     if col in df_master.columns:
# #         # Nếu cột đang ở dạng chuỗi, thay thế dấu phẩy ',' thành dấu chấm '.' (nếu có)
# #         if df_master[col].dtype == 'object':
# #             df_master[col] = df_master[col].astype(str).str.replace(',', '.', regex=False)
        
# #         # Chuyển đổi sang kiểu số thực (các giá trị lỗi chữ/khoảng trắng sẽ thành NaN)
# #         df_master[col] = pd.to_numeric(df_master[col], errors='coerce')
        
# #         # Gọi nội suy tuyến tính một cách an toàn trên cột dạng số thực
# #         df_master[col] = df_master[col].interpolate(method='linear').fillna(0)

# # # ==============================================================================
# # # 4. RECONSTITUTION ET SÉCURISATION DES VARIABLES TEMPORELLES
# # # ==============================================================================
# # df_master['year'] = df_master['date'].dt.year
# # df_master['month'] = df_master['date'].dt.month
# # df_master['day'] = df_master['date'].dt.day
# # df_master['min'] = df_master['date'].dt.minute
# # df_master['week'] = df_master['date'].dt.isocalendar().week

# # # Id unique séquentiel de 1 à 8760
# # df_master['id'] = range(1, len(df_master) + 1)

# # # ==============================================================================
# # # 5. STRUCTURATION DES COLONNES ET EXPORTATION FINALE
# # # ==============================================================================
# # ordre_colonnes = [
# #     'date', 'visitor_count', 'elec_1', 'elec_2', 'elec_3',
# #     'cycle_attraction_max', 'duty_cycle_max', 'capacite salle', 'capacite file d\'attente', 'capacite pre-salle',
# #     'is_weekend','jf', 'is_open', 'type_frequentation',
# #     'temperature', 'temp_max', 'temp_min', 'temp_moy',
# #     'humidite','humidite_max', 'humidite_min', 'humidite_moy',
# #     'rayonnement_solaire',
# #     'day_degree_cold', 'day_degree_hot',
# #     'hour', 'min', 'day', 'month', 'year', 'week'
# # ]

# # colonnes_finales = [c for c in ordre_colonnes if c in df_master.columns]
# # df_master_clean = df_master[colonnes_finales]

# # # Sauvegarde du Master File propre pour l'entraînement ML
# # output_clean_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_missing\master_H07_elec_missing.csv"
# # df_master_clean.to_csv(output_clean_path, index=False)

# # print(f"Master File nettoyé avec succès ! Lignes : {len(df_master_clean)}")
# # print(f"Nombre de colonnes finales : {len(df_master_clean.columns)}")