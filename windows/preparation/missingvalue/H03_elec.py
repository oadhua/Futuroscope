import os
import numpy as np
import pandas as pd

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & ĐỌC DỮ LIỆU
# ==============================================================================
TARGET_ATTRACTION = "H03"
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

# import os
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor

# # ==============================================================================
# # 1. LOAD DATA & TÍCH HỢP LỊCH HOẠT ĐỘNG (HORAIRE)
# # ==============================================================================
# master_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master\master_H03_elec.csv"
# horaire_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\cadence\horaire_all_years.csv"  # Thay đường dẫn tới file horaire của bạn nếu cần

# df_master = pd.read_csv(master_path)
# df_horaire = pd.read_csv(horaire_path)

# # Chuẩn hóa cột ngày để Merge chính xác
# df_master["date_dt"] = pd.to_datetime(df_master["date"])
# df_master["date_only"] = df_master["date_dt"].dt.strftime("%Y-%m-%d")

# df_horaire["date_dt"] = pd.to_datetime(df_horaire["date"], format="%d/%m/%Y", errors="coerce")
# df_horaire["date_only"] = df_horaire["date_dt"].dt.strftime("%Y-%m-%d")

# # Chọn các cột cần thiết từ file lịch để tránh trùng lặp cột khi merge
# cols_to_merge = ["date_only", "h_ouv", "h_ferm"]
# if "is_open" in df_horaire.columns and "is_open" not in df_master.columns:
#     cols_to_merge.append("is_open")
# if "type_frequentation" in df_horaire.columns and "type_frequentation" not in df_master.columns:
#     cols_to_merge.append("type_frequentation")

# # Bỏ các cột h_ouv, h_ferm cũ nếu đã có trong df_master để cập nhật dữ liệu mới từ horaire
# df_master = df_master.drop(columns=[c for c in ["h_ouv", "h_ferm", "h_ouv_h03", "h_ferm_h03"] if c in df_master.columns])

# # Merge lịch hoạt động vào df_master
# df_master = df_master.merge(df_horaire[cols_to_merge], on="date_only", how="left")

# # Trích xuất giờ (giờ bắt đầu và giờ kết thúc)
# df_master["h_ouv_dt"] = pd.to_datetime(df_master["h_ouv"], format="%H:%M:%S", errors="coerce")
# df_master["h_ferm_dt"] = pd.to_datetime(df_master["h_ferm"], format="%H:%M:%S", errors="coerce")

# # Giờ mở cửa: lấy giờ chính xác (vd: 09:15 -> 9h)
# df_master["h_ouv_hour"] = df_master["h_ouv_dt"].dt.hour.fillna(10)

# # Giờ đóng cửa: lấy giờ kết thúc (vd: 22:30 -> 23h để không bỏ sót nửa tiếng cuối)
# df_master["h_ferm_hour"] = df_master["h_ferm_dt"].dt.hour + np.where(df_master["h_ferm_dt"].dt.minute > 0, 1, 0)
# df_master["h_ferm_hour"] = df_master["h_ferm_hour"].fillna(20)

# # Xóa các cột tạm
# df_master = df_master.drop(columns=["date_dt", "date_only", "h_ouv_dt", "h_ferm_dt"])

# # Tạo các biến thời gian chi tiết
# df_master["date_dt"] = pd.to_datetime(df_master["date"])
# df_master["year"] = df_master["date_dt"].dt.year
# df_master["month"] = df_master["date_dt"].dt.month
# df_master["day"] = df_master["date_dt"].dt.day
# df_master["hour"] = df_master["date_dt"].dt.hour
# df_master["min"] = df_master["date_dt"].dt.minute
# df_master["week"] = df_master["date_dt"].dt.isocalendar().week
# df_master["id"] = range(1, len(df_master) + 1)
# df_master = df_master.drop(columns=["date_dt"])

# # Báo cáo missing data ban đầu
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
#     print("-> Tuyệt vời! File không có cột nào bị thiếu dữ liệu.")
# else:
#     print(missing_only_before.to_string(index=False))
# print("=" * 65 + "\n")


# # ==============================================================================
# # 2. XỬ LÝ NỘI SUY DỮ LIỆU MÉTÉO - ELEC - OPERATIONAL
# # ==============================================================================
# # --- VARIABLE 1 : MÉTÉO ---
# colonnes_meteo = [
#     "temperature", "humidite", "rayonnement_solaire", "temp_max", "temp_min",
#     "temp_moy", "humidite_max", "humidite_min", "humidite_moy", 
#     "day_degree_cold", "day_degree_hot"
# ]
# for col in colonnes_meteo:
#     if col in df_master.columns:
#         df_master[col] = df_master[col].interpolate(method="linear").ffill().bfill()

# # --- VARIABLE 2 : ELEC ---
# for col in ["elec_1", "elec_2"]:
#     if col in df_master.columns:
#         df_master[col] = df_master[col].interpolate(method="linear").fillna(0)

# # Xác định khung giờ ngoài biên hoạt động
# cond_parc_ferme = df_master["is_open"] == 0
# cond_hors_amplitude = (df_master["is_open"] == 1) & (
#     (df_master["hour"] < df_master["h_ouv_hour"]) | (df_master["hour"] >= df_master["h_ferm_hour"])
# )

# # --- VARIABLE 3 : TRẠNG THÁI HOẠT ĐỘNG (ouvert, interrompu, operation) ---
# if "interrompu" in df_master.columns:
#     df_master["interrompu"] = df_master["interrompu"].fillna(0.0).astype(float).round(1)
#     df_master.loc[cond_parc_ferme | cond_hors_amplitude, "interrompu"] = 0.0

# if "ouvert" in df_master.columns:
#     # Chỉ tính profil moyen trên các khung giờ mở cửa hợp lệ
#     mask_open_valid = (df_master["is_open"] == 1) & (~cond_hors_amplitude)
#     profil_moyen_ouvert = df_master[mask_open_valid].groupby(["type_frequentation", "hour"])["ouvert"].transform("mean")
    
#     df_master["ouvert"] = df_master["ouvert"].fillna(profil_moyen_ouvert)
#     df_master["ouvert"] = df_master["ouvert"].ffill().bfill().fillna(0.0).clip(0.0, 1.0).round(1)
    
#     # Ép về 0.0 nếu ngoài giờ mở cửa hoặc công viên đóng
#     df_master.loc[cond_parc_ferme | cond_hors_amplitude, "ouvert"] = 0.0

# if "ouvert" in df_master.columns and "interrompu" in df_master.columns:
#     df_master["operation"] = (df_master["ouvert"] - df_master["interrompu"]).clip(0.0, 1.0).round(1)
#     df_master.loc[cond_parc_ferme | cond_hors_amplitude, "operation"] = 0.0


# # ==============================================================================
# # 3. CHỈ DỰ BÁO CÁC VỊ TRÍ BỊ THÍを行 (NaN) TRONG VISITOR_COUNT
# # ==============================================================================
# if "visitor_count" in df_master.columns:
#     print("-> Tiến hành lọc và chỉ dự báo các vị trí bị thiếu trong visitor_count...")

#     # 1. Đánh dấu CHÍNH XÁC những dòng ban đầu bị NaN
#     mask_missing = df_master["visitor_count"].isnull()

#     # 2. Xử lý các dòng BỊ THÍEU theo điều kiện logic (Nếu công viên đóng hoặc H03 nghỉ -> = 0)
#     cond_parc_ferme = df_master["is_open"] == 0
#     cond_hors_amplitude = (df_master["is_open"] == 1) & (
#         (df_master["hour"] < df_master["h_ouv_hour"]) | (df_master["hour"] >= df_master["h_ferm_hour"])
#     )
#     cond_h03_ferme = (df_master["operation"] == 0) if "operation" in df_master.columns else False

#     # Chỉ gán 0 cho những ô THỰC SỰ BỊ THÍEU
#     df_master.loc[mask_missing & (cond_parc_ferme | cond_hors_amplitude | cond_h03_ferme), "visitor_count"] = 0.0

#     # 3. Chuẩn bị biến cho Random Forest
#     features_ml = ["hour", "month", "frequentation_park_daily", "is_weekend", "jf"]
#     for h03_feat in ["operation", "ouvert", "interrompu"]:
#         if h03_feat in df_master.columns:
#             features_ml.append(h03_feat)

#     for col_opt in ["temperature", "rayonnement_solaire"]:
#         if col_opt in df_master.columns and df_master[col_opt].isnull().sum() == 0:
#             features_ml.append(col_opt)

#     df_encoded = pd.get_dummies(
#         df_master[features_ml + ["type_frequentation"]],
#         columns=["type_frequentation"],
#         drop_first=True
#     )

#     # 4. Xác định tập Train và tập Predict
#     # Train: Những dòng ĐÃ CÓ sẵn dữ liệu visitor_count (không bị NaN ban đầu)
#     mask_train = (~mask_missing) & (df_master["is_open"] == 1) & (df_master["operation"] > 0)

#     # Predict: CHỈ những dòng VẪN CÒN BỊ NaN (đã loại trừ các dòng đóng cửa = 0 ở bước 2)
#     mask_predict = mask_missing & (df_master["visitor_count"].isnull()) & (df_master["is_open"] == 1) & (df_master["operation"] > 0)

#     print(f"   - Số dòng có sẵn dùng để huấn luyện (Train): {mask_train.sum()}")
#     print(f"   - Số dòng bị thiếu thực sự cần dự báo (Predict): {mask_predict.sum()}")

#     # 5. Huấn luyện & Dự báo
#     if mask_predict.sum() > 0:
#         X_train = df_encoded.loc[mask_train]
#         y_train = df_master.loc[mask_train, "visitor_count"]
#         X_pred = df_encoded.loc[mask_predict]

#         rf_model = RandomForestRegressor(n_estimators=150, max_depth=14, random_state=42, n_jobs=-1)
#         rf_model.fit(X_train, y_train)

#         # Điền kết quả dự báo CHỈ vào các ô bị thiếu
#         df_master.loc[mask_predict, "visitor_count"] = rf_model.predict(X_pred)
#         print("   -> Đã điền xong dự báo cho toàn bộ các ô bị thiếu!")

#     # Làm tròn số nguyên
#     df_master["visitor_count"] = df_master["visitor_count"].fillna(0).clip(lower=0).round().astype(int)


# # ==============================================================================
# # 4. EXPORT FILE KẾT QUẢ
# # ==============================================================================
# ordre_colonnes = [
#     "date", "visitor_count", "frequentation_park_daily", "surface",
#     "elec_1", "elec_2", "interrompu", "ouvert", "operation",
#     "cycle_attraction_max", "duty_cycle_max", "capacite salle",
#     "capacite file d'attente", "capacite pre-salle", "is_weekend", "jf",
#     "is_open", "type_frequentation", "h_ouv", "h_ferm", "temperature", "temp_max", "temp_min",
#     "temp_moy", "humidite", "humidite_max", "humidite_min", "humidite_moy",
#     "rayonnement_solaire", "day_degree_cold", "day_degree_hot",
#     "hour", "min", "day", "month", "year", "week"
# ]

# colonnes_finales = [c for c in ordre_colonnes if c in df_master.columns]
# df_master_clean = df_master[colonnes_finales]

# output_dir = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_missing"
# os.makedirs(output_dir, exist_ok=True)
# output_clean_path = os.path.join(output_dir, "master_H03_elec_missing.csv")

# df_master_clean.to_csv(output_clean_path, index=False)
# print(f"-> Xuất file thành công: {output_clean_path}")