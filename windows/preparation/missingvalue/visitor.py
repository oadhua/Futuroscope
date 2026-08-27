import os
import pandas as pd

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
DATA_DIR = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean"
INPUT_MERGED_PATH = os.path.join(DATA_DIR, "master", "visitor.csv")
OUTPUT_CLEAN_PATH = os.path.join(DATA_DIR, "master_missing", "visitor_missing.csv")

def print_missing_report(df, step_name):
    """Hàm hỗ trợ in báo cáo missing values"""
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    report = pd.DataFrame({
        'Kiểu dữ liệu': df.dtypes,
        'Số lượng thiếu': missing_count,
        'Tỷ lệ thiếu (%)': missing_percent.round(4)
    })
    missing_only = report[report['Số lượng thiếu'] > 0]
    
    print("\n" + "=" * 60)
    print(f"📊 BÁO CÁO MISSING VALUES: {step_name.upper()}")
    print("=" * 60)
    if missing_only.empty:
        print("🎉 Không phát hiện giá trị thiếu (0 Missing Value).")
    else:
        print(missing_only.to_string())
    print("=" * 60)

print("=== [BƯỚC 2] BẮT ĐẦU KIỂM TRA & XỬ LÝ MISSING VALUES ===")

# 1. Đọc file đã gộp ở Bước 1
if not os.path.exists(INPUT_MERGED_PATH):
    raise FileNotFoundError(f"Chưa tìm thấy file {INPUT_MERGED_PATH}. Hãy chạy file merge trước!")

df = pd.read_csv(INPUT_MERGED_PATH)
df['datetime'] = pd.to_datetime(df['datetime'])

# 2. Báo cáo trạng thái dữ liệu TRƯỚC khi xử lý
print_missing_report(df, "Trước khi xử lý")

# 3. Tiến hành xử lý Missing Values
print("\nĐang thực hiện làm sạch dữ liệu...")

# A. Nội suy các biến thời tiết (Interpolation theo chuỗi thời gian của từng điểm)
weather_cols = [
    'temperature', 'humidite', 'rayonnement_solaire',
    'day_degree_cold', 'day_degree_hot', 
    'temp_max', 'temp_min', 'temp_moy', 
    'humidite_max', 'humidite_min', 'humidite_moy'
]
weather_cols_in_df = [c for c in weather_cols if c in df.columns]

df[weather_cols_in_df] = df.groupby('id_attraction')[weather_cols_in_df].transform(
    lambda grp: grp.interpolate(method='linear').bfill().ffill()
)

# B. Điền giá trị '00:00' cho lịch mở/đóng cửa khi công viên đóng
if 'h_ouv' in df.columns:
    df['h_ouv'] = df['h_ouv'].fillna('00:00')
if 'h_ferm' in df.columns:
    df['h_ferm'] = df['h_ferm'].fillna('00:00')

# C. Xử lý logic nghiệp vụ cho các cột trạng thái (ouvert, interrompu, operation)
# Định nghĩa các điều kiện ràng buộc cơ bản nếu có (ví dụ: công viên đóng cửa khi is_open == 0)
cond_parc_ferme = (df["is_open"] == 0) if "is_open" in df.columns else pd.Series(False, index=df.index)

if "interrompu" in df.columns:
    df["interrompu"] = df["interrompu"].fillna(0.0).astype(float).round(1)
    df.loc[cond_parc_ferme, "interrompu"] = 0.0

if "ouvert" in df.columns:
    # Tính profil moyen dựa trên type_frequentation và heure (hoặc heure thay cho hour)
    if "type_frequentation" in df.columns and "heure" in df.columns:
        mask_open_valid = (df.get("is_open", 1) == 1)
        profil_moyen_ouvert = df[mask_open_valid].groupby(["type_frequentation", "heure"])["ouvert"].transform("mean")
        df["ouvert"] = df["ouvert"].fillna(profil_moyen_ouvert)
    
    df["ouvert"] = df["ouvert"].fillna(0.0).ffill().bfill().clip(0.0, 1.0).round(1)
    df.loc[cond_parc_ferme, "ouvert"] = 0.0

if "ouvert" in df.columns and "interrompu" in df.columns:
    df["operation"] = (df["ouvert"] - df["interrompu"]).clip(0.0, 1.0).round(1)
    df.loc[cond_parc_ferme, "operation"] = 0.0

# 4. Báo cáo trạng thái dữ liệu SAU khi xử lý
print_missing_report(df, "Sau khi xử lý")

# 5. Xuất file hoàn chỉnh
os.makedirs(os.path.dirname(OUTPUT_CLEAN_PATH), exist_ok=True)
df.to_csv(OUTPUT_CLEAN_PATH, index=False, encoding='utf-8-sig')

print("\n==================================================")
print("🎉 HOÀN THÀNH XỬ LÝ & LÀM SẠCH DỮ LIỆU!")
print(f"-> File sạch cuối cùng: {OUTPUT_CLEAN_PATH}")
print("==================================================")