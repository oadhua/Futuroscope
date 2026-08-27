import os
import pandas as pd

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ
# ==========================================
DATA_DIR = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean"
OUTPUT_PATH = os.path.join(DATA_DIR, "master", "visitor.csv")

TARGET_ATTRACTIONS = ['H03', 'H07']
START_DATE = '2022-01-01 00:00:00'
END_DATE = '2026-07-30 23:00:00'

def find_file(base_dir, file_name):
    """Hàm tự động quét tìm file trong các thư mục con"""
    for root, dirs, files in os.walk(base_dir):
        if file_name in files:
            return os.path.join(root, file_name)
    raise FileNotFoundError(f"Không tìm thấy file '{file_name}' trong {base_dir}")

print(f"=== BẮT ĐẦU TẠO MASTER GRID (TỪ {START_DATE} ĐẾN {END_DATE}) CHO: {TARGET_ATTRACTIONS} ===")

# ==========================================
# 2. ĐỌC DỮ LIỆU GỐC
# ==========================================
print("\n[1/5] Đang đọc dữ liệu từ các thư mục con...")
df_vis = pd.read_csv(find_file(DATA_DIR, "visitor.csv"))
df_etat = pd.read_csv(find_file(DATA_DIR, "etat.csv"))
df_weather = pd.read_csv(find_file(DATA_DIR, "weather_hourly.csv"))
df_horaire = pd.read_csv(find_file(DATA_DIR, "horaire_all_years.csv"))

# ==========================================
# 3. CHUẨN HÓA DATETIME CHO CÁC BẢNG GỐC
# ==========================================
print("\n[2/5] Chuẩn hóa cột Datetime và làm sạch bảng phụ...")

# Đổi date thành datetime và xóa cột date cũ của df_vis để tránh trùng lặp
df_vis['datetime'] = pd.to_datetime(df_vis['date'])
df_vis = df_vis.drop(columns=['date'])

etat_time_col = 'datetime' if 'datetime' in df_etat.columns else 'date'
df_etat['datetime'] = pd.to_datetime(df_etat[etat_time_col])
if etat_time_col != 'datetime' and etat_time_col in df_etat.columns:
    df_etat = df_etat.drop(columns=[etat_time_col])

weather_time_col = 'datetime' if 'datetime' in df_weather.columns else 'date'
df_weather['datetime'] = pd.to_datetime(df_weather[weather_time_col])
if weather_time_col != 'datetime' and weather_time_col in df_weather.columns:
    df_weather = df_weather.drop(columns=[weather_time_col])

df_horaire['date_key'] = pd.to_datetime(df_horaire['date']).dt.strftime('%Y-%m-%d')

# ==========================================
# 4. TẠO KHUNG THỜI GIAN CHUẨN (MASTER GRID) HOURLY CHO TỪNG ATTRACTION
# ==========================================
print("\n[3/5] Tạo khung thời gian chuẩn theo giờ từ 2022 đến 2026...")
full_hours = pd.date_range(start=START_DATE, end=END_DATE, freq='h')

# Tạo dataframe chứa tổ hợp tất cả các giờ cho từng attraction
grid_list = []
for attr in TARGET_ATTRACTIONS:
    temp_df = pd.DataFrame({'datetime': full_hours})
    temp_df['id_attraction'] = attr
    grid_list.append(temp_df)

master_grid = pd.concat(grid_list, ignore_index=True)
master_grid['date_key'] = master_grid['datetime'].dt.strftime('%Y-%m-%d')

# ==========================================
# 5. LÀM SẠCH VÀ CHUẨN BỊ CÁC BẢNG PHỤ ĐỂ JOIN
# ==========================================
# Lọc visitor theo attractions mục tiêu
df_vis = df_vis[df_vis['id_attraction'].isin(TARGET_ATTRACTIONS)].copy()
df_vis_clean = df_vis[['datetime', 'id_attraction', 'visitor_count']].drop_duplicates(subset=['datetime', 'id_attraction'])

# Lọc etat theo attractions mục tiêu
df_etat = df_etat[df_etat['id_attraction'].isin(TARGET_ATTRACTIONS)].copy()
df_etat_clean = df_etat[['datetime', 'id_attraction', 'ouvert', 'interrompu', 'operation']].drop_duplicates(subset=['datetime', 'id_attraction'])

# Weather hourly (loại bỏ trùng lặp theo datetime)
weather_drop_cols = ['date', 'date_key', 'heure']
df_weather_clean = df_weather.drop(columns=[c for c in weather_drop_cols if c in df_weather.columns]).drop_duplicates(subset=['datetime'])

# Horaire (loại bỏ trùng lặp theo ngày)
horaire_drop_cols = ['date', 'mois_nom', 'nom_jour']
df_horaire_clean = df_horaire.drop(columns=[c for c in horaire_drop_cols if c in df_horaire.columns]).drop_duplicates(subset=['date_key'])

# ==========================================
# 6. TIẾN HÀNH LEFT JOIN VÀO MASTER GRID
# ==========================================
print("\n[4/5] Gộp toàn bộ dữ liệu vào khung thời gian chuẩn...")
master = pd.merge(master_grid, df_vis_clean, on=['datetime', 'id_attraction'], how='left')
master = pd.merge(master, df_etat_clean, on=['datetime', 'id_attraction'], how='left')
master = pd.merge(master, df_weather_clean, on='datetime', how='left')
master = pd.merge(master, df_horaire_clean, on='date_key', how='left')

# Trích xuất lại các cột thời gian dạng số chuẩn từ datetime
master['heure'] = master['datetime'].dt.hour
master['jour'] = master['datetime'].dt.day
master['mois'] = master['datetime'].dt.month
master['annee'] = master['datetime'].dt.year

# ==========================================
# 7. LỰA CHỌN CỘT VÀ XUẤT FILE
# ==========================================
print("\n[5/5] Đang chuẩn hóa thứ tự cột và xuất file...")

FINAL_COLUMNS = [
    # 1. Định danh & Lượt khách
    'datetime', 'id_attraction', 'visitor_count',
    
    # 2. Trạng thái vận hành điểm
    'ouvert', 'interrompu', 'operation',
    
    # 3. Thời tiết theo giờ (Hourly Weather)
    'temperature', 'humidite', 'rayonnement_solaire',
    
    # 4. Thời tiết tổng hợp theo ngày (Daily Weather Aggregates)
    'day_degree_cold', 'day_degree_hot', 
    'temp_max', 'temp_min', 'temp_moy', 
    'humidite_max', 'humidite_min', 'humidite_moy',
    
    # 5. Lịch hoạt động công viên
    'jf', 'is_weekend', 'is_open', 'h_ouv', 'h_ferm', 'frequentation', 'type_frequentation',
    
    # 6. Thời gian dạng số
    'heure', 'jour', 'mois', 'annee'
]

# Lọc chỉ lấy các cột thực sự tồn tại trong DataFrame
existing_cols = [col for col in FINAL_COLUMNS if col in master.columns]
master_clean = master[existing_cols].sort_values(by=['datetime', 'id_attraction']).reset_index(drop=True)

# Tạo thư mục đầu ra nếu chưa có
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Xuất ra file CSV
master_clean.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

print("\n==================================================")
print(f"XỬ LÝ HOÀN TẤT!")
print(f"-> Khoảng thời gian: từ {START_DATE} đến {END_DATE}")
print(f"-> Tổng số dòng: {len(master_clean):,}")
print(f"-> Tổng số cột giữ lại: {len(master_clean.columns)} cột")
print(f"-> File đầu ra: {OUTPUT_PATH}")
print("==================================================")