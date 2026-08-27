import os
import pandas as pd
import numpy as np

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN & SỨC CHỨA HÀNG CHỜ
# ==========================================
DATA_DIR = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean"
INPUT_CLEAN_PATH = os.path.join(DATA_DIR, "master_missing", "visitor_missing.csv")
OUTPUT_FINAL_PATH = os.path.join(DATA_DIR, "master_outliers", "visitor_outliers.csv")

MAX_CAPACITY = {
    'H03': 750,
    'H07': 800
}

MIN_CONSECUTIVE_FLAT_HOURS = 3

print("=== [BƯỚC 3] XỬ LÝ OUTLIERS + ĐƯỜNG NGANG (LUỒNG KIỂM TRA TỐI ƯU) ===")

# 1. Đọc dữ liệu
if not os.path.exists(INPUT_CLEAN_PATH):
    raise FileNotFoundError(f"Chưa tìm thấy file {INPUT_CLEAN_PATH}")

os.makedirs(os.path.dirname(OUTPUT_FINAL_PATH), exist_ok=True)
df = pd.read_csv(INPUT_CLEAN_PATH)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(by=['id_attraction', 'datetime']).reset_index(drop=True)

print(f"\n[1/5] Đã tải dữ liệu: {len(df):,} dòng.")

# ==========================================
# 2. XỬ LÝ OUTLIERS VẬT LÝ
# ==========================================
print("\n[2/5] Kiểm tra các lỗi vật lý (Physical Outliers)...")
df['visitor_count'] = df['visitor_count'].clip(lower=0)

closed_mask = (df['ouvert'] == 0) | (df['is_open'] == 0)
df.loc[closed_mask, 'visitor_count'] = 0

# ==========================================
# 3. KẾT HỢP: CAP SỨC CHỨA THỰC TẾ & IQR METHOD
# ==========================================
print("\n[3/5] Ép ngưỡng theo Sức chứa Thực tế & Thống kê IQR...")

def process_outliers_combined(group, column='visitor_count', factor=1.5):
    attraction_id = group['id_attraction'].iloc[0]
    cap_capacity = MAX_CAPACITY.get(attraction_id, None)
    
    # Cap sức chứa thực tế
    if cap_capacity is not None:
        group[column] = group[column].clip(upper=cap_capacity)

    # IQR Capping
    open_data = group[group[column] > 0][column]
    if len(open_data) > 0:
        Q1 = open_data.quantile(0.25)
        Q3 = open_data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = max(0, Q1 - factor * IQR)
        iqr_upper_bound = Q3 + factor * IQR
        final_upper_bound = max(iqr_upper_bound, cap_capacity) if cap_capacity else iqr_upper_bound
        
        group[column] = group[column].clip(lower=lower_bound, upper=final_upper_bound)
        print(f" -> Point [{attraction_id}]: IQR Q1={Q1:.1f}, Q3={Q3:.1f} | Final Upper Bound = {final_upper_bound:.1f}")
        
    return group

df = df.groupby('id_attraction', group_keys=False).apply(process_outliers_combined, column='visitor_count', factor=1.5)

# ==========================================
# 4. KIỂM TRA & XỬ LÝ ĐƯỜNG NGANG (SAU KHI ĐÃ CAPPING)
# ==========================================
print(f"\n[4/5] Kiểm tra đường ngang (Flatlines >= {MIN_CONSECUTIVE_FLAT_HOURS} giờ) sau khi Capping...")

total_plateaus = 0
for att_id in df['id_attraction'].unique():
    att_mask = df['id_attraction'] == att_id
    sub_df = df[att_mask].copy()
    
    is_diff = (sub_df['visitor_count'] != sub_df['visitor_count'].shift(1)) | (sub_df['visitor_count'] == 0)
    sub_df['block_id'] = is_diff.cumsum()
    
    block_sizes = sub_df.groupby('block_id')['visitor_count'].transform('count')
    plateau_mask = (block_sizes >= MIN_CONSECUTIVE_FLAT_HOURS) & (sub_df['visitor_count'] > 0)
    plateau_indices = sub_df[plateau_mask].index
    
    count_flat = len(plateau_indices)
    total_plateaus += count_flat
    print(f" -> Point [{att_id}]: Phát hiện {count_flat} mốc giờ bị đường ngang.")
    
    if count_flat > 0:
        df.loc[plateau_indices, 'visitor_count'] = np.nan

if total_plateaus > 0:
    # Lấy dữ liệu mốc giờ này của ngày hôm trước (t - 24h)
    df['prev_day_visitor'] = df.groupby('id_attraction')['visitor_count'].shift(24)
    nan_mask = df['visitor_count'].isna()
    df.loc[nan_mask, 'visitor_count'] = df.loc[nan_mask, 'prev_day_visitor']
    
    # Dự phòng nội suy tuyến tính nếu ngày trước đó cũng thiếu
    if df['visitor_count'].isna().sum() > 0:
        df['visitor_count'] = df.groupby('id_attraction')['visitor_count'].transform(
            lambda grp: grp.interpolate(method='linear').bfill().ffill()
        )
    df.drop(columns=['prev_day_visitor'], inplace=True, errors='ignore')
    print(f" -> Đã sửa {total_plateaus} mốc giờ đường ngang bằng dữ liệu ngày hôm trước / nội suy thành công.")

# Đảm bảo làm tròn và ép lại các điều kiện vật lý lần cuối
df.loc[closed_mask, 'visitor_count'] = 0
df['visitor_count'] = df['visitor_count'].round().astype(int)

# ==========================================
# 5. XUẤT FILE HOÀN CHỈNH
# ==========================================
print("\n[5/5] Đang lưu file dữ liệu cuối cùng...")
df.to_csv(OUTPUT_FINAL_PATH, index=False, encoding='utf-8-sig')

print("\n==================================================")
print("🎉 HOÀN THÀNH TOÀN BỘ QUY TRÌNH XỬ LÝ!")
print(f"-> File đầu ra cuối cùng: {OUTPUT_FINAL_PATH}")
print("==================================================")