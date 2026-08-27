import os
import pandas as pd
import numpy as np

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN & SỨC CHỨA HÀNG CHỜ
# ==========================================
DATA_DIR = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean"
INPUT_PATH = os.path.join(DATA_DIR, "master_outliers", "visitor_outliers.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "master_ml", "visitor.csv")

# Sức chứa tối đa (Max Capacity) của từng điểm tham quan
MAX_CAPACITY = {
    'H03': 750,
    'H07': 800
}

print("=== BẮT ĐẦU TẠO FEATURES (TEMPORAL + CAPACITY + FREQUENTATION ENCODING) ===")

# 1. Đọc dữ liệu
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Chưa tìm thấy file {INPUT_PATH}. Hãy kiểm tra lại đường dẫn!")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

df = pd.read_csv(INPUT_PATH)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(by=['id_attraction', 'datetime']).reset_index(drop=True)

print(f"\n[1/4] Đã tải dữ liệu: {len(df):,} dòng.")

# ==========================================
# 2. TẠO BIẾN THỜI GIAN & CHU KỲ (TEMPORAL FEATURES)
# ==========================================
print("\n[2/4] Tạo các biến Thời gian & Mã hóa Chu kỳ (Sin/Cos)...")

df['hour'] = df['datetime'].dt.hour              # 0 - 23
df['dayofweek'] = df['datetime'].dt.dayofweek      # 0 - 6
df['month'] = df['datetime'].dt.month            # 1 - 12
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# Sin/Cos Encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

# ==========================================
# 3. TẠO TỶ LỆ LẤP ĐẦY (CAPACITY RATIO)
# ==========================================
print("\n[3/4] Tạo Tỷ lệ Lấp đầy (Capacity Ratio)...")

df['max_capacity'] = df['id_attraction'].map(MAX_CAPACITY)
df['capacity_ratio'] = (df['visitor_count'] / df['max_capacity']).round(4)

# ==========================================
# 4. ENCODING CHO TYPE_FREQUENTATION
# ==========================================
print("\n[4/4] Đang xử lý Encoding cho 'type_frequentation'...")

if 'type_frequentation' in df.columns:
    # --- LỰA CHỌN A: Ordinal Encoding (Nếu có thứ tự rõ ràng) ---
    # Bạn có thể điều chỉnh dict này theo đúng các giá trị thực tế có trong dataset
    freq_mapping = {
        'BF': 1,
        'MF': 2,
        'HF': 3,
        'THF': 4,
    }
    
    # Tạo cột số Ordinal
    df['type_freq_num'] = df['type_frequentation'].map(freq_mapping).fillna(0).astype(int)
    
    # --- LỰA CHỌN B: One-Hot Encoding (Tạo các cột 0/1 cho mô hình ML) ---
    df = pd.get_dummies(df, columns=['type_frequentation'], prefix='freq', dtype=int)
    
    print(" -> Đã hoàn thành One-Hot Encoding & Ordinal Mapping cho type_frequentation.")
else:
    print(" -> Cảnh báo: Không tìm thấy cột 'type_frequentation' trong file dữ liệu.")

# ==========================================
# 5. XUẤT FILE HOÀN CHỈNH
# ==========================================
df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

print("\n==================================================")
print("🎉 HOÀN THÀNH TẠO FEATURES!")
print(f"-> File đầu ra: {OUTPUT_PATH}")
print(f"-> Danh sách các cột hiện tại trong dataset:\n{list(df.columns)}")
print("==================================================")