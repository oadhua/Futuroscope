import pandas as pd
import numpy as np

# 1. Đọc dữ liệu gốc
df = pd.read_csv(r'D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_missing\master_H03_thermal_missing.csv')
total_rows = len(df)

print("="*60)
print("BÁO CÁO THỐNG KÊ XỬ LÝ OUTLIERS THEO TỪNG PHƯƠNG ÁN")
print("="*60)

# =====================================================================
# PHẦN 1: CÁC TÙY CHỌN XỬ LÝ EC_VALUE (Bật 1 trong 3 cách)
# =====================================================================

# --- CÁCH 1.1: IQR phẳng toàn năm (Cách cũ) ---
# q1_ec = df['ec_value'].quantile(0.25)
# q3_ec = df['ec_value'].quantile(0.75)
# iqr_ec = q3_ec - q1_ec
# upper_bound_ec = q3_ec + 5.0 * iqr_ec
# lower_bound_ec = max(0, q1_ec - 5.0 * iqr_ec)
# ec_outliers_upper = (df['ec_value'] > upper_bound_ec).sum()
# ec_outliers_lower = (df['ec_value'] < lower_bound_ec).sum()
# total_ec_outliers = ec_outliers_upper + ec_outliers_lower
# print(f"[EC_VALUE] Đang bật: [CÁCH 1.1] IQR phẳng toàn năm")
# print(f"  - Ngưỡng trần cố định: {upper_bound_ec:.4f} kWh")
# print(f"  - Số dòng vượt trần: {ec_outliers_upper} dòng ({ec_outliers_upper/total_rows*100:.2f}%)")
# print(f"  - Số dòng dưới sàn: {ec_outliers_lower} dòng ({ec_outliers_lower/total_rows*100:.2f}%)")
# print(f"  => Tổng số dòng bị thay thế: {total_ec_outliers} dòng ({total_ec_outliers/total_rows*100:.2f}%)")
# df['ec_value'] = np.where(df['ec_value'] > upper_bound_ec, upper_bound_ec, np.where(df['ec_value'] < lower_bound_ec, lower_bound_ec, df['ec_value']))


# --- CÁCH 1.2: IQR biến động theo từng THÁNG (Giữ xu hướng mùa vụ) ---
df['ec_value_clean'] = df['ec_value'].copy()
total_ec_outliers = 0

# Tạo mảng lưu thông tin chi tiết từng tháng nếu cần kiểm tra sâu
for month in sorted(df['month'].unique()):
    mask = df['month'] == month
    month_data = df.loc[mask, 'ec_value']
    
    q1 = month_data.quantile(0.25)
    q3 = month_data.quantile(0.75)
    iqr = q3 - q1
    up = q3 + 1.5 * iqr
    low = max(0, q1 - 1.5 * iqr)
    
    # Tính toán số lượng outlier của riêng tháng này
    month_outliers = ((month_data > up) | (month_data < low)).sum()
    total_ec_outliers += month_outliers
    
    # Ép trần/sàn động cho tháng đó
    df.loc[mask, 'ec_value_clean'] = np.where(
        df.loc[mask, 'ec_value'] > up, up, 
        np.where(df.loc[mask, 'ec_value'] < low, low, df.loc[mask, 'ec_value'])
    )

print(f"[EC_VALUE] Đang bật: [CÁCH 1.2] IQR biến động theo từng THÁNG")
print(f"  => Tổng số dòng ec_value bị thay thế (cộng dồn 12 tháng): {total_ec_outliers} dòng ({total_ec_outliers/total_rows*100:.2f}%)")
df['ec_value'] = df['ec_value_clean']
df.drop(columns=['ec_value_clean'], inplace=True)


# --- CÁCH 1.3: Hampel Filter / Trung vị trượt 24h (Bảo toàn chuỗi thời gian tối ưu) ---
# window_size = 24
# rolling_median = df['ec_value'].rolling(window=window_size, center=True, min_periods=1).median()
# rolling_mad = (df['ec_value'] - rolling_median).abs().rolling(window=window_size, center=True, min_periods=1).median()
# upper_hampel = rolling_median + 3 * rolling_mad
# lower_hampel = np.maximum(0, rolling_median - 3 * rolling_mad)
# total_ec_outliers = ((df['ec_value'] > upper_hampel) | (df['ec_value'] < lower_hampel)).sum()
# print(f"[EC_VALUE] Đang bật: [CÁCH 1.3] Hampel Filter (Rolling Median 24h)")
# print(f"  => Tổng số dòng ec_value bị thay thế (lệch khỏi xu hướng ngày): {total_ec_outliers} dòng ({total_ec_outliers/total_rows*100:.2f}%)")
# df['ec_value'] = np.where(df['ec_value'] > upper_hampel, rolling_median, np.where(df['ec_value'] < lower_hampel, rolling_median, df['ec_value']))

print("-" * 60)

# =====================================================================
# PHẦN 2: CÁC TÙY CHỌN XỬ LÝ VISITOR_COUNT (Tích hợp Logic Vận Hành)
# =====================================================================

# --- BƯỚC 2.1: ÉP LOGIC VẬN HÀNH THỰC TẾ (Đóng cửa & Khung giờ hoạt động) ---

# 1. Khởi tạo mặt nạ (mask) kiểm tra giờ mở cửa mặc định: chỉ có thể có khách từ 10h trở đi
# (Nếu giờ < 10, chắc chắn chưa mở cửa đón khách vào attraction)
valid_hours_mask = (df['hour'] >= 10)

# 2. Áp dụng điều kiện đóng cửa động theo loại ngày (type_frequentation)
# - Ngày BF: chỉ có khách từ 10h đến 19h (giờ kết thúc là 19, tức là khách ra hết lúc 19h)
valid_hours_mask = np.where(df['type_frequentation'] == 'BF', valid_hours_mask & (df['hour'] <= 19), valid_hours_mask)

# - Ngày MF: chỉ có khách từ 10h đến 20h
valid_hours_mask = np.where(df['type_frequentation'] == 'MF', valid_hours_mask & (df['hour'] <= 20), valid_hours_mask)

# - Ngày HF hoặc THF: chỉ có khách từ 10h đến 21h
valid_hours_mask = np.where(df['type_frequentation'].isin(['HF', 'THF']), valid_hours_mask & (df['hour'] <= 21), valid_hours_mask)

# 3. Kết hợp với điều kiện công viên mở cửa (is_open == 1)
# Hợp lệ = Công viên phải MỞ CỬA VÀ nằm trong KHUNG GIỜ HOẠT ĐỘNG của ngày đó
logic_valide = (df['is_open'] == 1) & valid_hours_mask

# 4. Thống kê số dòng bị lỗi logic (Có khách khi đáng lẽ phải bằng 0)
visitor_logic_faults = ((df['visitor_count'] > 0) & (~logic_valide)).sum()

print(f"[VISITOR_COUNT] Kiểm tra Logic Vận Hành:")
print(f"  - Số dòng có khách sai khung giờ hoặc sai ngày mở cửa: {visitor_logic_faults} dòng ({visitor_logic_faults/total_rows*100:.2f}%)")
print(f"  => Tiến hành ép về 0 khách cho các dòng sai logic này.")

# Thực hiện ép về 0 cho các dòng không hợp lệ
df['visitor_count'] = np.where(logic_valide, df['visitor_count'], 0)
print("-" * 40)


# --- BƯỚC 2.2: LỰA CHỌN PHƯƠNG ÁN CHẶN TRẦN OUTLIER TOÁN HỌC (Bật 1 trong 3 cách) ---

# --- PHƯƠNG ÁN A: Chỉ chặn bằng Logic vật lý (Sức chứa hàng chờ tối đa = 750) ---
max_physical_capacity = 750
v_outliers = (df['visitor_count'] > max_physical_capacity).sum()
print(f"[VISITOR_COUNT] Đang bật: [PHƯƠNG ÁN A] (Chặn logic vật lý hàng chờ)")
print(f"  - Ngưỡng trần vật lý cố định: {max_physical_capacity} khách")
print(f"  => Số dòng vượt ngưỡng vật lý bị hạ trần: {v_outliers} dòng ({v_outliers/total_rows*100:.2f}%)")
df['visitor_count'] = np.where(df['visitor_count'] > max_physical_capacity, max_physical_capacity, df['visitor_count'])


# --- PHƯƠNG ÁN B: Plafonnement IQR tiêu chuẩn 1.5x (Ngưỡng cắt = 265 khách) ---
# upper_bound_v_15 = df['visitor_count'].quantile(0.75) + 1.5 * (df['visitor_count'].quantile(0.75) - df['visitor_count'].quantile(0.25))
# v_outliers = (df['visitor_count'] > upper_bound_v_15).sum()
# print(f"[VISITOR_COUNT] Đang bật: [PHƯƠNG ÁN B] (Plafonnement IQR 1.5x)")
# print(f"  - Ngưỡng trần toán học: {upper_bound_v_15:.1f} khách")
# print(f"  => Số dòng bị ép trần toán học: {v_outliers} dòng ({v_outliers/total_rows*100:.2f}%)")
# df['visitor_count'] = np.where(df['visitor_count'] > upper_bound_v_15, upper_bound_v_15, df['visitor_count'])


# --- PHƯƠNG ÁN C: Dung hòa - Plafonnement IQR nới rộng 3.0x (Ngưỡng cắt = 424 khách) ---
# upper_bound_v_30 = df['visitor_count'].quantile(0.75) + 3.0 * (df['visitor_count'].quantile(0.75) - df['visitor_count'].quantile(0.25))
# v_outliers = (df['visitor_count'] > upper_bound_v_30).sum()
# print(f"[VISITOR_COUNT] Đang bật: [PHƯƠNG ÁN C] (Plafonnement IQR 3.0x)")
# print(f"  - Ngưỡng trần dung hòa: {upper_bound_v_30:.1f} khách")
# print(f"  => Số dòng bị ép trần dung hòa: {v_outliers} dòng ({v_outliers/total_rows*100:.2f}%)")
# df['visitor_count'] = np.where(df['visitor_count'] > upper_bound_v_30, upper_bound_v_30, df['visitor_count'])


# Đảm bảo lượng khách không bị âm (Chặn sàn dưới an toàn rộng)
df['visitor_count'] = np.maximum(0, df['visitor_count'])

print("="*60)

# =====================================================================
# PHẦN 3: XUẤT FILE KẾT QUẢ ĐỂ ĐỐI CHỨNG
# =====================================================================
df.to_csv(r'D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_outliers\master_H03_thermal_outliers.csv', index=False)
print("\nĐã xử lý xong và xuất file dữ liệu thành công!")