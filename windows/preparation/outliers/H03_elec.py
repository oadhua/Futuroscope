import pandas as pd
import numpy as np

# 1. Đọc dữ liệu gốc
df = pd.read_csv(r'D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_missing\master_H03_elec_missing.csv')
total_rows = len(df)

print("="*60)
print("BÁO CÁO THỐNG KÊ XỬ LÝ OUTLIERS THEO TỪNG PHƯƠNG ÁN")
print("="*60)

# =====================================================================
# PHẦN 1: CÁC TÙY CHỌN XỬ LÝ OUTLIER CHO ELEC_1 VÀ ELEC_2 ĐỘC LẬP
# =====================================================================

# Chú ý: Hãy BẬT (uncomment) 1 trong 3 cách dưới đây để chạy.
# Cuối mỗi cách, hệ thống sẽ tự động đồng bộ lại cột tổng: df['ec_value'] = df['elec_1'] + df['elec_2']


# =====================================================================
# --- CÁCH 1.1: IQR PHẲNG TOÀN NĂM (Tách biệt từng cột) ---
# =====================================================================
"""
print(f"[OUTLIER] Đang bật: [CÁCH 1.1] IQR phẳng toàn năm cho từng cột độc lập")

# --- Xử lý cho cột elec_1 ---
q1_1 = df['elec_1'].quantile(0.25)
q3_1 = df['elec_1'].quantile(0.75)
iqr_1 = q3_1 - q1_1
upper_1 = q3_1 + 5.0 * iqr_1
lower_1 = max(0, q1_1 - 5.0 * iqr_1)

outliers_up_1 = (df['elec_1'] > upper_1).sum()
outliers_low_1 = (df['elec_1'] < lower_1).sum()
total_outliers_1 = outliers_up_1 + outliers_low_1

print(f"  => Cột elec_1:")
print(f"     - Ngưỡng trần cố định: {upper_1:.4f} kWh")
print(f"     - Thay thế: {total_outliers_1} dòng ({total_outliers_1/total_rows*100:.2f}%)")

df['elec_1'] = np.where(df['elec_1'] > upper_1, upper_1, np.where(df['elec_1'] < lower_1, lower_1, df['elec_1']))

# --- Xử lý cho cột elec_2 ---
q1_2 = df['elec_2'].quantile(0.25)
q3_2 = df['elec_2'].quantile(0.75)
iqr_2 = q3_2 - q1_2
upper_2 = q3_2 + 5.0 * iqr_2
lower_2 = max(0, q1_2 - 5.0 * iqr_2)

outliers_up_2 = (df['elec_2'] > upper_2).sum()
outliers_low_2 = (df['elec_2'] < lower_2).sum()
total_outliers_2 = outliers_up_2 + outliers_low_2

print(f"  => Cột elec_2:")
print(f"     - Ngưỡng trần cố định: {upper_2:.4f} kWh")
print(f"     - Thay thế: {total_outliers_2} dòng ({total_outliers_2/total_rows*100:.2f}%)")

df['elec_2'] = np.where(df['elec_2'] > upper_2, upper_2, np.where(df['elec_2'] < lower_2, lower_2, df['elec_2']))

# Đồng bộ lại cột tổng
df['ec_value'] = df['elec_1'] + df['elec_2']
"""


# =====================================================================
# --- CÁCH 1.2: IQR BIẾN ĐỘNG THEO TỪNG THÁNG (Tách biệt từng cột) ---
# =====================================================================
df['elec_1_clean'] = df['elec_1'].copy()
df['elec_2_clean'] = df['elec_2'].copy()
outliers_elec_1 = 0
outliers_elec_2 = 0

for month in sorted(df['month'].unique()):
    mask = df['month'] == month
    
    # --- Xử lý tháng cho cột elec_1 ---
    month_data_1 = df.loc[mask, 'elec_1']
    q1_1 = month_data_1.quantile(0.25)
    q3_1 = month_data_1.quantile(0.75)
    iqr_1 = q3_1 - q1_1
    up_1 = q3_1 + 3.0* iqr_1
    low_1 = max(0, q1_1 - 3.0 * iqr_1)
    
    outliers_elec_1 += ((month_data_1 > up_1) | (month_data_1 < low_1)).sum()
    df.loc[mask, 'elec_1_clean'] = np.where(
        df.loc[mask, 'elec_1'] > up_1, up_1, 
        np.where(df.loc[mask, 'elec_1'] < low_1, low_1, df.loc[mask, 'elec_1'])
    )
    
    # --- Xử lý tháng cho cột elec_2 ---
    month_data_2 = df.loc[mask, 'elec_2']
    q1_2 = month_data_2.quantile(0.25)
    q3_2 = month_data_2.quantile(0.75)
    iqr_2 = q3_2 - q1_2
    up_2 = q3_2 + 3.0 * iqr_2
    low_2 = max(0, q1_2 - 3.0 * iqr_2)
    
    outliers_elec_2 += ((month_data_2 > up_2) | (month_data_2 < low_2)).sum()
    df.loc[mask, 'elec_2_clean'] = np.where(
        df.loc[mask, 'elec_2'] > up_2, up_2, 
        np.where(df.loc[mask, 'elec_2'] < low_2, low_2, df.loc[mask, 'elec_2'])
    )

print(f"[OUTLIER] Đang bật: [CÁCH 1.2] IQR biến động theo từng THÁNG")
print(f"  => Cột elec_1: Tổng số dòng bị thay thế (12 tháng): {outliers_elec_1} dòng ({outliers_elec_1/total_rows*100:.2f}%)")
print(f"  => Cột elec_2: Tổng số dòng bị thay thế (12 tháng): {outliers_elec_2} dòng ({outliers_elec_2/total_rows*100:.2f}%)")

df['elec_1'] = df['elec_1_clean']
df['elec_2'] = df['elec_2_clean']
df.drop(columns=['elec_1_clean', 'elec_2_clean'], inplace=True)

# Đồng bộ lại cột tổng
# df['ec_value'] = df['elec_1'] + df['elec_2']


# =====================================================================
# --- CÁCH 1.3: HAMPEL FILTER / TRUNG VỊ TRƯỢT 24H (Tách biệt từng cột) ---
# =====================================================================
"""
print(f"[OUTLIER] Đang bật: [CÁCH 1.3] Hampel Filter (Rolling Median 24h)")
window_size = 24

# --- Xử lý chuỗi thời gian cho cột elec_1 ---
rolling_median_1 = df['elec_1'].rolling(window=window_size, center=True, min_periods=1).median()
rolling_mad_1 = (df['elec_1'] - rolling_median_1).abs().rolling(window=window_size, center=True, min_periods=1).median()
upper_hampel_1 = rolling_median_1 + 3 * rolling_mad_1
lower_hampel_1 = np.maximum(0, rolling_median_1 - 3 * rolling_mad_1)

total_outliers_1 = ((df['elec_1'] > upper_hampel_1) | (df['elec_1'] < lower_hampel_1)).sum()
print(f"  => Cột elec_1: Thay thế {total_outliers_1} dòng lệch xu hướng ngày ({total_outliers_1/total_rows*100:.2f}%)")

df['elec_1'] = np.where(df['elec_1'] > upper_hampel_1, rolling_median_1, 
                        np.where(df['elec_1'] < lower_hampel_1, rolling_median_1, df['elec_1']))

# --- Xử lý chuỗi thời gian cho cột elec_2 ---
rolling_median_2 = df['elec_2'].rolling(window=window_size, center=True, min_periods=1).median()
rolling_mad_2 = (df['elec_2'] - rolling_median_2).abs().rolling(window=window_size, center=True, min_periods=1).median()
upper_hampel_2 = rolling_median_2 + 3 * rolling_mad_2
lower_hampel_2 = np.maximum(0, rolling_median_2 - 3 * rolling_mad_2)

total_outliers_2 = ((df['elec_2'] > upper_hampel_2) | (df['elec_2'] < lower_hampel_2)).sum()
print(f"  => Cột elec_2: Thay thế {total_outliers_2} dòng lệch xu hướng ngày ({total_outliers_2/total_rows*100:.2f}%)")

df['elec_2'] = np.where(df['elec_2'] > upper_hampel_2, rolling_median_2, 
                        np.where(df['elec_2'] < lower_hampel_2, rolling_median_2, df['elec_2']))

# Đồng bộ lại cột tổng
df['ec_value'] = df['elec_1'] + df['elec_2']
"""

print("-" * 60)

# # =====================================================================
# # PHẦN 2: CÁC TÙY CHỌN XỬ LÝ VISITOR_COUNT (Tích hợp Logic Vận Hành)
# # =====================================================================

# # --- BƯỚC 2.1: ÉP LOGIC VẬN HÀNH THỰC TẾ (Đóng cửa & Khung giờ hoạt động) ---

# # 1. Khởi tạo mặt nạ (mask) kiểm tra giờ mở cửa mặc định: chỉ có thể có khách từ 10h trở đi
# # (Nếu giờ < 10, chắc chắn chưa mở cửa đón khách vào attraction)
# valid_hours_mask = (df['hour'] >= 10)

# # 2. Áp dụng điều kiện đóng cửa động theo loại ngày (type_frequentation)
# # - Ngày BF: chỉ có khách từ 10h đến 19h (giờ kết thúc là 19, tức là khách ra hết lúc 19h)
# valid_hours_mask = np.where(df['type_frequentation'] == 'BF', valid_hours_mask & (df['hour'] <= 19), valid_hours_mask)

# # - Ngày MF: chỉ có khách từ 10h đến 20h
# valid_hours_mask = np.where(df['type_frequentation'] == 'MF', valid_hours_mask & (df['hour'] <= 20), valid_hours_mask)

# # - Ngày HF hoặc THF: chỉ có khách từ 10h đến 21h
# valid_hours_mask = np.where(df['type_frequentation'].isin(['HF', 'THF']), valid_hours_mask & (df['hour'] <= 21), valid_hours_mask)

# # 3. Kết hợp với điều kiện công viên mở cửa (is_open == 1)
# # Hợp lệ = Công viên phải MỞ CỬA VÀ nằm trong KHUNG GIỜ HOẠT ĐỘNG của ngày đó
# logic_valide = (df['is_open'] == 1) & valid_hours_mask

# # 4. Thống kê số dòng bị lỗi logic (Có khách khi đáng lẽ phải bằng 0)
# visitor_logic_faults = ((df['visitor_count'] > 0) & (~logic_valide)).sum()

# print(f"[VISITOR_COUNT] Kiểm tra Logic Vận Hành:")
# print(f"  - Số dòng có khách sai khung giờ hoặc sai ngày mở cửa: {visitor_logic_faults} dòng ({visitor_logic_faults/total_rows*100:.2f}%)")
# print(f"  => Tiến hành ép về 0 khách cho các dòng sai logic này.")

# # Thực hiện ép về 0 cho các dòng không hợp lệ
# df['visitor_count'] = np.where(logic_valide, df['visitor_count'], 0)
# print("-" * 40)


# # --- BƯỚC 2.2: LỰA CHỌN PHƯƠNG ÁN CHẶN TRẦN OUTLIER TOÁN HỌC (Bật 1 trong 3 cách) ---

# # --- PHƯƠNG ÁN A: Chỉ chặn bằng Logic vật lý (Sức chứa hàng chờ tối đa = 750) ---
# max_physical_capacity = 750
# v_outliers = (df['visitor_count'] > max_physical_capacity).sum()
# print(f"[VISITOR_COUNT] Đang bật: [PHƯƠNG ÁN A] (Chặn logic vật lý hàng chờ)")
# print(f"  - Ngưỡng trần vật lý cố định: {max_physical_capacity} khách")
# print(f"  => Số dòng vượt ngưỡng vật lý bị hạ trần: {v_outliers} dòng ({v_outliers/total_rows*100:.2f}%)")
# df['visitor_count'] = np.where(df['visitor_count'] > max_physical_capacity, max_physical_capacity, df['visitor_count'])


# # --- PHƯƠNG ÁN B: Plafonnement IQR tiêu chuẩn 1.5x (Ngưỡng cắt = 265 khách) ---
# # upper_bound_v_15 = df['visitor_count'].quantile(0.75) + 1.5 * (df['visitor_count'].quantile(0.75) - df['visitor_count'].quantile(0.25))
# # v_outliers = (df['visitor_count'] > upper_bound_v_15).sum()
# # print(f"[VISITOR_COUNT] Đang bật: [PHƯƠNG ÁN B] (Plafonnement IQR 1.5x)")
# # print(f"  - Ngưỡng trần toán học: {upper_bound_v_15:.1f} khách")
# # print(f"  => Số dòng bị ép trần toán học: {v_outliers} dòng ({v_outliers/total_rows*100:.2f}%)")
# # df['visitor_count'] = np.where(df['visitor_count'] > upper_bound_v_15, upper_bound_v_15, df['visitor_count'])


# # --- PHƯƠNG ÁN C: Dung hòa - Plafonnement IQR nới rộng 3.0x (Ngưỡng cắt = 424 khách) ---
# # upper_bound_v_30 = df['visitor_count'].quantile(0.75) + 3.0 * (df['visitor_count'].quantile(0.75) - df['visitor_count'].quantile(0.25))
# # v_outliers = (df['visitor_count'] > upper_bound_v_30).sum()
# # print(f"[VISITOR_COUNT] Đang bật: [PHƯƠNG ÁN C] (Plafonnement IQR 3.0x)")
# # print(f"  - Ngưỡng trần dung hòa: {upper_bound_v_30:.1f} khách")
# # print(f"  => Số dòng bị ép trần dung hòa: {v_outliers} dòng ({v_outliers/total_rows*100:.2f}%)")
# # df['visitor_count'] = np.where(df['visitor_count'] > upper_bound_v_30, upper_bound_v_30, df['visitor_count'])


# # Đảm bảo lượng khách không bị âm (Chặn sàn dưới an toàn rộng)
# df['visitor_count'] = np.maximum(0, df['visitor_count'])

print("="*60)

# =====================================================================
# PHẦN 3: XUẤT FILE KẾT QUẢ ĐỂ ĐỐI CHỨNG
# =====================================================================
df.to_csv(r'D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_outliers\master_H03_elec_outliers.csv', index=False)
print("\nĐã xử lý xong và xuất file dữ liệu thành công!")