import pandas as pd
import numpy as np

# 1. Đọc dữ liệu gốc
df_path = r'D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_missing\master_H07_elec_missing.csv'
horaire_path = r'D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\cadence\horaire_all_years.csv'  # Đổi lại đường dẫn tuyệt đối nếu cần

df = pd.read_csv(df_path)
df_horaire = pd.read_csv(horaire_path)
total_rows = len(df)

print("="*60)
print("BÁO CÁO THỐNG KÊ XỬ LÝ OUTLIERS THEO TỪNG PHƯƠNG ÁN")
print("="*60)

# =====================================================================
# CHUẨN BỊ VÀ GỘP GIỜ MỞ/ĐÓNG THỰC TẾ CỦA CÔNG VIÊN
# =====================================================================
# Đảm bảo cột date ở cả 2 dataframe đều ở định dạng datetime chính xác
df['date'] = pd.to_datetime(df['date'])
df_horaire['date'] = pd.to_datetime(df_horaire['date'], format='%m/%d/%Y')  # Định dạng MM/DD/YYYY từ file lịch

# Chuyển đổi h_ouv và h_ferm thành giờ nguyên (Integer)
df_horaire['h_ouv_parsed'] = pd.to_datetime(df_horaire['h_ouv'], format='%H:%M:%S', errors='coerce').dt.hour
df_horaire['h_ferm_parsed'] = pd.to_datetime(df_horaire['h_ferm'], format='%H:%M:%S', errors='coerce').dt.hour

# Tạo cột phụ chỉ chứa ngày để gộp chính xác
df['date_only'] = df['date'].dt.normalize()
df_horaire['date_only'] = df_horaire['date'].dt.normalize()

# Gộp thông tin giờ mở/đóng vào dataframe chính
df = df.merge(
    df_horaire[['date_only', 'h_ouv_parsed', 'h_ferm_parsed']], 
    on='date_only', 
    how='left'
)

# Gán giá trị fallback dự phòng nếu ngày đó bị thiếu dữ liệu lịch hoặc công viên đóng cửa
df['h_ouv_h07'] = df['h_ouv_parsed'].fillna(10)
df['h_ferm_h07'] = df['h_ferm_parsed'].fillna(18)

# Xóa các cột phụ trung gian
df.drop(columns=['date_only', 'h_ouv_parsed', 'h_ferm_parsed'], inplace=True)

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

# --- Xử lý cho cột elec_3 ---
q1_3 = df['elec_3'].quantile(0.25)
q3_3 = df['elec_3'].quantile(0.75)
iqr_3 = q3_3 - q1_3
upper_3 = q3_3 + 5.0 * iqr_3
lower_3 = max(0, q1_3 - 5.0 * iqr_3)

outliers_up_3 = (df['elec_3'] > upper_3).sum()
outliers_low_3 = (df['elec_3'] < lower_3).sum()
total_outliers_3 = outliers_up_3 + outliers_low_3

print(f"  => Cột elec_3:")
print(f"     - Ngưỡng trần cố định: {upper_3:.4f} kWh")
print(f"     - Thay thế: {total_outliers_3} dòng ({total_outliers_3/total_rows*100:.2f}%)")

df['elec_3'] = np.where(df['elec_3'] > upper_3, upper_3, np.where(df['elec_3'] < lower_3, lower_3, df['elec_3']))


"""


# =====================================================================
# --- CÁCH 1.2: IQR BIẾN ĐỘNG THEO TỪNG THÁNG (Tách biệt từng cột) ---
# =====================================================================
df['elec_1_clean'] = df['elec_1'].copy()
# df['elec_2_clean'] = df['elec_2'].copy()
# df['elec_3_clean'] = df['elec_3'].copy()
outliers_elec_1 = 0
# outliers_elec_2 = 0
# outliers_elec_3 = 0

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
    
    # # --- Xử lý tháng cho cột elec_2 ---
    # month_data_2 = df.loc[mask, 'elec_2']
    # q1_2 = month_data_2.quantile(0.25)
    # q3_2 = month_data_2.quantile(0.75)
    # iqr_2 = q3_2 - q1_2
    # up_2 = q3_2 + 3.0 * iqr_2
    # low_2 = max(0, q1_2 - 3.0 * iqr_2)
    
    # outliers_elec_2 += ((month_data_2 > up_2) | (month_data_2 < low_2)).sum()
    # df.loc[mask, 'elec_2_clean'] = np.where(
    #     df.loc[mask, 'elec_2'] > up_2, up_2, 
    #     np.where(df.loc[mask, 'elec_2'] < low_2, low_2, df.loc[mask, 'elec_2'])
    # )
    
    # # --- Xử lý tháng cho cột elec_3 ---
    # month_data_3 = df.loc[mask, 'elec_3']
    # q1_3 = month_data_3.quantile(0.25)
    # q3_3 = month_data_3.quantile(0.75)
    # iqr_3 = q3_3 - q1_3
    # up_3 = q3_3 + 3.0 * iqr_3
    # low_3 = max(0, q1_3 - 3.0 * iqr_3)

    # outliers_elec_3 += ((month_data_3 > up_3) | (month_data_3 < low_3)).sum()
    # df.loc[mask, 'elec_3_clean'] = np.where(
    #     df.loc[mask, 'elec_3'] > up_3, up_3,
    #     np.where(df.loc[mask, 'elec_3'] < low_3, low_3, df.loc[mask, 'elec_3'])
    # )

print(f"[OUTLIER] Đang bật: [CÁCH 1.2] IQR biến động theo từng THÁNG")
print(f"  => Cột elec_1: Tổng số dòng bị thay thế (12 tháng): {outliers_elec_1} dòng ({outliers_elec_1/total_rows*100:.2f}%)")
# print(f"  => Cột elec_2: Tổng số dòng bị thay thế (12 tháng): {outliers_elec_2} dòng ({outliers_elec_2/total_rows*100:.2f}%)")
# print(f"  => Cột elec_3: Tổng số dòng bị thay thế (12 tháng): {outliers_elec_3} dòng ({outliers_elec_3/total_rows*100:.2f}%)")

df['elec_1'] = df['elec_1_clean']
# df['elec_2'] = df['elec_2_clean']
# df['elec_3'] = df['elec_3_clean']
df.drop(columns=['elec_1_clean'], inplace=True)

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

# --- Xử lý chuỗi thời gian cho cột elec_3 ---
rolling_median_3 = df['elec_3'].rolling(window=window_size, center=True, min_periods=1).median()
rolling_mad_3 = (df['elec_3'] - rolling_median_3).abs().rolling(window=window_size, center=True, min_periods=1).median()
upper_hampel_3 = rolling_median_3 + 3 * rolling_mad_3
lower_hampel_3 = np.maximum(0, rolling_median_3 - 3 * rolling_mad_3)
total_outliers_3 = ((df['elec_3'] > upper_hampel_3) | (df['elec_3'] < lower_hampel_3)).sum()
print(f"  => Cột elec_3: Thay thế {total_outliers_3} dòng lệch xu hướng ngày ({total_outliers_3/total_rows*100:.2f}%)")
df['elec_3'] = np.where(df['elec_3'] > upper_hampel_3, rolling_median_3,
                        np.where(df['elec_3'] < lower_hampel_3, rolling_median_3, df['elec_3']))
"""

print("-" * 60)

# # =====================================================================
# # PHẦN 2: CÁC TÙY CHỌN XỬ LÝ VISITOR_COUNT (Tích hợp Logic Vận Hành)
# # =====================================================================

# # --- BƯỚC 2.1: ÉP LOGIC VẬN HÀNH THỰC TẾ (Đóng cửa & Khung giờ hoạt động thực tế từ lịch) ---

# # 1. Tạo điều kiện lọc khung giờ mở cửa động theo dữ liệu lịch thực tế của ngày đó
# # Khách chỉ có thể xuất hiện trong khoảng thời gian [h_ouv_h07, h_ferm_h07)
# valid_hours_mask = (df['hour'] >= df['h_ouv_h07']) & (df['hour'] < df['h_ferm_h07'])

# # 2. Kết hợp với điều kiện công viên mở cửa (is_open == 1)
# logic_valide = (df['is_open'] == 1) & valid_hours_mask

# # 3. Thống kê số dòng bị lỗi logic (Có khách khi đáng lẽ phải bằng 0)
# visitor_logic_faults = ((df['visitor_count'] > 0) & (~logic_valide)).sum()

# print(f"[VISITOR_COUNT] Kiểm tra Logic Vận Hành:")
# print(f"  - Số dòng có khách sai khung giờ hoặc sai ngày mở cửa: {visitor_logic_faults} dòng ({visitor_logic_faults/total_rows*100:.2f}%)")
# print(f"  => Tiến hành ép về 0 khách cho các dòng sai logic này.")

# # Thực hiện ép về 0 cho các dòng không hợp lệ
# df['visitor_count'] = np.where(logic_valide, df['visitor_count'], 0)
# print("-" * 40)


# # --- BƯỚC 2.2: LỰA CHỌN PHƯƠNG ÁN CHẶN TRẦN OUTLIER TOÁN HỌC ---

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
df.drop(columns=['h_ouv_h07', 'h_ferm_h07'], inplace=True)
df.to_csv(r'D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_outliers\master_H07_elec_outliers.csv', index=False)
print("\nĐã xử lý xong và xuất file dữ liệu thành công!")

# import pandas as pd
# import numpy as np

# # 1. Đọc dữ liệu gốc
# df_path = r'D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_missing\master_H07_elec_missing.csv'
# horaire_path = r'D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\cadence\horaire_all_years.csv'  # Đổi lại đường dẫn tuyệt đối nếu cần

# df = pd.read_csv(df_path)
# df_horaire = pd.read_csv(horaire_path)
# total_rows = len(df)

# print("="*60)
# print("BÁO CÁO THỐNG KÊ XỬ LÝ OUTLIERS THEO TỪNG PHƯƠNG ÁN")
# print("="*60)

# # =====================================================================
# # CHUẨN BỊ VÀ GỘP GIỜ MỞ/ĐÓNG THỰC TẾ CỦA CÔNG VIÊN
# # =====================================================================
# # Đảm bảo cột date ở cả 2 dataframe đều ở định dạng datetime chính xác
# df['date'] = pd.to_datetime(df['date'])
# df_horaire['date'] = pd.to_datetime(df_horaire['date'], format='%m/%d/%Y')  # Định dạng MM/DD/YYYY từ file lịch

# # Chuyển đổi h_ouv và h_ferm thành giờ nguyên (Integer)
# df_horaire['h_ouv_parsed'] = pd.to_datetime(df_horaire['h_ouv'], format='%H:%M:%S', errors='coerce').dt.hour
# df_horaire['h_ferm_parsed'] = pd.to_datetime(df_horaire['h_ferm'], format='%H:%M:%S', errors='coerce').dt.hour

# # Tạo cột phụ chỉ chứa ngày để gộp chính xác
# df['date_only'] = df['date'].dt.normalize()
# df_horaire['date_only'] = df_horaire['date'].dt.normalize()

# # Gộp thông tin giờ mở/đóng vào dataframe chính
# df = df.merge(
#     df_horaire[['date_only', 'h_ouv_parsed', 'h_ferm_parsed']], 
#     on='date_only', 
#     how='left'
# )

# # Gán giá trị fallback dự phòng nếu ngày đó bị thiếu dữ liệu lịch hoặc công viên đóng cửa
# df['h_ouv_h07'] = df['h_ouv_parsed'].fillna(10)
# df['h_ferm_h07'] = df['h_ferm_parsed'].fillna(18)

# # Xóa các cột phụ trung gian
# df.drop(columns=['date_only', 'h_ouv_parsed', 'h_ferm_parsed'], inplace=True)

# # =====================================================================
# # PHẦN 1: CÁC TÙY CHỌN XỬ LÝ OUTLIER CHO ELEC_1 VÀ ELEC_2 ĐỘC LẬP
# # =====================================================================

# # Chú ý: Hãy BẬT (uncomment) 1 trong 3 cách dưới đây để chạy.
# # Cuối mỗi cách, hệ thống sẽ tự động đồng bộ lại cột tổng: df['ec_value'] = df['elec_1'] + df['elec_2']


# # =====================================================================
# # --- CÁCH 1.1: IQR PHẲNG TOÀN NĂM (Tách biệt từng cột) ---
# # =====================================================================
# """
# print(f"[OUTLIER] Đang bật: [CÁCH 1.1] IQR phẳng toàn năm cho từng cột độc lập")

# # --- Xử lý cho cột elec_1 ---
# q1_1 = df['elec_1'].quantile(0.25)
# q3_1 = df['elec_1'].quantile(0.75)
# iqr_1 = q3_1 - q1_1
# upper_1 = q3_1 + 5.0 * iqr_1
# lower_1 = max(0, q1_1 - 5.0 * iqr_1)

# outliers_up_1 = (df['elec_1'] > upper_1).sum()
# outliers_low_1 = (df['elec_1'] < lower_1).sum()
# total_outliers_1 = outliers_up_1 + outliers_low_1

# print(f"  => Cột elec_1:")
# print(f"     - Ngưỡng trần cố định: {upper_1:.4f} kWh")
# print(f"     - Thay thế: {total_outliers_1} dòng ({total_outliers_1/total_rows*100:.2f}%)")

# df['elec_1'] = np.where(df['elec_1'] > upper_1, upper_1, np.where(df['elec_1'] < lower_1, lower_1, df['elec_1']))

# # --- Xử lý cho cột elec_2 ---
# q1_2 = df['elec_2'].quantile(0.25)
# q3_2 = df['elec_2'].quantile(0.75)
# iqr_2 = q3_2 - q1_2
# upper_2 = q3_2 + 5.0 * iqr_2
# lower_2 = max(0, q1_2 - 5.0 * iqr_2)

# outliers_up_2 = (df['elec_2'] > upper_2).sum()
# outliers_low_2 = (df['elec_2'] < lower_2).sum()
# total_outliers_2 = outliers_up_2 + outliers_low_2

# print(f"  => Cột elec_2:")
# print(f"     - Ngưỡng trần cố định: {upper_2:.4f} kWh")
# print(f"     - Thay thế: {total_outliers_2} dòng ({total_outliers_2/total_rows*100:.2f}%)")

# df['elec_2'] = np.where(df['elec_2'] > upper_2, upper_2, np.where(df['elec_2'] < lower_2, lower_2, df['elec_2']))

# # --- Xử lý cho cột elec_3 ---
# q1_3 = df['elec_3'].quantile(0.25)
# q3_3 = df['elec_3'].quantile(0.75)
# iqr_3 = q3_3 - q1_3
# upper_3 = q3_3 + 5.0 * iqr_3
# lower_3 = max(0, q1_3 - 5.0 * iqr_3)

# outliers_up_3 = (df['elec_3'] > upper_3).sum()
# outliers_low_3 = (df['elec_3'] < lower_3).sum()
# total_outliers_3 = outliers_up_3 + outliers_low_3

# print(f"  => Cột elec_3:")
# print(f"     - Ngưỡng trần cố định: {upper_3:.4f} kWh")
# print(f"     - Thay thế: {total_outliers_3} dòng ({total_outliers_3/total_rows*100:.2f}%)")

# df['elec_3'] = np.where(df['elec_3'] > upper_3, upper_3, np.where(df['elec_3'] < lower_3, lower_3, df['elec_3']))


# """


# # =====================================================================
# # --- CÁCH 1.2: IQR BIẾN ĐỘNG THEO TỪNG THÁNG (Tách biệt từng cột) ---
# # =====================================================================
# df['elec_1_clean'] = df['elec_1'].copy()
# df['elec_2_clean'] = df['elec_2'].copy()
# df['elec_3_clean'] = df['elec_3'].copy()
# outliers_elec_1 = 0
# outliers_elec_2 = 0
# outliers_elec_3 = 0

# for month in sorted(df['month'].unique()):
#     mask = df['month'] == month
    
#     # --- Xử lý tháng cho cột elec_1 ---
#     month_data_1 = df.loc[mask, 'elec_1']
#     q1_1 = month_data_1.quantile(0.25)
#     q3_1 = month_data_1.quantile(0.75)
#     iqr_1 = q3_1 - q1_1
#     up_1 = q3_1 + 3.0* iqr_1
#     low_1 = max(0, q1_1 - 3.0 * iqr_1)
    
#     outliers_elec_1 += ((month_data_1 > up_1) | (month_data_1 < low_1)).sum()
#     df.loc[mask, 'elec_1_clean'] = np.where(
#         df.loc[mask, 'elec_1'] > up_1, up_1, 
#         np.where(df.loc[mask, 'elec_1'] < low_1, low_1, df.loc[mask, 'elec_1'])
#     )
    
#     # --- Xử lý tháng cho cột elec_2 ---
#     month_data_2 = df.loc[mask, 'elec_2']
#     q1_2 = month_data_2.quantile(0.25)
#     q3_2 = month_data_2.quantile(0.75)
#     iqr_2 = q3_2 - q1_2
#     up_2 = q3_2 + 3.0 * iqr_2
#     low_2 = max(0, q1_2 - 3.0 * iqr_2)
    
#     outliers_elec_2 += ((month_data_2 > up_2) | (month_data_2 < low_2)).sum()
#     df.loc[mask, 'elec_2_clean'] = np.where(
#         df.loc[mask, 'elec_2'] > up_2, up_2, 
#         np.where(df.loc[mask, 'elec_2'] < low_2, low_2, df.loc[mask, 'elec_2'])
#     )
    
#     # --- Xử lý tháng cho cột elec_3 ---
#     month_data_3 = df.loc[mask, 'elec_3']
#     q1_3 = month_data_3.quantile(0.25)
#     q3_3 = month_data_3.quantile(0.75)
#     iqr_3 = q3_3 - q1_3
#     up_3 = q3_3 + 3.0 * iqr_3
#     low_3 = max(0, q1_3 - 3.0 * iqr_3)

#     outliers_elec_3 += ((month_data_3 > up_3) | (month_data_3 < low_3)).sum()
#     df.loc[mask, 'elec_3_clean'] = np.where(
#         df.loc[mask, 'elec_3'] > up_3, up_3,
#         np.where(df.loc[mask, 'elec_3'] < low_3, low_3, df.loc[mask, 'elec_3'])
#     )

# print(f"[OUTLIER] Đang bật: [CÁCH 1.2] IQR biến động theo từng THÁNG")
# print(f"  => Cột elec_1: Tổng số dòng bị thay thế (12 tháng): {outliers_elec_1} dòng ({outliers_elec_1/total_rows*100:.2f}%)")
# print(f"  => Cột elec_2: Tổng số dòng bị thay thế (12 tháng): {outliers_elec_2} dòng ({outliers_elec_2/total_rows*100:.2f}%)")
# print(f"  => Cột elec_3: Tổng số dòng bị thay thế (12 tháng): {outliers_elec_3} dòng ({outliers_elec_3/total_rows*100:.2f}%)")

# df['elec_1'] = df['elec_1_clean']
# df['elec_2'] = df['elec_2_clean']
# df['elec_3'] = df['elec_3_clean']
# df.drop(columns=['elec_1_clean', 'elec_2_clean', 'elec_3_clean'], inplace=True)

# # Đồng bộ lại cột tổng
# # df['ec_value'] = df['elec_1'] + df['elec_2']


# # =====================================================================
# # --- CÁCH 1.3: HAMPEL FILTER / TRUNG VỊ TRƯỢT 24H (Tách biệt từng cột) ---
# # =====================================================================
# """
# print(f"[OUTLIER] Đang bật: [CÁCH 1.3] Hampel Filter (Rolling Median 24h)")
# window_size = 24

# # --- Xử lý chuỗi thời gian cho cột elec_1 ---
# rolling_median_1 = df['elec_1'].rolling(window=window_size, center=True, min_periods=1).median()
# rolling_mad_1 = (df['elec_1'] - rolling_median_1).abs().rolling(window=window_size, center=True, min_periods=1).median()
# upper_hampel_1 = rolling_median_1 + 3 * rolling_mad_1
# lower_hampel_1 = np.maximum(0, rolling_median_1 - 3 * rolling_mad_1)

# total_outliers_1 = ((df['elec_1'] > upper_hampel_1) | (df['elec_1'] < lower_hampel_1)).sum()
# print(f"  => Cột elec_1: Thay thế {total_outliers_1} dòng lệch xu hướng ngày ({total_outliers_1/total_rows*100:.2f}%)")

# df['elec_1'] = np.where(df['elec_1'] > upper_hampel_1, rolling_median_1, 
#                         np.where(df['elec_1'] < lower_hampel_1, rolling_median_1, df['elec_1']))

# # --- Xử lý chuỗi thời gian cho cột elec_2 ---
# rolling_median_2 = df['elec_2'].rolling(window=window_size, center=True, min_periods=1).median()
# rolling_mad_2 = (df['elec_2'] - rolling_median_2).abs().rolling(window=window_size, center=True, min_periods=1).median()
# upper_hampel_2 = rolling_median_2 + 3 * rolling_mad_2
# lower_hampel_2 = np.maximum(0, rolling_median_2 - 3 * rolling_mad_2)

# total_outliers_2 = ((df['elec_2'] > upper_hampel_2) | (df['elec_2'] < lower_hampel_2)).sum()
# print(f"  => Cột elec_2: Thay thế {total_outliers_2} dòng lệch xu hướng ngày ({total_outliers_2/total_rows*100:.2f}%)")

# df['elec_2'] = np.where(df['elec_2'] > upper_hampel_2, rolling_median_2, 
#                         np.where(df['elec_2'] < lower_hampel_2, rolling_median_2, df['elec_2']))

# # --- Xử lý chuỗi thời gian cho cột elec_3 ---
# rolling_median_3 = df['elec_3'].rolling(window=window_size, center=True, min_periods=1).median()
# rolling_mad_3 = (df['elec_3'] - rolling_median_3).abs().rolling(window=window_size, center=True, min_periods=1).median()
# upper_hampel_3 = rolling_median_3 + 3 * rolling_mad_3
# lower_hampel_3 = np.maximum(0, rolling_median_3 - 3 * rolling_mad_3)
# total_outliers_3 = ((df['elec_3'] > upper_hampel_3) | (df['elec_3'] < lower_hampel_3)).sum()
# print(f"  => Cột elec_3: Thay thế {total_outliers_3} dòng lệch xu hướng ngày ({total_outliers_3/total_rows*100:.2f}%)")
# df['elec_3'] = np.where(df['elec_3'] > upper_hampel_3, rolling_median_3,
#                         np.where(df['elec_3'] < lower_hampel_3, rolling_median_3, df['elec_3']))
# """

# print("-" * 60)

# # # =====================================================================
# # # PHẦN 2: CÁC TÙY CHỌN XỬ LÝ VISITOR_COUNT (Tích hợp Logic Vận Hành)
# # # =====================================================================

# # # --- BƯỚC 2.1: ÉP LOGIC VẬN HÀNH THỰC TẾ (Đóng cửa & Khung giờ hoạt động thực tế từ lịch) ---

# # # 1. Tạo điều kiện lọc khung giờ mở cửa động theo dữ liệu lịch thực tế của ngày đó
# # # Khách chỉ có thể xuất hiện trong khoảng thời gian [h_ouv_h07, h_ferm_h07)
# # valid_hours_mask = (df['hour'] >= df['h_ouv_h07']) & (df['hour'] < df['h_ferm_h07'])

# # # 2. Kết hợp với điều kiện công viên mở cửa (is_open == 1)
# # logic_valide = (df['is_open'] == 1) & valid_hours_mask

# # # 3. Thống kê số dòng bị lỗi logic (Có khách khi đáng lẽ phải bằng 0)
# # visitor_logic_faults = ((df['visitor_count'] > 0) & (~logic_valide)).sum()

# # print(f"[VISITOR_COUNT] Kiểm tra Logic Vận Hành:")
# # print(f"  - Số dòng có khách sai khung giờ hoặc sai ngày mở cửa: {visitor_logic_faults} dòng ({visitor_logic_faults/total_rows*100:.2f}%)")
# # print(f"  => Tiến hành ép về 0 khách cho các dòng sai logic này.")

# # # Thực hiện ép về 0 cho các dòng không hợp lệ
# # df['visitor_count'] = np.where(logic_valide, df['visitor_count'], 0)
# # print("-" * 40)


# # # --- BƯỚC 2.2: LỰA CHỌN PHƯƠNG ÁN CHẶN TRẦN OUTLIER TOÁN HỌC ---

# # # --- PHƯƠNG ÁN A: Chỉ chặn bằng Logic vật lý (Sức chứa hàng chờ tối đa = 750) ---
# # max_physical_capacity = 750
# # v_outliers = (df['visitor_count'] > max_physical_capacity).sum()
# # print(f"[VISITOR_COUNT] Đang bật: [PHƯƠNG ÁN A] (Chặn logic vật lý hàng chờ)")
# # print(f"  - Ngưỡng trần vật lý cố định: {max_physical_capacity} khách")
# # print(f"  => Số dòng vượt ngưỡng vật lý bị hạ trần: {v_outliers} dòng ({v_outliers/total_rows*100:.2f}%)")
# # df['visitor_count'] = np.where(df['visitor_count'] > max_physical_capacity, max_physical_capacity, df['visitor_count'])


# # # --- PHƯƠNG ÁN B: Plafonnement IQR tiêu chuẩn 1.5x (Ngưỡng cắt = 265 khách) ---
# # # upper_bound_v_15 = df['visitor_count'].quantile(0.75) + 1.5 * (df['visitor_count'].quantile(0.75) - df['visitor_count'].quantile(0.25))
# # # v_outliers = (df['visitor_count'] > upper_bound_v_15).sum()
# # # print(f"[VISITOR_COUNT] Đang bật: [PHƯƠNG ÁN B] (Plafonnement IQR 1.5x)")
# # # print(f"  - Ngưỡng trần toán học: {upper_bound_v_15:.1f} khách")
# # # print(f"  => Số dòng bị ép trần toán học: {v_outliers} dòng ({v_outliers/total_rows*100:.2f}%)")
# # # df['visitor_count'] = np.where(df['visitor_count'] > upper_bound_v_15, upper_bound_v_15, df['visitor_count'])


# # # --- PHƯƠNG ÁN C: Dung hòa - Plafonnement IQR nới rộng 3.0x (Ngưỡng cắt = 424 khách) ---
# # # upper_bound_v_30 = df['visitor_count'].quantile(0.75) + 3.0 * (df['visitor_count'].quantile(0.75) - df['visitor_count'].quantile(0.25))
# # # v_outliers = (df['visitor_count'] > upper_bound_v_30).sum()
# # # print(f"[VISITOR_COUNT] Đang bật: [PHƯƠNG ÁN C] (Plafonnement IQR 3.0x)")
# # # print(f"  - Ngưỡng trần dung hòa: {upper_bound_v_30:.1f} khách")
# # # print(f"  => Số dòng bị ép trần dung hòa: {v_outliers} dòng ({v_outliers/total_rows*100:.2f}%)")
# # # df['visitor_count'] = np.where(df['visitor_count'] > upper_bound_v_30, upper_bound_v_30, df['visitor_count'])


# # # Đảm bảo lượng khách không bị âm (Chặn sàn dưới an toàn rộng)
# # df['visitor_count'] = np.maximum(0, df['visitor_count'])

# print("="*60)

# # =====================================================================
# # PHẦN 3: XUẤT FILE KẾT QUẢ ĐỂ ĐỐI CHỨNG
# # =====================================================================
# df.drop(columns=['h_ouv_h07', 'h_ferm_h07'], inplace=True)
# df.to_csv(r'D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_outliers\master_H07_elec_outliers.csv', index=False)
# print("\nĐã xử lý xong và xuất file dữ liệu thành công!")