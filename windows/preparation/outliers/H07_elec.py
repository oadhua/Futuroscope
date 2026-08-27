import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================================
# 0. CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU DÀNH CHO H07
# =====================================================================
INPUT_PATH = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_missing\H07_elec_missing.csv"
OUTPUT_DIR = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_outliers"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "H07_elec_outliers.csv")

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Không tìm thấy file đầu vào: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)
total_rows = len(df)

print("=" * 70)
print("BÁO CÁO THỐNG KÊ XỬ LÝ OUTLIERS & LỖI ĐỒNG HỒ ĐIỆN (H07 ELEC)")
print("=" * 70)


# =====================================================================
# PHẦN 1: LỌC 2 TẦNG ĐỘNG (LOCAL MAD + IQR THEO ANNEE VÀ WEEK)
# =====================================================================
print("\n[PHẦN 1] TRUY QUÉT & KHÔI PHỤC LỖI THEO TỪNG TUẦN CỦA TỪNG NĂM (ANNEE & WEEK)...")

elec_cols = ["elec_1", "elec_2"]
year_week_group = ["year", "week"]

# Đảm bảo có cột hour từ datetime
if "hour" not in df.columns and "datetime" in df.columns:
    df["hour"] = pd.to_datetime(df["datetime"]).dt.hour

for col in elec_cols:
    if col not in df.columns:
        continue

    s = df[col].copy()

    # -----------------------------------------------------------------
    # 1. TẦNG 1: LỌC CỤC BỘ (LOCAL MAD SPIKE & FLATLINE & ZERO DROP)
    # -----------------------------------------------------------------
    window_size = 7
    roll_med = s.rolling(window=window_size, center=True, min_periods=1).median()
    roll_mad = (s - roll_med).abs().rolling(window=window_size, center=True, min_periods=1).median()

    dynamic_threshold = np.maximum(4.0 * roll_mad, 0.15 * roll_med)
    mask_spike_local = (s - roll_med).abs() > dynamic_threshold

    # Phẳng lì (>= 2h) / Rớt về 0 khi vận hành / Giá trị âm
    mask_flatline = s.groupby(s.diff().ne(0).cumsum()).transform("count") >= 2
    mask_zero_op = (df.get("is_open", 0) == 1) & (s <= 0.5)
    mask_negative = s < 0.0

    mask_tier1 = mask_spike_local | mask_flatline | mask_zero_op | mask_negative

    # -----------------------------------------------------------------
    # 2. TẦNG 2: LỌC TỔNG THỂ BẰNG IQR THEO ANNEE & WEEK
    # -----------------------------------------------------------------
    s_temp = s.copy()
    s_temp[mask_tier1] = np.nan  # Loại bỏ trước rác Tier 1

    # Tính Q1, Q3, IQR tách biệt cho từng cặp (annee x week)
    q1_yw = df.groupby(year_week_group)[col].transform(lambda x: s_temp.loc[x.index].quantile(0.25))
    q3_yw = df.groupby(year_week_group)[col].transform(lambda x: s_temp.loc[x.index].quantile(0.75))
    iqr_yw = q3_yw - q1_yw

    # Ngưỡng trần/sàn động theo từng tuần của từng năm (2.5x IQR)
    upper_bound_iqr = q3_yw + 2.5 * iqr_yw
    lower_bound_iqr = np.maximum(0, q1_yw - 2.5 * iqr_yw)

    mask_iqr_outlier = (s_temp > upper_bound_iqr) | (s_temp < lower_bound_iqr)

    # -----------------------------------------------------------------
    # 3. TỔNG HỢP LỖI & ĐẮP DỮ LIỆU BẰNG HỒ SƠ TUẦN [ANNEE x WEEK x IS_OPEN x HOUR]
    # -----------------------------------------------------------------
    mask_total_anomaly = mask_tier1 | mask_iqr_outlier

    print(f"\n  => Cột [{col}]: Tổng phát hiện {mask_total_anomaly.sum()} điểm rác")
    print(f"     + Lỗi cục bộ Tier 1 (MAD/Flat/Zero): {mask_tier1.sum()} điểm")
    print(f"     + Outliers Tier 2 (IQR Theo {year_week_group}): {mask_iqr_outlier.sum()} điểm")

    # Tạo bản sao sạch
    df_clean = df.copy()
    df_clean.loc[mask_total_anomaly, col] = np.nan

    # Điền NaN bằng Trung vị theo đúng ngữ cảnh: [annee x week x is_open x hour]
    group_keys = [c for c in ["annee", "week", "is_open", "hour"] if c in df.columns]
    
    if group_keys:
        context_median = df_clean.groupby(group_keys)[col].transform("median")
        df[col] = df_clean[col].fillna(context_median)
    else:
        df[col] = df_clean[col].fillna(roll_med)

    # Dự phòng (Fallback) - Nếu tuần đó bị khuyết dữ liệu nhiều, lấy trung vị theo [week x hour] chung
    fallback_median = df_clean.groupby(["week", "hour"])[col].transform("median")
    df[col] = df[col].fillna(fallback_median)

    # -----------------------------------------------------------------
    # 4. NỘI SUY MƯỢT TẮC GIÁP RANH & KHÓA SÀN >= 0
    # -----------------------------------------------------------------
    df[col] = df[col].interpolate(method="linear", limit=2, limit_direction="both").clip(lower=0.0)

print("\n=> Đã xử lý xong hoàn hảo với cột 'annee' và 'week'!")


# =====================================================================
# PHẦN 2: XỬ LÝ OUTLIERS CHO VISITOR_COUNT H07 (CHỈ ÁP DỤNG <= 10/02/2024)
# =====================================================================
print("\n" + "-" * 70)
print("[PHẦN 2] BÁO CÁO TỔNG HỢP OUTLIERS VISITOR_COUNT H07 (CHỈ DỮ LIỆU <= 10/02/2024):")

if "date" in df.columns:
    df["date_dt"] = pd.to_datetime(df["date"])
elif "datetime" in df.columns:
    df["date_dt"] = pd.to_datetime(df["datetime"]).dt.date
    df["date_dt"] = pd.to_datetime(df["date_dt"])

cutoff_date = pd.to_datetime("2024-02-10")
date_mask = df["date_dt"] <= cutoff_date
total_target_rows = date_mask.sum()

print(f"  -> Tổng số dòng trong giai đoạn xử lý (<= 10/02/2024): {total_target_rows} / {total_rows} dòng")

# Tín hiệu mở/vận hành trò chơi
ouvert_signal = df.get("ouvert", 0) > 0
operation_signal = df.get("operation", 0) > 0
has_any_operation = ouvert_signal | operation_signal

# Đóng cửa thực sự (is_open == 0 VÀ không có vận hành trò chơi)
mask_pure_closed = (df["is_open"] == 0) & (~has_any_operation)

# 1. Reset lượt khách về 0 cho các khoảng đóng cửa thực sự
mask_logic_error = date_mask & (df["visitor_count"] > 0) & mask_pure_closed
outliers_logic = mask_logic_error.sum()
df.loc[date_mask & mask_pure_closed, "visitor_count"] = 0

# 2. Xử lý Outlier vượt sức chứa (Chỉnh ngưỡng phù hợp với H07 nếu cần)
max_physical_capacity = 750
mask_capacity_error = date_mask & (df["visitor_count"] > max_physical_capacity)
outliers_capacity = mask_capacity_error.sum()

df.loc[date_mask, "visitor_count"] = np.where(
    df.loc[date_mask, "visitor_count"] > max_physical_capacity,
    max_physical_capacity,
    df.loc[date_mask, "visitor_count"],
)
df.loc[date_mask, "visitor_count"] = np.maximum(0, df.loc[date_mask, "visitor_count"])

if "date_dt" in df.columns:
    df.drop(columns=["date_dt"], inplace=True)

total_visitor_outliers = outliers_logic + outliers_capacity

print(f"  - Lỗi logic (is_open=0 VÀ không có vận hành ouvert/operation): {outliers_logic} dòng ({outliers_logic/total_target_rows*100:.2f}%)")
print(f"  - Vượt sức chứa vật lý (> {max_physical_capacity} khách): {outliers_capacity} dòng ({outliers_capacity/total_target_rows*100:.2f}%)")
print(f"  => TỔNG SỐ DÒNG OUTLIER VISITOR ĐÃ XỬ LÝ (<= 10/02/2024): {total_visitor_outliers} dòng")
print("  => Các trường hợp có ouvert > 0 / operation > 0 được GIỮ NGUYÊN kết quả mô hình.")
print("=" * 70)


# =====================================================================
# PHẦN 3: XUẤT FILE KẾT QUẢ FOR H07
# =====================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n✅ Đã xuất file H07 thành công tại:\n -> {OUTPUT_PATH}")


# =====================================================================
# PHẦN 4: VẼ BIỂU ĐỒ TƯƠNG TÁC PLOTLY CHO H07
# =====================================================================
print("\n[PHẦN 4] KHỞI TẠO BIỂU ĐỒ TƯƠNG TÁC PLOTLY CHO H07...")

df_raw = pd.read_csv(INPUT_PATH)
df_clean = pd.read_csv(OUTPUT_PATH)

df_raw["datetime"] = pd.to_datetime(
    df_raw["date"].astype(str) + " " + df_raw["hour"].astype(str) + ":00:00"
)
df_clean["datetime"] = pd.to_datetime(
    df_clean["date"].astype(str) + " " + df_clean["hour"].astype(str) + ":00:00"
)

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.07,
    subplot_titles=(
        "1. Tiêu Thụ Điện Cột elec_1 (kWh) - H07 (Đã sửa Gai & Đoạn phẳng lì)",
        "2. Tiêu Thụ Điện Cột elec_2 (kWh) - H07 (Đã sửa Gai & Đoạn phẳng lì)",
        "3. Lượt Khách Visitor Count H07 (Lọc <= 10/02/2024)",
    ),
)

# --- SUBPLOT 1: ELEC_1 ---
fig.add_trace(
    go.Scatter(
        x=df_raw["datetime"],
        y=df_raw["elec_1"],
        name="Raw elec_1",
        line=dict(color="crimson", width=1),
        opacity=0.4,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_clean["datetime"],
        y=df_clean["elec_1"],
        name="Clean elec_1",
        line=dict(color="royalblue", width=1.5),
    ),
    row=1,
    col=1,
)

# --- SUBPLOT 2: ELEC_2 ---
fig.add_trace(
    go.Scatter(
        x=df_raw["datetime"],
        y=df_raw["elec_2"],
        name="Raw elec_2",
        line=dict(color="darkorange", width=1),
        opacity=0.4,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_clean["datetime"],
        y=df_clean["elec_2"],
        name="Clean elec_2",
        line=dict(color="forestgreen", width=1.5),
    ),
    row=2,
    col=1,
)

# --- SUBPLOT 3: VISITOR COUNT ---
fig.add_trace(
    go.Scatter(
        x=df_raw["datetime"],
        y=df_raw["visitor_count"],
        name="Raw Visitor",
        line=dict(color="mediumpurple", width=1),
        opacity=0.35,
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_clean["datetime"],
        y=df_clean["visitor_count"],
        name="Clean Visitor",
        line=dict(color="midnightblue", width=1.5),
    ),
    row=3,
    col=1,
)

fig.add_vline(
    x=pd.to_datetime("2024-02-10").timestamp() * 1000,
    line_width=2,
    line_dash="dash",
    line_color="red",
    annotation_text="Cutoff (10/02/2024)",
    annotation_position="top left",
    row=3,
    col=1,
)

fig.update_layout(
    height=850,
    title_text="<b>ĐỐI CHỨNG DỮ LIỆU TRƯỚC & SAU CLEAN OUTLIERS (H07 ELEC)</b>",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

fig.update_xaxes(rangeslider_visible=True, row=3, col=1)

html_output_path = os.path.join(OUTPUT_DIR, "H07_elec_outliers_interactive.html")
fig.write_html(html_output_path)
print(f"✅ Đã tạo xong biểu đồ HTML tương tác H07 tại:\n -> {html_output_path}")

fig.show()

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
# # df['elec_2_clean'] = df['elec_2'].copy()
# # df['elec_3_clean'] = df['elec_3'].copy()
# outliers_elec_1 = 0
# # outliers_elec_2 = 0
# # outliers_elec_3 = 0

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
    
#     # # --- Xử lý tháng cho cột elec_2 ---
#     # month_data_2 = df.loc[mask, 'elec_2']
#     # q1_2 = month_data_2.quantile(0.25)
#     # q3_2 = month_data_2.quantile(0.75)
#     # iqr_2 = q3_2 - q1_2
#     # up_2 = q3_2 + 3.0 * iqr_2
#     # low_2 = max(0, q1_2 - 3.0 * iqr_2)
    
#     # outliers_elec_2 += ((month_data_2 > up_2) | (month_data_2 < low_2)).sum()
#     # df.loc[mask, 'elec_2_clean'] = np.where(
#     #     df.loc[mask, 'elec_2'] > up_2, up_2, 
#     #     np.where(df.loc[mask, 'elec_2'] < low_2, low_2, df.loc[mask, 'elec_2'])
#     # )
    
#     # # --- Xử lý tháng cho cột elec_3 ---
#     # month_data_3 = df.loc[mask, 'elec_3']
#     # q1_3 = month_data_3.quantile(0.25)
#     # q3_3 = month_data_3.quantile(0.75)
#     # iqr_3 = q3_3 - q1_3
#     # up_3 = q3_3 + 3.0 * iqr_3
#     # low_3 = max(0, q1_3 - 3.0 * iqr_3)

#     # outliers_elec_3 += ((month_data_3 > up_3) | (month_data_3 < low_3)).sum()
#     # df.loc[mask, 'elec_3_clean'] = np.where(
#     #     df.loc[mask, 'elec_3'] > up_3, up_3,
#     #     np.where(df.loc[mask, 'elec_3'] < low_3, low_3, df.loc[mask, 'elec_3'])
#     # )

# print(f"[OUTLIER] Đang bật: [CÁCH 1.2] IQR biến động theo từng THÁNG")
# print(f"  => Cột elec_1: Tổng số dòng bị thay thế (12 tháng): {outliers_elec_1} dòng ({outliers_elec_1/total_rows*100:.2f}%)")
# # print(f"  => Cột elec_2: Tổng số dòng bị thay thế (12 tháng): {outliers_elec_2} dòng ({outliers_elec_2/total_rows*100:.2f}%)")
# # print(f"  => Cột elec_3: Tổng số dòng bị thay thế (12 tháng): {outliers_elec_3} dòng ({outliers_elec_3/total_rows*100:.2f}%)")

# df['elec_1'] = df['elec_1_clean']
# # df['elec_2'] = df['elec_2_clean']
# # df['elec_3'] = df['elec_3_clean']
# df.drop(columns=['elec_1_clean'], inplace=True)

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

# # import pandas as pd
# # import numpy as np

# # # 1. Đọc dữ liệu gốc
# # df_path = r'D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_missing\master_H07_elec_missing.csv'
# # horaire_path = r'D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\cadence\horaire_all_years.csv'  # Đổi lại đường dẫn tuyệt đối nếu cần

# # df = pd.read_csv(df_path)
# # df_horaire = pd.read_csv(horaire_path)
# # total_rows = len(df)

# # print("="*60)
# # print("BÁO CÁO THỐNG KÊ XỬ LÝ OUTLIERS THEO TỪNG PHƯƠNG ÁN")
# # print("="*60)

# # # =====================================================================
# # # CHUẨN BỊ VÀ GỘP GIỜ MỞ/ĐÓNG THỰC TẾ CỦA CÔNG VIÊN
# # # =====================================================================
# # # Đảm bảo cột date ở cả 2 dataframe đều ở định dạng datetime chính xác
# # df['date'] = pd.to_datetime(df['date'])
# # df_horaire['date'] = pd.to_datetime(df_horaire['date'], format='%m/%d/%Y')  # Định dạng MM/DD/YYYY từ file lịch

# # # Chuyển đổi h_ouv và h_ferm thành giờ nguyên (Integer)
# # df_horaire['h_ouv_parsed'] = pd.to_datetime(df_horaire['h_ouv'], format='%H:%M:%S', errors='coerce').dt.hour
# # df_horaire['h_ferm_parsed'] = pd.to_datetime(df_horaire['h_ferm'], format='%H:%M:%S', errors='coerce').dt.hour

# # # Tạo cột phụ chỉ chứa ngày để gộp chính xác
# # df['date_only'] = df['date'].dt.normalize()
# # df_horaire['date_only'] = df_horaire['date'].dt.normalize()

# # # Gộp thông tin giờ mở/đóng vào dataframe chính
# # df = df.merge(
# #     df_horaire[['date_only', 'h_ouv_parsed', 'h_ferm_parsed']], 
# #     on='date_only', 
# #     how='left'
# # )

# # # Gán giá trị fallback dự phòng nếu ngày đó bị thiếu dữ liệu lịch hoặc công viên đóng cửa
# # df['h_ouv_h07'] = df['h_ouv_parsed'].fillna(10)
# # df['h_ferm_h07'] = df['h_ferm_parsed'].fillna(18)

# # # Xóa các cột phụ trung gian
# # df.drop(columns=['date_only', 'h_ouv_parsed', 'h_ferm_parsed'], inplace=True)

# # # =====================================================================
# # # PHẦN 1: CÁC TÙY CHỌN XỬ LÝ OUTLIER CHO ELEC_1 VÀ ELEC_2 ĐỘC LẬP
# # # =====================================================================

# # # Chú ý: Hãy BẬT (uncomment) 1 trong 3 cách dưới đây để chạy.
# # # Cuối mỗi cách, hệ thống sẽ tự động đồng bộ lại cột tổng: df['ec_value'] = df['elec_1'] + df['elec_2']


# # # =====================================================================
# # # --- CÁCH 1.1: IQR PHẲNG TOÀN NĂM (Tách biệt từng cột) ---
# # # =====================================================================
# # """
# # print(f"[OUTLIER] Đang bật: [CÁCH 1.1] IQR phẳng toàn năm cho từng cột độc lập")

# # # --- Xử lý cho cột elec_1 ---
# # q1_1 = df['elec_1'].quantile(0.25)
# # q3_1 = df['elec_1'].quantile(0.75)
# # iqr_1 = q3_1 - q1_1
# # upper_1 = q3_1 + 5.0 * iqr_1
# # lower_1 = max(0, q1_1 - 5.0 * iqr_1)

# # outliers_up_1 = (df['elec_1'] > upper_1).sum()
# # outliers_low_1 = (df['elec_1'] < lower_1).sum()
# # total_outliers_1 = outliers_up_1 + outliers_low_1

# # print(f"  => Cột elec_1:")
# # print(f"     - Ngưỡng trần cố định: {upper_1:.4f} kWh")
# # print(f"     - Thay thế: {total_outliers_1} dòng ({total_outliers_1/total_rows*100:.2f}%)")

# # df['elec_1'] = np.where(df['elec_1'] > upper_1, upper_1, np.where(df['elec_1'] < lower_1, lower_1, df['elec_1']))

# # # --- Xử lý cho cột elec_2 ---
# # q1_2 = df['elec_2'].quantile(0.25)
# # q3_2 = df['elec_2'].quantile(0.75)
# # iqr_2 = q3_2 - q1_2
# # upper_2 = q3_2 + 5.0 * iqr_2
# # lower_2 = max(0, q1_2 - 5.0 * iqr_2)

# # outliers_up_2 = (df['elec_2'] > upper_2).sum()
# # outliers_low_2 = (df['elec_2'] < lower_2).sum()
# # total_outliers_2 = outliers_up_2 + outliers_low_2

# # print(f"  => Cột elec_2:")
# # print(f"     - Ngưỡng trần cố định: {upper_2:.4f} kWh")
# # print(f"     - Thay thế: {total_outliers_2} dòng ({total_outliers_2/total_rows*100:.2f}%)")

# # df['elec_2'] = np.where(df['elec_2'] > upper_2, upper_2, np.where(df['elec_2'] < lower_2, lower_2, df['elec_2']))

# # # --- Xử lý cho cột elec_3 ---
# # q1_3 = df['elec_3'].quantile(0.25)
# # q3_3 = df['elec_3'].quantile(0.75)
# # iqr_3 = q3_3 - q1_3
# # upper_3 = q3_3 + 5.0 * iqr_3
# # lower_3 = max(0, q1_3 - 5.0 * iqr_3)

# # outliers_up_3 = (df['elec_3'] > upper_3).sum()
# # outliers_low_3 = (df['elec_3'] < lower_3).sum()
# # total_outliers_3 = outliers_up_3 + outliers_low_3

# # print(f"  => Cột elec_3:")
# # print(f"     - Ngưỡng trần cố định: {upper_3:.4f} kWh")
# # print(f"     - Thay thế: {total_outliers_3} dòng ({total_outliers_3/total_rows*100:.2f}%)")

# # df['elec_3'] = np.where(df['elec_3'] > upper_3, upper_3, np.where(df['elec_3'] < lower_3, lower_3, df['elec_3']))


# # """


# # # =====================================================================
# # # --- CÁCH 1.2: IQR BIẾN ĐỘNG THEO TỪNG THÁNG (Tách biệt từng cột) ---
# # # =====================================================================
# # df['elec_1_clean'] = df['elec_1'].copy()
# # df['elec_2_clean'] = df['elec_2'].copy()
# # df['elec_3_clean'] = df['elec_3'].copy()
# # outliers_elec_1 = 0
# # outliers_elec_2 = 0
# # outliers_elec_3 = 0

# # for month in sorted(df['month'].unique()):
# #     mask = df['month'] == month
    
# #     # --- Xử lý tháng cho cột elec_1 ---
# #     month_data_1 = df.loc[mask, 'elec_1']
# #     q1_1 = month_data_1.quantile(0.25)
# #     q3_1 = month_data_1.quantile(0.75)
# #     iqr_1 = q3_1 - q1_1
# #     up_1 = q3_1 + 3.0* iqr_1
# #     low_1 = max(0, q1_1 - 3.0 * iqr_1)
    
# #     outliers_elec_1 += ((month_data_1 > up_1) | (month_data_1 < low_1)).sum()
# #     df.loc[mask, 'elec_1_clean'] = np.where(
# #         df.loc[mask, 'elec_1'] > up_1, up_1, 
# #         np.where(df.loc[mask, 'elec_1'] < low_1, low_1, df.loc[mask, 'elec_1'])
# #     )
    
# #     # --- Xử lý tháng cho cột elec_2 ---
# #     month_data_2 = df.loc[mask, 'elec_2']
# #     q1_2 = month_data_2.quantile(0.25)
# #     q3_2 = month_data_2.quantile(0.75)
# #     iqr_2 = q3_2 - q1_2
# #     up_2 = q3_2 + 3.0 * iqr_2
# #     low_2 = max(0, q1_2 - 3.0 * iqr_2)
    
# #     outliers_elec_2 += ((month_data_2 > up_2) | (month_data_2 < low_2)).sum()
# #     df.loc[mask, 'elec_2_clean'] = np.where(
# #         df.loc[mask, 'elec_2'] > up_2, up_2, 
# #         np.where(df.loc[mask, 'elec_2'] < low_2, low_2, df.loc[mask, 'elec_2'])
# #     )
    
# #     # --- Xử lý tháng cho cột elec_3 ---
# #     month_data_3 = df.loc[mask, 'elec_3']
# #     q1_3 = month_data_3.quantile(0.25)
# #     q3_3 = month_data_3.quantile(0.75)
# #     iqr_3 = q3_3 - q1_3
# #     up_3 = q3_3 + 3.0 * iqr_3
# #     low_3 = max(0, q1_3 - 3.0 * iqr_3)

# #     outliers_elec_3 += ((month_data_3 > up_3) | (month_data_3 < low_3)).sum()
# #     df.loc[mask, 'elec_3_clean'] = np.where(
# #         df.loc[mask, 'elec_3'] > up_3, up_3,
# #         np.where(df.loc[mask, 'elec_3'] < low_3, low_3, df.loc[mask, 'elec_3'])
# #     )

# # print(f"[OUTLIER] Đang bật: [CÁCH 1.2] IQR biến động theo từng THÁNG")
# # print(f"  => Cột elec_1: Tổng số dòng bị thay thế (12 tháng): {outliers_elec_1} dòng ({outliers_elec_1/total_rows*100:.2f}%)")
# # print(f"  => Cột elec_2: Tổng số dòng bị thay thế (12 tháng): {outliers_elec_2} dòng ({outliers_elec_2/total_rows*100:.2f}%)")
# # print(f"  => Cột elec_3: Tổng số dòng bị thay thế (12 tháng): {outliers_elec_3} dòng ({outliers_elec_3/total_rows*100:.2f}%)")

# # df['elec_1'] = df['elec_1_clean']
# # df['elec_2'] = df['elec_2_clean']
# # df['elec_3'] = df['elec_3_clean']
# # df.drop(columns=['elec_1_clean', 'elec_2_clean', 'elec_3_clean'], inplace=True)

# # # Đồng bộ lại cột tổng
# # # df['ec_value'] = df['elec_1'] + df['elec_2']


# # # =====================================================================
# # # --- CÁCH 1.3: HAMPEL FILTER / TRUNG VỊ TRƯỢT 24H (Tách biệt từng cột) ---
# # # =====================================================================
# # """
# # print(f"[OUTLIER] Đang bật: [CÁCH 1.3] Hampel Filter (Rolling Median 24h)")
# # window_size = 24

# # # --- Xử lý chuỗi thời gian cho cột elec_1 ---
# # rolling_median_1 = df['elec_1'].rolling(window=window_size, center=True, min_periods=1).median()
# # rolling_mad_1 = (df['elec_1'] - rolling_median_1).abs().rolling(window=window_size, center=True, min_periods=1).median()
# # upper_hampel_1 = rolling_median_1 + 3 * rolling_mad_1
# # lower_hampel_1 = np.maximum(0, rolling_median_1 - 3 * rolling_mad_1)

# # total_outliers_1 = ((df['elec_1'] > upper_hampel_1) | (df['elec_1'] < lower_hampel_1)).sum()
# # print(f"  => Cột elec_1: Thay thế {total_outliers_1} dòng lệch xu hướng ngày ({total_outliers_1/total_rows*100:.2f}%)")

# # df['elec_1'] = np.where(df['elec_1'] > upper_hampel_1, rolling_median_1, 
# #                         np.where(df['elec_1'] < lower_hampel_1, rolling_median_1, df['elec_1']))

# # # --- Xử lý chuỗi thời gian cho cột elec_2 ---
# # rolling_median_2 = df['elec_2'].rolling(window=window_size, center=True, min_periods=1).median()
# # rolling_mad_2 = (df['elec_2'] - rolling_median_2).abs().rolling(window=window_size, center=True, min_periods=1).median()
# # upper_hampel_2 = rolling_median_2 + 3 * rolling_mad_2
# # lower_hampel_2 = np.maximum(0, rolling_median_2 - 3 * rolling_mad_2)

# # total_outliers_2 = ((df['elec_2'] > upper_hampel_2) | (df['elec_2'] < lower_hampel_2)).sum()
# # print(f"  => Cột elec_2: Thay thế {total_outliers_2} dòng lệch xu hướng ngày ({total_outliers_2/total_rows*100:.2f}%)")

# # df['elec_2'] = np.where(df['elec_2'] > upper_hampel_2, rolling_median_2, 
# #                         np.where(df['elec_2'] < lower_hampel_2, rolling_median_2, df['elec_2']))

# # # --- Xử lý chuỗi thời gian cho cột elec_3 ---
# # rolling_median_3 = df['elec_3'].rolling(window=window_size, center=True, min_periods=1).median()
# # rolling_mad_3 = (df['elec_3'] - rolling_median_3).abs().rolling(window=window_size, center=True, min_periods=1).median()
# # upper_hampel_3 = rolling_median_3 + 3 * rolling_mad_3
# # lower_hampel_3 = np.maximum(0, rolling_median_3 - 3 * rolling_mad_3)
# # total_outliers_3 = ((df['elec_3'] > upper_hampel_3) | (df['elec_3'] < lower_hampel_3)).sum()
# # print(f"  => Cột elec_3: Thay thế {total_outliers_3} dòng lệch xu hướng ngày ({total_outliers_3/total_rows*100:.2f}%)")
# # df['elec_3'] = np.where(df['elec_3'] > upper_hampel_3, rolling_median_3,
# #                         np.where(df['elec_3'] < lower_hampel_3, rolling_median_3, df['elec_3']))
# # """

# # print("-" * 60)

# # # # =====================================================================
# # # # PHẦN 2: CÁC TÙY CHỌN XỬ LÝ VISITOR_COUNT (Tích hợp Logic Vận Hành)
# # # # =====================================================================

# # # # --- BƯỚC 2.1: ÉP LOGIC VẬN HÀNH THỰC TẾ (Đóng cửa & Khung giờ hoạt động thực tế từ lịch) ---

# # # # 1. Tạo điều kiện lọc khung giờ mở cửa động theo dữ liệu lịch thực tế của ngày đó
# # # # Khách chỉ có thể xuất hiện trong khoảng thời gian [h_ouv_h07, h_ferm_h07)
# # # valid_hours_mask = (df['hour'] >= df['h_ouv_h07']) & (df['hour'] < df['h_ferm_h07'])

# # # # 2. Kết hợp với điều kiện công viên mở cửa (is_open == 1)
# # # logic_valide = (df['is_open'] == 1) & valid_hours_mask

# # # # 3. Thống kê số dòng bị lỗi logic (Có khách khi đáng lẽ phải bằng 0)
# # # visitor_logic_faults = ((df['visitor_count'] > 0) & (~logic_valide)).sum()

# # # print(f"[VISITOR_COUNT] Kiểm tra Logic Vận Hành:")
# # # print(f"  - Số dòng có khách sai khung giờ hoặc sai ngày mở cửa: {visitor_logic_faults} dòng ({visitor_logic_faults/total_rows*100:.2f}%)")
# # # print(f"  => Tiến hành ép về 0 khách cho các dòng sai logic này.")

# # # # Thực hiện ép về 0 cho các dòng không hợp lệ
# # # df['visitor_count'] = np.where(logic_valide, df['visitor_count'], 0)
# # # print("-" * 40)


# # # # --- BƯỚC 2.2: LỰA CHỌN PHƯƠNG ÁN CHẶN TRẦN OUTLIER TOÁN HỌC ---

# # # # --- PHƯƠNG ÁN A: Chỉ chặn bằng Logic vật lý (Sức chứa hàng chờ tối đa = 750) ---
# # # max_physical_capacity = 750
# # # v_outliers = (df['visitor_count'] > max_physical_capacity).sum()
# # # print(f"[VISITOR_COUNT] Đang bật: [PHƯƠNG ÁN A] (Chặn logic vật lý hàng chờ)")
# # # print(f"  - Ngưỡng trần vật lý cố định: {max_physical_capacity} khách")
# # # print(f"  => Số dòng vượt ngưỡng vật lý bị hạ trần: {v_outliers} dòng ({v_outliers/total_rows*100:.2f}%)")
# # # df['visitor_count'] = np.where(df['visitor_count'] > max_physical_capacity, max_physical_capacity, df['visitor_count'])


# # # # --- PHƯƠNG ÁN B: Plafonnement IQR tiêu chuẩn 1.5x (Ngưỡng cắt = 265 khách) ---
# # # # upper_bound_v_15 = df['visitor_count'].quantile(0.75) + 1.5 * (df['visitor_count'].quantile(0.75) - df['visitor_count'].quantile(0.25))
# # # # v_outliers = (df['visitor_count'] > upper_bound_v_15).sum()
# # # # print(f"[VISITOR_COUNT] Đang bật: [PHƯƠNG ÁN B] (Plafonnement IQR 1.5x)")
# # # # print(f"  - Ngưỡng trần toán học: {upper_bound_v_15:.1f} khách")
# # # # print(f"  => Số dòng bị ép trần toán học: {v_outliers} dòng ({v_outliers/total_rows*100:.2f}%)")
# # # # df['visitor_count'] = np.where(df['visitor_count'] > upper_bound_v_15, upper_bound_v_15, df['visitor_count'])


# # # # --- PHƯƠNG ÁN C: Dung hòa - Plafonnement IQR nới rộng 3.0x (Ngưỡng cắt = 424 khách) ---
# # # # upper_bound_v_30 = df['visitor_count'].quantile(0.75) + 3.0 * (df['visitor_count'].quantile(0.75) - df['visitor_count'].quantile(0.25))
# # # # v_outliers = (df['visitor_count'] > upper_bound_v_30).sum()
# # # # print(f"[VISITOR_COUNT] Đang bật: [PHƯƠNG ÁN C] (Plafonnement IQR 3.0x)")
# # # # print(f"  - Ngưỡng trần dung hòa: {upper_bound_v_30:.1f} khách")
# # # # print(f"  => Số dòng bị ép trần dung hòa: {v_outliers} dòng ({v_outliers/total_rows*100:.2f}%)")
# # # # df['visitor_count'] = np.where(df['visitor_count'] > upper_bound_v_30, upper_bound_v_30, df['visitor_count'])


# # # # Đảm bảo lượng khách không bị âm (Chặn sàn dưới an toàn rộng)
# # # df['visitor_count'] = np.maximum(0, df['visitor_count'])

# # print("="*60)

# # # =====================================================================
# # # PHẦN 3: XUẤT FILE KẾT QUẢ ĐỂ ĐỐI CHỨNG
# # # =====================================================================
# # df.drop(columns=['h_ouv_h07', 'h_ferm_h07'], inplace=True)
# # df.to_csv(r'D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_outliers\master_H07_elec_outliers.csv', index=False)
# # print("\nĐã xử lý xong và xuất file dữ liệu thành công!")