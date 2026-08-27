import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================================
# 0. CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU DÀNH CHO H07 THERMAL
# =====================================================================
INPUT_PATH = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_missing\H07_thermal_missing.csv"
OUTPUT_DIR = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_outliers"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "H07_thermal_outliers.csv")

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Không tìm thấy file đầu vào: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)
total_rows = len(df)

if "datetime" in df.columns:
    df["datetime_dt"] = pd.to_datetime(df["datetime"])
else:
    df["datetime_dt"] = pd.to_datetime(df["date"].astype(str) + " " + df["hour"].astype(str) + ":00:00")

df["month"] = df["datetime_dt"].dt.month
df["day"] = df["datetime_dt"].dt.day

print("=" * 70)
print("BÁO CÁO TỔNG HỢP XỬ LÝ OUTLIERS & MÙA SƯỞI (H07 THERMAL - ĐA TẦNG BÙ DỰ PHÒNG)")
print("=" * 70)


# =====================================================================
# PHẦN 1: KHÔI PHỤC LỖI CẢM BIẾN & TÁI TẠO XU HƯỚNG (H07 THERMAL)
# =====================================================================
TARGET_COL = "ec_value"

if TARGET_COL not in df.columns:
    raise KeyError(f"Không tìm thấy cột '{TARGET_COL}' trong file H07_thermal_missing.csv!")

print(f"\n[PHẦN 1] TÁI TẠO DỮ LIỆU CẢM BIẾN CHUẨN XU HƯỚNG CHO [{TARGET_COL}]...")

# 1. Phân lập Mùa Sưởi & Mùa Tắt Sưởi
mask_summer = (
    (df["month"] > 5) & (df["month"] < 10)
) | (
    (df["month"] == 5) & (df["day"] >= 15)
) | (
    (df["month"] == 10) & (df["day"] <= 15)
)
mask_winter = ~mask_summer

# 2. Phát hiện Flatline lỗi
def detect_flatlines(series, mask_winter_series, min_duration=4):
    diff = series.diff().ne(0)
    run_id = diff.cumsum()
    run_lengths = series.groupby(run_id).transform("count")
    
    flat_positive = (run_lengths >= min_duration) & (series > 0)
    flat_zero_winter = (run_lengths >= min_duration) & (series == 0) & mask_winter_series
    return flat_positive | flat_zero_winter

mask_negative = df[TARGET_COL] < 0.0
mask_flatline = detect_flatlines(df[TARGET_COL], mask_winter, min_duration=4)

MAX_THERMAL_CAPACITY_H07 = 285.0

group_cols = ["type_frequentation", "is_open", "hour"]
valid_group_cols = [c for c in group_cols if c in df.columns]

if valid_group_cols:
    grp_q1 = df.groupby(valid_group_cols)[TARGET_COL].transform(lambda x: x.quantile(0.25))
    grp_q3 = df.groupby(valid_group_cols)[TARGET_COL].transform(lambda x: x.quantile(0.75))
    grp_iqr = grp_q3 - grp_q1
    dynamic_upper = np.maximum(grp_q3 + 8.0 * grp_iqr, MAX_THERMAL_CAPACITY_H07)
    mask_profile_outlier = (df[TARGET_COL] > dynamic_upper) & mask_winter
else:
    mask_profile_outlier = (df[TARGET_COL] > MAX_THERMAL_CAPACITY_H07) & mask_winter

mask_summer_noise = mask_summer & (df[TARGET_COL] > 0.5)

mask_anomaly = mask_negative | mask_flatline | mask_profile_outlier | mask_summer_noise
total_anomalies = mask_anomaly.sum()

print(f"  -> Phát hiện {total_anomalies} điểm lỗi thực sự ({total_anomalies/total_rows*100:.2f}% tổng dữ liệu).")

# 3. THỰC HIỆN TÁI TẠO XU HƯỚNG BẰNG THUẬT TOÁN ĐA TẦNG (MULTI-TIER IMPUTATION)
df.loc[mask_anomaly, TARGET_COL] = np.nan

# Tầng 1: Mùa hè mặc định ép về 0.0
df.loc[mask_summer & df[TARGET_COL].isna(), TARGET_COL] = 0.0

# Tầng 2: Điền Mùa Lạnh theo Profile Chi Tiết [month, type_frequentation, is_open, hour] bằng MEAN()
profile_keys = ["month", "type_frequentation", "is_open", "hour"]
valid_profile_keys = [k for k in profile_keys if k in df.columns]

if valid_profile_keys:
    mean_profile_l1 = df[mask_winter].groupby(valid_profile_keys)[TARGET_COL].transform("mean")
    df[TARGET_COL] = df[TARGET_COL].fillna(mean_profile_l1)

# Tầng 3: Dự phòng Cấp 1 theo Profile Rộng Hơn [is_open, hour]
backup_keys_l2 = [k for k in ["is_open", "hour"] if k in df.columns]
if backup_keys_l2 and df[TARGET_COL].isna().sum() > 0:
    mean_profile_l2 = df[mask_winter].groupby(backup_keys_l2)[TARGET_COL].transform("mean")
    df[TARGET_COL] = df[TARGET_COL].fillna(mean_profile_l2)

# Tầng 4: Dự phòng Cấp 2 theo Khung Giờ [hour]
if "hour" in df.columns and df[TARGET_COL].isna().sum() > 0:
    mean_profile_l3 = df[mask_winter].groupby("hour")[TARGET_COL].transform("mean")
    df[TARGET_COL] = df[TARGET_COL].fillna(mean_profile_l3)

# Tầng 5: Nội suy thời gian làm mượt nhẹ hai chiều
s_temp = df.set_index("datetime_dt")[TARGET_COL]
s_temp = s_temp.interpolate(method="time", limit_direction="both")
df[TARGET_COL] = s_temp.values

# Tầng 6: Vét cạn bằng Mean toàn hệ thống nếu còn thiếu ở đầu/cuối chuỗi
if df[TARGET_COL].isna().sum() > 0:
    df[TARGET_COL] = df[TARGET_COL].fillna(df[TARGET_COL].mean())

df[TARGET_COL] = df[TARGET_COL].clip(lower=0.0)

print(f"  -> Số lượng NaN ec_value còn lại sau xử lý: {df[TARGET_COL].isna().sum()} ô.")
print("  ✅ Đã tái tạo xong xu hướng dữ liệu cho cột ec_value H07!")


# =====================================================================
# PHẦN 2: XỬ LÝ OUTLIERS CHO VISITOR_COUNT H07 (GIỚI HẠN <= 10/02/2024)
# =====================================================================
cutoff_date = pd.to_datetime("2024-02-10")
date_mask = df["datetime_dt"].dt.floor("D") <= cutoff_date
total_target_rows = date_mask.sum()

ouvert_signal = df.get("ouvert", 0) > 0
operation_signal = df.get("operation", 0) > 0
has_any_operation = ouvert_signal | operation_signal

mask_pure_closed = (df["is_open"] == 0) & (~has_any_operation)

mask_logic_error = date_mask & (df["visitor_count"] > 0) & mask_pure_closed
outliers_logic = mask_logic_error.sum()
df.loc[date_mask & mask_pure_closed, "visitor_count"] = 0

max_physical_capacity = 1000  
mask_capacity_error = date_mask & (df["visitor_count"] > max_physical_capacity)
outliers_capacity = mask_capacity_error.sum()

df.loc[date_mask, "visitor_count"] = np.where(
    df.loc[date_mask, "visitor_count"] > max_physical_capacity,
    max_physical_capacity,
    df.loc[date_mask, "visitor_count"],
)
df.loc[date_mask, "visitor_count"] = np.maximum(0, df.loc[date_mask, "visitor_count"])

df.drop(columns=["datetime_dt"], inplace=True, errors="ignore")


# =====================================================================
# PHẦN 3: XUẤT FILE KẾT QUẢ FOR H07 THERMAL
# =====================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n✅ Đã xuất file H07 Thermal thành công tại:\n -> {OUTPUT_PATH}")


# =====================================================================
# PHẦN 4: VẼ BIỂU ĐỒ TƯƠNG TÁC PLOTLY
# =====================================================================
df_raw = pd.read_csv(INPUT_PATH)
df_clean = pd.read_csv(OUTPUT_PATH)

df_raw["datetime"] = pd.to_datetime(df_raw["date"].astype(str) + " " + df_raw["hour"].astype(str) + ":00:00")
df_clean["datetime"] = pd.to_datetime(df_clean["date"].astype(str) + " " + df_clean["hour"].astype(str) + ":00:00")

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        "1. Tiêu Thụ Nhiệt (ec_value - H07 Thermal)",
        "2. Lượt Khách Visitor Count (Lọc <= 10/02/2024)",
    ),
)

fig.add_trace(
    go.Scatter(
        x=df_raw["datetime"],
        y=df_raw[TARGET_COL],
        name=f"Raw {TARGET_COL}",
        line=dict(color="crimson", width=1),
        opacity=0.35,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_clean["datetime"],
        y=df_clean[TARGET_COL],
        name=f"Clean {TARGET_COL}",
        line=dict(color="royalblue", width=1.5),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=df_raw["datetime"],
        y=df_raw["visitor_count"],
        name="Raw Visitor",
        line=dict(color="mediumpurple", width=1),
        opacity=0.35,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_clean["datetime"],
        y=df_clean["visitor_count"],
        name="Clean Visitor",
        line=dict(color="midnightblue", width=1.5),
    ),
    row=2,
    col=1,
)

fig.add_vline(
    x=pd.to_datetime("2024-02-10").timestamp() * 1000,
    line_width=2,
    line_dash="dash",
    line_color="red",
    annotation_text="Cutoff (10/02/2024)",
    annotation_position="top left",
    row=2,
    col=1,
)

fig.update_layout(
    height=650,
    title_text="<b>ĐỐI CHỨNG DỮ LIỆU CẢM BIẾN NHIỆT EC_VALUE & LƯỢT KHÁCH (H07 THERMAL)</b>",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

fig.update_xaxes(rangeslider_visible=True, row=2, col=1)

html_output_path = os.path.join(OUTPUT_DIR, "H07_thermal_outliers_interactive.html")
fig.write_html(html_output_path)
print(f"✅ Đã tạo xong biểu đồ HTML tương tác H07 Thermal tại:\n -> {html_output_path}")

fig.show()