import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# =====================================================================
# BƯỚC 1: ĐỌC DỮ LIỆU CHUẨN TIME SERIES
# =====================================================================
file_path = r'D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_missing\master_H03_missing.csv'

# Đọc file và ép cột 'date' làm Datetime Index
df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')

# Sắp xếp lại Index theo đúng trình tự thời gian tăng dần
df = df.sort_index()

# Đảm bảo tần suất dữ liệu là theo giờ ('H') để thuật toán phân rã chạy mượt mà
df = df.asfreq('H')

# Nếu có ô trống phát sinh sau khi đổi tần suất, điền bằng giá trị trước đó (Forward Fill)
df['ec_value'] = df['ec_value'].ffill()
df['visitor_count'] = df['visitor_count'].ffill()

print("="*60)
print("1. THÔNG TIN TẬP DỮ LIỆU CHUỖI THỜI GIAN")
print("="*60)
print(f"Khoảng thời gian: Từ {df.index.min()} đến {df.index.max()}")
print(f"Tổng số giờ quan sát: {len(df)}")
print("-" * 60)


# =====================================================================
# BƯỚC 2: KIỂM TRA TÍNH DỪNG (STATIONARITY TEST - ADF TEST)
# =====================================================================
print("\n" + "="*60)
print("2. KIỂM TRA TÍNH DỪNG (AUGMENTED DICKEY-FULLER TEST)")
print("="*60)

def run_adf_test(series, name):
    print(f" Đang kiểm tra cột: [{name}]")
    result = adfuller(series.dropna())
    print(f"  - ADF Statistic: {result[0]:.4f}")
    print(f"  - p-value: {result[1]:.4f}")
    print("  - Critical Values (Các giá trị tới hạn):")
    for key, value in result[4].items():
        print(f"    {key}: {value:.4f}")
        
    if result[1] <= 0.05:
        print(f"  => Kết luận: p-value <= 0.05. Chuỗi dữ liệu [{name}] ĐÃ ĐẠT TÍNH DỪNG (Stationary).")
    else:
        print(f"  => Kết luận: p-value > 0.05. Chuỗi dữ liệu [{name}] KHÔNG DỪNG (Non-Stationary). Cần lấy sai phân nếu dùng ARIMA.")
    print("-" * 40)

# Chạy kiểm tra cho cả biến mục tiêu (Điện) và biến tương quan (Khách)
run_adf_test(df['ec_value'], 'ec_value')
run_adf_test(df['visitor_count'], 'visitor_count')


# =====================================================================
# BƯỚC 3: PHÂN RÃ CHUỖI THỜI GIAN (TIME SERIES DECOMPOSITION)
# =====================================================================
# Sử dụng chu kỳ sinh học 24 giờ (1 ngày) để tách biệt Trend và Seasonality trong ngày
decomposition = seasonal_decompose(df['ec_value'], model='additive', period=24)

# Vẽ biểu đồ phân rã
fig_decomp = decomposition.plot()
fig_decomp.set_size_inches(14, 10)
fig_decomp.suptitle('Phân rã Chuỗi Thời Gian Điện Năng Tiêu Thụ (ec_value) - Chu kỳ 24h', fontsize=16, y=1.02)
plt.tight_layout()


# =====================================================================
# BƯỚC 4: TÍNH TOÁN VÀ VẼ HÀM TỰ TƯƠNG QUAN (ACF & PACF)
# =====================================================================
# Xem xét độ trễ (lags) trong vòng 48 giờ (2 ngày) để tìm quy luật trễ giờ và trễ ngày
fig_corr, axes = plt.subplots(2, 1, figsize=(14, 10))

# Biểu đồ ACF
plot_acf(df['ec_value'], lags=48, ax=axes[0], title="Hàm Tự Tương Quan (ACF) - ec_value")
axes[0].grid(True, linestyle='--', alpha=0.6)

# Biểu đồ PACF
plot_pacf(df['ec_value'], lags=48, ax=axes[1], title="Hàm Tự Tương Quan Từng Phần (PACF) - ec_value")
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

# Hiển thị tất cả các biểu đồ đã vẽ
plt.show()

print("\n" + "="*60)
print("PHÂN TÍCH HOÀN TẤT! Hãy quan sát các biểu đồ hiển thị trên màn hình.")
print("="*60)