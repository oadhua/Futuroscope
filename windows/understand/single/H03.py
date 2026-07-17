# import pandas as pd
# from ydata_profiling import ProfileReport

# # 1. Đọc file dữ liệu của bạn
# df = pd.read_csv('D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master\master_H03_elec.csv')

# # 2. Tạo báo cáo (bạn có thể đặt tên cho dự án)
# profile = ProfileReport(df, title="Báo cáo EDA - Attraction H03 electricité", explorative=True)

# # 3. Xuất báo cáo ra file HTML để mở bằng trình duyệt
# profile.to_file("report_H03_elec.html")


# import pandas as pd
# from dataprep.eda import create_report

# # Đọc dữ liệu
# df = pd.read_csv(r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master\master_H03_elec.csv")

# # Tạo báo cáo
# report = create_report(df)

# # Lưu báo cáo thành file HTML (sẽ nằm cùng thư mục với file code của bạn)
# report.save("report_master_H03.html")

# print("Đã xuất báo cáo thành công! Bạn hãy mở file 'report_master_H03.html' bằng trình duyệt nhé.")

import numpy as np
import pandas as pd

# 1. Đọc dữ liệu
path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master\master_H07_elec.csv"
df = pd.read_csv(path)

total_rows = len(df)
print(f"Kích thước tập dữ liệu: {total_rows:,} dòng, {df.shape[1]} cột.\n")

# Tiêu đề bảng kết quả
print("-" * 95)
print(f"{'Tên Cột':<30} | {'Missing Count':<13} | {'Missing %':<10} | {'Outlier Count':<13} | {'Outlier %':<10}")
print("-" * 95)

# 2. Vòng lặp tính toán và hiển thị
for col in df.columns:
    # --- Tính Missing Value ---
    missing_count = df[col].isnull().sum()
    missing_pct = (missing_count / total_rows) * 100

    # --- Tính Outliers (chỉ áp dụng cho cột dữ liệu số) ---
    if pd.api.types.is_numeric_dtype(df[col]):
        col_clean = df[col].dropna()
        total_valid = len(col_clean) # Số lượng dòng thực tế không bị rỗng của cột

        if total_valid > 0:
            # Tính IQR
            q1 = col_clean.quantile(0.25)
            q3 = col_clean.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Đếm outlier
            outlier_count = ((col_clean < lower_bound) | (col_clean > upper_bound)).sum()
            # Tính % outlier dựa trên số lượng dữ liệu thực tế (đã loại trừ missing)
            outlier_pct = (outlier_count / total_valid) * 100
        else:
            outlier_count = 0
            outlier_pct = 0.0
        
        outlier_count_str = f"{outlier_count:,}"
        outlier_pct_str = f"{outlier_pct:.2f}%"
    else:
        outlier_count_str = "N/A"
        outlier_pct_str = "N/A"

    # In kết quả định dạng theo hàng thẳng lối
    print(f"{col:<30} | {missing_count:<13,} | {missing_pct:<10.2f}% | {outlier_count_str:<13} | {outlier_pct_str:<10}")

print("-" * 95)
