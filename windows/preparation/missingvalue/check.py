import pandas as pd

# 1. Đọc dữ liệu từ file master_missing
df_path = r'D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_missing\master_H07_elec_missing.csv'
df = pd.read_csv(df_path)

cols_to_check = ['elec_1', 'elec_2', 'elec_3']

print("="*80)
print("DANH SÁCH CÁC DÒNG BỊ LỖI KIỂU CHỮ (STRING) TRONG CỘT ĐIỆN:")
print("="*80)

any_error = False

for col in cols_to_check:
    if col in df.columns:
        # Tìm các giá trị không thể chuyển đổi thành số (bị lỗi chữ, khoảng trắng, ký tự đặc biệt)
        # pd.to_numeric với errors='coerce' sẽ biến các giá trị lỗi thành NaN
        coerced = pd.to_numeric(df[col], errors='coerce')
        
        # Những dòng mà dữ liệu gốc KHÔNG phải rỗng (notnull) nhưng sau khi ép số lại biến thành NaN (isnull)
        # chính là các dòng bị lỗi định dạng string
        error_mask = df[col].notnull() & coerced.isnull()
        
        error_rows = df[error_mask]
        
        if len(error_rows) > 0:
            any_error = True
            print(f"\n[!] Cột '{col}' phát hiện {len(error_rows)} dòng chứa giá trị lỗi kiểu String:")
            print("-" * 80)
            # In ra các thông tin nhận diện dòng lỗi: Chỉ số dòng, Date, Hour và Giá trị thực tế đang bị lỗi
            for idx, row in error_rows.iterrows():
                # Cộng thêm 2 vào index vì file CSV bắt đầu từ dòng 2 trong Excel (dòng 1 là header)
                excel_row = idx + 2 
                print(f"  - Dòng Excel: {excel_row:<6} | Index Pandas: {idx:<5} | Ngày: {row['date']} (Giờ: {row['hour']}) | Giá trị lỗi: '{row[col]}'")
        else:
            print(f"  - Cột '{col}': Sạch! Không có dòng nào bị lỗi kiểu chữ.")

if not any_error:
    print("\n[OK] Không phát hiện giá trị chữ nào bị lẫn vào 3 cột điện!")
print("="*80)