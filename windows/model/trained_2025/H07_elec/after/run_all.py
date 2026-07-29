import subprocess
import os
import sys  # <--- Thêm thư viện này vào

# Danh sách các file mô hình của bạn
model_files = [
    # "01. linear.py",
    # "02. svm.py",
    # "03. rf.py",
    # "04. gbr.py",
    # "05. knn.py",
    # "06. xg_boost.py",
    "07. lstm.py",
    "08. gru.py",
    "09. ann_mlp.py"
]

# sys.executable sẽ lấy chính xác đường dẫn của python.exe trong môi trường (.py310)
python_executable = sys.executable 

for file in model_files:
    if os.path.exists(file):
        print(f"=== Đang chạy: {file} ===")
        try:
            # Thay vì dùng chữ "python" chung chung, ta dùng đường dẫn cụ thể
            result = subprocess.run([python_executable, file], check=True)
            print(f"=== Hoàn thành: {file} ===\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Lỗi khi chạy {file}: {e}\n")
    else:
        print(f"⚠ Không tìm thấy file: {file}")