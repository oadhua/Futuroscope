import subprocess
import os
import sys  # <--- Thêm thư viện này vào

# Danh sách các file mô hình của bạn
model_files = [
    "01. linear.py",
    "03. svm.py",
    "04. rf.py",
    "05. gb.py",
    "06. knn.py",
    "07. xg_boost.py",
    # "08. lstm.py",
    # "09. gru.py",
    # "10. ann_mlp.py"
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