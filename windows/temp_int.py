import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
# ==========================================
def load_data(filepath):
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])

    # Lọc riêng dữ liệu các tháng mùa đông (Tháng 12, 1, 2)
    winter_df = (
        df[df["month"].isin([10, 11, 12, 1, 2, 3])].sort_values("date").reset_index(drop=True)
    )
    return winter_df


# ==========================================
# 2. MÔ HÌNH VẬT LÝ VỚI MẠNG RC (GREY-BOX)
# ==========================================
def simulate_indoor_temperature(
    data,
    R_eq=0.15,  # Điện trở nhiệt tương đương của tòa nhà (K/kW)
    C_eq=1000.0,  # Điện dung nhiệt tương đương của tòa nhà (kWh/K)
    COP=2.0,  # Hệ số hiệu suất sưởi (COP = 2.0 cho máy sưởi/bơm nhiệt)
    window_area=60.0,  # Diện tích cửa sổ thu nhiệt (m2)
    shgc=0.6,  # Hệ số hấp thụ bức xạ của kính (Solar Heat Gain Coefficient)
    q_person=0.08,  # Nhiệt lượng tỏa ra từ 1 người (kW/người ~ 80W)
    T_start=18.0,  # Nhiệt độ xuất phát ban đầu (°C)
):
    n = len(data)
    T_in = np.zeros(n)
    T_in[0] = T_start

    # Lấy các mảng dữ liệu đầu vào
    T_out = data["temperature"].values
    ec_val = data["ec_value"].values  # Điện năng tiêu thụ (kWh)
    solar = data["rayonnement_solaire"].values / 1000.0  # Đổi W/m2 -> kW/m2
    visitors = data["visitor_count"].values  # Số lượng khách

    dt = 1.0  # Bước thời gian: 1 giờ

    # Giải phương trình vi phân biến thiên nhiệt độ:
    # C * (dT_in/dt) = Q_heating + Q_solar + Q_internal - (T_in - T_out) / R
    for t in range(1, n):
        Q_heating = ec_val[t - 1] * COP
        Q_solar = solar[t - 1] * window_area * shgc
        Q_internal = visitors[t - 1] * q_person

        Q_gains = Q_heating + Q_solar + Q_internal
        Q_losses = (T_in[t - 1] - T_out[t - 1]) / R_eq

        dT_dt = (Q_gains - Q_losses) / C_eq
        T_in[t] = T_in[t - 1] + dT_dt * dt

    data["temperature_in_simulated"] = T_in
    return data


# ==========================================
# 3. CHẠY MÔ PHỎNG VÀ VẼ ĐỒ THỊ
# ==========================================
# Đọc file dữ liệu
df_winter = load_data("D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H03_predict_pure_thermal.csv")

# Chạy mô phỏng cho toàn bộ mùa đông
df_winter = simulate_indoor_temperature(df_winter)

# Lọc một khoảng thời gian 2 tuần mùa đông để trực quan hóa chi tiết (ví dụ: Đầu tháng 1/2025)
sample = df_winter[
    (df_winter["date"] >= "2024-12-01") & (df_winter["date"] < "2024-12-15")
].copy()

plt.figure(figsize=(14, 7))

# Đường nhiệt độ trong nhà mô phỏng
plt.plot(
    sample["date"],
    sample["temperature_in_simulated"],
    label="Nhiệt độ trong nhà mô phỏng ($T_{in}$)",
    color="crimson",
    linewidth=2.5,
)

# Đường nhiệt độ ngoài trời
plt.plot(
    sample["date"],
    sample["temperature"],
    label="Nhiệt độ ngoài trời ($T_{out}$)",
    color="royalblue",
    alpha=0.6,
    linestyle="-",
)

# Vùng mức yêu cầu tối thiểu duy trì (18°C)
plt.axhline(
    y=18.0,
    color="darkorange",
    linestyle="--",
    linewidth=1.5,
    label="Mục tiêu sưởi (18°C)",
)

# Trực quan hóa trạng thái mở cửa tòa nhà
plt.fill_between(
    sample["date"],
    0,
    30,
    where=(sample["is_open"] == 1),
    color="gold",
    alpha=0.15,
    label="Thời gian mở cửa (is_open = 1)",
)

plt.title(
    "MÔ PHỎNG NHIỆT ĐỘ TÒA NHÀ TRONG MÙA ĐÔNG (1300 m²)", fontsize=14, fontweight="bold"
)
plt.xlabel("Thời gian", fontsize=12)
plt.ylabel("Nhiệt độ (°C)", fontsize=12)
plt.ylim(sample["temperature"].min() - 2, sample["temperature_in_simulated"].max() + 5)
plt.legend(loc="upper right", frameon=True)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# ==========================================
# 4. ĐÁNH GIÁ KẾT QUẢ KHI TÒA NHÀ MỞ CỬA
# ==========================================
open_hours = df_winter[df_winter["is_open"] == 1]
print("--- THỐNG KÊ NHIỆT ĐỘ MÔ PHỎNG KHI MỞ CỬA MÙA ĐÔNG ---")
print(
    f"Nhiệt độ trung bình trong nhà : {open_hours['temperature_in_simulated'].mean():.2f} °C"
)
print(
    f"Nhiệt độ thấp nhất            : {open_hours['temperature_in_simulated'].min():.2f} °C"
)
print(
    f"Nhiệt độ cao nhất             : {open_hours['temperature_in_simulated'].max():.2f} °C"
)
