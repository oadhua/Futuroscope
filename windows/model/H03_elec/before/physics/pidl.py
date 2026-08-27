import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim

import plotly.graph_objects as go
from plotly.subplots import make_subplots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Đang chạy PiDL đã vá lỗi vật lý trên thiết bị: {device}")

# =====================================================================
# 1. LOAD VÀ TIỀN XỬ LÝ DỮ LIỆU
# =====================================================================
file_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H03_predict_pure_elec.csv"
df = pd.read_csv(file_path)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

# Tính toán đạo hàm sai phân nhiệt độ theo thời gian (dT/dt) phục vụ cho C_b (Quán tính nhiệt)
df["temp_diff"] = df["temperature"].diff().fillna(0)

features_elec = [
    "year", "month", "day", "hour", "week", "is_weekend", "is_open", "jf",
    "frequentation_park_daily", "surface", "duree", "ouvert", "interrompu",
    "operation", "visitor_count", "temperature", "temp_diff", "humidite",
    "rayonnement_solaire", "day_degree_cold", "day_degree_hot",
    "temp_max", "temp_min", "temp_moy", "humidite_max", "humidite_min",
    "humidite_moy", "freq_HF", "freq_MF", "freq_THF"
]

train_mask = df["year"] < 2026
test_mask = df["year"] >= 2026

# Chuẩn hóa dữ liệu đầu vào cho Mạng Nơ-ron
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(df.loc[train_mask, features_elec])
X_test_scaled = scaler_X.transform(df.loc[test_mask, features_elec])

# Chuyển đổi sang PyTorch Tensors
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_t = torch.tensor(df.loc[train_mask, "elec_1"].values, dtype=torch.float32).unsqueeze(1).to(device)

X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_t = torch.tensor(df.loc[test_mask, "elec_1"].values, dtype=torch.float32).unsqueeze(1).to(device)

# Các biến vật lý thô (chưa chuẩn hóa) dùng cho Physics Loss
T_ext_train = torch.tensor(df.loc[train_mask, "temperature"].values, dtype=torch.float32).unsqueeze(1).to(device)
temp_diff_train = torch.tensor(df.loc[train_mask, "temp_diff"].values, dtype=torch.float32).unsqueeze(1).to(device)
visitors_train = torch.tensor(df.loc[train_mask, "visitor_count"].values, dtype=torch.float32).unsqueeze(1).to(device)
op_train = torch.tensor(df.loc[train_mask, "operation"].values, dtype=torch.float32).unsqueeze(1).to(device)
is_open_train = torch.tensor(df.loc[train_mask, "is_open"].values, dtype=torch.float32).unsqueeze(1).to(device)

test_dates = df.loc[test_mask, "date"] if "date" in df.columns else df.loc[test_mask].index


# =====================================================================
# 2. KIẾN TRÚC MẠNG NƠ-RON & HÀM LOSS VẬT LÝ CÓ CHẶN BIÊN (CLAMP)
# =====================================================================
class BuildingEnergyPiDL(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

class PhysicsInformedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Khởi tạo thông số vật lý ban đầu
        self.UA = nn.Parameter(torch.tensor([4.5], dtype=torch.float32))
        self.C_b = nn.Parameter(torch.tensor([15.0], dtype=torch.float32))
        self.COP = 3.5
        self.q_person = 0.12
        self.E_standby = 5.0
        self.mse = nn.MSELoss()

    def forward(self, pred_elec, true_elec, T_ext, temp_diff, visitors, operation, is_open, T_ref=24.0):
        # 1. ÉP BUỘC AN TOÀN VẬT LÝ: Chặn biên UA và C_b luôn dương tuyệt đối!
        UA_safe = torch.clamp(self.UA, min=0.5, max=30.0)
        Cb_safe = torch.clamp(self.C_b, min=1.0, max=100.0)

        # 2. Data Loss (Sai số so với dữ liệu thực tế)
        l_data = self.mse(pred_elec, true_elec)

        # 3. Physics Loss (Phương trình nhiệt động lực học tòa nhà)
        q_env = UA_safe * (T_ext - T_ref)          # Truyền nhiệt vỏ bao che
        q_inertia = Cb_safe * temp_diff            # Quán tính nhiệt tích trữ
        q_occ = self.q_person * visitors * operation # Nhiệt lượng do con người

        e_physics_calc = torch.relu(q_env + q_inertia + q_occ) / self.COP + self.E_standby

        # Xử lý trạng thái đóng cửa / dừng hoạt động
        e_physics_target = torch.where(
            (is_open > 0) & (operation > 0),
            e_physics_calc,
            torch.tensor(self.E_standby, device=pred_elec.device)
        )

        l_physics = self.mse(pred_elec, e_physics_target)

        # Tổng Loss kết hợp Data-driven + Physics Constraint
        total_loss = l_data + 0.5 * l_physics
        return total_loss, l_data, l_physics


# =====================================================================
# 3. HUẤN LUYỆN MÔ HÌNH PiDL
# =====================================================================
model = BuildingEnergyPiDL(input_dim=len(features_elec)).to(device)
criterion = PhysicsInformedLoss().to(device)
optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=0.003)

epochs = 1200
print("\n🏋️ BẮT ĐẦU HUẤN LUYỆN PI-DL (ĐÃ KHÓA BIÊN VẬT LÝ)...")
print("-" * 75)

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()

    predictions = model(X_train_t)
    loss, l_data, l_phys = criterion(
        predictions, y_train_t, T_ext_train, temp_diff_train, 
        visitors_train, op_train, is_open_train
    )

    loss.backward()
    optimizer.step()

    if epoch % 150 == 0 or epoch == epochs:
        print(f"Epoch {epoch:4d}/{epochs} | Loss Total: {loss.item():.4f} | "
              f"Data MSE: {l_data.item():.4f} | Physics Res: {l_phys.item():.4f} | "
              f"UA: {torch.clamp(criterion.UA, 0.5, 30.0).item():.3f} kW/°C | "
              f"C_b: {torch.clamp(criterion.C_b, 1.0, 100.0).item():.3f} kWh/°C")

print("-" * 75)
learned_UA = float(torch.clamp(criterion.UA, 0.5, 30.0).item())
learned_Cb = float(torch.clamp(criterion.C_b, 1.0, 100.0).item())
print(f"✅ HOÀN TẤT HIỆU CHỈNH THAM SỐ VẬT LÝ:")
print(f"  ► Hệ số truyền nhiệt vỏ bao che (UA) : {learned_UA:.3f} kW/°C (Đã chuẩn dương)")
print(f"  ► Quán tính nhiệt tòa nhà (C_b)       : {learned_Cb:.3f} kWh/°C (Đã động học)")


# =====================================================================
# 4. DỰ BÁO VÀ ĐÁNH GIÁ CHỈ SỐ (METRICS: R2, MAE, RMSE)
# =====================================================================
model.eval()
with torch.no_grad():
    pred_test_tensor = model(X_test_t)
    pred_elec_1 = pred_test_tensor.cpu().numpy().flatten()

pred_elec_1 = np.maximum(pred_elec_1, 5.0) # Chặn dưới năng lượng standby
Y1_test = df.loc[test_mask, "elec_1"].values

rmse = np.sqrt(mean_squared_error(Y1_test, pred_elec_1))
mae = mean_absolute_error(Y1_test, pred_elec_1)
r2 = r2_score(Y1_test, pred_elec_1)

print("\n=========================================================")
print("--- ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH PiDL TRÊN TẬP TEST (2026) ---")
print(f"[elec_1] RMSE : {rmse:.4f} kWh")
print(f"[elec_1] MAE  : {mae:.4f} kWh")
print(f"[elec_1] R²   : {r2:.4f}")
print("=========================================================\n")


# =====================================================================
# 5. TRỰC QUAN HÓA BẰNG PLOTLY VÀ XUẤT FILE KẾT QUẢ
# =====================================================================
subtitle_pidl = f"<b>elec_1 : Réel vs PiDL (Physics-Informed Deep Learning)</b><br><span style='font-size: 11px; color: #555;'>RMSE: {rmse:.2f} kWh | MAE: {mae:.2f} kWh | R²: {r2:.4f} | UA: {learned_UA:.2f}</span>"

fig = make_subplots(rows=1, cols=1)

fig.add_trace(go.Scatter(
    x=test_dates, y=Y1_test, mode="lines",
    name="elec_1 Réel", line=dict(color="#1f77b4", width=1.5),
    hovertemplate="<b>Horodatage:</b> %{x}<br><b>Réel:</b> %{y:.2f} kWh<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=test_dates, y=pred_elec_1, mode="lines",
    name="elec_1 Prédiction (PiDL)", line=dict(color="#d62728", width=1.5, dash="dot"),
    customdata=np.stack((Y1_test, np.abs(Y1_test - pred_elec_1)), axis=-1),
    hovertemplate="<b>Horodatage:</b> %{x}<br><b>PiDL Pred:</b> %{y:.2f} kWh<br><b>Écart:</b> %{customdata[1]:.2f} kWh<extra></extra>"
))

fig.update_layout(
    height=550, margin=dict(t=80, b=60, l=60, r=40),
    title_text=f"<b>{subtitle_pidl}</b>", title_x=0.5,
    hovermode="x unified", template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.update_yaxes(title_text="Consommation (kWh)")
fig.update_xaxes(rangeslider_visible=True)

output_html = "05_pidl_interactive.html"
fig.write_html(output_html, include_plotlyjs="cdn")
print(f"[OK] Đồ thị tương tác PiDL đã xuất thành công: {os.path.abspath(output_html)}")

# Xuất tệp CSV kết quả chuẩn đầu vào cho bộ SLSQP Optimizer
df_results = df.loc[test_mask, ["date"] + [f for f in features_elec if f != "temp_diff"]].copy()
df_results["elec_1_real"] = Y1_test
df_results["elec_1_pred"] = np.round(pred_elec_1, 2)
df_results["elec_1_error_abs"] = np.abs(Y1_test - pred_elec_1)

output_csv = "predict_2026.csv"
df_results.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"[OK] File dự báo chuẩn cho Optimizer đã lưu: {os.path.abspath(output_csv)}")

fig.show()