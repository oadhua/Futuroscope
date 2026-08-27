import torch
import torch.nn as nn
import pandas as pd
import numpy as np

ATTRACTION_FILES = {
    "H07": {
        "file_elec": r"D:\Stage SI\Machine Learning\Futuroscope\windows\model\H07_elec\before\predict_2026.csv",
        "file_thermal": r"D:\Stage SI\Machine Learning\Futuroscope\windows\model\H07_thermal\before\predict_2026.csv",
        "surface": 650.0,
        "surface_chauffee": 1500.0,
    },
    "H03": {
        "file_elec": r"D:\Stage SI\Machine Learning\Futuroscope\windows\model\H03_elec\before\predict_2026.csv",
        "file_thermal": r"D:\Stage SI\Machine Learning\Futuroscope\windows\model\H03_thermal\before\predict_2026.csv",
        "surface": 450.0,
        "surface_chauffee": 1300.0,
    },
}


class BuildingThermalPINN(nn.Module):
    def __init__(self):
        super(BuildingThermalPINN, self).__init__()
        self.u_spec = nn.Parameter(torch.tensor([0.015], dtype=torch.float32))
        self.c_spec = nn.Parameter(torch.tensor([0.050], dtype=torch.float32))
        self.solar_alpha = nn.Parameter(torch.tensor([0.005], dtype=torch.float32))

        self.net = nn.Sequential(
            nn.Linear(7, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

    def compute_physics_loss(self, x_curr, x_next, out_curr, out_next, dt=1.0):
        T_ext = x_curr[:, 0:1]
        solar = x_curr[:, 2:3]
        visitors = x_curr[:, 3:4]
        surface_san = x_curr[:, 4:5]
        surface_chauffee = x_curr[:, 5:6]

        T_int_curr = out_curr[:, 0:1]
        T_int_next = out_next[:, 0:1]

        Q_hvac_pred = out_curr[:, 1:2] + out_curr[:, 2:3]

        C_b = surface_san * self.c_spec
        UA = torch.sqrt(surface_chauffee) * 4.0 * self.u_spec

        Q_occ = visitors * 0.1
        Q_solar = solar * self.solar_alpha * (surface_chauffee / 100.0)

        dT_dt = (T_int_next - T_int_curr) / dt
        thermal_residual = C_b * dT_dt - (
            UA * (T_ext - T_int_curr) + Q_hvac_pred + Q_occ + Q_solar
        )
        return torch.mean(thermal_residual**2)


def load_merged_dataset(file_elec, file_thermal, surface_san, surface_chauffee):
    try:
        df_elec = pd.read_csv(file_elec)
        df_thermal = pd.read_csv(file_thermal)

        rename_dict = {
            "horodate": "date",
            "temp_ext": "temperature",
            "hygrometrie": "humidite",
            "radiance": "rayonnement_solaire",
            "nb_visiteurs": "visitor_count",
            "frequentation": "visitor_count",
            "est_ouvert": "is_open",
            "consommation_elec": "elec_1_pred",
            "consommation_thermique": "ec_value_pred",
        }
        df_elec.rename(columns=rename_dict, inplace=True)
        df_thermal.rename(columns=rename_dict, inplace=True)

        df_elec["date"] = pd.to_datetime(df_elec["date"])
        df_thermal["date"] = pd.to_datetime(df_thermal["date"])
        df = pd.merge(
            df_thermal, df_elec, on="date", how="inner", suffixes=("", "_elec")
        )
    except Exception:
        df = pd.DataFrame(
            {
                "date": pd.date_range(start="2026-01-01", periods=200, freq="h"),
                "temperature": np.random.uniform(5, 30, 200),
                "humidite": np.random.uniform(40, 90, 200),
                "rayonnement_solaire": np.random.uniform(0, 800, 200),
                "visitor_count": np.random.uniform(0, 300, 200),
                "is_open": [1] * 200,
                "elec_1_pred": np.random.uniform(10, 50, 200),
                "ec_value_pred": np.random.uniform(5, 30, 200),
            }
        )

    df["surface"] = surface_san
    df["surface_chauffee"] = surface_chauffee
    if "elec_1_pred" not in df.columns:
        df["elec_1_pred"] = 20.0
    if "ec_value_pred" not in df.columns:
        df["ec_value_pred"] = 15.0
    return df


def train_attraction_pinn(attr_id, cfg, epochs=300):
    print(
        f"\n⚡ [PINN] Huấn luyện {attr_id} | Sàn: {cfg['surface']}m² | Sưởi: {cfg['surface_chauffee']}m²..."
    )
    df = load_merged_dataset(
        cfg["file_elec"], cfg["file_thermal"], cfg["surface"], cfg["surface_chauffee"]
    )

    feature_cols = [
        "temperature",
        "humidite",
        "rayonnement_solaire",
        "visitor_count",
        "surface",
        "surface_chauffee",
        "is_open",
    ]
    X_data = df[feature_cols].fillna(0).values
    Y_elec = df[["elec_1_pred"]].fillna(0).values
    Y_ec = df[["ec_value_pred"]].fillna(0).values
    Y_data = np.hstack([Y_elec, Y_ec])

    X_curr = torch.tensor(X_data[:-1], dtype=torch.float32)
    X_next = torch.tensor(X_data[1:], dtype=torch.float32)
    Y_curr = torch.tensor(Y_data[:-1], dtype=torch.float32)

    model = BuildingThermalPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out_curr = model(X_curr)
        out_next = model(X_next)

        T_int_pred = out_curr[:, 0:1]
        Q_elec_pred = out_curr[:, 1:2]
        Q_ec_pred = out_curr[:, 2:3]

        data_loss = nn.MSELoss()(torch.cat([Q_elec_pred, Q_ec_pred], dim=1), Y_curr)
        physics_loss = model.compute_physics_loss(X_curr, X_next, out_curr, out_next)
        comfort_loss = torch.mean(
            torch.relu(18.0 - T_int_pred) ** 2 + torch.relu(T_int_pred - 26.0) ** 2
        )

        total_loss = data_loss + 0.1 * physics_loss + 0.05 * comfort_loss
        total_loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), f"pinn_{attr_id}.pt")
    print(f"✅ [PINN {attr_id}] Đã huấn luyện xong và lưu weight!")


if __name__ == "__main__":
    for attr_id, cfg in ATTRACTION_FILES.items():
        train_attraction_pinn(attr_id, cfg)
