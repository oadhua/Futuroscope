import asyncio
import torch
import numpy as np
import pandas as pd
from asyncua import Client
from pinn_multi_attr import BuildingThermalPINN, ATTRACTION_FILES, load_merged_dataset

PANORAMA_URL = "opc.tcp://127.0.0.1:4840/freeopcua/server/"
URI_PANORAMA = "http://futuroscope.hvac.panorama"

GLOBAL_DATAFRAMES = {
    attr_id: load_merged_dataset(
        cfg["file_elec"], cfg["file_thermal"], cfg["surface"], cfg["surface_chauffee"]
    )
    for attr_id, cfg in ATTRACTION_FILES.items()
}

TOTAL_SAVINGS = {
    attr_id: {"elec_base": 0.0, "elec_opt": 0.0, "ec_base": 0.0, "ec_opt": 0.0}
    for attr_id in ATTRACTION_FILES
}


def get_attraction_open_status(row):
    """
    Xác định trạng thái mở cửa thực tế của điểm tham quan.
    """
    for col in ["operation", "ouvert", "attraction_open"]:
        if col in row and not pd.isna(row[col]):
            val = row[col]
            if isinstance(val, (int, float, np.number)):
                return int(val > 0)
            if isinstance(val, str):
                return 1 if val.strip().lower() in ["1", "true", "ouvert", "open"] else 0
            return int(bool(val))
    return int(row.get("is_open", 0))


class MultiAttrOptimizer:
    def __init__(self, attr_id):
        self.attr_id = attr_id
        self.pinn = BuildingThermalPINN()
        try:
            self.pinn.load_state_dict(
                torch.load(f"pinn_{attr_id}.pt", weights_only=True)
            )
        except Exception:
            pass
        self.pinn.eval()

    def optimize_consignes(
        self,
        df_window_24h,
        curr_temp,
        season="hiver",
    ):
        """
        Tối ưu hóa PINN và tính toán tiêu thụ năng lượng theo mô hình tương quan nhiệt thực tế.
        """
        row_curr = df_window_24h.iloc[0]
        attr_is_open = get_attraction_open_status(row_curr)
        visitors_curr = float(row_curr.get("visitor_count", 0))

        # 1. ANALYSE DU REGIME (Xác định trạng thái hoạt động)
        will_open_soon = False
        is_closing_soon = False

        if attr_is_open == 0:
            for ahead in range(1, 3):
                if ahead < len(df_window_24h):
                    if get_attraction_open_status(df_window_24h.iloc[ahead]) == 1:
                        will_open_soon = True
                        break
        else:
            if len(df_window_24h) > 1 and get_attraction_open_status(df_window_24h.iloc[1]) == 0:
                is_closing_soon = True

        # 2. KHỞI TẠO ĐIỂM CÀI ĐẶT CƠ SỞ (BASE) VÀ TỐI ƯU (OPT)
        if attr_is_open == 0:
            if will_open_soon:
                sp_chaud_base, sp_chaud_init = 19.0, 18.5
                sp_froid_base, sp_froid_init = 25.0, 26.0
                min_chaud, max_chaud = 17.5, 19.5
                min_froid, max_froid = 24.5, 26.5
                cta_state = "PRE_HEATING" if season == "hiver" else "PRE_COOLING"
            else:
                sp_chaud_base, sp_chaud_init = 19.0, 15.0
                sp_froid_base, sp_froid_init = 25.0, 28.0
                min_chaud, max_chaud = 15.0, 16.5
                min_froid, max_froid = 27.0, 29.0
                cta_state = "HORS_GEL" if season == "hiver" else "OFF"
        else:
            if is_closing_soon:
                sp_chaud_base, sp_chaud_init = 21.0, 19.5
                sp_froid_base, sp_froid_init = 24.0, 25.5
                min_chaud, max_chaud = 18.5, 20.0
                min_froid, max_froid = 24.5, 26.0
                cta_state = "EARLY_STOP"
            else:
                sp_chaud_base, sp_chaud_init = 21.0, 20.5
                sp_froid_base, sp_froid_init = 24.0, 24.5
                min_chaud, max_chaud = 20.0, 22.0
                min_froid, max_froid = 24.0, 25.0
                cta_state = "ON"

        T_chaud = torch.tensor([[sp_chaud_init]], dtype=torch.float32, requires_grad=True)
        T_froid = torch.tensor([[sp_froid_init]], dtype=torch.float32, requires_grad=True)

        optimizer = torch.optim.Adam([T_chaud, T_froid], lr=0.02)

        # Trích xuất dữ liệu môi trường
        T_ext = float(curr_temp - 4.0)
        humidite = float(row_curr.get("humidite", 65.0))
        solar = float(row_curr.get("rayonnement_solaire", 200.0))
        surf_san = float(row_curr.get("surface", 100))
        surf_chauff = float(row_curr.get("surface_chauffee", 100))

        x_in = torch.tensor(
            [[T_ext, humidite, solar, visitors_curr, surf_san, surf_chauff, attr_is_open]],
            dtype=torch.float32,
        )

        # 3. TỐI ƯU HÓA QUA HÀM LOSS THEO NGUYÊN LÝ VẬT LÝ
        for _ in range(40):
            optimizer.zero_grad()
            out = self.pinn(x_in)

            Q_elec_base = torch.relu(out[:, 1:2])
            Q_ec_base = torch.relu(out[:, 2:3])

            delta_T_chaud = torch.relu(T_chaud - T_ext)
            delta_T_froid = torch.relu(T_ext - T_froid)

            k_thermal = 0.06
            Q_ec_adj = Q_ec_base * (1.0 + k_thermal * delta_T_chaud)
            Q_elec_adj = Q_elec_base * (1.0 + k_thermal * delta_T_froid)

            eff_factor = 0.90 if is_closing_soon else 1.0
            total_cost = (Q_elec_adj * 0.15 + Q_ec_adj * 0.10) * eff_factor

            deadband_penalty = 20.0 * torch.relu(3.0 - (T_froid - T_chaud)) ** 2

            visitor_scale = 1.0 + (visitors_curr / 150.0)
            if attr_is_open == 1:
                comfort_penalty = 15.0 * visitor_scale * (
                    torch.relu(20.5 - T_chaud) ** 2 + torch.relu(T_froid - 24.5) ** 2
                )
            else:
                comfort_penalty = 2.0 * (torch.relu(15.0 - T_chaud) ** 2 + torch.relu(T_froid - 28.0) ** 2)

            if will_open_soon:
                target_pre_chaud = 18.5 if season == "hiver" else 16.0
                target_pre_froid = 26.0 if season == "hiver" else 24.5
                comfort_penalty += 25.0 * (
                    (T_chaud - target_pre_chaud) ** 2 + (T_froid - target_pre_froid) ** 2
                )

            loss = torch.mean(total_cost + comfort_penalty + deadband_penalty)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                T_chaud.clamp_(min_chaud, max_chaud)
                T_froid.clamp_(min_froid, max_froid)

        opt_chaud = round(T_chaud.item(), 1)
        opt_froid = round(T_froid.item(), 1)

        # 4. TÍNH NĂNG LƯỢNG TIÊU THỤ THỰC TẾ DỰA TRÊN CHÊNH LỆCH CONSIGNE
        with torch.no_grad():
            final_out = self.pinn(x_in)
            base_elec_val = max(10.0, final_out[0, 1].item())
            base_ec_val = max(10.0, final_out[0, 2].item())

            # Hệ số nhạy nhiệt năng lượng: Thay đổi 1°C làm tăng/giảm ~6% tải nhiệt
            gamma_thermal = 0.06

            dT_chaud = opt_chaud - sp_chaud_base
            dT_froid = sp_froid_base - opt_froid

            eff_factor = 0.90 if is_closing_soon else 1.0

            factor_ec = (1.0 + (gamma_thermal * dT_chaud)) * eff_factor
            factor_elec = (1.0 + (gamma_thermal * dT_froid)) * eff_factor

            # Giới hạn điều chỉnh vật lý thực tế từ -35% đến +20%
            factor_ec = max(0.65, min(1.20, factor_ec))
            factor_elec = max(0.65, min(1.20, factor_elec))

            opt_ec = round(base_ec_val * factor_ec, 2)
            opt_elec = round(base_elec_val * factor_elec, 2)

        return opt_chaud, opt_froid, opt_elec, opt_ec, cta_state


async def process_attraction(attr_id, config, client, idx):
    opt_engine = MultiAttrOptimizer(attr_id)

    node_time = client.get_node(f"ns={idx};s={attr_id}_timestamp")
    node_temp = client.get_node(f"ns={idx};s={attr_id}_temperature")
    node_visitors = client.get_node(f"ns={idx};s={attr_id}_visitor_count")
    node_chaud = client.get_node(f"ns={idx};s={attr_id}_Consigne_Chaud")
    node_froid = client.get_node(f"ns={idx};s={attr_id}_Consigne_Froid")

    scada_time_str = await node_time.get_value()
    if not scada_time_str:
        return

    curr_temp = await node_temp.get_value()
    visitors = await node_visitors.get_value()

    df_attr = GLOBAL_DATAFRAMES[attr_id]
    scada_dt = pd.to_datetime(scada_time_str).floor("h")

    df_window_24h = df_attr[df_attr["date"] >= scada_dt].head(24)
    if df_window_24h.empty:
        df_window_24h = df_attr.iloc[[0]]

    row_data = df_window_24h.iloc[0]
    base_elec = float(row_data.get("elec_1_pred", 25.0))
    base_ec = float(row_data.get("ec_value_pred", 20.0))

    season = "hiver" if curr_temp < 20.0 else "ete"

    opt_chaud, opt_froid, opt_elec, opt_ec, cta_state = opt_engine.optimize_consignes(
        df_window_24h=df_window_24h,
        curr_temp=curr_temp,
        season=season,
    )

    TOTAL_SAVINGS[attr_id]["elec_base"] += base_elec
    TOTAL_SAVINGS[attr_id]["elec_opt"] += opt_elec
    TOTAL_SAVINGS[attr_id]["ec_base"] += base_ec
    TOTAL_SAVINGS[attr_id]["ec_opt"] += opt_ec

    saved_elec = base_elec - opt_elec
    pct_elec = (saved_elec / base_elec * 100) if base_elec > 0 else 0.0

    saved_ec = base_ec - opt_ec
    pct_ec = (saved_ec / base_ec * 100) if base_ec > 0 else 0.0

    await node_chaud.write_value(opt_chaud)
    await node_froid.write_value(opt_froid)

    tot_elec_saved = TOTAL_SAVINGS[attr_id]["elec_base"] - TOTAL_SAVINGS[attr_id]["elec_opt"]
    tot_ec_saved = TOTAL_SAVINGS[attr_id]["ec_base"] - TOTAL_SAVINGS[attr_id]["ec_opt"]

    print(
        f"⏰ [MÔ PHỎNG TIME-SERIES: {scada_dt}] ➔ Attraction: {attr_id} | CTA State: [{cta_state}]"
    )
    print(f"   🎯 Consigne ➔ CHAUD: {opt_chaud:.1f}°C | FROID: {opt_froid:.1f}°C")
    print(
        f"   ⚡ Điện (Elec): Tức thời: {base_elec:.2f} kWh ➔ {opt_elec:.2f} kWh ({saved_elec:+.2f} kWh / {pct_elec:+.1f}%) | 📊 Lũy kế tiết kiệm: {tot_elec_saved:.2f} kWh"
    )
    print(
        f"   🔥 Nhiệt (EC) : Tức thời: {base_ec:.2f} kWh ➔ {opt_ec:.2f} kWh ({saved_ec:+.2f} kWh / {pct_ec:+.1f}%) | 📊 Lũy kế tiết kiệm: {tot_ec_saved:.2f} kWh\n"
    )


async def main():
    print("🚀 [CONTROL PINN] Bắt đầu chạy mô phỏng tối ưu hóa thời gian thực...")
    while True:
        try:
            async with Client(url=PANORAMA_URL) as client:
                idx = await client.get_namespace_index(URI_PANORAMA)
                tasks = [
                    process_attraction(attr_id, config, client, idx)
                    for attr_id, config in ATTRACTION_FILES.items()
                ]
                await asyncio.gather(*tasks)
        except Exception as e:
            print(f"❌ Kết nối SCADA: {e}")
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())