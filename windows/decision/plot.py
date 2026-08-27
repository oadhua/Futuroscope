import asyncio
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from asyncua import Client

# Import trực tiếp từ control.py (đảm bảo file control.py nằm cùng thư mục)
from control import (
    PANORAMA_URL,
    URI_PANORAMA,
    ATTRACTION_CONFIGS,
    OptimiseurConsoHVAC,
    GLOBAL_PREDICTIONS,
    construct_24h_dataframe
)

ATTRACTIONS = ["H03", "H07"]
data_store = {
    code: {"times": [], "temps_ext": [], "sp_chaud": [], "sp_froid": [], "saved_energy": []}
    for code in ATTRACTIONS
}

plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
fig.canvas.manager.set_window_title("SCADA Real-time HVAC Monitoring & Energy Savings - Futuroscope")

async def fetch_and_compute_data():
    results = {}
    try:
        async with Client(url=PANORAMA_URL) as client:
            idx = await client.get_namespace_index(URI_PANORAMA)
            for code in ATTRACTIONS:
                cfg = ATTRACTION_CONFIGS[code]
                optimizer = OptimiseurConsoHVAC(cfg)

                node_timestamp = client.get_node(f"ns={idx};s={code}_timestamp")
                node_temp = client.get_node(f"ns={idx};s={code}_temperature")
                node_is_open = client.get_node(f"ns={idx};s={code}_is_open")
                node_operation = client.get_node(f"ns={idx};s={code}_operation")
                node_visitors = client.get_node(f"ns={idx};s={code}_visitor_count")
                node_chaud = client.get_node(f"ns={idx};s={code}_Consigne_Chaud")
                node_froid = client.get_node(f"ns={idx};s={code}_Consigne_Froid")

                scada_time = await node_timestamp.get_value()
                curr_temp = await node_temp.get_value()
                is_open = await node_is_open.get_value()
                operation = await node_operation.get_value()
                visitors = await node_visitors.get_value()
                sp_c = await node_chaud.get_value()
                sp_f = await node_froid.get_value()

                # Sử dụng chung hàm construct_24h_dataframe từ control.py
                df_24h = construct_24h_dataframe(code, curr_temp, is_open, operation, visitors, scada_time)

                t_scada_prev = sp_f if curr_temp > 20.0 else sp_c
                _, _, _, e_saved_list, unit, _, _ = optimizer.optimiser_bloc_24h(df_24h, T_scada_prev=t_scada_prev)
                
                results[code] = (curr_temp, sp_c, sp_f, e_saved_list[0], unit)
        return results
    except Exception as e:
        print(f"⚠️ Erreur de communication SCADA: {e}")
        return {code: (None, None, None, 0.0, "kWh") for code in ATTRACTIONS}

def update(frame):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    opc_data = loop.run_until_complete(fetch_and_compute_data())
    now_str = datetime.now().strftime("%H:%M:%S")

    for col_idx, code in enumerate(ATTRACTIONS):
        t_ext, sp_c, sp_f, e_saved_val, unit = opc_data[code]
        store = data_store[code]
        cfg = ATTRACTION_CONFIGS[code]

        if t_ext is not None:
            store["times"].append(now_str)
            store["temps_ext"].append(t_ext)
            store["sp_chaud"].append(sp_c)
            store["sp_froid"].append(sp_f)
            store["saved_energy"].append(e_saved_val)

            if len(store["times"]) > 30:
                for key in store:
                    store[key].pop(0)

            # Subplot Haut: Température & Consignes
            ax_temp = axes[0, col_idx]
            ax_temp.clear()
            ax_temp.plot(store["times"], store["temps_ext"], label="Temp. Ext (°C)", color="tab:orange", marker="o", lw=1.5)
            ax_temp.plot(store["times"], store["sp_froid"], label="Consigne Froid (°C)", color="tab:blue", ls="--", lw=1.8)
            ax_temp.plot(store["times"], store["sp_chaud"], label="Consigne Chaud (°C)", color="tab:red", ls="--", lw=1.8)
            ax_temp.set_title(f"[{code}] {cfg['name']} - Consignes MPC", fontsize=11, fontweight="bold")
            ax_temp.set_ylabel("Température (°C)")
            ax_temp.legend(loc="upper left", fontsize=8)
            ax_temp.grid(True)

            # Subplot Bas: Économie d'énergie exacte (khớp 100% control.py)
            ax_power = axes[1, col_idx]
            ax_power.clear()
            ax_power.bar(store["times"], store["saved_energy"], color="tab:green", alpha=0.75, label=f"Économie ΔE ({unit})")
            
            total_session_saved = sum(store["saved_energy"])
            ax_power.set_title(f"Économie Instantanée | Total Session: {total_session_saved:+.2f} {unit}", fontsize=10, fontweight="bold")
            ax_power.set_xlabel("Heure (HH:MM:SS)")
            ax_power.set_ylabel(f"Économie ({unit})")
            ax_power.legend(loc="upper left", fontsize=8)
            ax_power.grid(True)
            ax_power.set_xticks(range(len(store["times"])))
            ax_power.set_xticklabels(store["times"], rotation=45, ha="right", fontsize=7)

    plt.tight_layout()

ani = animation.FuncAnimation(fig, update, interval=2000, save_count=100)

if __name__ == "__main__":
    print("📊 [DASHBOARD SCADA] Connexion directe au moteur de control.py...")
    plt.show()