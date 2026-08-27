# -*- coding: utf-8 -*-
"""
SERVEUR SIMULATEUR SCADA OPC-UA PANORAMA
Auteur: Hứa Đại Bảo (FMD / LIAS)
Description: Simulation du serveur OPC-UA Panorama diffusant en temps réel les données
             d'exploitation, météo et fréquentation des attractions du Parc du Futuroscope.
"""

import asyncio
import pandas as pd
from asyncua import Server, ua

URI_PANORAMA = "http://futuroscope.hvac.panorama"

ATTRACTION_FILES = {
    "H07": {
        "elec": r"D:\Stage SI\Machine Learning\Futuroscope\windows\model\H07_elec\before\predict_2026.csv",
        "thermal": r"D:\Stage SI\Machine Learning\Futuroscope\windows\model\H07_thermal\before\predict_2026.csv",
    },
    "H03": {
        "elec": r"D:\Stage SI\Machine Learning\Futuroscope\windows\model\H03_elec\before\predict_2026.csv",
        "thermal": r"D:\Stage SI\Machine Learning\Futuroscope\windows\model\H03_thermal\before\predict_2026.csv",
    },
}


def charger_donnees_attraction(attr_id):
    files = ATTRACTION_FILES[attr_id]
    try:
        df_elec = pd.read_csv(files["elec"])
        df_thermal = pd.read_csv(files["thermal"])
    except FileNotFoundError:
        df_elec = pd.read_csv("predict_2026.csv")
        df_thermal = df_elec.copy()

    df_elec["date"] = pd.to_datetime(df_elec["date"])
    df_thermal["date"] = pd.to_datetime(df_thermal["date"])

    df = pd.merge(df_thermal, df_elec, on="date", suffixes=("", "_elec_file"))
    if "operation" not in df.columns:
        df["operation"] = 1.0

    df = df.set_index("date").resample("10min").ffill().reset_index()
    return df


async def main():
    server = Server()
    await server.init()
    server.set_endpoint("opc.tcp://127.0.0.1:4840/freeopcua/server/")

    idx = await server.register_namespace(URI_PANORAMA)

    data_store = {}
    node_store = {}

    print("📊 [SERVEUR SCADA] Chargement des données multi-attractions...")

    for attr_id in ATTRACTION_FILES.keys():
        df = charger_donnees_attraction(attr_id)
        data_store[attr_id] = df

        folder = await server.nodes.objects.add_folder(idx, f"Attraction_{attr_id}")

        node_store[attr_id] = {
            "timestamp": await folder.add_variable(
                ua.NodeId(f"{attr_id}_timestamp", idx),
                f"{attr_id}_timestamp",
                str(df.iloc[0]["date"]),
            ),
            "temp": await folder.add_variable(
                ua.NodeId(f"{attr_id}_temperature", idx),
                f"{attr_id}_temperature",
                float(df.iloc[0]["temperature"]),
            ),
            "is_open": await folder.add_variable(
                ua.NodeId(f"{attr_id}_is_open", idx),
                f"{attr_id}_is_open",
                int(df.iloc[0]["is_open"]),
            ),
            "operation": await folder.add_variable(
                ua.NodeId(f"{attr_id}_operation", idx),
                f"{attr_id}_operation",
                float(df.iloc[0]["operation"]),
            ),
            "visitors": await folder.add_variable(
                ua.NodeId(f"{attr_id}_visitor_count", idx),
                f"{attr_id}_visitor_count",
                float(df.iloc[0].get("visitor_count", 0)),
            ),
            "chaud": await folder.add_variable(
                ua.NodeId(f"{attr_id}_Consigne_Chaud", idx),
                f"{attr_id}_Consigne_Chaud",
                18.0,
            ),
            "froid": await folder.add_variable(
                ua.NodeId(f"{attr_id}_Consigne_Froid", idx),
                f"{attr_id}_Consigne_Froid",
                23.5,
            ),
        }

        await node_store[attr_id]["chaud"].set_writable()
        await node_store[attr_id]["froid"].set_writable()

    async with server:
        print(
            "🚀 [SERVEUR SCADA] Serveur Multi-Attractions OPC-UA Panorama démarré"
            " sur opc.tcp://127.0.0.1:4840/freeopcua/server/\n"
        )

        row_idx = 0
        while True:
            print("📡 [BUS SCADA - ÉMISSION 10-MIN]")

            for attr_id, df in data_store.items():
                row = df.iloc[row_idx % len(df)]
                nodes = node_store[attr_id]

                await nodes["timestamp"].write_value(str(row["date"]))
                await nodes["temp"].write_value(float(row["temperature"]))
                await nodes["is_open"].write_value(int(row["is_open"]))
                await nodes["operation"].write_value(float(row["operation"]))
                await nodes["visitors"].write_value(float(row.get("visitor_count", 0)))

                val_chaud = await nodes["chaud"].get_value()
                val_froid = await nodes["froid"].get_value()

                status_parc = "OUVERT" if row["is_open"] == 1 else "FERMÉ"
                print(
                    f"  📍 [{attr_id} | {row['date']}] Temp: {row['temperature']:.1f}°C"
                    f" | Visiteurs: {int(row.get('visitor_count', 0))} | Parc:"
                    f" {status_parc} | Consignes actuelles -> Chaud: {val_chaud:.1f}°C,"
                    f" Froid: {val_froid:.1f}°C"
                )

            print("-" * 75)
            row_idx += 1
            await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
