import asyncio
from datetime import datetime
import pandas as pd
from asyncua import Server, ua
from pinn_multi_attr import ATTRACTION_FILES, load_merged_dataset

URI_PANORAMA = "http://futuroscope.hvac.panorama"


class SubHandler:
    def __init__(self, attr_id):
        self.attr_id = attr_id

    def datachange_notification(self, node, val, data):
        now_str = datetime.now().strftime("%H:%M:%S")
        print(
            f"📡 [SERVER SCADA {now_str}] <{self.attr_id}> Cập nhật Consigne: Node {node} ➔ {val:.1f}°C"
        )


async def main():
    server = Server()
    await server.init()
    # Tăng session timeout cấu hình server để tránh cảnh báo timeout
    server.set_endpoint("opc.tcp://127.0.0.1:4840/freeopcua/server/")
    idx = await server.register_namespace(URI_PANORAMA)

    # Load dữ liệu CSV chuỗi thời gian cho các Attraction
    datasets = {
        attr_id: load_merged_dataset(
            cfg["file_elec"],
            cfg["file_thermal"],
            cfg["surface"],
            cfg["surface_chauffee"],
        )
        for attr_id, cfg in ATTRACTION_FILES.items()
    }

    nodes = {}
    sub = await server.create_subscription(500, SubHandler("GENERAL"))

    for attr_id in ATTRACTION_FILES.keys():
        folder = await server.nodes.objects.add_folder(idx, f"Attraction_{attr_id}")

        nodes[attr_id] = {
            "time": await folder.add_variable(
                ua.NodeId(f"{attr_id}_timestamp", idx), f"{attr_id}_timestamp", ""
            ),
            "temp": await folder.add_variable(
                ua.NodeId(f"{attr_id}_temperature", idx), f"{attr_id}_temperature", 0.0
            ),
            "open": await folder.add_variable(
                ua.NodeId(f"{attr_id}_is_open", idx), f"{attr_id}_is_open", 1
            ),
            "visitors": await folder.add_variable(
                ua.NodeId(f"{attr_id}_visitor_count", idx),
                f"{attr_id}_visitor_count",
                0.0,
            ),
            "chaud": await folder.add_variable(
                ua.NodeId(f"{attr_id}_Consigne_Chaud", idx),
                f"{attr_id}_Consigne_Chaud",
                19.0,
            ),
            "froid": await folder.add_variable(
                ua.NodeId(f"{attr_id}_Consigne_Froid", idx),
                f"{attr_id}_Consigne_Froid",
                24.0,
            ),
        }

        await nodes[attr_id]["chaud"].set_writable()
        await nodes[attr_id]["froid"].set_writable()
        await sub.subscribe_data_change(nodes[attr_id]["chaud"])
        await sub.subscribe_data_change(nodes[attr_id]["froid"])

    async with server:
        print(
            "🚀 [SERVER SCADA] Bắt đầu mô phỏng chạy chuỗi thời gian từ file predict_2026.csv..."
        )

        # Giả định các file có độ dài giống nhau, lấy số dòng của file đầu tiên
        sample_df = list(datasets.values())[0]
        num_rows = len(sample_df)

        for step in range(num_rows):
            print(f"\n⏳ --- Mô phỏng Bước {step + 1}/{num_rows} ---")
            for attr_id, df in datasets.items():
                row = df.iloc[step]

                # Cập nhật thông số từ dòng CSV lên các Node OPC-UA
                timestamp_val = str(row["date"])
                temp_val = float(row.get("temperature", 18.0))
                open_val = int(row.get("is_open", 1))
                visitors_val = float(row.get("visitor_count", 100.0))

                await nodes[attr_id]["time"].write_value(timestamp_val)
                await nodes[attr_id]["temp"].write_value(temp_val)
                await nodes[attr_id]["open"].write_value(open_val)
                await nodes[attr_id]["visitors"].write_value(visitors_val)

            # Tốc độ phát mô phỏng: 2 giây chuyển đổi sang 1 giờ tiếp theo trong CSV
            await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
