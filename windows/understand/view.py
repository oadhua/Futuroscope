import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Danh sách 4 file master_ml chính xác
paths = [
    r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H03_predict_pure_elec.csv",
    r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H03_predict_pure_thermal.csv",
    r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H07_predict_pure_elec.csv",
    r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H07_predict_pure_thermal.csv"
]

titles = [
    "H03 - Électricité (Master ML)",
    "H03 - Thermique (Master ML)",
    "H07 - Électricité (Master ML)",
    "H07 - Thermique (Master ML)"
]

colors = ["royalblue", "darkorange", "forestgreen", "crimson"]

# Tạo subplots gồm 4 hàng dọc
fig = make_subplots(
    rows=4, 
    cols=1, 
    shared_xaxes=True, 
    subplot_titles=titles
)

for i, path in enumerate(paths):
    try:
        df = pd.read_csv(path)
        
        # Tìm cột thời gian
        time_col = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if time_col:
            t_col = time_col[0]
            df[t_col] = pd.to_datetime(df[t_col])
            df = df.sort_values(t_col).reset_index(drop=True)
            
            # Lấy cột mục tiêu điện/nhiệt (thường bắt đầu bằng elec hoặc th/therm hoặc các cột giá trị)
            # Hoặc ưu tiên tìm cột 'elec_1' hoặc 'thermal' nếu có, nếu không lấy cột số đầu tiên khác date
            target_cols = [col for col in df.columns if col.startswith('elec') or col.startswith('ec')]
            val_col = target_cols[0] if target_cols else [col for col in df.columns if col != t_col][0]
            
            fig.add_trace(
                go.Scatter(
                    x=df[t_col].dt.strftime("%Y-%m-%d %H:%M:%S"),
                    y=df[val_col],
                    mode="lines",
                    name=titles[i],
                    line=dict(color=colors[i], width=1)
                ),
                row=i+1, col=1
            )
            fig.update_yaxes(title_text=val_col, row=i+1, col=1)
    except Exception as e:
        print(f"Lỗi khi đọc file {path}: {e}")

# Cấu hình giao diện chung
fig.update_layout(
    title="<b>Xu hướng dữ liệu Master ML (H03 & H07 - Điện & Nhiệt)</b>",
    hovermode="x unified",
    height=900,
    showlegend=False
)

fig.update_xaxes(rangeslider=dict(visible=False), type="date", row=4, col=1)

# Lưu thành file HTML và mở trực tiếp
fig.write_html("visualize_master_ml_trends.html")
fig.show()