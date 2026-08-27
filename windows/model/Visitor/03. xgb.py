import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
DATA_DIR = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean"
DATA_OUT = r"D:\Stage SI\Machine Learning\Futuroscope\windows\model\Visitor"
INPUT_PATH = os.path.join(DATA_DIR, "master_ml", "visitor.csv")
MODEL_DIR = os.path.join(DATA_OUT, "03. xgboost")

os.makedirs(MODEL_DIR, exist_ok=True)

print("=== HUẤN LUYỆN XGBOOST & BACKCASTING QUÁ KHỨ (SỐ NGUYÊN) ===")

# 1. Đọc dữ liệu toàn bộ (để có thể backcast về giai đoạn trước đó nếu file master có)
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        f"Chưa tìm thấy file {INPUT_PATH}. Hãy kiểm tra lại file features!"
    )

df = pd.read_csv(INPUT_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values(by=["id_attraction", "datetime"]).reset_index(drop=True)

START_TRAIN_DATE = pd.to_datetime("2024-02-10")

print(f"\n[1/4] Đã tải toàn bộ dữ liệu. Tổng số dòng: {len(df):,}")

# ==========================================
# 2. XÁC ĐỊNH TARGET & FEATURES
# ==========================================
TARGET = "visitor_count"
EXCLUDE_COLS = [
    "datetime",
    "id_attraction",
    "visitor_count",
    "max_capacity",
    "capacity_ratio",
    "h_ouv",
    "h_ferm",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

FEATURE_COLS = [
    c
    for c in df.columns
    if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])
]

# Khởi tạo đối tượng đồ thị Plotly dùng chung
fig = go.Figure()

COLOR_PALETTE = {
    "H03": {"actual": "#1f77b4", "pred_xgb": "#ff7f0e"},
    "H07": {"actual": "#d62728", "pred_xgb": "#2ca02c"},
}

results_summary = []
filled_dfs = []

# ==========================================
# 3. VÒNG LẶP HUẤN LUYỆN, VALIDATE & BACKCAST
# ==========================================
for idx, att_id in enumerate(df["id_attraction"].unique()):
    print(f"\n📌 ĐANG XỬ LÝ ĐIỂM THAM QUAN: [{att_id}]")

    sub_df = (
        df[df["id_attraction"] == att_id]
        .copy()
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # Khởi tạo cột chứa giá trị dự báo backcast bằng XGBoost nếu chưa có
    sub_df["visitor_count_backcast_xgb"] = np.nan

    # Tách phần dữ liệu chuẩn để Train/Test (từ 10/02/2024 trở đi và có visitor_count thật)
    recent_valid_mask = (sub_df["datetime"] >= START_TRAIN_DATE) & (
        sub_df[TARGET].notnull()
    )
    train_eval_df = sub_df[recent_valid_mask].copy()

    if len(train_eval_df) == 0:
        print(
            f" -> ⚠️ Cảnh báo: Attraction {att_id} không có dữ liệu từ {START_TRAIN_DATE.strftime('%Y-%m-%d')} trở đi để train!"
        )
        filled_dfs.append(sub_df)
        continue

    # Chia Train / Test (80/20) trên phần dữ liệu chuẩn
    split_idx = int(len(train_eval_df) * 0.80)
    train_df = train_eval_df.iloc[:split_idx]
    test_df = train_eval_df.iloc[split_idx:]

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET]

    # Huấn luyện mô hình đánh giá (Validation) với XGBoost
    model_xgb_eval = XGBRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1
    )
    model_xgb_eval.fit(X_train, y_train)

    # Dự báo và làm tròn thành số nguyên (Validation)
    y_pred_xgb = np.round(
        np.clip(model_xgb_eval.predict(X_test), a_min=0, a_max=None)
    ).astype(int)

    # Lưu chỉ số đánh giá Validation
    mae = mean_absolute_error(y_test, y_pred_xgb)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2 = r2_score(y_test, y_pred_xgb)
    results_summary.append(
        {"Attraction": att_id, "Model": "XGBoost", "MAE": mae, "RMSE": rmse, "R2": r2}
    )
    print(f" -> [XGBoost] Validation R²: {r2:.4f} | MAE: {mae:.2f}")

    # Huấn luyện lại mô hình tối ưu trên toàn bộ tập dữ liệu hợp lệ (>= 10/02/2024) để backcast
    X_all_recent = train_eval_df[FEATURE_COLS]
    y_all_recent = train_eval_df[TARGET]

    final_model_xgb = XGBRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1
    )
    final_model_xgb.fit(X_all_recent, y_all_recent)

    # --- THỰC HIỆN BACKCASTING CHO QUÁ KHỨ VÀ CÁC DÒNG THIẾU ---
    backcast_mask = (sub_df["datetime"] < START_TRAIN_DATE) | (sub_df[TARGET].isnull())

    past_df = pd.DataFrame()
    y_past_pred_xgb = np.array([])

    if backcast_mask.any():
        X_backcast = sub_df.loc[backcast_mask, FEATURE_COLS]

        # Dự báo và làm tròn thành số nguyên (Backcasting)
        pred_xgb_backcast = np.round(
            np.clip(final_model_xgb.predict(X_backcast), 0, None)
        ).astype(int)

        sub_df.loc[backcast_mask, "visitor_count_backcast_xgb"] = pred_xgb_backcast

        # Lấp đầy trực tiếp vào cột TARGET nếu dòng đó đang trống / quá khứ
        sub_df.loc[backcast_mask, TARGET] = pred_xgb_backcast

        past_df = sub_df[backcast_mask].copy()
        y_past_pred_xgb = pred_xgb_backcast
        print(
            f" -> ⏪ Đã Backcast bằng XGBoost (số nguyên) cho {backcast_mask.sum():,} dòng (quá khứ hoặc thiếu)."
        )

    filled_dfs.append(sub_df)

    # --- VẼ ĐỒ THỊ PLOTLY ---
    colors = COLOR_PALETTE.get(att_id, {"actual": "#1f77b4", "pred_xgb": "#ff7f0e"})
    is_visible = True if idx == 0 else "legendonly"

    # 1. Trace Backcast quá khứ (XGBoost)
    if len(past_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=past_df["datetime"],
                y=y_past_pred_xgb,
                mode="lines",
                name=f"[{att_id}] Backcast (XGB)",
                line=dict(color=colors["pred_xgb"], width=1.5, dash="dashdot"),
                visible=is_visible,
                hovertemplate=f"<b>{att_id} Backcast (XGB):</b> %{{y}} người<br><b>Thời gian:</b> %{{x}}<extra></extra>",
            )
        )

    # 2. Trace Thực tế (Validation)
    fig.add_trace(
        go.Scatter(
            x=test_df["datetime"],
            y=y_test,
            mode="lines",
            name=f"[{att_id}] Thực tế",
            line=dict(color=colors["actual"], width=2),
            visible=is_visible,
            hovertemplate=f"<b>{att_id} Thực tế:</b> %{{y}} người<br><b>Thời gian:</b> %{{x}}<extra></extra>",
        )
    )

    # 3. Trace Dự báo Validation (XGBoost)
    fig.add_trace(
        go.Scatter(
            x=test_df["datetime"],
            y=y_pred_xgb,
            mode="lines",
            name=f"[{att_id}] Dự báo (XGB)",
            line=dict(color=colors["pred_xgb"], width=1.5, dash="dot"),
            visible=is_visible,
            hovertemplate=f"<b>{att_id} XGB:</b> %{{y}} người<br><b>Thời gian:</b> %{{x}}<extra></extra>",
        )
    )

# Gộp toàn bộ dữ liệu sau khi backcast xong
final_full_dataset = pd.concat(filled_dfs, ignore_index=True)

# Ép kiểu cột TARGET về số nguyên
final_full_dataset[TARGET] = final_full_dataset[TARGET].astype(int)

# ==========================================
# 4. TÙY CHỈNH TƯƠNG TÁC & LAYOUT GỘP
# ==========================================
fig.update_layout(
    title="<b>BACKCASTING QUÁ KHỨ & VALIDATION (XGBOOST)</b>",
    xaxis_title="Thời gian",
    yaxis_title="Lượt khách (visitor_count)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(
        x=1.02,
        y=1,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="LightGrey",
        borderwidth=1,
        title=dict(text="<b>Mẹo: Click nhãn để Bật/Ẩn</b>"),
    ),
    xaxis=dict(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all", label="Tất cả"),
                ]
            )
        ),
        rangeslider=dict(visible=True),
        type="date",
    ),
)

# Lưu file kết quả vào thư mục riêng của XGBoost
output_combined_html = os.path.join(MODEL_DIR, "03_xgboost_backcast_complete.html")
output_csv = os.path.join(MODEL_DIR, "visitor_fully_backcasted_xgb.csv")

fig.write_html(output_combined_html)
final_full_dataset.to_csv(output_csv, index=False, encoding="utf-8-sig")

summary_df = pd.DataFrame(results_summary)
summary_df.to_csv(
    os.path.join(MODEL_DIR, "xgb_baseline_results_summary.csv"),
    index=False,
    encoding="utf-8-sig",
)

print("\n==================================================")
print("🎉 HOÀN THÀNH HUẤN LUYỆN, VALIDATE VÀ BACKCASTING VỚI XGBOOST!")
print(f"-> File HTML tương tác lưu tại: {output_combined_html}")
print(f"-> File CSV đầy đủ dữ liệu backcast lưu tại: {output_csv}")
print("==================================================")
