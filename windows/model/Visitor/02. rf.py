import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
DATA_DIR = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean"
DATA_OUT = r"D:\Stage SI\Machine Learning\Futuroscope\windows\model\Visitor"
INPUT_PATH = os.path.join(DATA_DIR, "master_ml", "visitor.csv")
MODEL_DIR = os.path.join(DATA_OUT, "02. random_forest")

os.makedirs(MODEL_DIR, exist_ok=True)

print("=== HUẤN LUYỆN RANDOM FOREST & BACKCASTING QUÁ KHỨ (SỐ NGUYÊN) ===")

# 1. Đọc dữ liệu toàn bộ (để có thể backcast về giai đoạn trước đó nếu file master có)
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Chưa tìm thấy file {INPUT_PATH}. Hãy kiểm tra lại file features!")

df = pd.read_csv(INPUT_PATH)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(by=['id_attraction', 'datetime']).reset_index(drop=True)

START_TRAIN_DATE = pd.to_datetime("2024-02-10")

print(f"\n[1/4] Đã tải toàn bộ dữ liệu. Tổng số dòng: {len(df):,}")

# ==========================================
# 2. XÁC ĐỊNH TARGET & FEATURES
# ==========================================
TARGET = 'visitor_count'
EXCLUDE_COLS = ['datetime', 'id_attraction', 'visitor_count', 'max_capacity', 'capacity_ratio']

FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])]

# Khởi tạo đối tượng đồ thị Plotly dùng chung
fig = go.Figure()

COLOR_PALETTE = {
    'H03': {'actual': '#1f77b4', 'pred_rf': '#ff7f0e'}, 
    'H07': {'actual': '#d62728', 'pred_rf': '#2ca02c'} 
}

results_summary = []
filled_dfs = []

# ==========================================
# 3. VÒNG LẶP HUẤN LUYỆN, VALIDATE & BACKCAST
# ==========================================
for idx, att_id in enumerate(df['id_attraction'].unique()):
    print(f"\n📌 ĐANG XỬ LÝ ĐIỂM THAM QUAN: [{att_id}]")
    
    sub_df = df[df['id_attraction'] == att_id].copy().sort_values('datetime').reset_index(drop=True)
    
    # Khởi tạo cột chứa giá trị dự báo backcast bằng Random Forest nếu chưa có
    sub_df['visitor_count_backcast_rf'] = np.nan
    
    # Tách phần dữ liệu chuẩn để Train/Test (từ 10/02/2024 trở đi và có visitor_count thật)
    recent_valid_mask = (sub_df['datetime'] >= START_TRAIN_DATE) & (sub_df[TARGET].notnull())
    train_eval_df = sub_df[recent_valid_mask].copy()
    
    if len(train_eval_df) == 0:
        print(f" -> ⚠️ Cảnh báo: Attraction {att_id} không có dữ liệu từ {START_TRAIN_DATE.strftime('%Y-%m-%d')} trở đi để train!")
        filled_dfs.append(sub_df)
        continue

    # Chia Train / Test (80/20) trên phần dữ liệu chuẩn
    split_idx = int(len(train_eval_df) * 0.80)
    train_df = train_eval_df.iloc[:split_idx]
    test_df = train_eval_df.iloc[split_idx:]
    
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET]
    
    # Huấn luyện mô hình đánh giá (Validation)
    model_rf_eval = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_rf_eval.fit(X_train, y_train)
    
    # LÀM TRÒN THÀNH SỐ NGUYÊN (VALIDATION)
    y_pred_rf = np.round(np.clip(model_rf_eval.predict(X_test), a_min=0, a_max=None)).astype(int)
    
    # Lưu chỉ số đánh giá Validation
    mae = mean_absolute_error(y_test, y_pred_rf)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2 = r2_score(y_test, y_pred_rf)
    results_summary.append({'Attraction': att_id, 'Model': 'Random Forest', 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    print(f" -> [Random Forest] Validation R²: {r2:.4f} | MAE: {mae:.2f}")

    # Huấn luyện lại mô hình tối ưu trên toàn bộ tập dữ liệu hợp lệ (>= 10/02/2024) để backcast
    X_all_recent = train_eval_df[FEATURE_COLS]
    y_all_recent = train_eval_df[TARGET]
    
    final_model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    final_model_rf.fit(X_all_recent, y_all_recent)
    
    # --- THỰC HIỆN BACKCASTING CHO QUÁ KHỨ VÀ CÁC DÒNG THIẾU ---
    backcast_mask = (sub_df['datetime'] < START_TRAIN_DATE) | (sub_df[TARGET].isnull())
    
    past_df = pd.DataFrame()
    y_past_pred_rf = np.array([])
    
    if backcast_mask.any():
        X_backcast = sub_df.loc[backcast_mask, FEATURE_COLS]
        
        # LÀM TRÒN THÀNH SỐ NGUYÊN (BACKCASTING)
        pred_rf_backcast = np.round(np.clip(final_model_rf.predict(X_backcast), 0, None)).astype(int)
        
        sub_df.loc[backcast_mask, 'visitor_count_backcast_rf'] = pred_rf_backcast
        
        # Lấp đầy trực tiếp vào cột TARGET nếu dòng đó đang trống / quá khứ
        sub_df.loc[backcast_mask, TARGET] = pred_rf_backcast
        
        past_df = sub_df[backcast_mask].copy()
        y_past_pred_rf = pred_rf_backcast
        print(f" -> ⏪ Đã Backcast bằng Random Forest (số nguyên) cho {backcast_mask.sum():,} dòng (quá khứ hoặc thiếu).")

    filled_dfs.append(sub_df)

    # --- VẼ ĐỒ THỊ PLOTLY ---
    colors = COLOR_PALETTE.get(att_id, {'actual': '#1f77b4', 'pred_rf': '#ff7f0e'})
    is_visible = True if idx == 0 else 'legendonly'

    # 1. Trace Backcast quá khứ (Random Forest) - Đã bỏ :.1f thành %{y}
    if len(past_df) > 0:
        fig.add_trace(go.Scatter(
            x=past_df['datetime'], y=y_past_pred_rf,
            mode='lines', name=f'[{att_id}] Backcast (RF)',
            line=dict(color=colors['pred_rf'], width=1.5, dash='dashdot'),
            visible=is_visible,
            hovertemplate=f'<b>{att_id} Backcast (RF):</b> %{{y}} người<br><b>Thời gian:</b> %{{x}}<extra></extra>'
        ))

    # 2. Trace Thực tế (Validation)
    fig.add_trace(go.Scatter(
        x=test_df['datetime'], y=y_test,
        mode='lines', name=f'[{att_id}] Thực tế',
        line=dict(color=colors['actual'], width=2),
        visible=is_visible,
        hovertemplate=f'<b>{att_id} Thực tế:</b> %{{y}} người<br><b>Thời gian:</b> %{{x}}<extra></extra>'
    ))

    # 3. Trace Dự báo Validation (Random Forest) - Đã bỏ :.1f thành %{y}
    fig.add_trace(go.Scatter(
        x=test_df['datetime'], y=y_pred_rf,
        mode='lines', name=f'[{att_id}] Dự báo (RF)',
        line=dict(color=colors['pred_rf'], width=1.5, dash='dot'),
        visible=is_visible,
        hovertemplate=f'<b>{att_id} RF:</b> %{{y}} người<br><b>Thời gian:</b> %{{x}}<extra></extra>'
    ))

# Gộp toàn bộ dữ liệu sau khi backcast xong
final_full_dataset = pd.concat(filled_dfs, ignore_index=True)

# Ép kiểu toàn bộ cột TARGET thành số nguyên hẳn hoi trước khi xuất file
final_full_dataset[TARGET] = final_full_dataset[TARGET].astype(int)

# ==========================================
# 4. TÙY CHỈNH TƯƠNG TÁC & LAYOUT GỘP
# ==========================================
fig.update_layout(
    title='<b>BACKCASTING QUÁ KHỨ & VALIDATION (RANDOM FOREST)</b>',
    xaxis_title='Thời gian',
    yaxis_title='Lượt khách (visitor_count)',
    hovermode='x unified',
    template='plotly_white',
    legend=dict(
        x=1.02, y=1,
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='LightGrey',
        borderwidth=1,
        title=dict(text='<b>Mẹo: Click nhãn để Bật/Ẩn</b>')
    ),
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(step="all", label="Tất cả")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)

# Lưu file kết quả vào thư mục riêng của Random Forest
output_combined_html = os.path.join(MODEL_DIR, "02_random_forest_backcast_complete.html")
output_csv = os.path.join(MODEL_DIR, "visitor_fully_backcasted_rf.csv")

fig.write_html(output_combined_html)
final_full_dataset.to_csv(output_csv, index=False, encoding='utf-8-sig')

summary_df = pd.DataFrame(results_summary)
summary_df.to_csv(os.path.join(MODEL_DIR, "rf_baseline_results_summary.csv"), index=False, encoding='utf-8-sig')

print("\n==================================================")
print("🎉 HOÀN THÀNH HUẤN LUYỆN, VALIDATE VÀ BACKCASTING VỚI RANDOM FOREST (SỐ NGUYÊN)!")
print(f"-> File HTML tương tác lưu tại: {output_combined_html}")
print(f"-> File CSV đầy đủ dữ liệu backcast lưu tại: {output_csv}")
print("==================================================")