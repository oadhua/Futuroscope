import pandas as pd


class RealtimePanoramaController:
    def __init__(self, baseline_file="setpoint_2026.csv"):
        # Tên file CSV baseline của bạn
        self.baseline_file = baseline_file
        self.deadband = 0.5  # Ngưỡng chênh lệch để đưa ra quyết định Override (°C)

    def _load_baseline_for_hour(self, current_hour: int) -> float:
        try:
            df = pd.read_csv(self.baseline_file)

            # -------------------------------------------------------------------------
            # ⚠️ LƯU Ý BỔ SUNG NẾU TÊN CỘT KHÁC NHAU:
            # - Nếu cột giờ tên là 'hour', 'heure', 'time'... hãy sửa lại bên dưới.
            # - Nếu cột nhiệt độ tên là 'consigne_baseline', 'setpoint', 'T_set'... hãy sửa lại.
            # -------------------------------------------------------------------------
            row = df[df["hour"] == current_hour]
            if not row.empty:
                # Lấy giá trị consigne tương ứng với giờ hiện tại
                return float(row["consigne_affichee"].values[0])

        except Exception as e:
            print(f"⚠️ Không đọc được file baseline ({e}), dùng giá trị mặc định 24.0°C")

        return 24.0  # Giá trị mặc định an toàn

    def _mo_hinh_vat_ly_1h(
        self, temp_ext: float, visitors_per_hour: float, temp_int: float
    ) -> float:
        """Mô hình tính toán lại Setpoint tối ưu thời gian thực cho 1 giờ."""
        if visitors_per_hour > 1200 and temp_ext > 28.0:
            return 23.0  # Lượng khách đông + Ngoài trời nóng -> Tăng cường làm mát
        elif visitors_per_hour < 300 and temp_ext < 22.0:
            return 25.0  # Vắng khách -> Giảm tải tiết kiệm điện
        return 24.0

    def process_15min_reading(
        self,
        current_hour: int,
        current_minute: int,
        temp_ext_15m: float,
        visitors_15m: int,
        temp_int_15m: float,
    ):
        # 1. Đọc Baseline từ file CSV cho giờ hiện tại
        t_baseline = self._load_baseline_for_hour(current_hour)

        # 2. Quy đổi dữ liệu 15 phút -> Tốc độ quy mô 1 giờ
        visitor_rate_hourly = visitors_15m * 4

        # 3. Tính toán Setpoint thời gian thực từ mô hình
        t_calculated = self._mo_hinh_vat_ly_1h(
            temp_ext_15m, visitor_rate_hourly, temp_int_15m
        )

        # 4. So sánh với Baseline và đưa ra quyết định Override
        diff = abs(t_calculated - t_baseline)
        if diff >= self.deadband:
            final_setpoint = t_calculated
            decision = f"OVERRIDE GTB (Lệch {diff:.1f}°C >= {self.deadband}°C)"
        else:
            final_setpoint = t_baseline
            decision = "GIỮ BASELINE (Biến động nhỏ)"

        return {
            "timestamp": f"{current_hour:02d}:{current_minute:02d}",
            "visitors_15m": visitors_15m,
            "visitor_rate_hourly": visitor_rate_hourly,
            "temp_ext": temp_ext_15m,
            "t_baseline": t_baseline,
            "t_calculated": t_calculated,
            "final_setpoint_sent": final_setpoint,
            "decision": decision,
        }
