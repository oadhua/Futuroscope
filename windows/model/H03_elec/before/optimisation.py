import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize


# =====================================================================
# 0. MODULE ĐÁNH GIÁ ĐỘ CHÍNH XÁC MÔ HÌNH VẬT LÝ
# =====================================================================
def evaluer_precision_modele(
    y_true, y_pred, title="ÉVALUATION DU MODÈLE PHYSIQUE"
):
    """Calcule les métriques d'erreur RMSE, MAE, MAPE et R² entre données réelles et prédictions."""
    mask = (y_true > 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        print(f"[{title}] Données réelles insuffisantes pour l'évaluation.")
        return None

    rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100.0

    ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
    ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    print("\n" + "=" * 65)
    print(f"   {title}")
    print("=" * 65)
    print(f"   • Nombre d'échantillons (N)  : {len(y_true_clean)} heures")
    print(f"   • Erreur Absolue Moyenne (MAE): {mae:.3f} kWh")
    print(f"   • Erreur Quadratique (RMSE)   : {rmse:.3f} kWh")
    print(f"   • Erreur Relative (MAPE)      : {mape:.2f} %")
    print(f"   • Coefficient R²              : {r2:.4f}")
    print("=" * 65 + "\n")

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


# =====================================================================
# 1. BỘ TỐI ƯU NHIỆT - ĐIỆN NĂNG HVAC ĐỘNG (OPTIMISEUR THERMO-ÉLECTRIQUE)
# =====================================================================
class OptimiseurConsoHVAC:

    def __init__(self, site="tapis", e_standby=None):
        if e_standby is None:
            raise ValueError(
                "Le paramètre e_standby doit être fourni à l'optimiseur."
            )

        self.site = site
        self.e_standby_base = e_standby
        self.alpha_extinction = 0.25  # Hệ số suy giảm công suất dư sau khi tắt

        # Thông số kỹ thuật của hệ thống CTA (CIAT)
        if site == "vienne":
            self.p_froid_tot = 410.0
            self.p_chaud_tot = 285.0
            self.p_elec_tot = 32.5
            self.ua_val = 6.5
            self.cb_val = 25.0
        else:  # Tapis Magique
            self.p_froid_tot = 300.0
            self.p_chaud_tot = 325.0
            self.p_elec_tot = 30.2
            self.ua_val = 4.5
            self.cb_val = 18.0

    def _obtenir_config_saison(self, month):
        """Xác định các thông số nhiệt động học và giới hạn tiện nghi theo mùa."""
        if month in [11, 12, 1, 2, 3]:
            saison = "hiver"
            p_therm = self.p_chaud_tot
            t_ref = 19.0
            t_comf = 19.5
            bounds_open = (18.5, 20.5)
            bounds_off = (15.0, 15.0)  # Duy trì chống đóng băng (Hors-gel)
        elif month in [5, 6, 7, 8, 9]:
            saison = "ete"
            p_therm = self.p_froid_tot
            t_ref = 24.0
            t_comf = 23.5
            bounds_open = (23.0, 25.0)  # Tích lạnh 23°C (sáng) -> Duy trì 25°C (chiều đắt)
            bounds_off = (26.0, 26.0)  # Tắt lạnh đêm
        else:
            saison = "intersaison"
            p_therm = (self.p_froid_tot + self.p_chaud_tot) / 2.0
            t_ref = 21.0
            t_comf = 21.0
            bounds_open = (20.5, 21.5)
            bounds_off = (21.0, 21.0)

        cop_installe = p_therm / self.p_elec_tot

        return {
            "saison": saison,
            "COP": cop_installe,
            "P_elec_max": self.p_elec_tot,
            "UA": self.ua_val,
            "C_b": self.cb_val,
            "q_person": 0.12,  # Phát nhiệt du khách (kW/người)
            "N_base_avg": 1000,
            "T_ref": t_ref,
            "T_comf": t_comf,
            "bounds_open": bounds_open,
            "bounds_off": bounds_off,
            "w_comfort": 0.0008,  # Trọng số phạt lệch nhiệt độ tiện nghi
            "w_smooth": 2.5,     # Trọng số phạt thay đổi nhiệt độ đột ngột
        }

    def _calculer_energie_veille(self, row, baseline_e, p):
        """Tính toán công suất tiêu thụ nền (standby/đêm)."""
        e_tail = max(self.e_standby_base, baseline_e * 0.12)
        temp_ext = row.get("temperature", 15.0)

        delta_hors_gel = 0.0
        if p["saison"] == "hiver" and temp_ext < 5.0:
            delta_hors_gel = 0.08 * (5.0 - temp_ext)

        return round(e_tail + delta_hors_gel, 2)

    def _cout_horaire(
        self, T_curr, T_prev, row, baseline_e, tariff, p, e_prev_calc=None
    ):
        """Tính chi phí năng lượng và độ tiện nghi từng giờ dựa hoàn toàn vào T_set (T_curr)."""
        is_open = row["is_open"]
        operation = row.get("operation", 1.0)
        visitors = (
            row["visitor_count"] if (is_open == 1 and operation > 0) else 0.0
        )
        saison = p["saison"]

        # -------------------------------------------------------------
        # 1. TRẠNG THÁI ĐÓNG CỬA HOẶC TẮT (OFF / VEILLE / PRE-CONDITIONING)
        # -------------------------------------------------------------
        if operation == 0:
            is_off_target = (
                (saison == "ete" and T_curr >= 25.5)
                or (saison == "hiver" and T_curr <= 16.0)
                or (saison == "intersaison" and abs(T_curr - p["T_ref"]) < 0.5)
            )

            if is_off_target:
                e_veille = self._calculer_energie_veille(row, baseline_e, p)
                if e_prev_calc is not None and e_prev_calc > e_veille:
                    e_total = e_veille + (e_prev_calc - e_veille) * self.alpha_extinction
                else:
                    e_total = e_veille
                etat_cta = "OFF"
            else:
                # Chế độ Tiền điều hòa đón đầu (Pré-conditionnement)
                if saison == "hiver":
                    delta_env = (p["UA"] / p["COP"]) * (T_curr - p["T_ref"])
                    delta_inertia = (p["C_b"] / p["COP"]) * (T_curr - T_prev)
                    etat_cta = "PRE_HEATING"
                else:
                    delta_env = (p["UA"] / p["COP"]) * (p["T_ref"] - T_curr)
                    delta_inertia = (p["C_b"] / p["COP"]) * (T_prev - T_curr)
                    etat_cta = "PRE_COOLING"

                e_calc = baseline_e + delta_env + delta_inertia
                e_min = self._calculer_energie_veille(row, baseline_e, p)
                e_total = min(max(e_min, e_calc), baseline_e + p["P_elec_max"])

        # -------------------------------------------------------------
        # 2. TRẠNG THÁI MỞ CỬA HOẠT ĐỘNG (ON)
        # -------------------------------------------------------------
        else:
            if saison == "hiver":
                delta_env = (p["UA"] / p["COP"]) * (T_curr - p["T_ref"])
                delta_inertia = (p["C_b"] / p["COP"]) * (T_curr - T_prev)
            else:
                delta_env = (p["UA"] / p["COP"]) * (p["T_ref"] - T_curr)
                delta_inertia = (p["C_b"] / p["COP"]) * (T_prev - T_curr)

            delta_occ = (
                p["q_person"] * (visitors - p["N_base_avg"]) / p["COP"]
            )
            e_calc = baseline_e + delta_env + delta_inertia + delta_occ
            e_min = self._calculer_energie_veille(row, baseline_e, p)
            e_total = min(max(e_min, e_calc), baseline_e + p["P_elec_max"])

            # Đơn giản gán duy nhất 1 trạng thái là ON khi công viên mở cửa
            etat_cta = "ON"

        # -------------------------------------------------------------
        # 3. PÉNALITÉS (HÀM PHẠT TIỆN NGHI VÀ DAO ĐỘNG NHIỆT)
        # -------------------------------------------------------------
        if saison == "hiver":
            comfort_diff = max(0.0, p["T_comf"] - T_curr)
        else:
            comfort_diff = max(0.0, T_curr - p["T_comf"])

        comfort_penalty = (
            p["w_comfort"] * visitors * operation * (comfort_diff**2)
            if (visitors > 0)
            else 0.0
        )
        smooth_penalty = p["w_smooth"] * ((T_curr - T_prev) ** 2)

        total_cost = (e_total * tariff) + comfort_penalty + smooth_penalty
        return total_cost, e_total, etat_cta

    def optimiser_bloc_24h(self, df_24h):
        """Tối ưu hóa toàn bộ 24 giờ trong ngày bằng thuật toán SLSQP."""
        month = (
            df_24h["datetime_temp"].dt.month.iloc[0]
            if "datetime_temp" in df_24h.columns
            else 7
        )
        p = self._obtenir_config_saison(month)

        # Biểu giá điện theo giờ cao điểm (EDF Heures de Pointe: 8h-12h và 17h-20h)
        tariffs = [
            1.5 if (8 <= h < 12) or (17 <= h < 20) else 1.0
            for h in df_24h["hour"]
        ]
        baselines = df_24h["elec_1_pred"].values

        def objective(T_setpoints):
            total_obj = 0.0
            e_last = None
            for t in range(24):
                T_curr = T_setpoints[t]
                T_prev = T_setpoints[t - 1] if t > 0 else T_curr
                cost, e_curr, _ = self._cout_horaire(
                    T_curr,
                    T_prev,
                    df_24h.iloc[t],
                    baselines[t],
                    tariffs[t],
                    p,
                    e_prev_calc=e_last,
                )
                total_obj += cost
                e_last = e_curr
            return total_obj

        # -------------------------------------------------------------
        # THIẾT LẬP GIỚI HẠN (BOUNDS) CHO 24 GIỜ - NHÌN TRƯỚC 2 TIẾNG
        # -------------------------------------------------------------
        bounds = []
        x0 = []

        for t in range(24):
            is_open_curr = df_24h.iloc[t]["is_open"]
            op_curr = df_24h.iloc[t].get("operation", 1.0)

            # Quét trước 2 tiếng xem công viên có sắp mở cửa không
            is_preconditioning_window = False
            for lookahead in range(1, 3):
                if t + lookahead < 24:
                    if (
                        df_24h.iloc[t + lookahead]["is_open"] == 1
                        and df_24h.iloc[t + lookahead].get("operation", 1.0) > 0
                    ):
                        is_preconditioning_window = True
                        break

            if is_open_curr == 0 or op_curr == 0:
                if is_preconditioning_window:
                    bounds.append((p["bounds_open"][0], p["bounds_open"][1]))
                    x0.append(p["bounds_open"][0])  # Khởi tạo nhiệt độ đón đầu
                else:
                    bounds.append(p["bounds_off"])
                    x0.append(p["bounds_off"][0])
            else:
                bounds.append(p["bounds_open"])
                x0.append(p["T_comf"])

        # Ràng buộc tốc độ thay đổi nhiệt độ (Tối đa ±1.5°C mỗi giờ)
        constraints = []
        for t in range(1, 24):
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda T, idx=t: 1.5 - (T[idx] - T[idx - 1]),
                }
            )
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda T, idx=t: 1.5 - (T[idx - 1] - T[idx]),
                }
            )

        # Chạy thuật toán tối ưu SLSQP
        res = minimize(
            objective,
            np.array(x0),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 350},
        )

        opt_sp = res.x
        e_opt_list, cta_states, advices = [], [], []
        e_last = None

        for t in range(24):
            T_curr = opt_sp[t]
            T_prev = opt_sp[t - 1] if t > 0 else T_curr
            row = df_24h.iloc[t]

            _, e_calc, etat_cta = self._cout_horaire(
                T_curr,
                T_prev,
                row,
                baselines[t],
                tariffs[t],
                p,
                e_prev_calc=e_last,
            )
            e_opt_list.append(round(float(e_calc), 2))
            cta_states.append(etat_cta)
            e_last = e_calc

            saison_str = p["saison"].upper()
            if etat_cta == "OFF":
                adv = f"[{saison_str}] CTA OFF: Duy trì công suất nền (Đêm/Đóng cửa)"
            elif etat_cta in ["PRE_COOLING", "PRE_HEATING"]:
                adv = f"[{saison_str}] {etat_cta}: Tiền điều hòa đón đầu, cài {T_curr:.1f}°C"
            else:
                adv = f"[{saison_str}] CTA ON: Vận hành tối ưu năng lượng, cài {T_curr:.1f}°C"

            advices.append(adv)

        return np.round(opt_sp, 1), e_opt_list, cta_states, advices, p["saison"]


# =====================================================================
# 2. PIPELINE ĐIỀU HÀNH VÀ XUẤT BÁO CÁO FULL 2026
# =====================================================================
def run_full_2026_pipeline(input_filepath, output_filepath, site_name="tapis"):
    start_time = time.time()

    if not os.path.exists(input_filepath):
        raise FileNotFoundError(
            f"Fichier d'entrée introuvable : {input_filepath}"
        )

    print(f"Chargement du fichier source : {input_filepath}...")
    df = (
        pd.read_csv(input_filepath)
        if input_filepath.endswith(".csv")
        else pd.read_excel(input_filepath)
    )

    if "operation" not in df.columns:
        df["operation"] = 1.0

    df["datetime_temp"] = pd.to_datetime(
        df["date"], dayfirst=True, errors="coerce"
    )
    df["date_clean"] = df["datetime_temp"].dt.strftime("%Y-%m-%d")
    df = df.sort_values(by=["date_clean", "hour"]).reset_index(drop=True)

    real_col = next(
        (col for col in ["elec_1_real", "mesure_e", "E_real"] if col in df.columns),
        None,
    )
    if real_col and "elec_1_pred" in df.columns:
        evaluer_precision_modele(
            y_true=df[real_col].values,
            y_pred=df["elec_1_pred"].values,
            title=f"ÉVALUATION DU MODÈLE PHYSIQUE [{site_name.upper()}] vs MESURES RÉELLES ({real_col})",
        )

    # Tính toán công suất điện nền tiêu chuẩn ban đêm
    df_deep_night = df[
        ((df["hour"] >= 22) | (df["hour"] <= 5)) & (df["operation"] == 0)
    ]

    if not df_deep_night.empty and "elec_1_pred" in df_deep_night.columns:
        e_standby_calc = float(df_deep_night["elec_1_pred"].quantile(0.20))
    else:
        df_standby = df[df["operation"] == 0]
        e_standby_calc = (
            float(df_standby["elec_1_pred"].quantile(0.15))
            if not df_standby.empty
            else float(df["elec_1_pred"].quantile(0.05))
        )

    print(
        f"Consommation de talon électrique de base (E_standby_base): {e_standby_calc:.2f} kWh"
    )

    optimizer = OptimiseurConsoHVAC(site=site_name, e_standby=e_standby_calc)

    (
        recommended_setpoints,
        elec_1_optimized,
        cta_status_list,
        action_advices,
        seasons_list,
    ) = ([], [], [], [], [])
    unique_dates = df["date_clean"].unique()

    print(
        f"Optimisation multi-saisons avec décroissance en cours pour [{site_name}] ({len(unique_dates)} jours)..."
    )

    for current_date in unique_dates:
        df_day = df[df["date_clean"] == current_date]
        if len(df_day) != 24:
            recommended_setpoints.extend([24.0] * len(df_day))
            elec_1_optimized.extend(df_day["elec_1_pred"].values)
            cta_status_list.extend(["OFF"] * len(df_day))
            action_advices.extend(
                ["Données incomplètes (24h requises)"] * len(df_day)
            )
            seasons_list.extend(["inconnu"] * len(df_day))
            continue

        sp_opt, e_opt, cta_st, adv, saison = optimizer.optimiser_bloc_24h(
            df_day
        )
        recommended_setpoints.extend(sp_opt)
        elec_1_optimized.extend(e_opt)
        cta_status_list.extend(cta_st)
        action_advices.extend(adv)
        seasons_list.extend([saison] * 24)

    # Gán dữ liệu đầu ra vào DataFrame
    df["mode_saison"] = seasons_list
    df["statut_cta"] = cta_status_list
    df["consigne_recommandee"] = recommended_setpoints
    df["consigne_affichee"] = [
        "OFF" if st == "OFF" else f"{sp}°C"
        for st, sp in zip(df["statut_cta"], df["consigne_recommandee"])
    ]
    df["elec_1_optimise"] = elec_1_optimized
    df["economie_kwh"] = np.round(df["elec_1_pred"] - df["elec_1_optimise"], 2)
    df["conseil_action"] = action_advices

    total_baseline = df["elec_1_pred"].sum()
    total_optimised = df["elec_1_optimise"].sum()
    total_savings = df["economie_kwh"].sum()
    pct_savings = (
        (total_savings / total_baseline) * 100.0 if total_baseline > 0 else 0.0
    )

    print("\n" + "=" * 65)
    print("   RAPPORT SYNTHÉTIQUE D'ÉCONOMIE D'ÉNERGIE MULTI-SAISON")
    print("=" * 65)
    print(
        f"   • Consommation initiale (Baseline)  : {total_baseline:,.2f} kWh"
    )
    print(
        f"   • Consommation après optimisation   : {total_optimised:,.2f} kWh"
    )
    print(
        f"   • Économie d'énergie globale        : {total_savings:,.2f} kWh ({pct_savings:.2f}%)"
    )

    for s in ["hiver", "ete", "intersaison"]:
        df_s = df[df["mode_saison"] == s]
        if not df_s.empty:
            s_base = df_s["elec_1_pred"].sum()
            s_opt = df_s["elec_1_optimise"].sum()
            s_gain = s_base - s_opt
            s_pct = (s_gain / s_base) * 100.0 if s_base > 0 else 0.0
            print(
                f"    -> Saison [{s.upper():<11}]: Gain de {s_gain:,.2f} kWh ({s_pct:.2f}%)"
            )

    print("=" * 65 + "\n")

    export_cols = [
        "date",
        "hour",
        "mode_saison",
        "is_open",
        "operation",
        "visitor_count",
        "temperature",
        "elec_1_pred",
        "statut_cta",
        "consigne_affichee",
        "elec_1_optimise",
        "economie_kwh",
        "conseil_action",
    ]
    df_export = df[[c for c in export_cols if c in df.columns]]

    if output_filepath.endswith(".csv"):
        df_export.to_csv(output_filepath, index=False)
    else:
        df_export.to_excel(output_filepath, index=False)

    print(
        f"Processus terminé en {time.time() - start_time:.2f} s. Fichier généré : {output_filepath}\n"
    )


if __name__ == "__main__":
    run_full_2026_pipeline(
        "predict_2026.csv", "setpoint_H03_2026.csv", site_name="tapis"
    )