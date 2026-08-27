import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class OptimiseurConsoThermique:
    """
    Optimiseur d'Énergie Calorifique / Thermique (EC)
    Applicable au bâtiment H07 (La vienne) et au Pavillon de la Vienne.
    
    Tích hợp:
    - Tự động xác định công suất điện/nhiệt nền (Standby/Talon) ban đêm theo Quantile.
    - Cửa sổ dự báo đón đầu (2h Lookahead) cho trạng thái PRE_HEATING.
    - Tối ưu hóa chuỗi thời gian 24h sử dụng thuật toán SLSQP có ràng buộc gradient nhiệt.
    """
    def __init__(self, site="vienne", ec_standby=2.0):
        self.site = site.lower()
        self.ec_standby_base = ec_standby
        self.alpha_extinction = 0.20  # Coefficient d'atténuation thermique lors de l'arrêt CTA

        # Configuration technique issue du tableau d'audit CTA
        if self.site == "vienne":
            self.p_chaud_tot = 285.0   # CTA1 (165 kW) + CTA2 (120 kW)
            self.p_froid_tot = 410.0   # CTA1 (215 kW) + CTA2 (195 kW)
            self.ua_val = 6.5          # Déperdition enveloppe (kW/K)
            self.cb_val = 25.0         # Inertie thermique du bâtiment (kWh/K)
            self.surface_glass = 120.0 # Surface vitrée exposée (m²)
        else:  # Tapis Magique (H03)
            self.p_chaud_tot = 325.0   # CTA1 (105 kW) + CTA2 (220 kW) - Norme Audit
            self.p_froid_tot = 300.0   # CTA1 (160 kW) + CTA2 (140 kW)
            self.ua_val = 4.5          # kW/K
            self.cb_val = 18.0         # kWh/K
            self.surface_glass = 85.0  # m²

    def _xac_dinh_ec_standby_dynamique(self, df_24h, quantile=0.15):
        """
        Tự động xác định công suất nền (standby/talon) dựa trên phân vị (quantile)
        của dữ liệu dự báo/lịch sử trong khung giờ đêm (22h -> 05h).
        """
        # Lọc khung giờ đêm (22h - 23h và 00h - 04h)
        mask_nuit = (df_24h["hour"] >= 22) | (df_24h["hour"] < 5)
        df_nuit = df_24h[mask_nuit]

        if not df_nuit.empty and "ec_value_pred" in df_nuit.columns:
            # Phân vị Q15 giúp loại bỏ các đỉnh tải bất thường ban đêm
            ec_standby_calc = df_nuit["ec_value_pred"].quantile(quantile)
            return max(0.5, float(ec_standby_calc))
        
        return self.ec_standby_base

    def _obtenir_config_saison(self, month):
        """Détermine les seuils de confort et paramètres selon la saison."""
        if month in [11, 12, 1, 2, 3]:
            saison = "hiver"
            t_ref = 19.0
            t_comf = 19.5
            bounds_open = (18.5, 20.5)
            bounds_off = (15.0, 15.0)  # Seuil Hors-gel
        elif month in [5, 6, 7, 8, 9]:
            saison = "ete"
            t_ref = 24.0
            t_comf = 23.5
            bounds_open = (23.0, 25.0)
            bounds_off = (26.0, 26.0)
        else:
            saison = "intersaison"
            t_ref = 20.5
            t_comf = 20.5
            bounds_open = (19.5, 21.5)
            bounds_off = (17.0, 17.0)

        return {
            "saison": saison,
            "P_chaud_max": self.p_chaud_tot,
            "UA": self.ua_val,
            "C_b": self.cb_val,
            "q_person": 0.08,        # Apport thermique par occupant (kW/personne)
            "eta_sol": 0.35,          # Facteur d'absorption du rayonnement solaire
            "T_ref": t_ref,
            "T_comf": t_comf,
            "bounds_open": bounds_open,
            "bounds_off": bounds_off,
            "w_comfort": 0.005,       # Pénalité d'inconfort ajustée
            "w_smooth": 0.5,          # Pénalité de variation brusque de consigne
        }

    def _calculer_energie_veille(self, row, ec_standby_dynamique, p):
        """Calcule la consommation de talon/hors-gel nocturne avec linh hoạt nhiệt độ ngoài trời."""
        temp_ext = row.get("temperature", 10.0)

        delta_hors_gel = 0.0
        if p["saison"] == "hiver" and temp_ext < 3.0:
            delta_hors_gel = 0.12 * (3.0 - temp_ext)

        return round(ec_standby_dynamique + delta_hors_gel, 2)

    def _cout_horaire_thermique(self, T_curr, T_prev, row, baseline_ec, tariff, p, etat_cta, ec_prev_calc=None):
        """
        Calcule la consommation d'énergie thermique EC (kWh) et le coût horaire.
        Le paramètre etat_cta est explicite pour garantir la continuité du gradient SLSQP.
        """
        is_open = row.get("is_open", 0)
        operation = row.get("operation", 1.0)
        visitors = row.get("visitor_count", 0.0) if (is_open == 1 and operation > 0) else 0.0
        sol_rad = row.get("rayonnement_solaire", 0.0) if operation > 0 else 0.0

        if p["saison"] == "ete":
            return 0.0, 0.0, "OFF"

        # 1. ÉTAT ÉTEINT (OFF)
        if etat_cta == "OFF":
            ec_veille = self._calculer_energie_veille(row, p["ec_standby"], p)
            if ec_prev_calc is not None and ec_prev_calc > ec_veille:
                ec_total = ec_veille + (ec_prev_calc - ec_veille) * self.alpha_extinction
            else:
                ec_total = ec_veille

        # 2. ÉTAT PRÉ-CHAUFFAGE (PRE_HEATING)
        elif etat_cta == "PRE_HEATING":
            delta_env = p["UA"] * (T_curr - p["T_ref"])
            delta_inertia = p["C_b"] * max(0.0, T_curr - T_prev)
            ec_calc = baseline_ec + delta_env + delta_inertia
            ec_min = self._calculer_energie_veille(row, p["ec_standby"], p)
            ec_total = min(max(ec_min, ec_calc), p["P_chaud_max"])

        # 3. ÉTAT OUVERT AUX VISITEURS (ON)
        else:
            delta_env = p["UA"] * (T_curr - p["T_ref"])
            delta_inertia = p["C_b"] * max(0.0, T_curr - T_prev)

            # Apports gratuits (visiteurs et rayonnement solaire)
            gain_occ = p["q_person"] * (visitors / 100.0)
            gain_sol = p["eta_sol"] * (sol_rad / 1000.0) * self.surface_glass

            ec_calc = baseline_ec + delta_env + delta_inertia - gain_occ - gain_sol
            ec_min = self._calculer_energie_veille(row, p["ec_standby"], p)
            ec_total = min(max(ec_min, ec_calc), p["P_chaud_max"])

        # 4. FONCTIONS DE PÉNALITÉ (Régularisation)
        comfort_diff = max(0.0, p["T_comf"] - T_curr)
        comfort_penalty = (p["w_comfort"] * visitors * operation * (comfort_diff ** 2)) if visitors > 0 else 0.0
        smooth_penalty = p["w_smooth"] * ((T_curr - T_prev) ** 2)

        total_cost = (ec_total * tariff) + comfort_penalty + smooth_penalty
        return total_cost, ec_total, etat_cta

    def optimiser_bloc_24h(self, df_24h):
        """Optimisation par fenêtre de 24 heures via l'algorithme SLSQP."""
        month = df_24h["datetime_temp"].dt.month.iloc[0] if "datetime_temp" in df_24h.columns else 1
        p = self._obtenir_config_saison(month)

        if p["saison"] == "ete":
            return np.full(24, 24.0), [0.0]*24, ["OFF"]*24, ["Saison estivale : Chauffage EC OFF"]*24, "ete"

        # --- TỰ ĐỘNG XÁC ĐỊNH CÔNG SUẤT NỀN (STANDBY) DYNAMIC CHO 24H HÔM NAY ---
        ec_standby_day = self._xac_dinh_ec_standby_dynamique(df_24h, quantile=0.15)
        p["ec_standby"] = ec_standby_day

        tariffs = [1.5 if (8 <= h < 12) or (17 <= h < 20) else 1.0 for h in df_24h["hour"]]
        baselines = df_24h["ec_value_pred"].values

        # Deterministic identification of CTA states and preheating windows (2h Lookahead)
        cta_states_pred = []
        bounds = []

        for t in range(24):
            is_open_curr = df_24h.iloc[t]["is_open"]
            op_curr = df_24h.iloc[t].get("operation", 1.0)

            # Lookahead 2h window
            is_preheating_window = False
            for lookahead in range(1, 3):
                if t + lookahead < 24:
                    if df_24h.iloc[t + lookahead]["is_open"] == 1 and df_24h.iloc[t + lookahead].get("operation", 1.0) > 0:
                        is_preheating_window = True
                        break

            if is_open_curr == 0 or op_curr == 0:
                if is_preheating_window:
                    cta_states_pred.append("PRE_HEATING")
                    bounds.append((p["bounds_off"][0], p["bounds_open"][1]))
                else:
                    cta_states_pred.append("OFF")
                    bounds.append(p["bounds_off"])
            else:
                cta_states_pred.append("ON")
                bounds.append(p["bounds_open"])

        # Construct feasible initial guess (x0) satisfying ramp constraints
        x0 = []
        for t in range(24):
            state = cta_states_pred[t]
            if state == "OFF":
                x0.append(bounds[t][0])
            elif state == "ON":
                x0.append(p["T_comf"])
            else:  # PRE_HEATING
                prev_val = x0[t-1] if t > 0 else bounds[t][0]
                x0.append(min(prev_val + 1.2, p["bounds_open"][0]))

        x0 = np.array(x0)

        # Objective Function
        def objective(T_setpoints):
            total_obj = 0.0
            ec_last = None
            for t in range(24):
                T_curr = T_setpoints[t]
                T_prev = T_setpoints[t - 1] if t > 0 else T_curr
                cost, ec_curr, _ = self._cout_horaire_thermique(
                    T_curr, T_prev, df_24h.iloc[t], baselines[t], tariffs[t], p,
                    etat_cta=cta_states_pred[t], ec_prev_calc=ec_last
                )
                total_obj += cost
                ec_last = ec_curr
            return total_obj

        # Thermal gradient constraints (max 1.5°C/h)
        constraints = []
        for t in range(1, 24):
            constraints.append({"type": "ineq", "fun": lambda T, idx=t: 1.5 - (T[idx] - T[idx - 1])})
            constraints.append({"type": "ineq", "fun": lambda T, idx=t: 1.5 - (T[idx - 1] - T[idx])})

        res = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 350}
        )

        opt_sp = res.x if res.success else x0
        ec_opt_list, cta_states, advices = [], [], []
        ec_last = None

        for t in range(24):
            T_curr = opt_sp[t]
            T_prev = opt_sp[t - 1] if t > 0 else T_curr
            row = df_24h.iloc[t]

            _, ec_calc, etat_cta = self._cout_horaire_thermique(
                T_curr, T_prev, row, baselines[t], tariffs[t], p,
                etat_cta=cta_states_pred[t], ec_prev_calc=ec_last
            )
            ec_opt_list.append(round(float(ec_calc), 2))
            cta_states.append(etat_cta)
            ec_last = ec_calc

            saison_str = p["saison"].upper()
            if etat_cta == "OFF":
                adv = f"[{saison_str}] EC OFF : Maintien hors-gel / Talon calculé ({ec_standby_day:.2f} kWh)"
            elif etat_cta == "PRE_HEATING":
                adv = f"[{saison_str}] PRE_HEATING : Pré-chauffage d'anticipation, consigne à {T_curr:.1f}°C"
            else:
                adv = f"[{saison_str}] EC ON : Chauffage optimisé, consigne à {T_curr:.1f}°C"

            advices.append(adv)

        return np.round(opt_sp, 1), ec_opt_list, cta_states, advices, p["saison"]


# --- EXÉCUTION & EXPORTATION ---
if __name__ == "__main__":
    input_file = "predict_2026.csv"

    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        df["datetime_temp"] = pd.to_datetime(df["date"])

        optimizer = OptimiseurConsoThermique(site="vienne", ec_standby=2.0)

        all_results = []
        for start_idx in range(0, len(df), 24):
            df_day = df.iloc[start_idx:start_idx + 24].copy()
            if len(df_day) < 24:
                break
                
            sp_opt, ec_opt, cta_states, advices, saison = optimizer.optimiser_bloc_24h(df_day)
            
            df_day["EC_optimisee (kWh)"] = ec_opt
            df_day["Consigne_T (°C)"] = sp_opt
            df_day["Etat_CTA"] = cta_states
            df_day["Recommandation_Operationnelle"] = advices
            df_day["Economie (kWh)"] = (df_day["ec_value_pred"] - df_day["EC_optimisee (kWh)"]).round(2)
            
            all_results.append(df_day)

        df_final = pd.concat(all_results, ignore_index=True)

        output_cols = {
            "date": "Horodatage",
            "hour": "Heure",
            "temperature": "Température extérieure (°C)",
            "is_open": "Statut ouverture",
            "visitor_count": "Fréquentation prévue",
            "ec_value_pred": "Consommation EC prédite (kWh)",
            "EC_optimisee (kWh)": "Consommation EC optimisée (kWh)",
            "Consigne_T (°C)": "Consigne de température (°C)",
            "Etat_CTA": "État CTA",
            "Economie (kWh)": "Économie réalisée (kWh)",
            "Recommandation_Operationnelle": "Recommandations opérationnelles"
        }

        df_export = df_final[list(output_cols.keys())].rename(columns=output_cols)

        excel_path = "Planning_Optimisation_EC_H07.xlsx"
        csv_path = "Planning_Optimisation_EC_H07.csv"

        df_export.to_excel(excel_path, index=False, engine='openpyxl')
        df_export.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print("=== OPTIMISATION ET EXPORTATION TERMINÉES ===")
        print(f"1. Fichier Excel : {excel_path}")
        print(f"2. Fichier CSV   : {csv_path}")
    else:
        print(f"Fichier '{input_file}' introuvable. Veuillez vérifier le chemin du fichier.")