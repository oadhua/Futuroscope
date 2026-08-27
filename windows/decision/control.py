import asyncio
import time
import numpy as np
import pandas as pd
from asyncua import Client
from scipy.optimize import minimize

# =====================================================================
# 1. CONFIGURATION DU SYSTÈME ET PARAMÈTRES PHYSIQUES
# =====================================================================
PANORAMA_URL = "opc.tcp://127.0.0.1:4840/freeopcua/server/"
URI_PANORAMA = "http://futuroscope.hvac.panorama"

# Amplitude maximale de variation de la consigne par pas de calcul (°C)
DELTA_T_MAX = 3.0  # Modifier à 1.5 si vous souhaitez des variations plus douces

ATTRACTION_CONFIGS = {
    "H07": {
        "name": "La Vienne Dynamique",
        "p_froid_tot": 410.0,
        "p_chaud_tot": 285.0,
        "p_elec_tot": 32.5,
        "cop_chaud": 3.2,  # COP vận hành mùa đông (Chauffage)
        "cop_froid": 3.8,  # EER/COP vận hành mùa hè (Refroidissement)
        "ua_val": 6.5,
        "cb_val": 25.0,
        "e_standby": 4.0,
        "ratio_hvac_elec": 0.55,
        "file_elec": r"D:\Stage SI\Machine Learning\Futuroscope\windows\model\H07_elec\before\predict_2026.csv",
        "file_thermal": r"D:\Stage SI\Machine Learning\Futuroscope\windows\model\H07_thermal\before\predict_2026.csv",
    },
    "H03": {
        "name": "Tapis Magique",
        "p_froid_tot": 300.0,
        "p_chaud_tot": 325.0,
        "p_elec_tot": 30.2,
        "cop_chaud": 3.0,  # COP vận hành mùa đông (Chauffage)
        "cop_froid": 3.5,  # EER/COP vận hành mùa hè (Refroidissement)
        "ua_val": 4.5,
        "cb_val": 18.0,
        "e_standby": 3.5,
        "ratio_hvac_elec": 0.65,
        "file_elec": r"D:\Stage SI\Machine Learning\Futuroscope\windows\model\H03_elec\before\predict_2026.csv",
        "file_thermal": r"D:\Stage SI\Machine Learning\Futuroscope\windows\model\H03_thermal\before\predict_2026.csv",
    },
}


class OptimiseurConsoHVAC:
    def __init__(self, config):
        self.config = config
        self.cop_chaud = config.get("cop_chaud", 3.0)
        self.cop_froid = config.get("cop_froid", 3.5)
        self.ua_val = config["ua_val"]
        self.cb_val = config["cb_val"]
        self.e_standby_base = config["e_standby"]
        self.ratio_hvac = config.get("ratio_hvac_elec", 0.60)

    def _obtenir_config_saison(self, month):
        """Définit la température de confort cible globale selon la saison."""
        if month in [11, 12, 1, 2, 3]:
            saison = "hiver"
            t_comf = 19.0  # Cible de confort hivernale
            cop_active = self.cop_chaud
        elif month in [5, 6, 7, 8, 9]:
            saison = "ete"
            t_comf = 24.0  # Cible de confort estivale idéale
            cop_active = self.cop_froid
        else:
            saison = "intersaison"
            t_comf = 22.5
            cop_active = (self.cop_chaud + self.cop_froid) / 2.0

        return {
            "saison": saison,
            "COP": cop_active,
            "UA": self.ua_val,
            "C_b": self.cb_val,
            "T_comf": t_comf,
            "w_comfort": 20.0,
            "w_smooth": 0.5,
            "w_scada": 1.2,
        }

    def _cout_horaire(
        self,
        T_curr,
        T_prev,
        row,
        baseline_val,
        tariff,
        p,
        T_scada_prev=None,
        hours_closed=0,
        is_closing_soon=False,
    ):
        is_open = int(row.get("is_open", 0))
        operation = float(row.get("operation", 1.0))
        visitors = (
            float(row.get("visitor_count", 0.0))
            if (is_open == 1 and operation > 0)
            else 0.0
        )
        saison = p["saison"]
        base_load = self.e_standby_base * 1.2

        # -------------------------------------------------------------
        # 1. ATTRACTION FERMÉE (Hors exploitation)
        # -------------------------------------------------------------
        if operation == 0 or is_open == 0:
            decay_factor = np.exp(-0.4 * hours_closed)
            if saison == "hiver":
                base_hg = max(self.e_standby_base, baseline_val * 0.15)
                delta_q = p["UA"] * (T_curr - p["T_comf"]) + p["C_b"] * (
                    T_curr - T_prev
                )
                q_opt = (
                    base_hg
                    + (baseline_val - base_hg) * decay_factor
                    + max(0.0, delta_q)
                )
                off_penalty = 100.0 * ((T_curr - 15.0) ** 2)
                return (
                    (q_opt * tariff) + off_penalty,
                    baseline_val,
                    q_opt,
                    baseline_val - q_opt,
                    "kWh_th",
                    "HORS_GEL",
                )
            else:
                e_opt = (
                    self.e_standby_base
                    + (baseline_val - self.e_standby_base) * decay_factor
                )
                return (
                    e_opt * tariff,
                    baseline_val,
                    e_opt,
                    baseline_val - e_opt,
                    "kWh_el",
                    "OFF",
                )

        # -------------------------------------------------------------
        # 2. ATTRACTION EN EXPLOITATION
        # -------------------------------------------------------------
        if saison == "hiver":
            q_pred = float(row.get("ec_value_pred", baseline_val))
            delta_q = p["UA"] * (T_curr - p["T_comf"]) + p["C_b"] * (T_curr - T_prev)
            q_opt = (
                max(base_load, (q_pred + delta_q) * 0.3)
                if is_closing_soon
                else max(base_load, q_pred + delta_q)
            )
            energy_pred, energy_opt, unit = q_pred, q_opt, "kWh_th"
            cost_base = energy_opt * tariff
            etat_cta = "EARLY_STOP" if is_closing_soon else "ON"
        else:
            e_pred_tot = float(row.get("elec_1_pred", baseline_val))
            e_hvac_baseline = e_pred_tot * self.ratio_hvac
            e_non_hvac = e_pred_tot * (1.0 - self.ratio_hvac)

            # Bilan thermique RC appliqué au chiller avec COP_froid
            delta_e_hvac = (p["UA"] / p["COP"]) * (p["T_comf"] - T_curr) + (
                p["C_b"] / p["COP"]
            ) * (T_prev - T_curr)

            e_hvac_opt = (
                max(base_load, (e_hvac_baseline + delta_e_hvac) * 0.3)
                if is_closing_soon
                else max(base_load, e_hvac_baseline + delta_e_hvac)
            )

            energy_pred = e_pred_tot
            energy_opt = e_non_hvac + e_hvac_opt
            unit = "kWh_el"
            cost_base = energy_opt * tariff
            etat_cta = "EARLY_STOP" if is_closing_soon else "ON"

        # Pénalité de confort centrée sur T_comf
        comfort_diff = (
            max(0.0, 18.0 - T_curr)
            if saison == "hiver"
            else max(0.0, T_curr - p["T_comf"])
        )
        scale_visitor = max(1.0, visitors / 50.0)
        w_comfort_eff = p["w_comfort"] * 0.20 if is_closing_soon else p["w_comfort"]
        comfort_penalty = w_comfort_eff * scale_visitor * (comfort_diff**4)

        smooth_penalty = p["w_smooth"] * ((T_curr - T_prev) ** 2)
        scada_penalty = (
            p["w_scada"] * ((T_curr - T_scada_prev) ** 2) if T_scada_prev else 0.0
        )

        total_cost = cost_base + comfort_penalty + smooth_penalty + scada_penalty
        return (
            total_cost,
            energy_pred,
            energy_opt,
            energy_pred - energy_opt,
            unit,
            etat_cta,
        )

    def optimiser_bloc_24h(self, df_24h, T_scada_prev=None):
        month = pd.to_datetime(df_24h["datetime_temp"].iloc[0]).month
        p = self._obtenir_config_saison(month)

        tariffs = [
            1.5 if (8 <= h < 12) or (17 <= h < 20) else 1.0 for h in df_24h["hour"]
        ]

        col_baseline = "ec_value_pred" if p["saison"] == "hiver" else "elec_1_pred"
        if col_baseline not in df_24h.columns:
            col_baseline = (
                "elec_1_pred" if "elec_1_pred" in df_24h.columns else "ec_value_pred"
            )

        baselines = df_24h[col_baseline].values
        n_steps = len(df_24h)

        hours_closed_list = []
        c_closed = 0
        for idx in range(n_steps):
            row = df_24h.iloc[idx]
            if float(row.get("operation", 1.0)) == 0 or int(row.get("is_open", 0)) == 0:
                c_closed += 1
            else:
                c_closed = 0
            hours_closed_list.append(c_closed)

        bounds, x0 = [], []
        PRE_HOURS = 2

        for t in range(n_steps):
            row = df_24h.iloc[t]
            op = float(row.get("operation", 1.0))
            is_op = int(row.get("is_open", 0))

            will_open_soon = False
            for lookahead in range(1, PRE_HOURS + 1):
                if t + lookahead < n_steps:
                    next_row = df_24h.iloc[t + lookahead]
                    if (
                        float(next_row.get("operation", 1.0)) > 0
                        and int(next_row.get("is_open", 0)) == 1
                    ):
                        will_open_soon = True
                        break

            if op == 0 or is_op == 0:
                if will_open_soon:
                    if p["saison"] == "hiver":
                        bounds.append((15.0, 19.0))
                        x0.append(17.0)
                    else:
                        bounds.append((22.0, 26.0))
                        x0.append(24.5)
                else:
                    if p["saison"] == "hiver":
                        bounds.append((15.0, 15.0))
                        x0.append(15.0)
                    else:
                        bounds.append((26.0, 30.0))
                        x0.append(28.0)
            else:
                t_int_ref = (
                    T_scada_prev
                    if (T_scada_prev is not None and T_scada_prev > 10.0)
                    else p["T_comf"]
                )

                if p["saison"] == "hiver":
                    b_min = max(17.0, t_int_ref - DELTA_T_MAX)
                    b_max = min(21.0, t_int_ref + DELTA_T_MAX)
                else:
                    b_min = max(22.0, t_int_ref - DELTA_T_MAX)
                    b_max = min(26.0, t_int_ref + DELTA_T_MAX)

                if b_min > b_max:
                    b_min, b_max = (
                        (18.0, 20.0) if p["saison"] == "hiver" else (23.0, 25.0)
                    )

                x0_val = np.clip(t_int_ref, b_min, b_max)
                bounds.append((b_min, b_max))
                x0.append(x0_val)

        def objective(T_setpoints):
            total_obj = 0.0
            for t in range(n_steps):
                T_curr = T_setpoints[t]
                T_prev = (
                    T_scada_prev
                    if (t == 0 and T_scada_prev is not None)
                    else (T_setpoints[t - 1] if t > 0 else T_curr)
                )

                cost, _, _, _, _, _ = self._cout_horaire(
                    T_curr,
                    T_prev,
                    df_24h.iloc[t],
                    baselines[t],
                    tariffs[t],
                    p,
                    hours_closed=hours_closed_list[t],
                )

                smooth_penalty = p["w_smooth"] * ((T_curr - T_prev) ** 2)

                if t + 1 < n_steps:
                    next_row = df_24h.iloc[t + 1]
                    if (
                        float(next_row.get("operation", 1.0)) > 0
                        and int(next_row.get("is_open", 0)) == 1
                        and (
                            float(df_24h.iloc[t].get("operation", 1.0)) == 0
                            or int(df_24h.iloc[t].get("is_open", 0)) == 0
                        )
                    ):
                        arrival_penalty = 50.0 * (max(0.0, p["T_comf"] - T_curr) ** 2)
                        cost += arrival_penalty

                total_obj += cost + smooth_penalty
            return total_obj

        res = minimize(
            objective,
            np.array(x0),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-5},
        )
        opt_sp = res.x if res.success else np.array(x0)

        e_pred_list, e_opt_list, e_saved_list, cta_states = [], [], [], []
        unit_res = "kWh"

        for t in range(n_steps):
            T_curr = opt_sp[t]
            T_prev = (
                T_scada_prev
                if (t == 0 and T_scada_prev is not None)
                else (opt_sp[t - 1] if t > 0 else T_curr)
            )

            _, e_pred, e_opt, e_saved, unit_res, etat_cta = self._cout_horaire(
                T_curr,
                T_prev,
                df_24h.iloc[t],
                baselines[t],
                tariffs[t],
                p,
                hours_closed=hours_closed_list[t],
            )

            row = df_24h.iloc[t]
            if float(row.get("operation", 1.0)) == 0 and t + 1 < n_steps:
                if float(df_24h.iloc[t + 1].get("operation", 1.0)) > 0:
                    etat_cta = (
                        "PRE_HEATING" if p["saison"] == "hiver" else "PRE_COOLING"
                    )

            e_pred_list.append(round(float(e_pred), 2))
            e_opt_list.append(round(float(e_opt), 2))
            e_saved_list.append(round(float(e_saved), 2))
            cta_states.append(etat_cta)

        return (
            np.round(opt_sp, 1),
            e_pred_list,
            e_opt_list,
            e_saved_list,
            unit_res,
            cta_states,
            p["saison"],
        )


# =====================================================================
# 2. CHARGEMENT DES DONNÉES ET PRÉDICTIONS
# =====================================================================
def load_and_merge_predictions(file_elec, file_thermal):
    try:
        df_elec = pd.read_csv(file_elec)
        df_thermal = pd.read_csv(file_thermal)
    except Exception:
        df_thermal = pd.read_csv("predict_2026.csv")
        df_elec = df_thermal.copy()

    df_elec["date"] = pd.to_datetime(df_elec["date"])
    df_thermal["date"] = pd.to_datetime(df_thermal["date"])

    df_merged = pd.merge(
        df_thermal,
        df_elec,
        on="date",
        how="inner",
        suffixes=("", "_elec"),
    )

    if "hour" not in df_merged.columns:
        df_merged["hour"] = df_merged["date"].dt.hour

    if "elec_1_pred" not in df_merged.columns and "ec_value_pred" in df_merged.columns:
        df_merged["elec_1_pred"] = df_merged["ec_value_pred"]
    elif (
        "ec_value_pred" not in df_merged.columns and "elec_1_pred" in df_merged.columns
    ):
        df_merged["ec_value_pred"] = df_merged["elec_1_pred"]

    return df_merged


GLOBAL_PREDICTIONS = {
    attr_id: load_and_merge_predictions(cfg["file_elec"], cfg["file_thermal"])
    for attr_id, cfg in ATTRACTION_CONFIGS.items()
}


def construct_24h_dataframe(
    attr_id, curr_temp, is_open, operation, visitors, scada_time
):
    start_dt = pd.to_datetime(scada_time).floor("h")
    df_global = GLOBAL_PREDICTIONS[attr_id]

    df_24h = df_global[df_global["date"] >= start_dt].head(24).copy()
    if len(df_24h) < 24:
        df_24h = df_global.tail(24).copy()

    df_24h = df_24h.reset_index(drop=True)

    df_24h.loc[0, "temperature"] = curr_temp
    df_24h.loc[0, "is_open"] = is_open
    df_24h.loc[0, "operation"] = operation
    df_24h.loc[0, "visitor_count"] = visitors

    df_24h["datetime_temp"] = df_24h["date"]
    return df_24h


# =====================================================================
# 3. ÉMISSION DES PRÉCONISATIONS VERS LE SCADA PANORAMA
# =====================================================================
async def traiter_attraction(attr_id, config, client, idx):
    optimizer = OptimiseurConsoHVAC(config)

    node_timestamp = client.get_node(f"ns={idx};s={attr_id}_timestamp")
    node_temp = client.get_node(f"ns={idx};s={attr_id}_temperature")
    node_is_open = client.get_node(f"ns={idx};s={attr_id}_is_open")
    node_operation = client.get_node(f"ns={idx};s={attr_id}_operation")
    node_visitors = client.get_node(f"ns={idx};s={attr_id}_visitor_count")

    node_chaud = client.get_node(f"ns={idx};s={attr_id}_Consigne_Chaud")
    node_froid = client.get_node(f"ns={idx};s={attr_id}_Consigne_Froid")

    scada_time = await node_timestamp.get_value()
    curr_temp = await node_temp.get_value()
    is_open = await node_is_open.get_value()
    operation = await node_operation.get_value()
    visitors = await node_visitors.get_value()

    val_chaud_prev = await node_chaud.get_value()
    val_froid_prev = await node_froid.get_value()
    t_scada_prev = val_froid_prev if curr_temp > 20.0 else val_chaud_prev

    df_24h = construct_24h_dataframe(
        attr_id, curr_temp, is_open, operation, visitors, scada_time
    )

    opt_sp, e_pred, e_opt, e_saved, unit, cta_st, saison = optimizer.optimiser_bloc_24h(
        df_24h, T_scada_prev=t_scada_prev
    )

    target_setpoint = float(opt_sp[0])
    current_cta_state = cta_st[0]

    e_pred_now = e_pred[0]
    e_opt_now = e_opt[0]
    e_saved_now = e_saved[0]
    pct_saved_now = (e_saved_now / e_pred_now * 100) if e_pred_now > 0 else 0.0

    total_pred_24h = sum(e_pred)
    total_opt_24h = sum(e_opt)
    total_saved_24h = sum(e_saved)
    pct_saved_24h = (
        (total_saved_24h / total_pred_24h * 100) if total_pred_24h > 0 else 0.0
    )

    if saison == "hiver":
        sp_chaud, sp_froid = target_setpoint, 28.0
    elif saison == "ete":
        sp_chaud, sp_froid = 12.0, target_setpoint
    else:
        sp_chaud, sp_froid = min(target_setpoint, 19.5), max(target_setpoint, 23.5)

    await node_chaud.write_value(sp_chaud)
    await node_froid.write_value(sp_froid)

    status_parc = "OUVERT" if is_open == 1 else "FERMÉ"
    saison_label = "CHAUFFAGE" if saison == "hiver" else "REFROIDISSEMENT"

    print(
        f"  🏛️ [{attr_id} - {config['name']}] Temp Ext: {curr_temp:.1f}°C |"
        f" Parc: {status_parc} | Régime: {saison_label}"
    )
    print(
        f"     ➔ Consigne optimale recommandée: {target_setpoint:.1f}°C (Chaud: {sp_chaud:.1f}°C"
        f" / Froid: {sp_froid:.1f}°C) | CTA: {current_cta_state}"
    )
    print(
        f"     📊 [HEURE ACTUELLE] Initial ML (Pred): {e_pred_now:.2f} {unit} |"
        f" Optimisé (Opt): {e_opt_now:.2f} {unit}"
    )
    print(
        f"     💡 [ÉCONOMIE DÉTECTÉE] ΔE: {e_saved_now:+.2f} {unit}"
        f" ({pct_saved_now:+.1f}%)"
    )
    print(
        f"     📈 [CUMUL 24H] Économie totale projetée 24h:"
        f" {total_saved_24h:+.2f} {unit} ({pct_saved_24h:+.1f}%)\n"
    )


# =====================================================================
# 4. BOUCLE PRINCIPALE D'OPTIMISATION (MAIN LOOP)
# =====================================================================
async def main():
    print(
        "🚀 [MOTEUR DE CONTRÔLE CVC] Démarrage du moteur d'optimisation"
        " Multi-Attractions..."
    )
    INTERVALLE_TEST = 2.0

    while True:
        t_debut = time.time()

        try:
            async with Client(url=PANORAMA_URL) as client:
                idx = await client.get_namespace_index(URI_PANORAMA)

                print("\n" + "=" * 75)
                print(
                    "⏱️ [CYCLE DE CALCUL SCADA] Exécution de l'optimisation"
                    " multi-attractions"
                )
                print("-" * 75)

                tasks = [
                    traiter_attraction(attr_id, config, client, idx)
                    for attr_id, config in ATTRACTION_CONFIGS.items()
                ]
                await asyncio.gather(*tasks)

                print("=" * 75)

        except Exception as e:
            print(f"❌ Erreur lors du traitement SCADA: {e}")

        t_ecoule = time.time() - t_debut
        t_attente = max(0.0, INTERVALLE_TEST - t_ecoule)
        await asyncio.sleep(t_attente)


if __name__ == "__main__":
    asyncio.run(main())