import os
import io
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy import text
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import plotly.graph_objects as go
import plotly.io as pio

from app.core.database import engine
from app.modules.ml_pipeline.training.training_services import (
    SCHEMA_NAME,
    MODEL_STORAGE_DIR,
    MLTrainingService,
    create_sequences,
)
from app.modules.ml_pipeline.ml_inference.ml_inference_schemas import ExportFileFormat

logger = logging.getLogger(__name__)


class MLInferenceService:
    @classmethod
    def load_model_artifacts(cls, model_id: str) -> Dict[str, Any]:
        """Charge le fichier modèle (.joblib) et les poids Keras (.keras pour LSTM/GRU)."""
        joblib_path = os.path.join(MODEL_STORAGE_DIR, f"{model_id}.joblib")
        if not os.path.exists(joblib_path):
            raise ValueError(
                f"Fichier modèle introuvable '{model_id}.joblib' dans '{MODEL_STORAGE_DIR}'."
            )

        saved_data = joblib.load(joblib_path)
        m_type = saved_data.get("model_type")

        if m_type in ["lstm", "gru"]:
            keras_path = os.path.join(MODEL_STORAGE_DIR, f"{model_id}.keras")
            if not os.path.exists(keras_path):
                raise ValueError(
                    f"Fichier modèle Keras introuvable '{model_id}.keras'."
                )
            saved_data["tf_model"] = tf.keras.models.load_model(keras_path)

        return saved_data

    @classmethod
    def get_model_info(cls, model_id: str) -> dict:
        """Extrait la colonne cible réelle (target_column) du modèle."""
        target_col = None
        try:
            query = text("""
                SELECT model_id, target_column, target_type
                FROM data_prep.ml_model_registry 
                WHERE model_id = :model_id
            """)
            with engine.connect() as conn:
                result = conn.execute(query, {"model_id": model_id}).fetchone()
                if result:
                    res_dict = dict(result._mapping)
                    target_col = res_dict.get("target_column")
        except Exception as e:
            logger.warning(
                f"Impossible d'interroger le registre ml_model_registry : {str(e)}"
            )

        if not target_col:
            m_lower = model_id.lower()
            if "visitor" in m_lower or "freq" in m_lower:
                target_col = "visitor_count"
            elif "ec" in m_lower:
                target_col = "ec_kwh"
            elif "elec" in m_lower:
                target_col = "elec_kwh"
            else:
                target_col = "visitor_count"

        t_lower = target_col.lower()
        if "visitor" in t_lower or "freq" in t_lower:
            target_type = "visitor"
        elif "ec" in t_lower:
            target_type = "ec"
        else:
            target_type = "elec"

        return {"target_column": target_col, "target_type": target_type}

    @classmethod
    def run_prediction_array(cls, model_id: str, df: pd.DataFrame) -> np.ndarray:
        """Exécute la prédiction pour le DataFrame et retourne un tableau NumPy."""
        artifacts = cls.load_model_artifacts(model_id)
        m_type = artifacts.get("model_type")
        feature_names = artifacts.get("feature_names", [])
        scaler = artifacts.get("scaler")
        model_artifact = artifacts.get("model_artifact")

        missing = [col for col in feature_names if col not in df.columns]
        if missing:
            raise ValueError(
                f"Données manquantes pour les colonnes requises : {missing}"
            )

        X_df = df[feature_names].copy().ffill().bfill().fillna(0)

        if m_type in ["lstm", "gru"]:
            tf_model = artifacts["tf_model"]
            X_scaled = scaler.transform(X_df) if scaler else X_df.values
            time_steps = 12
            dummy_y = np.zeros(len(X_scaled))
            X_seq, _ = create_sequences(X_scaled, dummy_y, time_steps=time_steps)

            if len(X_seq) == 0:
                raise ValueError(
                    f"Données insuffisantes (longueur < {time_steps} pas) pour prédire la séquence."
                )

            preds = tf_model.predict(X_seq, verbose=0).flatten()
            pad_width = len(df) - len(preds)
            if pad_width > 0:
                preds = np.pad(preds, (pad_width, 0), mode="edge")
        else:
            X_input = scaler.transform(X_df) if scaler else X_df
            preds = model_artifact.predict(X_input)

        return np.maximum(0, preds)

    @classmethod
    def run_prediction(
        cls,
        model_id: str,
        version_id: str,
        id_attraction: str = "ALL",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Charge les données de la BD, exécute la prédiction et formate la sortie."""
        artifacts = cls.load_model_artifacts(model_id)
        model_info = cls.get_model_info(model_id)
        target_column = artifacts.get("target_column") or model_info["target_column"]
        target_type = model_info["target_type"]

        df = MLTrainingService.load_data_from_db(
            version_id=str(version_id),
            id_attraction=id_attraction,
            start_date=start_date,
            end_date=end_date,
        )

        if df.empty:
            raise ValueError(f"Aucune donnée trouvée dans la table '{version_id}'.")

        preds = cls.run_prediction_array(model_id=model_id, df=df)

        predictions = []
        for idx, row in df.iterrows():
            actual_val = row.get(target_column, None)
            actual_clean = (
                None
                if (pd.isna(actual_val) or actual_val is None)
                else float(actual_val)
            )

            dt_str = (
                row["datetime"].strftime("%Y-%m-%dT%H:%M:%S")
                if isinstance(row["datetime"], (pd.Timestamp, np.datetime64))
                else str(row["datetime"])
            )

            raw_val = float(preds[idx])
            pred_val = (
                float(max(0, int(round(raw_val))))
                if target_type == "visitor"
                else float(max(0.0, raw_val))
            )

            predictions.append(
                {"datetime": dt_str, "actual": actual_clean, "predicted": pred_val}
            )

        return {
            "status": "success",
            "model_id": model_id,
            "version_id": version_id,
            "id_attraction": id_attraction,
            "target_column": target_column,
            "count": len(predictions),
            "predictions": predictions,
        }

    # ==================== COMPARAISON GÉNÉRALE PAR TARGET COLUMN DYNAMIQUE ====================

    @classmethod
    def compare_and_evaluate(
        cls,
        version_id: str,
        model_ids: List[str],
        id_attraction: str = "ALL",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_html: bool = True,
    ) -> Dict[str, Any]:
        """Prédit et compare Réel vs Prédit selon la target_column respective des modèles."""
        df = MLTrainingService.load_data_from_db(
            version_id=str(version_id),
            id_attraction=id_attraction,
            start_date=start_date,
            end_date=end_date,
        )

        if df.empty:
            raise ValueError(
                f"La table '{version_id}' ne contient aucune donnée sur la période sélectionnée."
            )

        df["datetime"] = pd.to_datetime(df["datetime"])
        # Clé de tri par mois/année et libellé d'affichage MM/YYYY
        df["period_key"] = df["datetime"].dt.to_period("M")
        df["period_label"] = df["datetime"].dt.strftime("%m/%Y")

        metrics_results = {}
        model_details = []

        for m_id in model_ids:
            info = cls.get_model_info(m_id)
            target_col = info["target_column"]
            target_type = info["target_type"]

            if target_col not in df.columns:
                raise ValueError(
                    f"La colonne cible '{target_col}' du modèle '{m_id}' n'existe pas dans la table '{version_id}'."
                )

            preds = cls.run_prediction_array(m_id, df)

            act_col_name = f"{m_id}__actual"
            pred_col_name = f"{m_id}__pred"

            df[act_col_name] = df[target_col].fillna(0).astype(float)
            df[pred_col_name] = preds

            y_true = df[act_col_name].values
            y_pred = df[pred_col_name].values

            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae = float(mean_absolute_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))
            t_act = float(np.sum(y_true))
            t_prd = float(np.sum(y_pred))
            diff = t_prd - t_act
            err_pct = (diff / t_act * 100) if t_act != 0 else 0.0

            metric_obj = {
                "model_id": m_id,
                "target_column": target_col,
                "target_type": target_type,
                "rmse": round(rmse, 2),
                "mae": round(mae, 2),
                "r2": round(r2, 4),
                "total_actual": round(t_act, 2),
                "total_predicted": round(t_prd, 2),
                "total_difference": round(diff, 2),
                "error_percentage": round(err_pct, 2),
            }
            metrics_results[m_id] = metric_obj
            model_details.append(metric_obj)

        # Regroupement des données par period_key et conservation de period_label (MM/YYYY)
        agg_dict = {}
        for m_id in model_ids:
            agg_dict[f"{m_id}__actual"] = "sum"
            agg_dict[f"{m_id}__pred"] = "sum"

        monthly_df = (
            df.groupby(["period_key", "period_label"]).agg(agg_dict).reset_index()
        )
        monthly_df = monthly_df.sort_values("period_key")

        monthly_summary = []
        for _, r in monthly_df.iterrows():
            row_dict = {
                "period": str(r["period_label"])
            }  # Format MM/YYYY (ex: 01/2026)
            for m_id in model_ids:
                act_val = float(r[f"{m_id}__actual"])
                prd_val = float(r[f"{m_id}__pred"])
                diff_val = prd_val - act_val

                # Tính phần trăm sai lệch hàng tháng (Tránh chia cho 0)
                pct_val = (diff_val / act_val * 100) if act_val != 0 else 0.0

                row_dict[f"{m_id}_actual"] = round(act_val, 2)
                row_dict[f"{m_id}_pred"] = round(prd_val, 2)
                row_dict[f"{m_id}_diff"] = round(diff_val, 2)
                row_dict[f"{m_id}_pct"] = round(pct_val, 2)
            monthly_summary.append(row_dict)

        # Génération du tableau de bord HTML si demandé
        html_dashboard = None
        if include_html:
            html_dashboard = cls._generate_dynamic_html_dashboard(
                monthly_summary, model_details
            )

        return {
            "status": "success",
            "version_id": version_id,
            "metrics": metrics_results,
            "monthly_summary": monthly_summary,
            "html_dashboard": html_dashboard,
        }

    @classmethod
    def _generate_dynamic_html_dashboard(
        cls, monthly_summary: List[Dict], model_details: List[Dict]
    ) -> str:
        """Génère le tableau de bord HTML Plotly basé sur la target_column de chaque modèle."""
        months = [r["period"] for r in monthly_summary]
        fig = go.Figure()

        for m in model_details:
            m_id = m["model_id"]
            col_name = m["target_column"]
            act_vals = [r[f"{m_id}_actual"] for r in monthly_summary]
            prd_vals = [r[f"{m_id}_pred"] for r in monthly_summary]

            fig.add_trace(
                go.Bar(x=months, y=act_vals, name=f"{col_name} Réel ({m_id})")
            )
            fig.add_trace(
                go.Bar(x=months, y=prd_vals, name=f"{col_name} Prévision ({m_id})")
            )

        fig.update_layout(
            barmode="group",
            title="Comparaison Valeurs Réelles vs Prévisions par Colonne Cible",
            xaxis_title="Temps (Mois)",
            yaxis_title="Valeur",
            height=550,  # Tăng nhẹ chiều cao để chứa legend phía dưới
            template="plotly_white",
            margin=dict(
                t=50, l=40, r=30, b=80
            ),  # Tăng lề dưới (bottom) để legend không bị đè
            legend=dict(
                orientation="h",  # Hiển thị chú thích theo chiều ngang
                y=-0.25,  # Đẩy legend xuống dưới trục X
                x=0.5,  # Căn giữa theo chiều ngang
                xanchor="center",
                yanchor="top",
            ),
        )

        chart_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

        # Génération des cartes KPI
        kpi_cards_html = ""
        for m in model_details:
            kpi_cards_html += f"""
            <div class="kpi-card">
                <div class="kpi-title">{m["target_column"]} ({m["model_id"]})</div>
                <div class="kpi-value">Prévision : {m["total_predicted"]:,}</div>
                <div style="font-size: 13px; color: #4a5568; margin-top: 4px;">Réel : {m["total_actual"]:,}</div>
                <div style="font-size: 12px; color: #718096;">Écart : {m["error_percentage"]:+.2f}% | R² : {m["r2"]}</div>
            </div>
            """

        # Génération des onglets pour chaque modèle
        tab_buttons = [
            '<button class="tab-btn active" onclick="openTab(event, \'tab-chart\')">Graphique de Comparaison</button>',
            '<button class="tab-btn" onclick="openTab(event, \'tab-kpi\')">Indicateurs KPI & Évaluation</button>',
        ]

        tab_contents = [
            f'<div id="tab-chart" class="tab-content active">{chart_html}</div>',
            f'<div id="tab-kpi" class="tab-content"><div class="kpi-grid">{kpi_cards_html}</div></div>',
        ]

        for idx, m in enumerate(model_details):
            m_id = m["model_id"]
            col_name = m["target_column"]
            tab_id = f"tab-m-{idx}"

            tab_buttons.append(
                f'<button class="tab-btn" onclick="openTab(event, \'{tab_id}\')">Tableau : {col_name}</button>'
            )

            rows_html = "".join(
                [
                    f"<tr>"
                    f"<td>{r['period']}</td>"
                    f"<td class='num'>{r[f'{m_id}_actual']:,}</td>"
                    f"<td class='num'>{r[f'{m_id}_pred']:,}</td>"
                    f"<td class='num'>{r[f'{m_id}_diff']:,}</td>"
                    f"<td class='num'>{r[f'{m_id}_pct']:+.2f}%</td>"
                    f"</tr>"
                    for r in monthly_summary
                ]
            )

            table_html = f"""
            <div id="{tab_id}" class="tab-content">
                <h3>Tableau Statistique Détaillé de la Colonne : <span style="color:#2b6cb0;">{col_name}</span> (Modèle : {m_id})</h3>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Période / Mois</th>
                            <th class="num">Réel</th>
                            <th class="num">Prévision</th>
                            <th class="num">Écart (Préd - Réel)</th>
                            <th class="num">Écart (%)</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            """
            tab_contents.append(table_html)

        return f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; background: #f8fafc; padding: 20px; border-radius: 10px;">
            <style>
                .tab-nav {{ display: flex; gap: 8px; border-bottom: 2px solid #e2e8f0; margin-bottom: 20px; flex-wrap: wrap; }}
                .tab-btn {{ padding: 10px 18px; border: none; background: #edf2f7; font-weight: 600; cursor: pointer; border-radius: 6px 6px 0 0; }}
                .tab-btn.active {{ background: #2b6cb0; color: white; }}
                .tab-content {{ display: none; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
                .tab-content.active {{ display: block; }}
                .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 15px; }}
                .kpi-card {{ background: #f7fafc; border-left: 4px solid #2b6cb0; padding: 15px; border-radius: 6px; border: 1px solid #e2e8f0; }}
                .kpi-title {{ font-size: 13px; color: #718096; font-weight: bold; text-transform: uppercase; }}
                .kpi-value {{ font-size: 20px; font-weight: bold; color: #2d3748; margin-top: 5px; }}
                .data-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                .data-table th, .data-table td {{ padding: 10px 14px; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
                .data-table th {{ background: #edf2f7; text-align: left; }}
                .num {{ text-align: right; }}
            </style>

            <div class="tab-nav">{"".join(tab_buttons)}</div>
            {"".join(tab_contents)}

            <script>
                function openTab(evt, tabId) {{
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tab-content");
                    for (i = 0; i < tabcontent.length; i++) {{ tabcontent[i].classList.remove("active"); }}
                    tablinks = document.getElementsByClassName("tab-btn");
                    for (i = 0; i < tablinks.length; i++) {{ tablinks[i].classList.remove("active"); }}
                    document.getElementById(tabId).classList.add("active");
                    evt.currentTarget.classList.add("active");
                    window.dispatchEvent(new Event('resize'));
                }}
            </script>
        </div>
        """

    @classmethod
    def export_predictions_to_schema(
        cls,
        model_id: str,
        version_id: str,
        id_attraction: str = "ALL",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        target_schema: str = "predictions",
        output_table_name: Optional[str] = None,
        if_exists: str = "replace",
    ) -> Dict[str, Any]:
        """Exporte les données de prévision vers un schéma distinct."""
        version_id_str = str(version_id)
        model_id_str = str(model_id)

        model_info = cls.get_model_info(model_id_str)
        target_col = model_info.get("target_column")
        target_type = model_info.get("target_type")

        df_raw = MLTrainingService.load_data_from_db(
            version_id=version_id_str,
            id_attraction=id_attraction,
            start_date=start_date,
            end_date=end_date,
        )

        if df_raw.empty:
            raise ValueError(
                f"Aucune donnée à exporter depuis la table '{version_id_str}'."
            )

        raw_preds = cls.run_prediction_array(model_id=model_id_str, df=df_raw)

        if target_type == "visitor":
            formatted_preds = [max(0, int(round(val))) for val in raw_preds]
        else:
            formatted_preds = [max(0.0, float(val)) for val in raw_preds]

        df_export = df_raw.copy()
        df_export["predicted_value"] = formatted_preds

        if target_col and target_col in df_export.columns:
            df_export["residual_error"] = (
                df_export[target_col] - df_export["predicted_value"]
            )
        else:
            df_export["residual_error"] = None

        df_export["predicted_by_model"] = model_id_str
        df_export["prediction_created_at"] = pd.Timestamp.now()

        if not output_table_name:
            clean_ver = version_id_str.replace("-", "_").lower()
            clean_mod = model_id_str.replace("-", "_").lower()
            output_table_name = f"{clean_ver}_pred_{clean_mod}"

        with engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{target_schema}";'))
            df_export.to_sql(
                name=output_table_name,
                con=connection,
                schema=target_schema,
                if_exists=if_exists,
                index=False,
            )

        return {
            "status": "success",
            "message": f"Table exportée avec succès vers {target_schema}.{output_table_name}",
            "target_schema": str(target_schema),
            "target_table": str(output_table_name),
            "total_rows": int(len(df_export)),
            "kept_target_column": str(target_col) if target_col else None,
            "dropped_columns": [],
        }

    @classmethod
    def export_predictions_to_file(
        cls,
        version_id: str,
        model_ids: List[str],
        file_format: ExportFileFormat,
        id_attraction: str = "ALL",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[io.BytesIO, str, str]:
        """
        Lấy toàn bộ các cột từ bảng nguồn gốc, chạy dự đoán cho danh sách các mô hình,
        ghép các cột kết quả dự đoán và trả về (file_stream, filename, media_type).
        """
        df_source = MLTrainingService.load_data_from_db(
            version_id=str(version_id),
            id_attraction=id_attraction,
            start_date=start_date,
            end_date=end_date,
        )

        if df_source.empty:
            raise ValueError(f"Aucune donnée trouvée dans la table '{version_id}'.")

        df_export = df_source.copy()

        for m_id in model_ids:
            try:
                info = cls.get_model_info(m_id)
                target_col = info["target_column"]
                target_type = info["target_type"]

                raw_preds = cls.run_prediction_array(m_id, df_source.copy())

                if target_type == "visitor":
                    formatted_preds = [max(0, int(round(val))) for val in raw_preds]
                else:
                    formatted_preds = [max(0.0, float(val)) for val in raw_preds]

                col_pred_name = f"pred_{m_id}"
                col_err_name = f"residual_{m_id}"

                df_export[col_pred_name] = formatted_preds

                if target_col in df_export.columns:
                    df_export[col_err_name] = df_export[target_col] - df_export[col_pred_name]

            except Exception as e:
                logger.error(f"Lỗi khi dự đoán mô hình {m_id} để xuất file: {str(e)}")
                df_export[f"pred_{m_id}_error"] = str(e)

        output = io.BytesIO()

        if file_format == ExportFileFormat.EXCEL:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_export.to_excel(writer, index=False, sheet_name='Predictions')
            output.seek(0)
            filename = f"export_{version_id}_predictions.xlsx"
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:
            csv_bytes = df_export.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            output = io.BytesIO(csv_bytes)
            filename = f"export_{version_id}_predictions.csv"
            media_type = "text/csv"

        return output, filename, media_type

    @classmethod
    def apply_imputation_to_db(
        cls,
        version_id: str,
        id_attraction: str,
        target_column: str,
        mode: str,
        data: List[Dict[str, Any]],
    ) -> int:
        """Écrit / Remplace les résultats de prévision dans la base de données PostgreSQL."""
        version_id_str = str(version_id)
        existing_cols = MLTrainingService.get_version_columns(version_id_str)

        if target_column not in existing_cols:
            raise ValueError(
                f"La colonne '{target_column}' n'existe pas dans la table '{SCHEMA_NAME}.{version_id_str}'."
            )

        attr_val = (id_attraction or "ALL").strip().upper()
        attr_conditions = []
        if attr_val != "ALL":
            if "id_attraction" in existing_cols:
                attr_conditions.append("id_attraction = :attr")
            elif "attraction_id" in existing_cols:
                attr_conditions.append("attraction_id = :attr")

        attr_clause = (
            f" AND ({' OR '.join(attr_conditions)})" if attr_conditions else ""
        )

        if mode == "fill_missing":
            condition_sql = (
                f'("{target_column}" IS NULL OR "{target_column}"::text = \'NaN\')'
            )
        else:
            condition_sql = "1=1"

        update_query = text(f"""
            UPDATE {SCHEMA_NAME}."{version_id_str}"
            SET "{target_column}" = :pred_val
            WHERE datetime = :dt_val {attr_clause} AND ({condition_sql});
        """)

        updated_count = 0
        with engine.begin() as conn:
            for item in data:
                params = {
                    "pred_val": item["predicted_value"],
                    "dt_val": item["datetime"],
                }
                if attr_conditions:
                    params["attr"] = id_attraction

                res = conn.execute(update_query, params)
                updated_count += res.rowcount

        return updated_count