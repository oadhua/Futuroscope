import json
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.core.database import engine
from app.modules.data_pipeline.queries import get_dynamic_gathering_query

logger = logging.getLogger(__name__)

SCHEMA_NAME = "data_prep"


class PreprocessingTraceabilityService:

    # ==================== CHARGEMENT AUTONOME DE LA BASE DE DONNÉES ====================

    @classmethod
    def load_version_dataframe(
        cls, version_id: str, id_attraction: str = "ALL"
    ) -> pd.DataFrame:
        """
        Charge de manière autonome un DataFrame depuis PostgreSQL (schema data_prep) 
        selon la version et l'attraction spécifiées.
        """
        if version_id == "v0_raw":
            try:
                query_sql = get_dynamic_gathering_query(
                    engine, id_attraction=id_attraction
                )
                with engine.connect() as conn:
                    return pd.read_sql(text(query_sql), conn)
            except Exception as e:
                raise ValueError(
                    f"Erreur lors de la requête v0_raw depuis la base de données : {str(e)}"
                )

        if not re.match(r"^[a-zA-Z0-9_]+$", version_id):
            raise ValueError("Le nom version_id contient des caractères invalides.")

        with engine.connect() as conn:
            check_sql = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = :s AND table_name = :t
                );
            """)
            exists = conn.execute(
                check_sql, {"s": SCHEMA_NAME, "t": version_id}
            ).scalar()

            if not exists:
                raise FileNotFoundError(
                    f"La version '{version_id}' est introuvable dans la base de données."
                )

            order_clause = ""
            cols_query = text("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_schema = :s AND table_name = :t AND column_name = 'datetime'
            """)
            if conn.execute(
                cols_query, {"s": SCHEMA_NAME, "t": version_id}
            ).fetchone():
                order_clause = " ORDER BY datetime ASC"

            df = pd.read_sql(
                text(f'SELECT * FROM {SCHEMA_NAME}."{version_id}"{order_clause}'), conn
            )

            if (
                id_attraction
                and id_attraction != "ALL"
                and "id_attraction" in df.columns
            ):
                df = df[
                    df["id_attraction"].astype(str) == str(id_attraction)
                ].reset_index(drop=True)

            return df

    # ==================== CALCULS STATISTIQUES ====================

    @classmethod
    def _calculate_overview_stats(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcule les indicateurs statistiques globaux du DataFrame."""
        numeric_df = df.select_dtypes(include=[np.number])
        outliers_total = 0

        if not numeric_df.empty:
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1
            outliers_mask = (numeric_df < (Q1 - 1.5 * IQR)) | (
                numeric_df > (Q3 + 1.5 * IQR)
            )
            outliers_total = int(outliers_mask.sum().sum())

        return {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "outliers_count": outliers_total,
            "zero_values": int((df == 0).sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        }

    @classmethod
    def _calculate_column_stats(
        cls, df: pd.DataFrame, col_name: str
    ) -> Optional[Dict[str, Any]]:
        """Calcule l'analyse statistique détaillée d'une colonne donnée."""
        if col_name not in df.columns:
            return None

        series = df[col_name]
        is_numeric = pd.api.types.is_numeric_dtype(series)

        min_v = float(series.min()) if is_numeric and not series.empty else None
        max_v = float(series.max()) if is_numeric and not series.empty else None
        mean_v = float(series.mean()) if is_numeric and not series.empty else None

        outliers_c = 0
        neg_c = 0
        zero_c = 0

        if is_numeric:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            outliers_c = int(
                ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()
            )
            neg_c = int((series < 0).sum())
            zero_c = int((series == 0).sum())

        return {
            "column_name": col_name,
            "min_val": min_v,
            "max_val": max_v,
            "mean_val": mean_v,
            "unique_count": int(series.nunique()),
            "missing_values": int(series.isnull().sum()),
            "outliers_count": outliers_c,
            "negative_count": neg_c,
            "zero_count": zero_c,
        }

    # ==================== RAPPORT DE TRAÇABILITÉ ====================

    @classmethod
    def get_comparison_traceability(
        cls,
        version_id: str,
        id_attraction: Optional[str] = "ALL",
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Génère le rapport comparatif entre la version d'origine et la version modifiée."""
        id_attr = id_attraction or "ALL"

        # 1. Lecture des métadonnées depuis le registre PostgreSQL
        query = text(f"""
            SELECT version_id, parent_version_id, id_attraction, step_type, method, 
                   target_columns, parameters, stats
            FROM {SCHEMA_NAME}.data_version_registry
            WHERE version_id = :v_id
        """)

        with engine.connect() as conn:
            row = conn.execute(query, {"v_id": version_id}).mappings().fetchone()

        if not row:
            raise FileNotFoundError(
                f"La version '{version_id}' est introuvable dans le registre de données."
            )

        parent_v_id = row["parent_version_id"] or "v0_raw"
        registered_id_attr = row["id_attraction"] or id_attr

        # 2. Chargement autonome des deux DataFrames
        df_processed = cls.load_version_dataframe(
            version_id, id_attraction=registered_id_attr
        )
        df_original = cls.load_version_dataframe(
            parent_v_id, id_attraction=registered_id_attr
        )

        # 3. Identification de la colonne cible
        target_cols_str = row.get("target_columns") or ""
        cols_list = [c.strip() for c in target_cols_str.split(",") if c.strip()]
        selected_col = (
            target_column
            if target_column
            else (cols_list[0] if cols_list else df_processed.columns[0])
        )

        # 4. Calcul des métriques globales et détaillées
        orig_overview = cls._calculate_overview_stats(df_original)
        proc_overview = cls._calculate_overview_stats(df_processed)

        orig_col_stats = cls._calculate_column_stats(df_original, selected_col)
        proc_col_stats = cls._calculate_column_stats(df_processed, selected_col)

        # 5. Extraction des points pour le graphique temporel (échantillonnage de 500 points)
        chart_points = []
        time_col = "datetime" if "datetime" in df_processed.columns else None

        if time_col and selected_col in df_processed.columns:
            sample_df_proc = (
                df_processed[[time_col, selected_col]]
            )

            for _, r in sample_df_proc.iterrows():
                t_val = str(r[time_col])
                p_val = (
                    float(r[selected_col])
                    if pd.notnull(r[selected_col])
                    else None
                )

                orig_match = (
                    df_original[df_original[time_col] == r[time_col]]
                    if time_col in df_original.columns
                    else pd.DataFrame()
                )
                o_val = (
                    float(orig_match[selected_col].values[0])
                    if not orig_match.empty
                    and selected_col in orig_match.columns
                    and pd.notnull(orig_match[selected_col].values[0])
                    else None
                )

                chart_points.append(
                    {
                        "timestamp": t_val,
                        "original_value": o_val,
                        "processed_value": p_val,
                    }
                )

        return {
            "version_id": version_id,
            "parent_version_id": parent_v_id,
            "id_attraction": registered_id_attr,
            "operation": row.get("step_type", "Prétraitement"),
            "column_name": selected_col,
            "method": row.get("method", "Standard"),
            "original_overview": orig_overview,
            "processed_overview": proc_overview,
            "original_column_stats": orig_col_stats,
            "processed_column_stats": proc_col_stats,
            "chart_data": chart_points,
        }