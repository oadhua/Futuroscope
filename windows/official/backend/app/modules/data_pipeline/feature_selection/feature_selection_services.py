import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from app.core.database import engine
from app.modules.data_pipeline.queries import get_dynamic_gathering_query

logger = logging.getLogger(__name__)

SCHEMA_NAME = "data_prep"


class FeatureSelectionService:
    METHOD_LABELS_FR = {
        "correlation": "Sélection par Corrélations (Pearson/Spearman)",
        "mutual_info": "Sélection par Information Mutuelle",
        "random_forest": "Sélection par Importance Random Forest",
        "randomforest": "Sélection par Importance Random Forest",
        "xgboost": "Sélection par XGBoost Feature Importance",
        "shap": "Sélection par Valeurs SHAP",
        "lasso": "Sélection par Régularisation Lasso (L1)",
    }

    @classmethod
    def _init_db_schema(cls):
        """Khởi tạo schema data_prep và bảng registry với auto-commit độc lập."""
        queries = [
            f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA_NAME}";',
            f"""
            CREATE TABLE IF NOT EXISTS "{SCHEMA_NAME}"."data_version_registry" (
                version_id VARCHAR(100) PRIMARY KEY,
                parent_version_id VARCHAR(100),
                id_attraction VARCHAR(50) DEFAULT 'ALL',
                step_type VARCHAR(100),
                method VARCHAR(100),
                method_label_fr VARCHAR(255),
                target_columns TEXT,
                parameters JSONB,
                stats JSONB,
                file_path VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
        ]
        try:
            with engine.begin() as conn:
                for q in queries:
                    conn.execute(text(q))
        except Exception as e:
            logger.error(f"Erreur init schema: {e}")

    @classmethod
    def load_version_dataframe(
        cls, version_id: str, id_attraction: str = "ALL"
    ) -> pd.DataFrame:
        """Tải dữ liệu của một phiên bản từ PostgreSQL."""
        cls._init_db_schema()

        if version_id == "v0_raw":
            try:
                query_sql = get_dynamic_gathering_query(
                    engine, id_attraction=id_attraction
                )
                with engine.begin() as conn:
                    return pd.read_sql(text(query_sql), conn)
            except Exception as e:
                raise ValueError(
                    f"Erreur lors de la requête v0_raw depuis la base de données : {str(e)}"
                )

        if not re.match(r"^[a-zA-Z0-9_]+$", version_id):
            raise ValueError("Le nom version_id contient des caractères invalides.")

        with engine.begin() as conn:
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
            if conn.execute(cols_query, {"s": SCHEMA_NAME, "t": version_id}).fetchone():
                order_clause = " ORDER BY datetime ASC"

            df = pd.read_sql(
                text(f'SELECT * FROM "{SCHEMA_NAME}"."{version_id}"{order_clause}'),
                conn,
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

    @classmethod
    def get_valid_candidate_features(
        cls, df: pd.DataFrame, target_column: str = "visitor_count"
    ) -> List[str]:
        """
        Trích xuất các đặc trưng số hợp lệ dựa trên quy tắc Pipeline 2 giai đoạn (Cascade):
        1. Target là visitor_count -> Loại bỏ TẤT CẢ các cột năng lượng (elec_*, ec_*,...).
        2. Target là Năng lượng (elec/ec) -> Loại bỏ TẤT CẢ các cột năng lượng khác, GIỮ LẠI visitor_count.
        """
        candidate_cols = []

        # Các cột định danh / thời gian luôn loại trừ
        always_exclude = {"datetime", "id_attraction", target_column}

        # Danh sách từ khóa/tiền tố nhận diện biến năng lượng
        energy_keywords = ("elec", "ec", "energy", "power", "kwh", "mwh")

        target_lower = target_column.lower()
        is_target_energy = any(
            target_lower.startswith(kw) or kw in target_lower for kw in energy_keywords
        )

        for col in df.columns:
            if col in always_exclude:
                continue

            col_lower = col.lower()
            is_col_energy = any(
                col_lower.startswith(kw) or kw in col_lower for kw in energy_keywords
            )

            # 1. QUY TẮC CHO VISITOR_COUNT: Không chứa bất kỳ cột năng lượng nào
            if target_column == "visitor_count" and is_col_energy:
                continue

            # 2. QUY TẮC CHO NĂNG LƯỢNG (Elec / EC): Không chứa các cột năng lượng khác
            elif is_target_energy and is_col_energy:
                continue

            # Chỉ lấy các cột kiểu số
            if np.issubdtype(df[col].dtype, np.number):
                candidate_cols.append(col)

        return candidate_cols

    @classmethod
    def analyze_feature_importance(
        cls,
        df: pd.DataFrame,
        target_column: str,
        method: str,
        corr_method: str = "pearson",
        mi_neighbors: int = 3,
        rf_n_estimators: int = 100,
        rf_max_depth: Optional[int] = None,
        rf_min_samples_split: int = 2,
        lasso_cv: int = 5,
        lasso_max_iter: int = 3000,
    ) -> List[Dict[str, Any]]:
        """Tính toán điểm quan trọng của đặc trưng dựa trên các cột hợp lệ."""
        if target_column not in df.columns:
            raise ValueError(f"La colonne cible '{target_column}' est introuvable.")

        # Lọc danh sách cột hợp lệ theo Target Column
        candidate_cols = cls.get_valid_candidate_features(
            df, target_column=target_column
        )
        if not candidate_cols:
            return []

        clean_df = df[candidate_cols + [target_column]].dropna()
        if len(clean_df) == 0:
            raise ValueError(
                "Données insuffisantes après suppression des valeurs manquantes."
            )

        X = clean_df[candidate_cols]
        y = clean_df[target_column]

        scores = []
        normalized_method = method.lower().replace("_", "").replace(" ", "")

        if "corr" in normalized_method:
            c_method = (
                corr_method
                if corr_method in ["pearson", "spearman", "kendall"]
                else "pearson"
            )
            corr_series = (
                X.apply(lambda col: col.corr(y, method=c_method)).abs().fillna(0)
            )
            for col, val in corr_series.items():
                scores.append({"name": col, "score": float(val)})

        elif "mutual" in normalized_method:
            mi_scores = mutual_info_regression(
                X, y, n_neighbors=max(1, mi_neighbors), random_state=42
            )
            for col, val in zip(candidate_cols, mi_scores):
                scores.append({"name": col, "score": float(val)})

        elif any(
            m in normalized_method
            for m in ["randomforest", "tree", "xgb", "xgboost", "shap"]
        ):
            rf = RandomForestRegressor(
                n_estimators=max(10, rf_n_estimators),
                max_depth=rf_max_depth,
                min_samples_split=max(2, rf_min_samples_split),
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X, y)
            for col, val in zip(candidate_cols, rf.feature_importances_):
                scores.append({"name": col, "score": float(val)})

        elif "lasso" in normalized_method:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            lasso = LassoCV(
                cv=max(2, lasso_cv),
                max_iter=max(1000, lasso_max_iter),
                random_state=42,
                n_jobs=-1,
            ).fit(X_scaled, y)

            coefs = np.abs(lasso.coef_)
            for col, val in zip(candidate_cols, coefs):
                scores.append({"name": col, "score": float(val)})

        else:
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            for col, val in zip(candidate_cols, rf.feature_importances_):
                scores.append({"name": col, "score": float(val)})

        scores = sorted(scores, key=lambda x: x["score"], reverse=True)

        total_score = sum(s["score"] for s in scores) or 1.0
        running_sum = 0.0
        for item in scores:
            running_sum += item["score"]
            item["cumulative_score"] = float(running_sum / total_score)

        return scores

    @classmethod
    def get_version_stats(
        cls, version_id: str, id_attraction: str = "ALL"
    ) -> Tuple[List[str], List[str], int, List[str]]:
        """Lấy thống kê dữ liệu phiên bản."""
        df_full = cls.load_version_dataframe(version_id, id_attraction="ALL")

        attractions = []
        if "id_attraction" in df_full.columns:
            attractions = sorted(
                [str(x) for x in df_full["id_attraction"].unique() if pd.notnull(x)]
            )

        df_target = df_full
        if (
            id_attraction
            and id_attraction != "ALL"
            and "id_attraction" in df_full.columns
        ):
            df_target = df_full[
                df_full["id_attraction"].astype(str) == str(id_attraction)
            ]

        available_columns = list(df_target.columns)
        numeric_columns = [
            c for c in available_columns if np.issubdtype(df_target[c].dtype, np.number)
        ]

        total_records = len(df_target)

        return available_columns, numeric_columns, total_records, attractions

    @classmethod
    def create_feature_selected_version(
        cls,
        parent_version_id: str,
        target_column: str,
        selected_features: List[str],
        id_attraction: str = "ALL",
        method: str = "random_forest",
        custom_version_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tạo phiên bản mới tự động tăng chỉ số fs_1, fs_2,... dựa trên phiên bản cha."""
        cls._init_db_schema()

        # 1. Tải dữ liệu nguồn
        try:
            df_parent = cls.load_version_dataframe(
                parent_version_id, id_attraction=id_attraction
            )
        except Exception as e:
            logger.error(f"Erreur lecture parent_version: {e}")
            raise ValueError(
                f"Impossible de lire la version source '{parent_version_id}': {str(e)}"
            )

        original_cols = list(df_parent.columns)

        # 2. Giữ các cột bắt buộc
        mandatory_cols = [
            c
            for c in ["datetime", "id_attraction", target_column]
            if c in original_cols
        ]

        final_cols_to_keep = list(dict.fromkeys(mandatory_cols + selected_features))
        final_cols_to_keep = [c for c in final_cols_to_keep if c in original_cols]

        df_filtered = df_parent[final_cols_to_keep].copy()

        # 3. THUẬT TOÁN ĐẶT TÊN TỰ ĐỘNG TĂNG SUFFIX (_fs_1, _fs_2,...)
        if custom_version_id and custom_version_id.strip():
            # Nếu người dùng tự nhập tên riêng, chuẩn hóa ký tự
            new_version_id = re.sub(r"[-\s]+", "_", custom_version_id.strip())
            new_version_id = re.sub(r"[^a-zA-Z0-9_]", "", new_version_id)
        else:
            # Tự động tìm suffix số tiếp theo của phiên bản cha
            prefix = f"{parent_version_id}_fs_"
            existing_numbers = []

            with engine.begin() as conn:
                # Tìm tất cả bảng hiện có khớp định dạng <parent_version_id>_fs_%
                tables_sql = text("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = :s AND table_name LIKE :p
                """)
                rows = conn.execute(
                    tables_sql, {"s": SCHEMA_NAME, "p": f"{prefix}%"}
                ).fetchall()

                for row in rows:
                    t_name = row[0]
                    num_part = t_name[len(prefix) :]
                    if num_part.isdigit():
                        existing_numbers.append(int(num_part))

            next_number = max(existing_numbers) + 1 if existing_numbers else 1
            new_version_id = f"{prefix}{next_number}"

        # Đảm bảo tuyệt đối không đè bảng cũ nếu có trùng lặp ngoài ý muốn
        base_id = new_version_id
        counter = 1
        with engine.begin() as conn:
            while True:
                check_sql = text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = :s AND table_name = :t
                    );
                """)
                exists = conn.execute(
                    check_sql, {"s": SCHEMA_NAME, "t": new_version_id}
                ).scalar()
                if not exists:
                    break
                new_version_id = f"{base_id}_{counter}"
                counter += 1

        dropped_cols = [c for c in original_cols if c not in final_cols_to_keep]
        stats = {
            "target_column": target_column,
            "original_feature_count": len(original_cols),
            "selected_feature_count": len(final_cols_to_keep),
            "retained_features": final_cols_to_keep,
            "dropped_features": dropped_cols,
            "rows_affected": len(df_filtered),
        }

        method_label = cls.METHOD_LABELS_FR.get(
            method.lower(), "Sélection de Caractéristiques"
        )

        # 4. GHI BẢNG DỮ LIỆU MỚI
        try:
            with engine.begin() as conn:
                df_filtered.to_sql(
                    name=new_version_id,
                    con=engine,
                    schema=SCHEMA_NAME,
                    if_exists="replace",
                    index=False,
                )

            # 5. GHI REGISTRY METADATA
            insert_registry_sql = text(f"""
                INSERT INTO "{SCHEMA_NAME}"."data_version_registry" 
                (version_id, parent_version_id, id_attraction, step_type, method, method_label_fr, target_columns, parameters, stats, file_path, created_at)
                VALUES (:vid, :pvid, :attr, :step, :method, :label, :target, :params, :stats, :path, CURRENT_TIMESTAMP)
                ON CONFLICT (version_id) DO UPDATE SET
                    parent_version_id = EXCLUDED.parent_version_id,
                    id_attraction = EXCLUDED.id_attraction,
                    stats = EXCLUDED.stats,
                    created_at = CURRENT_TIMESTAMP;
            """)

            with engine.begin() as conn:
                conn.execute(
                    insert_registry_sql,
                    {
                        "vid": new_version_id,
                        "pvid": parent_version_id,
                        "attr": id_attraction,
                        "step": "feature_selection",
                        "method": method,
                        "label": method_label,
                        "target": target_column,
                        "params": json.dumps({"selected_features": selected_features}),
                        "stats": json.dumps(stats),
                        "path": f"postgresql://{SCHEMA_NAME}.{new_version_id}",
                    },
                )
        except Exception as e:
            logger.error(f"Lỗi khi ghi dữ liệu phiên bản mới: {e}", exc_info=True)
            raise ValueError(f"Erreur de sauvegarde PostgreSQL : {str(e)}")

        return {
            "version_id": new_version_id,
            "parent_version_id": parent_version_id,
            "id_attraction": id_attraction,
            "step_type": "feature_selection",
            "method": method,
            "method_label_fr": method_label,
            "target_columns": [target_column],
            "parameters": {"selected_features": selected_features},
            "stats": stats,
            "file_path": f"postgresql://{SCHEMA_NAME}.{new_version_id}",
            "created_at": datetime.now().isoformat(),
        }

    @classmethod
    def delete_version(cls, version_id: str) -> bool:
        """Xóa một phiên bản khỏi cơ sở dữ liệu."""
        if version_id == "v0_raw":
            raise ValueError("Impossible de supprimer la version initiale 'v0_raw'.")

        if not re.match(r"^[a-zA-Z0-9_]+$", version_id):
            raise ValueError("Le nom version_id contient des caractères invalides.")

        cls._init_db_schema()

        with engine.begin() as conn:
            drop_sql = text(f'DROP TABLE IF EXISTS "{SCHEMA_NAME}"."{version_id}";')
            conn.execute(drop_sql)

            delete_reg_sql = text(f"""
                DELETE FROM "{SCHEMA_NAME}"."data_version_registry" 
                WHERE version_id = :vid;
            """)
            conn.execute(delete_reg_sql, {"vid": version_id})

        return True