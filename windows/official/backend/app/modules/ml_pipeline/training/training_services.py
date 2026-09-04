import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from app.core.database import engine

logger = logging.getLogger(__name__)

SCHEMA_NAME = "data_prep"
MODEL_STORAGE_DIR = "storage/models"
os.makedirs(MODEL_STORAGE_DIR, exist_ok=True)


def is_energy_column(col_name: str) -> bool:
    """Kiểm tra xem một cột có phải là thông số năng lượng/điện năng hay không."""
    col_clean = col_name.lower().strip()
    energy_prefixes = (
        "elec_",
        "ec_",
        "energy_",
        "power_",
        "p_kw",
        "kwh",
        "temp_",
        "hvac_",
    )
    return (
        col_clean.startswith(energy_prefixes)
        or "elec" in col_clean
        or "power" in col_clean
    )


def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int = 12):
    """Tạo cửa sổ trượt (Slide Window) dạng 3D Tensor cho DL."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i : i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def build_tf_rnn_model(
    input_shape: Tuple[int, int],
    hidden_dim: int = 64,
    rnn_type: str = "lstm",
    learning_rate: float = 0.001,
    dropout: float = 0.2,
) -> Sequential:
    """Xây dựng mô hình Deep Learning chuẩn với Adam learning rate và Dropout."""
    model = Sequential()
    if rnn_type.lower() == "gru":
        model.add(GRU(hidden_dim, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(GRU(hidden_dim // 2, return_sequences=False))
        model.add(Dropout(dropout))
    else:
        model.add(LSTM(hidden_dim, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(LSTM(hidden_dim // 2, return_sequences=False))
        model.add(Dropout(dropout))

    model.add(Dense(1))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"]
    )
    return model


class MLTrainingService:
    @classmethod
    def list_available_versions(cls, id_attraction: Optional[str] = "ALL") -> List[Dict[str, Any]]:
        """Truy vấn danh sách các bảng phiên bản dữ liệu trong schema data_prep."""
        query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = :schema_name
              AND table_name NOT IN ('ml_model_registry')
            ORDER BY table_name DESC;
        """)
        try:
            with engine.connect() as conn:
                rows = conn.execute(query, {"schema_name": SCHEMA_NAME}).fetchall()
                versions = []
                for row in rows:
                    tbl_name = row[0]
                    versions.append({
                        "version_id": tbl_name,
                        "table_name": tbl_name,
                        "id_attraction": id_attraction or "ALL"
                    })
            return versions
        except Exception as e:
            logger.error(
                f"Lỗi khi đọc danh sách phiên bản từ schema {SCHEMA_NAME}: {str(e)}"
            )
            return []

    @classmethod
    def get_version_columns(cls, version_id: str, id_attraction: Optional[str] = "ALL") -> List[str]:
        """Lấy danh sách tất cả các cột của một bảng phiên bản."""
        inspect_query = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = :schema_name AND table_name = :table_name;
        """)
        with engine.connect() as conn:
            rows = conn.execute(
                inspect_query, {"schema_name": SCHEMA_NAME, "table_name": version_id}
            ).fetchall()
        return [r[0] for r in rows]

    @classmethod
    def delete_prep_version(cls, version_id: str) -> bool:
        """Xóa bảng phiên bản trong schema data_prep."""
        if version_id.lower() == "v0_raw":
            raise ValueError("Không thể xóa phiên bản gốc 'v0_raw'.")

        drop_query = text(f'DROP TABLE IF EXISTS {SCHEMA_NAME}."{version_id}" CASCADE;')
        with engine.begin() as conn:
            conn.execute(drop_query)
        return True

    @classmethod
    def load_data_from_db(
        cls,
        version_id: str,
        id_attraction: Optional[str] = "ALL",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Đọc dữ liệu sạch trực tiếp từ Bảng trong PostgreSQL Database."""

        inspect_query = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = :schema_name AND table_name = :table_name;
        """)

        with engine.connect() as conn:
            columns_rows = conn.execute(
                inspect_query, {"schema_name": SCHEMA_NAME, "table_name": version_id}
            ).fetchall()

        existing_cols = [row[0] for row in columns_rows]

        conditions = []
        params = {}

        attr_val = (id_attraction or "ALL").strip().upper()

        if attr_val != "ALL":
            attr_conditions = []
            if "id_attraction" in existing_cols:
                attr_conditions.append("id_attraction = :attr")
            if "attraction_id" in existing_cols:
                attr_conditions.append("attraction_id = :attr")

            if attr_conditions:
                conditions.append(f"({' OR '.join(attr_conditions)})")
                params["attr"] = id_attraction

        if start_date and "datetime" in existing_cols:
            conditions.append("datetime >= :start_date")
            params["start_date"] = start_date

        if end_date and "datetime" in existing_cols:
            conditions.append("datetime <= :end_date")
            params["end_date"] = end_date

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        order_clause = "ORDER BY datetime ASC" if "datetime" in existing_cols else ""

        query_str = f"""
            SELECT * 
            FROM {SCHEMA_NAME}."{version_id}"
            {where_clause}
            {order_clause};
        """

        with engine.connect() as conn:
            df = pd.read_sql(text(query_str), conn, params=params)

        if df.empty:
            raise ValueError(
                f"Không tìm thấy dữ liệu phù hợp trong phiên bản '{version_id}'."
            )

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])

        return df

    @classmethod
    def prepare_training_data(
        cls,
        version_id: str,
        id_attraction: str,
        target_column: str,
        target_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Lấy dữ liệu từ DB và chuẩn bị ma trận đặc trưng X, y."""
        df = cls.load_data_from_db(version_id, id_attraction, start_date, end_date)

        if target_column not in df.columns:
            raise ValueError(
                f"Cột mục tiêu '{target_column}' không tồn tại trong dữ liệu."
            )

        ignore_cols = ["datetime", "id_attraction", "attraction_id", target_column]
        target_lower = target_column.lower()

        feature_cols = []
        for c in df.columns:
            if c in ignore_cols or not pd.api.types.is_numeric_dtype(df[c]):
                continue

            if target_type == "visitor" and is_energy_column(c):
                continue

            if target_type == "energy" and is_energy_column(c):
                if not c.lower().startswith(target_lower):
                    continue

            feature_cols.append(c)

        if not feature_cols:
            raise ValueError(
                "Không tìm thấy thuộc tính (feature) hợp lệ nào sau khi lọc."
            )

        df_clean = df.copy()
        df_clean[feature_cols] = df_clean[feature_cols].ffill().bfill().fillna(0)
        df_clean[target_column] = df_clean[target_column].ffill().bfill().fillna(0)

        return df_clean, df_clean[feature_cols], df_clean[target_column], feature_cols

    @classmethod
    def train_and_evaluate(
        cls,
        version_id: str,
        id_attraction: str,
        target_column: str,
        target_type: str,
        model_type: str,
        split_method: str = "ratio",
        test_size: float = 0.2,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        split_date: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if hyperparameters is None:
            hyperparameters = {}

        df_full, X_df, y_s, feature_names = cls.prepare_training_data(
            version_id, id_attraction, target_column, target_type, start_date, end_date
        )

        if split_method == "date" and split_date:
            split_dt = pd.to_datetime(split_date)
            train_mask = df_full["datetime"] < split_dt
            test_mask = df_full["datetime"] >= split_dt

            X_train_raw, X_test_raw = X_df[train_mask], X_df[test_mask]
            y_train, y_test = y_s[train_mask].values, y_s[test_mask].values

            if len(X_train_raw) == 0 or len(X_test_raw) == 0:
                raise ValueError(
                    f"Mốc ngày chia 'split_date' ({split_date}) khiến tập Train hoặc Test bị rỗng."
                )
        else:
            split_idx = int(len(X_df) * (1 - test_size))
            X_train_raw, X_test_raw = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
            y_train, y_test = y_s.iloc[:split_idx].values, y_s.iloc[split_idx:].values

        m_type = model_type.lower()
        model_artifact = None
        scaler = None

        if m_type in ["xgboost", "lightgbm", "random_forest", "ridge"]:
            if m_type == "xgboost":
                params = {
                    "n_estimators": 100,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "n_jobs": -1,
                }
                params.update(hyperparameters)
                model_artifact = xgb.XGBRegressor(**params)

            elif m_type == "lightgbm":
                params = {
                    "n_estimators": 100,
                    "learning_rate": 0.05,
                    "max_depth": -1,
                    "num_leaves": 31,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "verbose": -1,
                    "n_jobs": -1,
                }
                params.update(hyperparameters)
                model_artifact = lgb.LGBMRegressor(**params)

            elif m_type == "random_forest":
                params = {
                    "n_estimators": 100,
                    "max_depth": 12,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 42,
                    "n_jobs": -1,
                }
                params.update(hyperparameters)
                model_artifact = RandomForestRegressor(**params)

            elif m_type == "ridge":
                params = {"alpha": 1.0, "solver": "auto", "random_state": 42}
                params.update(hyperparameters)
                model_artifact = Ridge(**params)

            model_artifact.fit(X_train_raw, y_train)
            y_pred = model_artifact.predict(X_test_raw)

        elif m_type in ["lstm", "gru"]:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)

            tf.random.set_seed(42)
            np.random.seed(42)

            time_steps = int(hyperparameters.get("time_steps", 12))
            epochs = int(hyperparameters.get("epochs", 20))
            batch_size = int(hyperparameters.get("batch_size", 32))
            hidden_dim = int(hyperparameters.get("hidden_dim", 64))
            learning_rate = float(hyperparameters.get("learning_rate", 0.001))
            dropout = float(hyperparameters.get("dropout", 0.2))

            X_tr_seq, y_tr_seq = create_sequences(X_train_scaled, y_train, time_steps)
            X_te_seq, y_te_seq = create_sequences(X_test_scaled, y_test, time_steps)

            if len(X_tr_seq) == 0 or len(X_te_seq) == 0:
                raise ValueError("Không đủ dữ liệu tạo chuỗi thời gian (time sequences).")

            input_shape = (X_tr_seq.shape[1], X_tr_seq.shape[2])
            tf_model = build_tf_rnn_model(
                input_shape=input_shape,
                hidden_dim=hidden_dim,
                rnn_type=m_type,
                learning_rate=learning_rate,
                dropout=dropout,
            )

            tf_model.fit(
                X_tr_seq,
                y_tr_seq,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                shuffle=False,
            )

            y_pred_seq = tf_model.predict(X_te_seq, verbose=0)
            y_pred = y_pred_seq.flatten()
            y_test = y_te_seq
            model_artifact = tf_model
        else:
            raise ValueError(f"Loại mô hình '{model_type}' chưa được hỗ trợ.")

        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        eps = 1e-5
        mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + eps))) * 100)

        metrics = {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{target_type}_{m_type}_{version_id}_{target_column}_{timestamp}"
        file_path = os.path.join(MODEL_STORAGE_DIR, f"{model_id}.joblib")

        if m_type in ["lstm", "gru"]:
            tf_model_path = os.path.join(MODEL_STORAGE_DIR, f"{model_id}.keras")
            model_artifact.save(tf_model_path)
            model_artifact_to_save = tf_model_path
        else:
            model_artifact_to_save = model_artifact

        save_dict = {
            "model_id": model_id,
            "model_type": m_type,
            "target_type": target_type,
            "model_artifact": model_artifact_to_save,
            "scaler": scaler,
            "feature_names": feature_names,
            "target_column": target_column,
            "metrics": metrics,
        }
        joblib.dump(save_dict, file_path)

        with engine.begin() as conn:
            conn.execute(
                text(f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.ml_model_registry (
                    model_id VARCHAR(255) PRIMARY KEY,
                    model_type VARCHAR(50),
                    target_type VARCHAR(50),
                    version_id VARCHAR(100),
                    id_attraction VARCHAR(50),
                    target_column VARCHAR(100),
                    r2_score FLOAT,
                    rmse FLOAT,
                    mae FLOAT,
                    metrics_json TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            )

            insert_sql = text(f"""
                INSERT INTO {SCHEMA_NAME}.ml_model_registry
                (model_id, model_type, target_type, version_id, id_attraction, target_column, r2_score, rmse, mae, metrics_json, created_at)
                VALUES (:m_id, :m_type, :t_type, :v_id, :attr, :target, :r2, :rmse, :mae, :m_json, NOW());
            """)
            conn.execute(
                insert_sql,
                {
                    "m_id": model_id,
                    "m_type": m_type,
                    "t_type": target_type,
                    "v_id": version_id,
                    "attr": id_attraction,
                    "target": target_column,
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                    "m_json": json.dumps(metrics),
                },
            )

        return {
            "status": "success",
            "model_id": model_id,
            "model_type": m_type,
            "target_type": target_type,
            "version_id": version_id,
            "id_attraction": id_attraction,
            "target_column": target_column,
            "train_rows": len(X_train_raw),
            "test_rows": len(X_test_raw),
            "metrics": metrics,
            "feature_names": feature_names,
            "created_at": datetime.now().isoformat(),
        }

    @classmethod
    def delete_model(cls, model_id: str) -> bool:
        """Xóa mô hình khỏi Database Registry và xóa các file đĩa liên quan (.joblib, .keras)."""
        delete_sql = text(f"""
            DELETE FROM {SCHEMA_NAME}.ml_model_registry
            WHERE model_id = :m_id;
        """)

        with engine.begin() as conn:
            result = conn.execute(delete_sql, {"m_id": model_id})
            if result.rowcount == 0:
                raise ValueError(
                    f"Không tìm thấy mô hình với ID '{model_id}' trong registry."
                )

        joblib_path = os.path.join(MODEL_STORAGE_DIR, f"{model_id}.joblib")
        keras_path = os.path.join(MODEL_STORAGE_DIR, f"{model_id}.keras")

        if os.path.exists(joblib_path):
            try:
                os.remove(joblib_path)
            except Exception as e:
                logger.warning(f"Không thể xóa file {joblib_path}: {str(e)}")

        if os.path.exists(keras_path):
            try:
                os.remove(keras_path)
            except Exception as e:
                logger.warning(f"Không thể xóa file {keras_path}: {str(e)}")

        return True

    @classmethod
    def get_dataset_date_range(
        cls, version_id: str, id_attraction: Optional[str] = "ALL"
    ) -> Dict[str, Optional[str]]:
        """Lấy ngày bắt đầu (MIN) và ngày kết thúc (MAX) của tập dữ liệu."""
        inspect_query = text(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = :schema_name AND table_name = :table_name;
        """)

        with engine.connect() as conn:
            columns = [
                row[0]
                for row in conn.execute(
                    inspect_query,
                    {"schema_name": SCHEMA_NAME, "table_name": version_id},
                ).fetchall()
            ]

        if "datetime" not in columns:
            return {"min_date": None, "max_date": None}

        conditions = []
        params = {}
        attr_val = (id_attraction or "ALL").strip().upper()

        if attr_val != "ALL":
            attr_conditions = []
            if "id_attraction" in columns:
                attr_conditions.append("id_attraction = :attr")
            if "attraction_id" in columns:
                attr_conditions.append("attraction_id = :attr")
            if attr_conditions:
                conditions.append(f"({' OR '.join(attr_conditions)})")
                params["attr"] = id_attraction

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = text(f"""
            SELECT 
                TO_CHAR(MIN(datetime), 'YYYY-MM-DD') AS min_date,
                TO_CHAR(MAX(datetime), 'YYYY-MM-DD') AS max_date
            FROM {SCHEMA_NAME}."{version_id}"
            {where_clause};
        """)

        with engine.connect() as conn:
            row = conn.execute(query, params).mappings().fetchone()
            return {
                "min_date": row["min_date"] if row else None,
                "max_date": row["max_date"] if row else None,
            }
    
    @classmethod
    def get_distinct_attractions(cls) -> List[str]:
        """
        Truy vấn tất cả các giá trị id_attraction/attraction_id duy nhất 
        từ các bảng trong schema data_prep.
        """
        # 1. Lấy danh sách tất cả các bảng trong schema data_prep
        tables_query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = :schema
              AND table_name NOT IN ('ml_model_registry');
        """)
        
        distinct_attractions = set()

        try:
            with engine.connect() as conn:
                tables = [r[0] for r in conn.execute(tables_query, {"schema": SCHEMA_NAME}).fetchall()]

                for table in tables:
                    # Kiểm tra xem bảng có cột id_attraction hoặc attraction_id không
                    cols_query = text("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_schema = :schema AND table_name = :table;
                    """)
                    cols = [r[0] for r in conn.execute(cols_query, {"schema": SCHEMA_NAME, "table": table}).fetchall()]

                    attr_col = None
                    if "id_attraction" in cols:
                        attr_col = "id_attraction"
                    elif "attraction_id" in cols:
                        attr_col = "attraction_id"

                    # Nếu có cột attraction, thực hiện SELECT DISTINCT
                    if attr_col:
                        data_query = text(f'SELECT DISTINCT "{attr_col}" FROM {SCHEMA_NAME}."{table}" WHERE "{attr_col}" IS NOT NULL;')
                        rows = conn.execute(data_query).fetchall()
                        for row in rows:
                            if row[0]:
                                distinct_attractions.add(str(row[0]).strip())

            # Sắp xếp danh sách kết quả
            result = sorted(list(distinct_attractions))
            return result
        except Exception as e:
            logger.error(f"Lỗi khi truy vấn danh sách attraction: {str(e)}")
            return []