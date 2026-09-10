import os
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import requests
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import HTTPException
from app.core.database import engine
from sqlalchemy import text

from app.modules.data_pipeline.missing_value.missing_value_services import (
    MissingValueService,
)
from app.modules.ml_pipeline.training.training_services import (
    MODEL_STORAGE_DIR,
    MLTrainingService,
    create_sequences,
)
from app.modules.ml_pipeline.future_forecast.future_forecast_schemas import (
    ForecastRequest,
    ModelMetadataResponse,
    FeatureMetadata,
    FeatureImportanceItem,
    PredictRequest,
    PredictResponse,
    ModelListItem,
)

logger = logging.getLogger(__name__)


class FutureFeatureService:
    LATITUDE = 46.6697
    LONGITUDE = 0.3603

    @classmethod
    def get_weather_forecast(
        cls, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={cls.LATITUDE}&longitude={cls.LONGITUDE}"
            f"&hourly=temperature_2m,relative_humidity_2m,direct_normal_irradiance"
            f"&start_date={start_str}&end_date={end_str}&timezone=Europe/Paris"
        )

        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                raise ValueError(f"Erreur API Open-Meteo HTTP {response.status_code}")

            data = response.json()["hourly"]
            df_weather = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(data["time"]),
                    "temperature": [
                        round(t, 1) if t is not None else 20.0
                        for t in data["temperature_2m"]
                    ],
                    "humidite": [
                        round(h, 1) if h is not None else 65.0
                        for h in data["relative_humidity_2m"]
                    ],
                    "rayonnement_solaire": [
                        round(r, 1) if r is not None else 0.0
                        for r in data["direct_normal_irradiance"]
                    ],
                }
            )
        except Exception as e:
            logger.warning(f"Fallback météo simulée suite à une erreur : {str(e)}")
            dates = pd.date_range(start=start_date, end=end_date, freq="1h")
            np.random.seed(42)
            base_temp = 20.0 + 5.0 * np.sin(np.pi * dates.hour / 12)
            df_weather = pd.DataFrame(
                {
                    "datetime": dates,
                    "temperature": np.round(base_temp, 1),
                    "humidite": np.random.uniform(50, 80, len(dates)).round(1),
                    "rayonnement_solaire": np.maximum(
                        0, 500 * np.sin(np.pi * (dates.hour - 6) / 12)
                    ).round(1),
                }
            )

        df_weather["day_degree_cold"] = np.maximum(
            0.0, 18.0 - df_weather["temperature"]
        ).round(1)
        df_weather["day_degree_hot"] = np.maximum(
            0.0, df_weather["temperature"] - 24.0
        ).round(1)

        df_weather["date_day"] = df_weather["datetime"].dt.date

        daily_stats = (
            df_weather.groupby("date_day")
            .agg(
                temp_max=("temperature", "max"),
                temp_min=("temperature", "min"),
                temp_moy=("temperature", "mean"),
                humidite_max=("humidite", "max"),
                humidite_min=("humidite", "min"),
                humidite_moy=("humidite", "mean"),
            )
            .reset_index()
        )

        df_weather = pd.merge(df_weather, daily_stats, on="date_day", how="left")
        df_weather.drop(columns=["date_day"], inplace=True)

        for col in ["temp_moy", "humidite_moy"]:
            if col in df_weather.columns:
                df_weather[col] = df_weather[col].round(1)

        # ĐẢM BẢO LUÔN CÓ ĐỦ CÁC CỘT THỜI TIẾT TỔNG HỢP VÀ KHÔNG BỊ KHUYẾT
        required_weather_cols = {
            "temp_min": "temperature",
            "temp_max": "temperature",
            "temp_moy": "temperature",
            "humidite_min": "humidite",
            "humidite_max": "humidite",
            "humidite_moy": "humidite",
        }
        for col, fallback_src in required_weather_cols.items():
            if col not in df_weather.columns or df_weather[col].isna().any():
                df_weather[col] = df_weather.get(col, df_weather[fallback_src]).fillna(
                    df_weather[fallback_src]
                )

        return df_weather

    @classmethod
    def get_historical_ouvert_profile(cls, id_attraction: str) -> pd.DataFrame:
        try:
            df_hist = MissingValueService.load_version_dataframe(
                "v0_raw", id_attraction=id_attraction
            )

            if df_hist.empty or "ouvert" not in df_hist.columns:
                return None

            if "is_open" in df_hist.columns:
                df_hist = df_hist[df_hist["is_open"] == 1]

            if (
                "type_frequentation" not in df_hist.columns
                or "heure" not in df_hist.columns
            ):
                return None

            # BẢNG ÁNH XẠ CHỮ -> SỐ
            freq_map = {
                "BF": 1,
                "MF": 2,
                "HF": 3,
                "THF": 4,
                "bf": 1,
                "mf": 2,
                "hf": 3,
                "thf": 4,
            }

            if df_hist["type_frequentation"].dtype == "object":
                df_hist["type_frequentation"] = (
                    df_hist["type_frequentation"]
                    .astype(str)
                    .str.strip()
                    .map(freq_map)
                    .fillna(1)
                    .astype("int64")
                )
            else:
                df_hist["type_frequentation"] = (
                    pd.to_numeric(df_hist["type_frequentation"], errors="coerce")
                    .fillna(1)
                    .astype("int64")
                )

            df_hist["heure"] = (
                pd.to_numeric(df_hist["heure"], errors="coerce")
                .fillna(0)
                .astype("int64")
            )

            group_cols = ["type_frequentation", "heure"]
            if "id_attraction" in df_hist.columns:
                group_cols = ["id_attraction"] + group_cols

            profile_df = (
                df_hist.groupby(group_cols)["ouvert"]
                .mean()
                .reset_index()
                .rename(columns={"ouvert": "ouvert_profil"})
            )
            return profile_df

        except Exception as e:
            logger.warning(
                f"Không thể lấy profil historique cho attraction {id_attraction}: {str(e)}"
            )
            return None

    @classmethod
    def get_dim_horaire_data(
        cls, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Lấy thông tin dim_horaire từ PostgreSQL theo khoảng thời gian dự báo.
        """
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        query = text("""
            SELECT 
                date::date AS date,
                frequentation,
                type_frequentation,
                jf,
                is_open
            FROM public.dim_horaire
            WHERE date::date BETWEEN :start_date AND :end_date
        """)
        try:
            with engine.connect() as conn:
                df_dim = pd.read_sql(
                    query,
                    con=conn,
                    params={"start_date": start_str, "end_date": end_str},
                )

            if not df_dim.empty:
                df_dim["date"] = pd.to_datetime(df_dim["date"]).dt.date

                freq_map = {
                    "BF": 1,
                    "MF": 2,
                    "HF": 3,
                    "THF": 4,
                    "bf": 1,
                    "mf": 2,
                    "hf": 3,
                    "thf": 4,
                }
                if df_dim["type_frequentation"].dtype == "object":
                    df_dim["type_frequentation"] = (
                        df_dim["type_frequentation"]
                        .astype(str)
                        .str.strip()
                        .map(freq_map)
                    )

                df_dim["type_frequentation"] = (
                    pd.to_numeric(df_dim["type_frequentation"], errors="coerce")
                    .fillna(1)
                    .astype("int64")
                )

                return df_dim
        except Exception as e:
            logger.warning(
                f"Không thể truy vấn public.dim_horaire từ DB: {str(e)}. Sử dụng giá trị fallback."
            )

        return pd.DataFrame()

    @classmethod
    def generate_future_dataframe(cls, req: ForecastRequest) -> pd.DataFrame:
        dates = pd.date_range(start=req.start_date, end=req.end_date, freq="1h")
        df_future = pd.DataFrame(
            {"datetime": dates, "id_attraction": req.id_attraction}
        )

        # 1. Tạo các cột thời gian chuẩn
        df_future["heure"] = df_future["datetime"].dt.hour
        df_future["hour"] = df_future["heure"]
        df_future["jour"] = df_future["datetime"].dt.day
        df_future["day"] = df_future["jour"]
        df_future["mois"] = df_future["datetime"].dt.month
        df_future["month"] = df_future["mois"]
        df_future["annee"] = df_future["datetime"].dt.year
        df_future["year"] = df_future["annee"]
        df_future["semaine"] = df_future["datetime"].dt.isocalendar().week.astype(int)
        df_future["week"] = df_future["semaine"]
        df_future["dayofweek"] = df_future["datetime"].dt.dayofweek
        df_future["is_weekend"] = (
            df_future["datetime"].dt.dayofweek.isin([5, 6]).astype(int)
        )
        df_future["interrompu"] = 0.0

        # 2. TỰ ĐỘNG ĐIỀN TỪ public.dim_horaire
        df_future["date"] = df_future["datetime"].dt.date
        df_dim = cls.get_dim_horaire_data(req.start_date, req.end_date)

        if not df_dim.empty:
            df_future = pd.merge(df_future, df_dim, on="date", how="left")

        # Fallback giá trị mặc định nếu ngày đó không có trong dim_horaire hoặc DB bị lỗi
        if (
            "type_frequentation" not in df_future.columns
            or df_future["type_frequentation"].isna().any()
        ):
            df_future["type_frequentation"] = (
                df_future.get("type_frequentation", pd.Series())
                .fillna(req.type_frequentation_default)
                .astype("int64")
            )

        if "jf" not in df_future.columns or df_future["jf"].isna().any():
            df_future["jf"] = df_future.get("jf", pd.Series()).fillna(0).astype("int64")

        if "is_open" not in df_future.columns or df_future["is_open"].isna().any():
            df_future["is_open"] = (
                df_future.get("is_open", pd.Series()).fillna(1).astype("int64")
            )

        if (
            "frequentation" not in df_future.columns
            or df_future["frequentation"].isna().any()
        ):
            freq_default_map = {1: 8000, 2: 15000, 3: 25000, 4: 35000}
            default_freq_series = (
                df_future["type_frequentation"].map(freq_default_map).fillna(10000)
            )
            df_future["frequentation"] = df_future.get(
                "frequentation", pd.Series()
            ).fillna(default_freq_series)

        df_future.drop(columns=["date"], inplace=True, errors="ignore")

        ouvert_mode_str = (
            req.ouvert_mode.value
            if hasattr(req.ouvert_mode, "value")
            else str(req.ouvert_mode)
        )

        # 3. Lấy profil historique cho ouvert
        df_profile = cls.get_historical_ouvert_profile(req.id_attraction)
        if df_profile is not None and not df_profile.empty:
            merge_cols = ["type_frequentation", "heure"]
            if "id_attraction" in df_profile.columns:
                merge_cols = ["id_attraction"] + merge_cols

            df_future = pd.merge(df_future, df_profile, on=merge_cols, how="left")
            df_future["ouvert_profil"] = df_future["ouvert_profil"].fillna(
                1.0 if ouvert_mode_str == "ideal" else 0.0
            )

            if ouvert_mode_str == "ideal":
                df_future["ouvert"] = np.where(
                    df_future["ouvert_profil"] >= float(req.open_threshold), 1.0, 0.0
                )
            else:
                df_future["ouvert"] = df_future["ouvert_profil"].astype(float)

            df_future.drop(columns=["ouvert_profil"], inplace=True, errors="ignore")
        else:
            df_future["ouvert"] = 1.0 if ouvert_mode_str == "ideal" else 0.0

        # 4. Thời tiết
        df_weather = cls.get_weather_forecast(req.start_date, req.end_date)
        df_future = pd.merge(df_future, df_weather, on="datetime", how="left")

        if req.temp_offset != 0.0:
            for col in ["temperature", "temp_max", "temp_min", "temp_moy"]:
                if col in df_future.columns:
                    df_future[col] = (df_future[col] + req.temp_offset).round(1)
            df_future["day_degree_cold"] = np.maximum(
                0.0, 18.0 - df_future["temperature"]
            ).round(1)
            df_future["day_degree_hot"] = np.maximum(
                0.0, df_future["temperature"] - 24.0
            ).round(1)

        # 5. Quy tắc nghiệp vụ & Operation factor
        df_future = MissingValueService.apply_visitor_domain_rules(df_future)

        if req.operation_factor < 1.0:
            df_future["operation"] = (
                df_future["operation"] * req.operation_factor
            ).round(1)

        return df_future


class ForecastService:
    STORAGE_DIR = MODEL_STORAGE_DIR

    @classmethod
    def generate_future_dataframe(cls, req: ForecastRequest) -> pd.DataFrame:
        return FutureFeatureService.generate_future_dataframe(req)

    @classmethod
    def get_available_attractions(cls) -> List[str]:
        attractions = set()
        if os.path.exists(cls.STORAGE_DIR):
            for file in os.listdir(cls.STORAGE_DIR):
                if file.endswith(".joblib"):
                    try:
                        saved_data = joblib.load(os.path.join(cls.STORAGE_DIR, file))
                        attr = saved_data.get("id_attraction")
                        if attr and attr != "ALL":
                            attractions.add(str(attr))
                    except Exception:
                        continue

        if not attractions:
            try:
                db_attrs = MLTrainingService.get_distinct_attractions()
                attractions.update(db_attrs)
            except Exception as e:
                logger.warning(f"Không thể đọc attractions từ DB: {str(e)}")

        return sorted(list(attractions))

    @classmethod
    def load_model_artifacts(cls, model_id: str) -> Dict[str, Any]:
        clean_id = model_id.replace(".joblib", "").replace(".keras", "")
        joblib_path = os.path.join(cls.STORAGE_DIR, f"{clean_id}.joblib")

        if not os.path.exists(joblib_path):
            raise HTTPException(
                status_code=404,
                detail=f"Fichier modèle introuvable : '{clean_id}.joblib'.",
            )

        try:
            saved_data = joblib.load(joblib_path)
            model_type = saved_data.get("model_type", "")

            if model_type in ["lstm", "gru"]:
                keras_path = os.path.join(cls.STORAGE_DIR, f"{clean_id}.keras")
                if not os.path.exists(keras_path):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Fichier modèle Keras introuvable : '{clean_id}.keras'.",
                    )
                saved_data["tf_model"] = tf.keras.models.load_model(keras_path)

            return saved_data
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Erreur lors du chargement des artefacts du modèle '{clean_id}': {str(e)}",
            )

    @classmethod
    def _predict_with_artifacts(
        cls, artifacts: Dict[str, Any], df_input: pd.DataFrame
    ) -> np.ndarray:
        feature_names = artifacts.get("feature_names", [])
        model_type = artifacts.get("model_type", "")
        scaler = artifacts.get("scaler", None)

        missing_cols = [c for c in feature_names if c not in df_input.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes d'entrée manquantes pour la prédiction : {missing_cols}",
            )

        X_df = df_input[feature_names].copy()
        X_df = X_df.ffill().bfill().fillna(0)

        if model_type in ["lstm", "gru"]:
            tf_model = artifacts.get("tf_model")
            X_val = X_df.values
            if scaler is not None:
                X_val = scaler.transform(X_val)

            time_steps = 12
            y_dummy = np.zeros(len(X_val))

            if len(X_val) < time_steps:
                padding = np.tile(X_val[:1], (time_steps - len(X_val), 1))
                X_val_padded = np.vstack([padding, X_val])
                y_dummy_padded = np.zeros(len(X_val_padded))
                X_seq, _ = create_sequences(
                    X_val_padded, y_dummy_padded, time_steps=time_steps
                )
            else:
                X_seq, _ = create_sequences(X_val, y_dummy, time_steps=time_steps)

            if len(X_seq) == 0:
                preds = np.zeros(len(df_input))
            else:
                preds_seq = tf_model.predict(X_seq, verbose=0).flatten()
                if len(preds_seq) < len(df_input):
                    preds = np.pad(
                        preds_seq,
                        (len(df_input) - len(preds_seq), 0),
                        mode="edge",
                    )
                else:
                    preds = preds_seq[-len(df_input) :]
        else:
            model_obj = artifacts.get("model_artifact")
            X_val = X_df
            if scaler is not None:
                X_val = scaler.transform(X_df)

            preds = model_obj.predict(X_val)

        return np.maximum(0.0, preds)

    @classmethod
    def _find_best_model_for_target(cls, target_type: str) -> str:
        if not os.path.exists(cls.STORAGE_DIR):
            raise HTTPException(
                status_code=404, detail="Répertoire modèle introuvable."
            )

        matching_files = [
            f.replace(".joblib", "")
            for f in os.listdir(cls.STORAGE_DIR)
            if f.endswith(".joblib") and target_type in f
        ]

        if not matching_files:
            all_files = [
                f.replace(".joblib", "")
                for f in os.listdir(cls.STORAGE_DIR)
                if f.endswith(".joblib")
            ]
            if not all_files:
                raise HTTPException(status_code=404, detail="Aucun modèle disponible.")
            return all_files[0]

        matching_files.sort(reverse=True)
        return matching_files[0]

    @classmethod
    def list_trained_models(cls) -> List[ModelListItem]:
        if not os.path.exists(cls.STORAGE_DIR):
            os.makedirs(cls.STORAGE_DIR, exist_ok=True)
            return []

        models_found = []
        for file in os.listdir(cls.STORAGE_DIR):
            if file.endswith(".joblib"):
                model_id = file.replace(".joblib", "")
                try:
                    artifacts = cls.load_model_artifacts(model_id)
                    target = artifacts.get(
                        "target_type",
                        "visitor"
                        if "visitor" in model_id
                        else "electricity"
                        if "elec" in model_id
                        else "thermal"
                        if "thermal" in model_id
                        else "general",
                    )
                    algorithm = artifacts.get("model_type", "Unknown")
                    metrics = artifacts.get("metrics", {})

                    models_found.append(
                        ModelListItem(
                            model_id=model_id,
                            name=f"Modèle {algorithm.upper()} ({model_id})",
                            target=target,
                            algorithm=algorithm,
                            metrics=metrics,
                        )
                    )
                except Exception:
                    continue

        return models_found

    @classmethod
    def get_model_metadata(cls, model_id: str) -> ModelMetadataResponse:
        artifacts = cls.load_model_artifacts(model_id)
        feature_names = artifacts.get("feature_names", [])
        model_obj = artifacts.get("model_artifact")
        algorithm_name = artifacts.get("model_type", "ML Model").upper()
        target_variable = artifacts.get("target_column", "Valeur Cible")

        features_meta = [
            FeatureMetadata(
                name=name,
                label=name.replace("_", " ").title(),
                type="number",
                default_value=0.0,
            )
            for name in feature_names
        ]

        importances_list = []
        if model_obj is not None and hasattr(model_obj, "feature_importances_"):
            importances = model_obj.feature_importances_
            for name, imp in zip(feature_names, importances):
                importances_list.append(
                    FeatureImportanceItem(name=name, importance=float(round(imp, 4)))
                )
            importances_list.sort(key=lambda x: x.importance, reverse=True)

        return ModelMetadataResponse(
            model_id=model_id,
            name=f"Modèle {algorithm_name} ({model_id})",
            algorithm=algorithm_name,
            target_variable=target_variable,
            features=features_meta,
            feature_importances=importances_list if importances_list else None,
        )

    @classmethod
    def predict_single_instance(cls, req: PredictRequest) -> PredictResponse:
        artifacts = cls.load_model_artifacts(req.model_id)
        required_cols = artifacts.get("feature_names", [])

        row_data = {}
        for col in required_cols:
            if col not in req.features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Variable d'entrée requise manquante : '{col}'",
                )
            row_data[col] = [float(req.features[col])]

        df_input = pd.DataFrame(row_data)
        preds = cls._predict_with_artifacts(artifacts, df_input)
        raw_prediction = preds[0]

        target_type = artifacts.get("target_type", "")
        is_energy = (
            "elec" in target_type
            or "thermal" in target_type
            or "elec" in req.model_id
            or "thermal" in req.model_id
        )

        unit = (
            "kWh"
            if is_energy
            else (
                "pers."
                if "visitor" in target_type or "visitor" in req.model_id
                else "unités"
            )
        )

        # Năng lượng > 0, các chỉ số khác >= 0
        if is_energy:
            pred_val = float(np.round(np.maximum(0.01, raw_prediction), 2))
        else:
            pred_val = float(np.round(np.maximum(0.0, raw_prediction), 2))

        return PredictResponse(
            status="success",
            model_id=req.model_id,
            predicted_value=pred_val,
            unit=unit,
            confidence_interval=[round(pred_val * 0.95, 2), round(pred_val * 1.05, 2)],
            timestamp=datetime.now().isoformat(),
        )

    @classmethod
    def predict(
        cls, req: ForecastRequest
    ) -> Tuple[pd.DataFrame, str, str, Optional[str]]:
        if req.custom_hourly_features and len(req.custom_hourly_features) > 0:
            df_features = pd.DataFrame(req.custom_hourly_features)
            if "datetime" in df_features.columns:
                df_features["datetime"] = pd.to_datetime(df_features["datetime"])
            if "id_attraction" not in df_features.columns:
                df_features["id_attraction"] = req.id_attraction
        else:
            df_features = FutureFeatureService.generate_future_dataframe(req)

        visitor_model_used = None

        # ==========================================
        # TRƯỜNG HỢP 1: NĂNG LƯỢNG (ELECTRICITY / THERMAL)
        # ==========================================
        if req.target_type in ["electricity", "thermal"]:
            visitor_model_id = req.visitor_model_id or cls._find_best_model_for_target(
                "visitor"
            )
            visitor_artifacts = cls.load_model_artifacts(visitor_model_id)
            visitor_model_used = visitor_model_id

            pred_vis_hourly = cls._predict_with_artifacts(
                visitor_artifacts, df_features
            )

            # 🟢 1. VISITOR_COUNT PHẢI LÀ SỐ NGUYÊN (INT >= 0)
            pred_vis_series = (
                pd.Series(pred_vis_hourly).clip(lower=0).round().astype(int)
            )

            df_features["visitor_count"] = pred_vis_series
            df_features["nb_visiteurs"] = pred_vis_series
            df_features["visitors"] = pred_vis_series

            target_model_id = req.model_id or cls._find_best_model_for_target(
                req.target_type
            )
            target_artifacts = cls.load_model_artifacts(target_model_id)

            preds_energy_hourly = cls._predict_with_artifacts(
                target_artifacts, df_features
            )

            # 🟢 2. NĂNG LƯỢNG PHẢI LỚN HƠN 0 (clip tối thiểu 0.01)
            df_features["predicted_value"] = (
                pd.Series(preds_energy_hourly).clip(lower=0.01).round(2)
            )

            return df_features, req.target_type, target_model_id, visitor_model_used

        # ==========================================
        # TRƯỜNG HỢP 2: LƯỢNG KHÁCH (VISITOR)
        # ==========================================
        elif req.target_type == "visitor":
            target_model_id = req.model_id or cls._find_best_model_for_target("visitor")
            target_artifacts = cls.load_model_artifacts(target_model_id)

            preds_hourly = cls._predict_with_artifacts(target_artifacts, df_features)

            # 🟢 1. VISITOR LÀ SỐ NGUYÊN (ROUND VÀ INT)
            pred_visitor_series = (
                pd.Series(preds_hourly).clip(lower=0).round().astype(int)
            )

            df_features["predicted_value"] = pred_visitor_series.astype(float)
            df_features["visitor_count"] = pred_visitor_series

            return df_features, req.target_type, target_model_id, None
