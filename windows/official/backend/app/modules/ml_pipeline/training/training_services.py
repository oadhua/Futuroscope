import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from dotenv import load_dotenv
from google import genai

# Scikit-learn algorithms & metrics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

# XGBoost & SHAP
import xgboost as xgb
import shap

# TensorFlow / Keras (RNNs: LSTM, GRU)
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
    col_clean = col_name.lower().strip()
    energy_prefixes = (
        "elec_",
        "ec_",
        "energy_",
        "power_",
        "p_kw",
        "kwh",
        "hvac_",
    )
    return (
        col_clean.startswith(energy_prefixes)
        or "elec" in col_clean
        or "power" in col_clean
    )


def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int = 12):
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
    def compute_shap_importance(
        cls, model, X_test: pd.DataFrame, feature_names: List[str], m_type: str
    ) -> List[Dict[str, float]]:
        """Calcule l'importance des variables selon la méthode SHAP."""
        try:
            sample_X = (
                X_test.sample(min(100, len(X_test)), random_state=42)
                if len(X_test) > 100
                else X_test
            )

            if m_type in ["xgboost", "random_forest", "gradient_boosting"]:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(sample_X)
            elif m_type in ["mlr", "linear_regression"]:
                explainer = shap.LinearExplainer(model, sample_X)
                shap_values = explainer.shap_values(sample_X)
            else:
                # KernelExplainer pour SVR, KNN, ANN (MLP), etc.
                explainer = shap.KernelExplainer(
                    model.predict,
                    sample_X.sample(min(20, len(sample_X)), random_state=42),
                )
                shap_values = explainer.shap_values(sample_X)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importance_list = [
                {"feature": feat, "importance": float(val)}
                for feat, val in zip(feature_names, mean_abs_shap)
            ]
            return sorted(importance_list, key=lambda x: x["importance"], reverse=True)
        except Exception as e:
            logger.warning(f"Impossible de calculer les valeurs SHAP : {str(e)}")
            return []

    @classmethod
    def compute_pfi_importance(
        cls, model, X_test: pd.DataFrame, y_test: np.ndarray, feature_names: List[str]
    ) -> List[Dict[str, float]]:
        """Calcule l'importance par permutation des variables (PFI)."""
        try:
            result = permutation_importance(
                model, X_test, y_test, n_repeats=5, random_state=42, scoring="r2"
            )
            importance_list = [
                {"feature": feat, "importance": float(val)}
                for feat, val in zip(feature_names, result.importances_mean)
            ]
            return sorted(importance_list, key=lambda x: x["importance"], reverse=True)
        except Exception as e:
            logger.warning(f"Impossible de calculer le PFI : {str(e)}")
            return []

    @classmethod
    def compare_with_previous_run(
        cls,
        target_type: str,
        target_column: str,
        id_attraction: str,
        current_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compare les performances avec le dernier entraînement partageant la même configuration."""
        query = text(f"""
            SELECT model_id, r2_score, rmse, mae
            FROM {SCHEMA_NAME}.ml_model_registry
            WHERE target_type = :t_type AND target_column = :t_col AND id_attraction = :attr
            ORDER BY created_at DESC
            LIMIT 1;
        """)
        try:
            with engine.connect() as conn:
                row = (
                    conn.execute(
                        query,
                        {
                            "t_type": target_type,
                            "t_col": target_column,
                            "attr": id_attraction,
                        },
                    )
                    .mappings()
                    .fetchone()
                )

            if not row:
                return {
                    "previous_model_id": None,
                    "r2_diff": None,
                    "rmse_diff": None,
                    "mae_diff": None,
                    "improvement_summary": "Il s'agit du premier modèle entraîné pour cette configuration.",
                }

            r2_diff = current_metrics["r2"] - float(row["r2_score"])
            rmse_diff = current_metrics["rmse"] - float(
                row["rmse_score"] if "rmse_score" in row else row["rmse"]
            )
            mae_diff = current_metrics["mae"] - float(row["mae"])

            if r2_diff > 0.01 and rmse_diff < 0:
                summary = f"Le nouveau modèle améliore nettement les performances par rapport à la version précédente ({row['model_id']}) : R² augmente de {r2_diff:+.4f}, RMSE diminue de {abs(rmse_diff):.4f}."
            elif r2_diff < -0.01:
                summary = f"Les performances du modèle ont diminué par rapport à la version précédente ({row['model_id']}) : R² diminue de {abs(r2_diff):.4f}."
            else:
                summary = f"Performances équivalentes à celles du modèle précédent ({row['model_id']})."

            return {
                "previous_model_id": row["model_id"],
                "r2_diff": round(r2_diff, 4),
                "rmse_diff": round(rmse_diff, 4),
                "mae_diff": round(mae_diff, 4),
                "improvement_summary": summary,
            }
        except Exception as e:
            logger.error(f"Erreur lors de la comparaison des performances : {str(e)}")
            return {
                "previous_model_id": None,
                "r2_diff": None,
                "rmse_diff": None,
                "mae_diff": None,
                "improvement_summary": "Impossible de récupérer l'historique d'entraînement pour la comparaison.",
            }

    @classmethod
    def generate_ai_explanation(
        cls,
        model_type: str,
        target_column: str,
        target_type: str,
        id_attraction: str,
        metrics: Dict[str, float],
        shap_imp: List[Dict[str, float]],
        pfi_imp: List[Dict[str, float]],
        comp: Dict[str, Any],
    ) -> str:
        """Génère automatiquement une analyse détaillée via l'API Gemini AI."""
        top_shap = [
            f"{x['feature']} (SHAP: {x['importance']:.4f})" for x in shap_imp[:5]
        ]
        top_pfi = [
            f"{x['feature']} (ΔR² PFI: {x['importance']:.4f})" for x in pfi_imp[:5]
        ]

        if target_type.lower() == "visitor":
            domain_context = """
- **Rôle d'Expert** : Expert Senior en Data Science, Analyse de Fréquentation, Marketing Prédictif et Operational Management dans les Parcs d'Attractions.
- **Unité Cible probable** : Nombre de visiteurs / Passages (Personnes).
- **Mécanismes à analyser** : Influence des conditions météo (pluie, température), de la saisonnalité (jours fériés, vacances scolaires, jour de la semaine), du calendrier des événements, et de l'inertie de fréquentation.
            """
        else:
            domain_context = """
- **Rôle d'Expert** : Expert Senior en Data Science, Génie Énergétique et Ingénierie CVC / HVAC (Chauffage, Ventilation, Climatisation).
- **Unité Cible probable** : Consommation électrique/thermique (kWh, kW) ou Température (°C).
- **Mécanismes à analyser** : Influence des conditions météo extérieures (température, inertie thermique), des plages d'occupation, des charges thermiques internes liées à la fréquentation (apport calorique humain), et du pilotage des équipements CVC.
            """

        prompt = f"""
Vous êtes un expert reconnu dans votre domaine. Veuillez fournir une analyse technique détaillée, rigoureuse et structurée en français expliquant les résultats d'entraînement du modèle de Machine Learning suivant :

---
### 1. CONTEXTE DE L'EXPERTISE & CONFIGURATION
{domain_context.strip()}

- **Variable Cible (Target Variable)** : `{target_column}` (Catégorie : {target_type.upper()})
- **Périmètre d'Application** : Attraction / Zone `{id_attraction}`
- **Algorithme d'Apprentissage** : {model_type.upper()}

---
### 2. PERFORMANCES DE PRÉDICTION (MÉTRIQUES)
- **Score R² (Coefficient de Détermination)** : {metrics.get("r2", 0):.4f}
- **RMSE (Root Mean Squared Error)** : {metrics.get("rmse", 0):.4f}
- **MAE (Mean Absolute Error)** : {metrics.get("mae", 0):.4f}
- **Comparaison Historique** : {comp.get("improvement_summary", "Aucun historique disponible")}

---
### 3. EXPLICABILITÉ & IMPACT DES VARIABLES D'ENTRÉE (SHAP & PFI)
- **Top 5 des variables décisionnelles (SHAP - Magnitude moyenne de l'impact sur `{target_column}`)** : 
  {", ".join(top_shap) if top_shap else "Non disponible"}

- **Top 5 des variables les plus critiques (PFI - Chute de performance R² par permutation)** : 
  {", ".join(top_pfi) if top_pfi else "Non disponible"}

---
### CONSIGNES DE RÉDACTION :
Veuillez rédiger une analyse structurée en 3 paragraphes principaux :

1. **Évaluation de la Précision et des Erreurs de Prédiction** :
   - Évaluez la valeur du score R² pour la prédiction de la variable `{target_column}`.
   - Interprétez concrètement les marges d'erreur MAE et RMSE dans l'unité réelle de la variable cible `{target_column}` (ex: écart moyen en nombre de visiteurs pour 'visitor', ou en kW/kWh pour 'energy').
   - Expliquez l'évolution par rapport au modèle précédent.

2. **Analyse Causalité / SHAP & Relations d'Impact Direct** :
   - Identifiez la **première variable du classement SHAP** et expliquez en détail **comment et pourquoi** elle influence la variable cible `{target_column}`.
   - Clarifiez le sens de la contribution (ex: relation directe positive où l'augmentation de la variable augmente `{target_column}`, ou relation inverse/non-linéaire).
   - Reliez cette analyse aux mécanismes métier (ex: météo/vacances sur la fréquentation VS température extérieure/affluence sur la consommation énergétique).

3. **Robustesse PFI & Recommandations Opérationnelles** :
   - Comparez le classement SHAP avec la Permutation Importance (PFI).
   - Expliquez ce que la dégradation du R² révèle sur la dépendance du modèle envers ces variables clés.
   - Formulez 1 à 2 conseils pratiques d'exploitation ou de gestion fondés sur ces résultats.

Gardez un ton professionnel, scientifique et directement exploitable par les équipes techniques et d'exploitation.
"""

        try:
            env_path = Path(__file__).resolve().parents[3] / ".env"
            load_dotenv(dotenv_path=env_path, override=True)
            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                logger.error("Clé GEMINI_API_KEY introuvable dans le fichier .env !")
                raise ValueError("Clé GEMINI_API_KEY manquante")

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text.strip()

        except Exception as e:
            logger.error(f"Erreur lors de l'appel à l'API Gemini : {str(e)}")
            quality = "élevée" if metrics.get("r2", 0) > 0.8 else "moyenne"
            return (
                f"[Fallback] Le modèle {model_type.upper()} atteint une précision {quality} (R² = {metrics.get('r2', 0):.4f}). "
                f"Comparaison : {comp.get('improvement_summary', '')}"
            )

    @classmethod
    def list_available_versions(
        cls, id_attraction: Optional[str] = "ALL"
    ) -> List[Dict[str, Any]]:
        query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = :schema_name AND table_name NOT IN ('ml_model_registry')
            ORDER BY table_name DESC;
        """)
        try:
            with engine.connect() as conn:
                rows = conn.execute(query, {"schema_name": SCHEMA_NAME}).fetchall()
                return [
                    {
                        "version_id": row[0],
                        "table_name": row[0],
                        "id_attraction": id_attraction or "ALL",
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Erreur lors de la lecture des versions : {str(e)}")
            return []

    @classmethod
    def get_version_columns(
        cls, version_id: str, id_attraction: Optional[str] = "ALL"
    ) -> List[str]:
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
        if version_id.lower() == "v0_raw":
            raise ValueError("Impossible de supprimer la version originale 'v0_raw'.")
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
        conditions, params = [], {}
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

        query_str = (
            f'SELECT * FROM {SCHEMA_NAME}."{version_id}" {where_clause} {order_clause};'
        )

        with engine.connect() as conn:
            df = pd.read_sql(text(query_str), conn, params=params)

        if df.empty:
            raise ValueError(f"Aucune donnée trouvée pour la version '{version_id}'.")

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
    ):
        df = cls.load_data_from_db(version_id, id_attraction, start_date, end_date)
        if target_column not in df.columns:
            raise ValueError(
                f"La colonne cible '{target_column}' n'existe pas dans le jeu de données."
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
                "Aucune variable explicative (feature) valide n'a été trouvée."
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
                    f"La date de séparation 'split_date' ({split_date}) génère un ensemble d'entraînement ou de test vide."
                )
        else:
            split_idx = int(len(X_df) * (1 - test_size))
            X_train_raw, X_test_raw = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
            y_train, y_test = y_s.iloc[:split_idx].values, y_s.iloc[split_idx:].values

        m_type = model_type.lower()
        model_artifact = None
        scaler = None

        # Standardisation obligatoire pour certains modèles (SVR, KNN, ANN, MLR)
        needs_scaling = m_type in [
            "svr",
            "knn",
            "ann",
            "mlp",
            "mlr",
            "linear_regression",
        ]
        if needs_scaling:
            scaler = StandardScaler()
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train_raw),
                columns=feature_names,
                index=X_train_raw.index,
            )
            X_test = pd.DataFrame(
                scaler.transform(X_test_raw),
                columns=feature_names,
                index=X_test_raw.index,
            )
        else:
            X_train, X_test = X_train_raw, X_test_raw

        # --- 1. XGBOOST ---
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
            model_artifact.fit(X_train, y_train)
            y_pred = model_artifact.predict(X_test)

        # --- 2. GRADIENT BOOSTING (Sklearn) ---
        elif m_type in ["gradient_boosting", "gb"]:
            params = {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 5,
                "random_state": 42,
            }
            params.update(hyperparameters)
            model_artifact = GradientBoostingRegressor(**params)
            model_artifact.fit(X_train, y_train)
            y_pred = model_artifact.predict(X_test)

        # --- 3. RANDOM FOREST ---
        elif m_type in ["random_forest", "rf"]:
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
            model_artifact.fit(X_train, y_train)
            y_pred = model_artifact.predict(X_test)

        # --- 4. MULTIPLE LINEAR REGRESSION (MLR) ---
        elif m_type in ["mlr", "linear_regression"]:
            model_artifact = LinearRegression()
            model_artifact.fit(X_train, y_train)
            y_pred = model_artifact.predict(X_test)

        # --- 5. SUPPORT VECTOR REGRESSION (SVR) ---
        elif m_type in ["svr", "support_vector"]:
            params = {"kernel": "rbf", "C": 1.0, "epsilon": 0.1}
            params.update(hyperparameters)
            model_artifact = SVR(**params)
            model_artifact.fit(X_train, y_train)
            y_pred = model_artifact.predict(X_test)

        # --- 6. K-NEAREST NEIGHBORS (KNN) ---
        elif m_type in ["knn", "k_neighbors"]:
            params = {"n_neighbors": 5, "weights": "uniform", "n_jobs": -1}
            params.update(hyperparameters)
            model_artifact = KNeighborsRegressor(**params)
            model_artifact.fit(X_train, y_train)
            y_pred = model_artifact.predict(X_test)

        # --- 7. ARTIFICIAL NEURAL NETWORK (ANN / MLP) ---
        elif m_type in ["ann", "mlp"]:
            params = {
                "hidden_layer_sizes": (64, 32),
                "activation": "relu",
                "max_iter": 300,
                "random_state": 42,
            }
            params.update(hyperparameters)
            model_artifact = MLPRegressor(**params)
            model_artifact.fit(X_train, y_train)
            y_pred = model_artifact.predict(X_test)

        # --- 8 & 9. LSTM & GRU (TensorFlow / Keras) ---
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
                raise ValueError(
                    "Données insuffisantes pour créer les séquences temporelles."
                )

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
            raise ValueError(
                f"Le type de modèle '{model_type}' n'est pas pris en charge."
            )

        # Calcul des métriques
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        metrics = {"r2": r2, "rmse": rmse, "mae": mae}

        # --- CALCUL SHAP ET PFI ---
        if m_type in ["lstm", "gru"]:
            shap_imp, pfi_imp = [], []
        else:
            shap_imp = cls.compute_shap_importance(
                model_artifact, X_test, feature_names, m_type
            )
            pfi_imp = cls.compute_pfi_importance(
                model_artifact, X_test, y_test, feature_names
            )

        perf_comp = cls.compare_with_previous_run(
            target_type, target_column, id_attraction, metrics
        )
        ai_exp = cls.generate_ai_explanation(
            m_type,
            target_column,
            target_type,
            id_attraction,
            metrics,
            shap_imp,
            pfi_imp,
            perf_comp,
        )

        # Sauvegarde
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
            "shap_importance": shap_imp,
            "pfi_importance": pfi_imp,
            "performance_improvement": perf_comp,
            "ai_explanation": ai_exp,
            "created_at": datetime.now().isoformat(),
        }

    @classmethod
    def delete_model(cls, model_id: str) -> bool:
        delete_sql = text(
            f"DELETE FROM {SCHEMA_NAME}.ml_model_registry WHERE model_id = :m_id;"
        )
        with engine.begin() as conn:
            result = conn.execute(delete_sql, {"m_id": model_id})
            if result.rowcount == 0:
                raise ValueError(
                    f"Modèle introuvable avec l'identifiant '{model_id}' dans le registre."
                )

        joblib_path = os.path.join(MODEL_STORAGE_DIR, f"{model_id}.joblib")
        keras_path = os.path.join(MODEL_STORAGE_DIR, f"{model_id}.keras")
        if os.path.exists(joblib_path):
            os.remove(joblib_path)
        if os.path.exists(keras_path):
            os.remove(keras_path)
        return True

    @classmethod
    def get_dataset_date_range(
        cls, version_id: str, id_attraction: Optional[str] = "ALL"
    ) -> Dict[str, Optional[str]]:
        inspect_query = text(f"""
            SELECT column_name FROM information_schema.columns 
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

        conditions, params = [], {}
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
            SELECT TO_CHAR(MIN(datetime), 'YYYY-MM-DD') AS min_date, TO_CHAR(MAX(datetime), 'YYYY-MM-DD') AS max_date
            FROM {SCHEMA_NAME}."{version_id}" {where_clause};
        """)
        with engine.connect() as conn:
            row = conn.execute(query, params).mappings().fetchone()
            return {
                "min_date": row["min_date"] if row else None,
                "max_date": row["max_date"] if row else None,
            }

    @classmethod
    def get_distinct_attractions(cls) -> List[str]:
        tables_query = text(
            f"SELECT table_name FROM information_schema.tables WHERE table_schema = :schema AND table_name NOT IN ('ml_model_registry');"
        )
        distinct_attractions = set()
        try:
            with engine.connect() as conn:
                tables = [
                    r[0]
                    for r in conn.execute(
                        tables_query, {"schema": SCHEMA_NAME}
                    ).fetchall()
                ]
                for table in tables:
                    cols_query = text(
                        f"SELECT column_name FROM information_schema.columns WHERE table_schema = :schema AND table_name = :table;"
                    )
                    cols = [
                        r[0]
                        for r in conn.execute(
                            cols_query, {"schema": SCHEMA_NAME, "table": table}
                        ).fetchall()
                    ]
                    attr_col = (
                        "id_attraction"
                        if "id_attraction" in cols
                        else ("attraction_id" if "attraction_id" in cols else None)
                    )
                    if attr_col:
                        data_query = text(
                            f'SELECT DISTINCT "{attr_col}" FROM {SCHEMA_NAME}."{table}" WHERE "{attr_col}" IS NOT NULL;'
                        )
                        for row in conn.execute(data_query).fetchall():
                            if row[0]:
                                distinct_attractions.add(str(row[0]).strip())
            return sorted(list(distinct_attractions))
        except Exception as e:
            logger.error(f"Erreur lors de la requête des attractions : {str(e)}")
            return []
