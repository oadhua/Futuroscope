import re
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from app.modules.data_pipeline.queries import get_dynamic_gathering_query


def get_data_prep_versions_service(db: Session) -> List[str]:
    """Lấy danh sách tất cả các bảng phiên bản trong schema data_prep (bỏ qua các bảng hệ thống)."""
    db.commit()
    connection = db.connection()
    inspector = inspect(connection)

    try:
        tables = inspector.get_table_names(schema="data_prep")
    except Exception:
        tables = []

    if not tables:
        query = text("""
            SELECT c.relname 
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'data_prep' 
              AND c.relkind = 'r'
            ORDER BY c.relname ASC;
        """)
        result = db.execute(query).fetchall()
        tables = [row[0] for row in result]

    # 1. LỌC BỎ các bảng hệ thống / registry
    EXCLUDED_TABLES = {"data_version_registry"}
    tables = [t for t in tables if t.lower() not in EXCLUDED_TABLES]

    tables = sorted(tables)
    if "v0_raw" not in tables:
        tables.insert(0, "v0_raw")

    return tables


def get_dataframe_from_version(
    db: Session, version_id: str, id_attraction: str = None
) -> pd.DataFrame:
    """Đọc dữ liệu từ Database và lọc chính xác biến theo id_attraction mà không làm mất biến."""
    conn = db.connection()
    target_attr = id_attraction.upper() if id_attraction else "ALL"

    if version_id == "v0_raw":
        query_sql = get_dynamic_gathering_query(db.bind, id_attraction=target_attr)
        df = pd.read_sql(text(query_sql), conn)
        return df

    schema_name = "data_prep"
    query_str = f'SELECT * FROM "{schema_name}"."{version_id}"'

    cols_query = text("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_schema = :s AND table_name = :t AND column_name = 'id_attraction'
    """)
    has_id_attr_col = conn.execute(
        cols_query, {"s": schema_name, "t": version_id}
    ).fetchone()

    if target_attr != "ALL" and has_id_attr_col:
        query_str += " WHERE id_attraction = :attr_id"
        df = pd.read_sql(text(query_str), conn, params={"attr_id": target_attr})
    else:
        df = pd.read_sql(text(query_str), conn)

    return df


# 🌟 HÀM MỚI: Tính tỷ lệ phân bổ của 1 Attraction qua tất cả các Version
def compute_version_distribution_for_attraction(
    db: Session, id_attraction: str
) -> List[Dict[str, Any]]:
    all_versions = get_data_prep_versions_service(db)
    attr_upper = id_attraction.upper()

    # Lọc danh sách phiên bản liên quan đến attraction này
    relevant_versions = [
        v for v in all_versions if v == "v0_raw" or attr_upper in v.upper()
    ]

    version_counts = {}
    total_across_versions = 0

    for v in relevant_versions:
        try:
            df_v = get_dataframe_from_version(
                db, version_id=v, id_attraction=id_attraction
            )
            count = len(df_v)
            version_counts[v] = count
            total_across_versions += count
        except Exception:
            version_counts[v] = 0

    repartition = []
    for v, count in version_counts.items():
        pct = (
            round((count / total_across_versions * 100), 2)
            if total_across_versions > 0
            else 0.0
        )
        repartition.append({"name": v, "total_lignes": count, "pourcentage": pct})

    return repartition


def generate_profilage_service(
    db: Session, version_id: str, id_attraction: str = None, nb_lignes_apercu: int = 100
) -> Dict[str, Any]:
    """Tính toán toàn bộ thống kê cho một phiên bản."""
    df = get_dataframe_from_version(db, version_id, id_attraction)

    total_lignes = len(df)
    total_colonnes = len(df.columns)

    total_missing = int(df.isnull().sum().sum())
    total_cells = total_lignes * total_colonnes if total_colonnes > 0 else 1
    pct_missing = round((total_missing / total_cells) * 100, 2)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    total_outliers = 0
    total_negatives = 0

    for col in numeric_cols:
        col_data = df[col].dropna()
        total_negatives += int((col_data < 0).sum())
        if len(col_data) > 0:
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            outliers = col_data[
                (col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))
            ]
            total_outliers += int(len(outliers))

    pct_outliers = round((total_outliers / total_cells) * 100, 2)
    pct_negatives = round((total_negatives / total_cells) * 100, 2)

    total_duplicates = int(df.duplicated().sum())
    pct_duplicates = (
        round((total_duplicates / total_lignes) * 100, 2) if total_lignes > 0 else 0.0
    )

    mem_usage = df.memory_usage(deep=True, index=False)
    total_mem = mem_usage.sum() if mem_usage.sum() > 0 else 1
    memoire_list = [
        {
            "nom_colonne": col,
            "octets": int(mem_usage[col]),
            "pourcentage": round((mem_usage[col] / total_mem) * 100, 2),
        }
        for col in df.columns
    ]

    comparaison_colonnes = []
    for col in df.columns:
        col_missing = int(df[col].isnull().sum())
        pct_col_missing = (
            round((col_missing / total_lignes) * 100, 2) if total_lignes > 0 else 0.0
        )

        is_num = col in numeric_cols
        col_outliers = 0
        if is_num and total_lignes > 0:
            col_data = df[col].dropna()
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            col_outliers = int(
                len(
                    col_data[
                        (col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))
                    ]
                )
            )

        pct_col_outliers = (
            round((col_outliers / total_lignes) * 100, 2) if total_lignes > 0 else 0.0
        )

        comparaison_colonnes.append(
            {
                "nom_colonne": col,
                "type_donnees": str(df[col].dtype),
                "valeurs_manquantes": col_missing,
                "pourcentage_manquants": pct_col_missing,
                "valeurs_aberrantes": col_outliers,
                "pourcentage_aberrants": pct_col_outliers,
                "moyenne": round(float(df[col].mean()), 2)
                if is_num and not np.isnan(df[col].mean())
                else None,
                "ecart_type": round(float(df[col].std()), 2)
                if is_num and not np.isnan(df[col].std())
                else None,
                "val_min": round(float(df[col].min()), 2)
                if is_num and not np.isnan(df[col].min())
                else None,
                "mediane": round(float(df[col].median()), 2)
                if is_num and not np.isnan(df[col].median())
                else None,
                "val_max": round(float(df[col].max()), 2)
                if is_num and not np.isnan(df[col].max())
                else None,
            }
        )

    liste_attractions = []
    attraction_distribution = []
    if "id_attraction" in df.columns:
        liste_attractions = [
            str(a) for a in df["id_attraction"].dropna().unique().tolist()
        ]
        counts = df["id_attraction"].value_counts()
        for attr_code, count in counts.items():
            attraction_distribution.append(
                {
                    "id_attraction": str(attr_code),
                    "total_lignes": int(count),
                    "pourcentage": round((count / total_lignes) * 100, 2)
                    if total_lignes > 0
                    else 0,
                }
            )

    # 🌟 NẾU CHỌN 1 ATTRACTION CỤ THỂ -> TÍNH PHÂN BỔ QUA CÁC VERSION
    repartition_par_version = None
    if id_attraction and id_attraction.upper() != "ALL":
        repartition_par_version = compute_version_distribution_for_attraction(
            db, id_attraction
        )

    date_debut, date_fin = "N/A", "N/A"
    datetime_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns
    if len(datetime_cols) > 0:
        first_dt_col = datetime_cols[0]
        date_debut = str(df[first_dt_col].min())
        date_fin = str(df[first_dt_col].max())
    elif "horodatage" in df.columns:
        date_debut = str(df["horodatage"].min())
        date_fin = str(df["horodatage"].max())

    apercu = df.head(nb_lignes_apercu).replace({np.nan: None}).to_dict(orient="records")

    return {
        "statut": "succes",
        "version_id": version_id,
        "id_attraction_filtre": id_attraction or "ALL",
        "total_lignes": total_lignes,
        "total_colonnes": total_colonnes,
        "nb_valeurs_manquantes": total_missing,
        "pourcentage_manquants": pct_missing,
        "nb_outliers": total_outliers,
        "pourcentage_outliers": pct_outliers,
        "nb_valeurs_negatives": total_negatives,
        "pourcentage_negatifs": pct_negatives,
        "nb_doublons": total_duplicates,
        "pourcentage_dupliques": pct_duplicates,
        "date_debut": date_debut,
        "date_fin": date_fin,
        "liste_attractions": liste_attractions,
        "repartition_par_attraction": attraction_distribution,
        "repartition_par_version": repartition_par_version,  # 🌟 THÊM TRƯỜNG NÀY VÀO RESPONSE
        "utilisation_memoire": memoire_list,
        "comparaison_colonnes": comparaison_colonnes,
        "apercu_donnees": apercu,
    }


def compare_versions_service(
    db: Session, v1: str, v2: str, id_attraction: str = None
) -> Dict[str, Any]:
    """So sánh chênh lệch giữa 2 phiên bản trong schema data_prep."""
    p1 = generate_profilage_service(db, v1, id_attraction, nb_lignes_apercu=1)
    p2 = generate_profilage_service(db, v2, id_attraction, nb_lignes_apercu=1)

    delta_global = {
        "diff_total_lignes": p2["total_lignes"] - p1["total_lignes"],
        "diff_total_colonnes": p2["total_colonnes"] - p1["total_colonnes"],
        "diff_nb_valeurs_manquantes": p2["nb_valeurs_manquantes"]
        - p1["nb_valeurs_manquantes"],
        "diff_pourcentage_manquants": round(
            p2["pourcentage_manquants"] - p1["pourcentage_manquants"], 2
        ),
        "diff_nb_outliers": p2["nb_outliers"] - p1["nb_outliers"],
        "diff_pourcentage_outliers": round(
            p2["pourcentage_outliers"] - p1["pourcentage_outliers"], 2
        ),
    }

    cols_v1 = {c["nom_colonne"]: c for c in p1["comparaison_colonnes"]}
    cols_v2 = {c["nom_colonne"]: c for c in p2["comparaison_colonnes"]}

    all_cols = set(cols_v1.keys()).union(set(cols_v2.keys()))
    delta_colonnes = []

    for col in all_cols:
        c1 = cols_v1.get(col, {})
        c2 = cols_v2.get(col, {})

        m1 = c1.get("moyenne")
        m2 = c2.get("moyenne")
        diff_moy = round(m2 - m1, 2) if m1 is not None and m2 is not None else None

        med1 = c1.get("mediane")
        med2 = c2.get("mediane")
        diff_med = (
            round(med2 - med1, 2) if med1 is not None and med2 is not None else None
        )

        delta_colonnes.append(
            {
                "nom_colonne": col,
                "type_v1": c1.get("type_donnees", "N/A"),
                "type_v2": c2.get("type_donnees", "N/A"),
                "diff_valeurs_manquantes": c2.get("valeurs_manquantes", 0)
                - c1.get("valeurs_manquantes", 0),
                "diff_valeurs_aberrantes": c2.get("valeurs_aberrantes", 0)
                - c1.get("valeurs_aberrantes", 0),
                "diff_moyenne": diff_moy,
                "diff_mediane": diff_med,
            }
        )

    return {
        "statut": "succes",
        "v1_version_id": v1,
        "v2_version_id": v2,
        "id_attraction": id_attraction or "ALL",
        "delta_global": delta_global,
        "delta_colonnes": delta_colonnes,
    }
