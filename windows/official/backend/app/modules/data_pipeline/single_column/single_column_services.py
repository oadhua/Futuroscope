from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.modules.data_pipeline.queries import get_dynamic_gathering_query
from app.modules.data_pipeline.utils import clean_nan_and_inf


def get_sorted_versions(db: Session) -> List[str]:
    """Lấy danh sách các bảng phiên bản trong schema data_prep được sắp xếp theo thời gian tạo (c.oid ASC)."""
    db.commit()
    query = text("""
        SELECT c.relname
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'data_prep' 
          AND c.relkind = 'r'
          AND c.relname NOT IN ('data_version_registry', 'ml_model_registry')
        ORDER BY c.oid ASC;
    """)
    try:
        result = db.execute(query).fetchall()
        tables = [row[0] for row in result]
    except Exception:
        db.rollback()
        tables = []

    if "v0_raw" in tables:
        tables.remove("v0_raw")
        tables.insert(0, "v0_raw")
    elif not tables:
        tables = ["v0_raw"]

    return tables


def get_dataframe_from_version_local(
    db: Session, version_id: str = "v0_raw", id_attraction: Optional[str] = None
) -> pd.DataFrame:
    """Tự đọc dữ liệu từ Database theo version và id_attraction (Tách biệt hoàn toàn với data_analysis)."""
    conn = db.connection()
    target_attr = id_attraction.upper() if id_attraction else "ALL"

    # 1. Trường hợp phiên bản thô v0_raw: gọi câu query Pivot động
    if not version_id or version_id == "v0_raw":
        query_sql = get_dynamic_gathering_query(
            db.bind, id_attraction=target_attr, version="v0_raw"
        )
        return pd.read_sql(text(query_sql), conn)

    # 2. Trường hợp các version khác trong schema data_prep
    schema_name = "data_prep"
    query_str = f'SELECT * FROM "{schema_name}"."{version_id}"'

    # Kiểm tra xem bảng trong data_prep có cột id_attraction hay không
    cols_query = text("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_schema = :s AND table_name = :t AND column_name = 'id_attraction'
    """)
    has_id_attr_col = conn.execute(
        cols_query, {"s": schema_name, "t": version_id}
    ).fetchone()

    # Lọc theo id_attraction nếu có cột id_attraction và target != 'ALL'
    if target_attr != "ALL" and has_id_attr_col:
        query_str += " WHERE id_attraction = :attr_id"
        df = pd.read_sql(text(query_str), conn, params={"attr_id": target_attr})
    else:
        df = pd.read_sql(text(query_str), conn)

    return df


def _cast_val(v: Any, is_int: bool = False) -> Optional[Any]:
    if pd.isnull(v) or np.isnan(v):
        return None
    return int(round(float(v))) if is_int else round(float(v), 2)


def _check_is_integer_col(col_name: str, serie_valide: pd.Series) -> bool:
    if serie_valide.empty:
        return False
    if col_name == "visitor_count":
        return True
    return bool((serie_valide % 1 == 0).all())


def analyser_profil_colonne_seule(
    df: pd.DataFrame,
    col_name: str,
    id_attraction: Optional[str] = None,
    version: Optional[str] = "v0_raw",
) -> Dict[str, Any]:
    """Phân tích chi tiết 1 cột (Single Column Profiling) hỗ trợ Versioning & Attraction Filter."""
    if df.empty or col_name not in df.columns:
        return {
            "erreur": f"Cột '{col_name}' không tồn tại trong phiên bản '{version}'."
        }

    df_target = df.copy()

    if (
        id_attraction
        and str(id_attraction).upper() != "ALL"
        and "id_attraction" in df_target.columns
    ):
        df_target = df_target[
            df_target["id_attraction"] == str(id_attraction).strip()
        ].copy()

    total_rows = len(df_target)
    if total_rows == 0:
        return {
            "erreur": f"Không có dữ liệu cho id_attraction = '{id_attraction}' và version = '{version}'"
        }

    df_target[col_name] = pd.to_numeric(df_target[col_name], errors="coerce")
    serie = df_target[col_name]
    serie_valide = serie.dropna()

    is_numeric = pd.api.types.is_numeric_dtype(serie)
    is_integer_col = (
        _check_is_integer_col(col_name, serie_valide) if is_numeric else False
    )
    dtype_str = "int64" if is_integer_col else str(serie.dtype)

    nb_missing = int(serie.isnull().sum())
    pct_missing = round((nb_missing / total_rows) * 100, 2)
    nb_unique = int(serie.nunique(dropna=True))

    nb_zeros, pct_zeros, nb_negatifs, pct_negatifs, nb_outliers, pct_outliers = (
        0, 0.0, 0, 0.0, 0, 0.0,
    )
    box_plot_data, histogram_data = None, None
    line_plots_data = {
        "par_heure": [],
        "par_jour": [],
        "par_mois": [],
        "par_annee": [],
    }

    if is_numeric and not serie_valide.empty:
        nb_zeros = int((serie_valide == 0).sum())
        pct_zeros = round((nb_zeros / total_rows) * 100, 2)
        nb_negatifs = int((serie_valide < 0).sum())
        pct_negatifs = round((nb_negatifs / total_rows) * 100, 2)

        q1 = float(serie_valide.quantile(0.25))
        q3 = float(serie_valide.quantile(0.75))
        iqr = q3 - q1
        borne_inf, borne_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        outliers_mask = (serie_valide < borne_inf) | (serie_valide > borne_sup)
        nb_outliers = int(outliers_mask.sum())
        pct_outliers = round((nb_outliers / total_rows) * 100, 2)

        box_plot_data = {
            "min": _cast_val(serie_valide.min(), is_integer_col),
            "q25": _cast_val(q1, is_integer_col),
            "mediane": _cast_val(serie_valide.median(), is_integer_col),
            "q75": _cast_val(q3, is_integer_col),
            "max": _cast_val(serie_valide.max(), is_integer_col),
            "borne_inf": _cast_val(borne_inf, is_integer_col),
            "borne_sup": _cast_val(borne_sup, is_integer_col),
            "outliers_samples": [
                _cast_val(x, is_integer_col)
                for x in serie_valide[outliers_mask].head(50).tolist()
            ],
        }

        counts, bin_edges = np.histogram(serie_valide, bins=20)
        histogram_data = []
        for i in range(len(counts)):
            if is_integer_col:
                low = int(np.round(bin_edges[i]))
                high = int(np.round(bin_edges[i + 1]))
                bin_str = f"{low} - {high}"
            else:
                bin_str = f"{round(bin_edges[i], 1)} - {round(bin_edges[i + 1], 1)}"

            histogram_data.append(
                {
                    "bin_range": bin_str,
                    "count": int(counts[i]),
                }
            )

        if "datetime" in df_target.columns:
            df_target["datetime"] = pd.to_datetime(df_target["datetime"])

            col_lower = col_name.lower()
            is_sum_metric = (
                col_name == "visitor_count"
                or "elec" in col_lower
                or "ec_" in col_lower
                or "puissance" in col_lower
                or "chaleur" in col_lower
                or "kwh" in col_lower
                or "m3" in col_lower
                or "consommation" in col_lower
            )
            agg_func = "sum" if is_sum_metric else "mean"
            is_result_int = is_integer_col and agg_func == "sum"

            heure_agg = (
                df_target.groupby("datetime")[col_name]
                .agg(agg_func)
                .reset_index()
                .sort_values("datetime")
            )
            line_plots_data["par_heure"] = [
                {
                    "heure": (
                        r["datetime"].strftime("%Y-%m-%d %H:%M")
                        if pd.notnull(r["datetime"])
                        else ""
                    ),
                    "valeur_moyenne": _cast_val(r[col_name], is_result_int),
                }
                for _, r in heure_agg.iterrows()
            ]

            df_target["date_only"] = df_target["datetime"].dt.date.astype(str)
            jour_agg = (
                df_target.groupby("date_only")[col_name]
                .agg(agg_func)
                .reset_index()
                .sort_values("date_only")
            )
            line_plots_data["par_jour"] = [
                {
                    "date": r["date_only"],
                    "valeur_moyenne": _cast_val(r[col_name], is_result_int),
                }
                for _, r in jour_agg.iterrows()
            ]

            df_target["year_month_str"] = df_target["datetime"].dt.strftime("%Y-%m")
            mois_agg = (
                df_target.groupby("year_month_str")[col_name]
                .agg(agg_func)
                .reset_index()
                .sort_values("year_month_str")
            )
            line_plots_data["par_mois"] = [
                {
                    "mois": str(r["year_month_str"]),
                    "valeur_moyenne": _cast_val(r[col_name], is_result_int),
                }
                for _, r in mois_agg.iterrows()
            ]

            df_target["year_str"] = df_target["datetime"].dt.strftime("%Y")
            annee_agg = (
                df_target.groupby("year_str")[col_name]
                .agg(agg_func)
                .reset_index()
                .sort_values("year_str")
            )
            line_plots_data["par_annee"] = [
                {
                    "annee": str(r["year_str"]),
                    "valeur_moyenne": _cast_val(r[col_name], is_result_int),
                }
                for _, r in annee_agg.iterrows()
            ]

    return {
        "nom_colonne": col_name,
        "id_attraction": id_attraction or "ALL",
        "selected_version": version or "v0_raw",
        "statistiques": {
            "type_donnees": dtype_str,
            "total_lignes": total_rows,
            "valeurs_manquantes": nb_missing,
            "pourcentage_manquants": pct_missing,
            "valeurs_uniques": nb_unique,
            "valeurs_zero": nb_zeros,
            "pourcentage_zeros": pct_zeros,
            "valeurs_negatives": nb_negatifs,
            "pourcentage_negatifs": pct_negatifs,
            "valeurs_aberrantes": nb_outliers,
            "pourcentage_outliers": pct_outliers,
        },
        "graphiques": {
            "box_plot": box_plot_data,
            "histogramme": histogram_data,
            "courbes_temporelles": line_plots_data,
        },
    }


def analyser_multi_versions(
    db: Session,
    col_name: str,
    versions: List[str],
    id_attraction: Optional[str] = "ALL"
) -> Dict[str, Any]:
    """Phân tích đồng thời một cột trên nhiều version khác nhau theo đúng thứ tự thời gian khởi tạo."""
    # 1. Lấy danh sách thứ tự version chuẩn từ PostgreSQL (sắp xếp theo thời gian tạo bảng c.oid)
    sorted_db_versions = get_sorted_versions(db)

    # 2. Sắp xếp danh sách versions đầu vào theo đúng thứ tự khởi tạo
    ordered_versions = [v for v in sorted_db_versions if v in versions]
    # Bổ sung các version nếu có trong danh sách truyền vào nhưng chưa tìm thấy ở DB query
    for v in versions:
        if v not in ordered_versions:
            ordered_versions.append(v)

    results = {}
    for ver in ordered_versions:
        try:
            df = get_dataframe_from_version_local(
                db=db, version_id=ver, id_attraction=id_attraction
            )
            if df.empty:
                continue
            
            res = analyser_profil_colonne_seule(
                df=df, col_name=col_name, id_attraction=id_attraction, version=ver
            )
            if "erreur" not in res:
                results[ver] = clean_nan_and_inf(res)
        except Exception as e:
            print(f"Erreur analyse version {ver}: {str(e)}")
            continue
            
    return results