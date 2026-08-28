from typing import Optional
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.modules.data_pipeline.utils import format_alias


def get_dynamic_gathering_query(
    engine: Engine,
    id_attraction: Optional[str] = None,
    version: Optional[str] = "v0_raw",  # Bổ sung tham số version
) -> str:
    """Tự động tạo câu SQL Pivot ghép id_attraction vào đầu tên cột năng lượng, có hỗ trợ versioning."""

    # Xác định tên bảng dữ liệu cơ sở theo version
    # Ví dụ: nếu version = "v0_raw" -> bảng gốc 'fact_attraction_hourly'
    # Nếu version dạng khác -> tự động map sang bảng tương ứng
    table_fact_attraction = "fact_attraction_hourly"
    table_fact_elec = "fact_elec_hourly"
    table_fact_ec = "fact_ec_hourly"

    if version and version != "v0_raw":
        # Có thể tùy chỉnh logic prefix/suffix bảng theo hệ thống của bạn tại đây
        table_fact_attraction = f"fact_attraction_hourly_{version}"

    has_filter = (
        id_attraction is not None
        and str(id_attraction).strip() != ""
        and str(id_attraction).upper() != "ALL"
    )

    clean_id = str(id_attraction).strip() if has_filter else None

    where_clause = (
        f"WHERE id_attraction = '{clean_id}' AND metric_name IS NOT NULL"
        if has_filter
        else "WHERE metric_name IS NOT NULL"
    )

    with engine.connect() as conn:
        try:
            df_elec = pd.read_sql(
                text(
                    f"SELECT DISTINCT id_attraction, metric_name FROM {table_fact_elec} {where_clause}"
                ),
                conn,
            )
            elec_pairs = df_elec.dropna().to_dict(orient="records")
        except Exception:
            elec_pairs = []

        try:
            df_ec = pd.read_sql(
                text(
                    f"SELECT DISTINCT id_attraction, metric_name FROM {table_fact_ec} {where_clause}"
                ),
                conn,
            )
            ec_pairs = df_ec.dropna().to_dict(orient="records")
        except Exception:
            ec_pairs = []

    elec_pivot_list = []
    elec_select_cols = []
    for item in elec_pairs:
        attr = item["id_attraction"]
        m = item["metric_name"]
        alias = format_alias(attr, m)
        elec_pivot_list.append(
            f"MAX(CASE WHEN id_attraction = '{attr}' AND metric_name = '{m}' THEN value END) AS \"{alias}\""
        )
        elec_select_cols.append(f'e."{alias}"')

    ec_pivot_list = []
    ec_select_cols = []
    for item in ec_pairs:
        attr = item["id_attraction"]
        m = item["metric_name"]
        alias = format_alias(attr, m)
        ec_pivot_list.append(
            f"MAX(CASE WHEN id_attraction = '{attr}' AND metric_name = '{m}' THEN value END) AS \"{alias}\""
        )
        ec_select_cols.append(f'c."{alias}"')

    elec_pivot_cols = (
        ",\n            ".join(elec_pivot_list)
        if elec_pivot_list
        else "NULL AS _no_elec"
    )
    ec_pivot_cols = (
        ",\n            ".join(ec_pivot_list) if ec_pivot_list else "NULL AS _no_ec"
    )

    energy_select_cols = elec_select_cols + ec_select_cols
    energy_select_str = ""
    if energy_select_cols:
        energy_select_str = ",\n        ".join(energy_select_cols) + ",\n        "

    cte_where = f"WHERE id_attraction = '{clean_id}'" if has_filter else ""
    main_where = f"WHERE f.id_attraction = '{clean_id}'" if has_filter else ""

    return f"""
    WITH elec_pivoted AS (
        SELECT 
            temps_id,
            id_attraction,
            {elec_pivot_cols}
        FROM {table_fact_elec}
        {cte_where}
        GROUP BY temps_id, id_attraction
    ),
    ec_pivoted AS (
        SELECT 
            temps_id,
            id_attraction,
            {ec_pivot_cols}
        FROM {table_fact_ec}
        {cte_where}
        GROUP BY temps_id, id_attraction
    )
    SELECT 
        f.datetime,
        f.id_attraction,
        f.visitor_count,
        f.ouvert,
        f.interrompu,
        f.operation,
        
        {energy_select_str}
        w.temperature,
        w.humidite,
        w.rayonnement_solaire,
        w.day_degree_cold,
        w.day_degree_hot,
        w.temp_max,
        w.temp_min,
        w.temp_moy,
        w.humidite_max,
        w.humidite_min,
        w.humidite_moy,

        h.jf,
        t.is_weekend,
        h.is_open,
        h.h_ouv,
        h.h_ferm,
        h.frequentation,
        h.type_frequentation,
        t.heure,
        t.jour,
        t.mois,
        t.annee
    FROM {table_fact_attraction} f
    LEFT JOIN dim_temps t ON f.temps_id = t.temps_id
    LEFT JOIN dim_weather w ON f.temps_id = w.temps_id
    LEFT JOIN dim_horaire h ON CAST(f.datetime AS DATE) = h.date
    LEFT JOIN elec_pivoted e ON f.temps_id = e.temps_id AND f.id_attraction = e.id_attraction
    LEFT JOIN ec_pivoted c ON f.temps_id = c.temps_id AND f.id_attraction = c.id_attraction
    {main_where}
    ORDER BY f.datetime ASC
    """
