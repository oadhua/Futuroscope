import traceback
import pandas as pd
from typing import Dict, List, Optional
from sqlalchemy import text, bindparam
from sqlalchemy.engine import Engine

from app.modules.data_pipeline.utils import (
    load_file_safely,
    create_temps_id,
    prepare_energy_fact,
)


def process_and_store_static_data(
    surface_bytes: bytes,
    cadence_bytes: bytes,
    engine: Engine,
    target_attractions: List[str] = ["H03", "H07"],
) -> Dict[str, int]:
    """ETL Nhóm Tĩnh: Đọc surface.xlsx và cadence.xlsx để nạp vào dim_attraction & dim_cadence."""
    df_surface = load_file_safely(surface_bytes, "surface.xlsx")
    df_cadence = load_file_safely(cadence_bytes, "cadence.xlsx")

    df_cadence.columns = (
        df_cadence.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )

    if "id_attraction" in df_cadence.columns:
        df_cadence["id_attraction"] = (
            df_cadence["id_attraction"].astype(str).str.strip()
        )
    if "id_attraction" in df_surface.columns:
        df_surface["id_attraction"] = (
            df_surface["id_attraction"].astype(str).str.strip()
        )

    df_cadence_merged = df_cadence.merge(df_surface, on="id_attraction", how="left")
    df_cadence_merged = df_cadence_merged[
        df_cadence_merged["id_attraction"].isin(target_attractions)
    ].copy()

    rename_dict = {
        "surface": "surface",
        "capacite salle": "capacite_salle",
        "capacite file d'attente": "capacite_file_attente",
        "capacite pre-salle": "capacite_pre_salle",
        "duree_longue": "duree_longue",
        "duree_courte": "duree_courte",
        "cycle de fonc d'une duree max_vl": "cycle_max_vl",
        "duty_cycle_max_vl": "duty_cycle_max_vl",
        "cycle de fonc d'une duree max_vc": "cycle_max_vc",
        "duty_cycle_max_vc": "duty_cycle_max_vc",
    }
    df_cadence_to_save = df_cadence_merged.rename(columns=rename_dict)

    expected_cols = ["id_attraction"] + list(rename_dict.values())
    df_cadence_to_save = df_cadence_to_save[
        [c for c in expected_cols if c in df_cadence_to_save.columns]
    ]

    df_attraction = pd.DataFrame(
        {
            "id_attraction": target_attractions,
            "name": [f"Attraction {attr}" for attr in target_attractions],
        }
    )

    with engine.begin() as conn:
        conn.execute(
            text("TRUNCATE TABLE dim_cadence, dim_attraction RESTART IDENTITY CASCADE;")
        )
        df_attraction.to_sql(
            "dim_attraction", con=conn, if_exists="append", index=False
        )
        df_cadence_to_save.to_sql(
            "dim_cadence", con=conn, if_exists="append", index=False
        )

    return {
        "dim_attraction": len(df_attraction),
        "dim_cadence": len(df_cadence_to_save),
    }


def process_and_store_dynamic_data(
    visitor_bytes: bytes,
    etat_bytes: bytes,
    weather_bytes: bytes,
    horaire_bytes: bytes,
    elec_bytes: bytes,
    ec_bytes: bytes,
    engine: Engine,
    target_attractions: List[str] = ["H03", "H07"],
) -> Dict[str, int]:
    """ETL Nhóm Động: Đọc 6 file biến đổi theo thời gian, nạp vào Fact Tables & Dynamic Dims."""
    df_visitor = load_file_safely(visitor_bytes, "visitor.csv")
    df_etat = load_file_safely(etat_bytes, "etat.csv")
    df_weather = load_file_safely(weather_bytes, "weather.csv")
    df_horaire = load_file_safely(horaire_bytes, "horaire.csv")
    df_elec_raw = load_file_safely(elec_bytes, "electricite.xlsx")
    df_ec_raw = load_file_safely(ec_bytes, "thermal.xlsx")

    target_attractions = [str(x).strip() for x in target_attractions]

    if "id_attraction" in df_visitor.columns:
        df_visitor["id_attraction"] = (
            df_visitor["id_attraction"].astype(str).str.strip()
        )
        df_visitor = df_visitor[df_visitor["id_attraction"].isin(target_attractions)]
    if "id_attraction" in df_etat.columns:
        df_etat["id_attraction"] = df_etat["id_attraction"].astype(str).str.strip()
        df_etat = df_etat[df_etat["id_attraction"].isin(target_attractions)]

    all_raw_dates = pd.concat(
        [
            df_weather["datetime"]
            if "datetime" in df_weather.columns
            else pd.Series(dtype="datetime64[ns]"),
            df_visitor["datetime"]
            if "datetime" in df_visitor.columns
            else pd.Series(dtype="datetime64[ns]"),
            df_elec_raw["datetime"]
            if "datetime" in df_elec_raw.columns
            else pd.Series(dtype="datetime64[ns]"),
            df_ec_raw["datetime"]
            if "datetime" in df_ec_raw.columns
            else pd.Series(dtype="datetime64[ns]"),
        ]
    ).dropna()

    if not all_raw_dates.empty:
        min_date = all_raw_dates.min().floor("h")
        max_date = all_raw_dates.max().ceil("h")
        full_time_range = pd.date_range(start=min_date, end=max_date, freq="h")
    else:
        full_time_range = pd.Series(dtype="datetime64[ns]")

    df_temps = pd.DataFrame({"datetime": full_time_range})
    df_temps = create_temps_id(df_temps)
    df_temps = df_temps.drop_duplicates(subset=["temps_id"]).reset_index(drop=True)

    df_temps["date"] = df_temps["datetime"].dt.date
    df_temps["annee"] = df_temps["datetime"].dt.year
    df_temps["mois"] = df_temps["datetime"].dt.month
    df_temps["semaine"] = df_temps["datetime"].dt.week
    df_temps["jour"] = df_temps["datetime"].dt.day
    df_temps["heure"] = df_temps["datetime"].dt.hour
    df_temps["jour_semaine"] = df_temps["datetime"].dt.dayofweek + 1
    df_temps["is_weekend"] = df_temps["jour_semaine"].isin([6, 7]).astype(int)

    if "jf" in df_horaire.columns and "datetime" in df_horaire.columns:
        df_horaire["date"] = pd.to_datetime(df_horaire["datetime"]).dt.date
        jf_map = df_horaire.set_index("date")["jf"].to_dict()
        df_temps["jf"] = df_temps["date"].map(jf_map).fillna(0).astype(int)
    else:
        df_temps["jf"] = 0

    for col in ["h_ouv", "h_ferm", "type_frequentation"]:
        if col in df_horaire.columns:
            df_horaire[col] = (
                df_horaire[col]
                .astype(str)
                .str.strip()
                .str[:20]
                .replace({"nan": None, "None": None, "": None})
            )

    horaire_cols = [
        "datetime",
        "date",
        "is_open",
        "h_ouv",
        "h_ferm",
        "frequentation",
        "type_frequentation",
        "jf",
    ]
    df_horaire_to_save = df_horaire.drop_duplicates(subset=["datetime"])[
        [c for c in horaire_cols if c in df_horaire.columns]
    ]

    df_weather = create_temps_id(df_weather)
    weather_cols = [
        "temps_id",
        "datetime",
        "date_key",
        "heure",
        "temperature",
        "humidite",
        "rayonnement_solaire",
        "day_degree_cold",
        "day_degree_hot",
        "temp_max",
        "temp_min",
        "temp_moy",
        "humidite_max",
        "humidite_min",
        "humidite_moy",
    ]
    df_weather_to_save = df_weather.drop_duplicates(subset=["temps_id"])[
        [c for c in weather_cols if c in df_weather.columns]
    ]

    df_visitor = create_temps_id(df_visitor)
    df_etat = create_temps_id(df_etat)

    df_fact_raw = pd.merge(
        df_visitor,
        df_etat,
        on=["temps_id", "id_attraction"],
        how="outer",
        suffixes=("", "_dup"),
    )
    if "datetime_dup" in df_fact_raw.columns:
        df_fact_raw = df_fact_raw.drop(columns=["datetime_dup"])

    fact_cols = [
        "temps_id",
        "id_attraction",
        "datetime",
        "visitor_count",
        "ouvert",
        "interrompu",
        "operation",
    ]
    df_fact_to_save = df_fact_raw[
        [c for c in fact_cols if c in df_fact_raw.columns]
    ].dropna(subset=["temps_id", "id_attraction"])

    for col in ["visitor_count", "ouvert", "interrompu", "operation"]:
        if col in df_fact_to_save.columns:
            df_fact_to_save[col] = pd.to_numeric(
                df_fact_to_save[col], errors="coerce"
            ).fillna(0)

    df_elec_to_save = prepare_energy_fact(df_elec_raw, target_attractions)
    df_ec_to_save = prepare_energy_fact(df_ec_raw, target_attractions)

    df_temps["temps_id"] = df_temps["temps_id"].astype("int64")

    for df_target in [
        df_weather_to_save,
        df_fact_to_save,
        df_elec_to_save,
        df_ec_to_save,
    ]:
        if "temps_id" in df_target.columns:
            df_target["temps_id"] = pd.to_numeric(
                df_target["temps_id"], errors="coerce"
            )
            df_target.dropna(subset=["temps_id"], inplace=True)
            df_target["temps_id"] = df_target["temps_id"].astype("int64")

    valid_temps_ids = set(df_temps["temps_id"])
    df_weather_to_save = df_weather_to_save[
        df_weather_to_save["temps_id"].isin(valid_temps_ids)
    ]
    df_fact_to_save = df_fact_to_save[df_fact_to_save["temps_id"].isin(valid_temps_ids)]
    df_elec_to_save = df_elec_to_save[df_elec_to_save["temps_id"].isin(valid_temps_ids)]
    df_ec_to_save = df_ec_to_save[df_ec_to_save["temps_id"].isin(valid_temps_ids)]

    df_fact_to_save = df_fact_to_save.drop_duplicates(
        subset=["temps_id", "id_attraction"]
    )
    df_elec_to_save = df_elec_to_save.drop_duplicates(
        subset=["temps_id", "id_attraction", "metric_name"]
    )
    df_ec_to_save = df_ec_to_save.drop_duplicates(
        subset=["temps_id", "id_attraction", "metric_name"]
    )

    for df_fact in [df_fact_to_save, df_elec_to_save, df_ec_to_save]:
        if "fact_id" in df_fact.columns:
            df_fact.drop(columns=["fact_id"], inplace=True)

    try:
        with engine.begin() as conn:
            for attr_id in target_attractions:
                conn.execute(
                    text("""
                    INSERT INTO dim_attraction (id_attraction, name)
                    VALUES (:attr_id, :nom)
                    ON CONFLICT (id_attraction) DO NOTHING;
                """),
                    {"attr_id": attr_id, "nom": f"Attraction {attr_id}"},
                )

            conn.execute(
                text("""
                TRUNCATE TABLE 
                    fact_attraction_hourly,
                    fact_elec_hourly,
                    fact_ec_hourly,
                    dim_weather, 
                    dim_horaire, 
                    dim_temps
                RESTART IDENTITY CASCADE;
            """)
            )

            df_temps.to_sql(
                "dim_temps",
                con=conn,
                if_exists="append",
                index=False,
                chunksize=5000,
                method="multi",
            )
            df_weather_to_save.to_sql(
                "dim_weather",
                con=conn,
                if_exists="append",
                index=False,
                chunksize=5000,
                method="multi",
            )
            df_horaire_to_save.to_sql(
                "dim_horaire",
                con=conn,
                if_exists="append",
                index=False,
                chunksize=5000,
                method="multi",
            )

            df_fact_to_save.to_sql(
                "fact_attraction_hourly",
                con=conn,
                if_exists="append",
                index=False,
                chunksize=5000,
                method="multi",
            )
            df_elec_to_save.to_sql(
                "fact_elec_hourly",
                con=conn,
                if_exists="append",
                index=False,
                chunksize=5000,
                method="multi",
            )
            df_ec_to_save.to_sql(
                "fact_ec_hourly",
                con=conn,
                if_exists="append",
                index=False,
                chunksize=5000,
                method="multi",
            )

        print("✅ Nạp thành công toàn bộ dữ liệu vào PostgreSQL!")

    except Exception as e:
        print(f"\n❌ LỖI TRONG QUÁ TRÌNH NẠP DATABASE: {e}")
        traceback.print_exc()
        raise e

    return {
        "dim_temps": len(df_temps),
        "fact_attraction": len(df_fact_to_save),
        "fact_elec": len(df_elec_to_save),
        "fact_ec": len(df_ec_to_save),
    }


def generate_all_fact_shells_in_db(
    start_date: str,
    end_date: str,
    engine: Engine,
    target_attractions: Optional[List[str]] = None,
) -> Dict[str, int]:
    """ETL Shell Generator: Tạo khung thời gian trống (Fact Shells) tự động quét và thích ứng với biến thể metric_name."""
    filter_attraction_sql = ""
    if target_attractions:
        filter_attraction_sql = "AND a.id_attraction IN :attractions"

    sql_attraction = f"""
        INSERT INTO fact_attraction_hourly (
            temps_id, id_attraction, datetime, visitor_count, ouvert, interrompu, operation
        )
        SELECT t.temps_id, a.id_attraction, t.datetime, NULL, NULL, NULL, NULL
        FROM dim_temps t
        CROSS JOIN dim_attraction a
        LEFT JOIN fact_attraction_hourly f 
            ON t.temps_id = f.temps_id AND a.id_attraction = f.id_attraction
        WHERE t.datetime >= CAST(:start_date AS TIMESTAMP) 
          AND t.datetime <= (CAST(:end_date AS DATE) + INTERVAL '1 day' - INTERVAL '1 second')
          {filter_attraction_sql}
          AND f.temps_id IS NULL;
    """

    sql_elec = f"""
        WITH existing_metrics AS (
            SELECT DISTINCT id_attraction, metric_name 
            FROM fact_elec_hourly
            WHERE metric_name IS NOT NULL
        )
        INSERT INTO fact_elec_hourly (temps_id, id_attraction, datetime, metric_name, value)
        SELECT t.temps_id, m.id_attraction, t.datetime, m.metric_name, NULL
        FROM dim_temps t
        CROSS JOIN existing_metrics m
        LEFT JOIN fact_elec_hourly f 
            ON t.temps_id = f.temps_id 
           AND m.id_attraction = f.id_attraction 
           AND m.metric_name = f.metric_name
        WHERE t.datetime >= CAST(:start_date AS TIMESTAMP) 
          AND t.datetime <= (CAST(:end_date AS DATE) + INTERVAL '1 day' - INTERVAL '1 second')
          {"AND m.id_attraction IN :attractions" if target_attractions else ""}
          AND f.temps_id IS NULL;
    """

    sql_ec = f"""
        WITH existing_metrics AS (
            SELECT DISTINCT id_attraction, metric_name 
            FROM fact_ec_hourly
            WHERE metric_name IS NOT NULL
        )
        INSERT INTO fact_ec_hourly (temps_id, id_attraction, datetime, metric_name, value)
        SELECT t.temps_id, m.id_attraction, t.datetime, m.metric_name, NULL
        FROM dim_temps t
        CROSS JOIN existing_metrics m
        LEFT JOIN fact_ec_hourly f 
            ON t.temps_id = f.temps_id 
           AND m.id_attraction = f.id_attraction 
           AND m.metric_name = f.metric_name
        WHERE t.datetime >= CAST(:start_date AS TIMESTAMP) 
          AND t.datetime <= (CAST(:end_date AS DATE) + INTERVAL '1 day' - INTERVAL '1 second')
          {"AND m.id_attraction IN :attractions" if target_attractions else ""}
          AND f.temps_id IS NULL;
    """

    sql_weather = """
        INSERT INTO dim_weather (temps_id, datetime, temperature, humidite, rayonnement_solaire,day_degree_cold,
        day_degree_hot, temp_max, temp_min, temp_moy, humidite_max, humidite_min, humidite_moy)
        SELECT t.temps_id, t.datetime, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
        FROM dim_temps t
        LEFT JOIN dim_weather w ON t.temps_id = w.temps_id
        WHERE t.datetime >= CAST(:start_date AS TIMESTAMP) 
          AND t.datetime <= (CAST(:end_date AS DATE) + INTERVAL '1 day' - INTERVAL '1 second')
          AND w.temps_id IS NULL;
    """

    params = {"start_date": start_date, "end_date": end_date}
    if target_attractions:
        params["attractions"] = target_attractions

    def prepare_stmt(query_str: str):
        stmt = text(query_str)
        if target_attractions:
            stmt = stmt.bindparams(bindparam("attractions", expanding=True))
        return stmt

    with engine.begin() as conn:
        res_attr = conn.execute(prepare_stmt(sql_attraction), params)
        res_elec = conn.execute(prepare_stmt(sql_elec), params)
        res_ec = conn.execute(prepare_stmt(sql_ec), params)
        res_weather = conn.execute(
            text(sql_weather), {"start_date": start_date, "end_date": end_date}
        )

    return {
        "fact_attraction_created": res_attr.rowcount,
        "fact_elec_created": res_elec.rowcount,
        "fact_ec_created": res_ec.rowcount,
        "dim_weather_created": res_weather.rowcount,
    }