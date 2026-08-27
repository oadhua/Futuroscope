import io
import re
import traceback
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import text, bindparam
from sqlalchemy.engine import Engine


# ==============================================================================
# 1. HELPER FUNCTIONS (BÓC TÁCH & CHUẨN HÓA DỮ LIỆU)
# ==============================================================================


def parse_energy_column(col_name: str) -> Tuple[Optional[str], str]:
    """Bóc tách tên cột năng lượng dài (điện/nhiệt)."""
    col_str = str(col_name).strip()
    match_id = re.match(r"^(H\d{2})", col_str)
    id_attraction = match_id.group(1) if match_id else None

    clean_name = re.sub(r"\[.*?\]", "", col_str).strip()
    clean_name = re.sub(r"^H\d{2}[_ ]*", "", clean_name).strip()
    clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", clean_name)
    clean_name = re.sub(r"_+", "_", clean_name).strip("_").lower()

    return id_attraction, clean_name


def load_file_safely(file_bytes: bytes, file_name: str = "") -> pd.DataFrame:
    """Tự động nhận diện định dạng (CSV hoặc Excel) và đọc dữ liệu an toàn."""
    is_excel = (
        file_name.endswith(".xlsx")
        or file_name.endswith(".xls")
        or file_bytes.startswith(b"PK")
    )

    if is_excel:
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        try:
            df = pd.read_csv(
                io.BytesIO(file_bytes), encoding="utf-8-sig", sep=None, engine="python"
            )
        except Exception:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig", sep=";")
            except Exception:
                df = pd.read_csv(
                    io.BytesIO(file_bytes), encoding="latin1", sep=None, engine="python"
                )

    df.columns = (
        df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    )

    date_col = None
    for col in df.columns:
        if col.lower() in [
            "datetime",
            "date",
            "timestamp",
            "date_heure",
            "horodate",
            "Date",
        ]:
            date_col = col
            break

    if date_col and date_col != "datetime":
        df = df.rename(columns={date_col: "datetime"})

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif "Date" in df.columns:
        df["datetime"] = pd.to_datetime(df["Date"])

    return df


def create_temps_id(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo khóa chính thời gian cấp giờ YYYYMMDDHH (ví dụ: 2026081210)."""
    if "datetime" in df.columns:
        df["temps_id"] = df["datetime"].dt.strftime("%Y%m%d%H").astype("int64")
    return df


def prepare_energy_fact(
    df_raw: pd.DataFrame, target_attractions: List[str]
) -> pd.DataFrame:
    df = df_raw.copy()
    if "temps_id" not in df.columns:
        df = create_temps_id(df)

    val_cols = [
        c
        for c in df.columns
        if c.lower() not in ["date", "hour", "datetime", "temps_id"]
    ]

    # Chuyển từ Wide sang Long trong 1 dòng duy nhất bằng pd.melt
    df_melted = df.melt(
        id_vars=["temps_id", "datetime"],
        value_vars=val_cols,
        var_name="raw_col",
        value_name="value",
    )

    # Parse id_attraction và clean_metric bằng vectorize/map
    parsed_meta = df_melted["raw_col"].apply(parse_energy_column)
    df_melted["id_attraction"] = [p[0] for p in parsed_meta]
    df_melted["metric_name"] = [p[1] for p in parsed_meta]

    # Lọc và ép kiểu
    mask = df_melted["id_attraction"].isin(target_attractions)
    final_df = df_melted[mask].copy()
    final_df["value"] = pd.to_numeric(final_df["value"], errors="coerce").fillna(0.0)

    cols = ["temps_id", "datetime", "id_attraction", "metric_name", "value"]
    return final_df[cols].drop_duplicates(
        subset=["temps_id", "id_attraction", "metric_name"]
    )


# ==============================================================================
# 2. ETL SERVICE - NHÓM DỮ LIỆU TĨNH (STATIC DATA)
# ==============================================================================
def process_and_store_static_data(
    surface_bytes: bytes,
    cadence_bytes: bytes,
    engine: Engine,
    target_attractions: List[str] = ["H03", "H07"],
) -> Dict[str, int]:
    """
    ETL Nhóm Tĩnh: Đọc surface.xlsx và cadence.xlsx để nạp vào dim_attraction & dim_cadence.
    Chỉ chạy 1 lần hoặc khi có sự thay đổi hạ tầng/thông số kỹ thuật công trình.
    """
    df_surface = load_file_safely(surface_bytes, "surface.xlsx")
    df_cadence = load_file_safely(cadence_bytes, "cadence.xlsx")

    # Clean tên cột cadence
    df_cadence.columns = (
        df_cadence.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )

    # Làm sạch khoảng trắng id_attraction
    if "id_attraction" in df_cadence.columns:
        df_cadence["id_attraction"] = (
            df_cadence["id_attraction"].astype(str).str.strip()
        )
    if "id_attraction" in df_surface.columns:
        df_surface["id_attraction"] = (
            df_surface["id_attraction"].astype(str).str.strip()
        )

    # Merge Surface + Cadence thành Bảng Dữ Liệu Tĩnh dim_cadence
    df_cadence_merged = df_cadence.merge(df_surface, on="id_attraction", how="left")
    df_cadence_merged = df_cadence_merged[
        df_cadence_merged["id_attraction"].isin(target_attractions)
    ].copy()

    # Chuẩn hóa tên cột khớp với Postgres DB
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

    # Nạp Bảng dim_attraction Tĩnh
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


# ==============================================================================
# 3. ETL SERVICE - NHÓM DỮ LIỆU ĐỘNG (DYNAMIC TIME-SERIES DATA)
# ==============================================================================
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
    """
    ETL Nhóm Động: Đọc 6 file biến đổi theo thời gian, nạp vào Fact Tables & Dynamic Dims.
    """
    # 1. ĐỌC DỮ LIỆU ĐỘNG
    df_visitor = load_file_safely(visitor_bytes, "visitor.csv")
    df_etat = load_file_safely(etat_bytes, "etat.csv")
    df_weather = load_file_safely(weather_bytes, "weather.csv")
    df_horaire = load_file_safely(horaire_bytes, "horaire.csv")
    df_elec_raw = load_file_safely(elec_bytes, "electricite.xlsx")
    df_ec_raw = load_file_safely(ec_bytes, "thermal.xlsx")

    # 2. CHUẨN HÓA KHÓA LỌC ID_ATTRACTION
    target_attractions = [str(x).strip() for x in target_attractions]

    if "id_attraction" in df_visitor.columns:
        df_visitor["id_attraction"] = (
            df_visitor["id_attraction"].astype(str).str.strip()
        )
        df_visitor = df_visitor[df_visitor["id_attraction"].isin(target_attractions)]
    if "id_attraction" in df_etat.columns:
        df_etat["id_attraction"] = df_etat["id_attraction"].astype(str).str.strip()
        df_etat = df_etat[df_etat["id_attraction"].isin(target_attractions)]

    # 3. CHUẨN HÓA DIM_TEMPS (SINH DÃY THỜI GIAN LIÊN TỤC 100% TRÁNH LỖI MẤT GIỜ/DST)
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

    # 4. CHUẨN HÓA DIM_HORAIRE & DIM_WEATHER
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

    # 5. CHUẨN HÓA FACT_ATTRACTION_HOURLY
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

    # 6. CHUẨN HÓA FACT ELEC & FACT EC
    df_elec_to_save = prepare_energy_fact(df_elec_raw, target_attractions)
    df_ec_to_save = prepare_energy_fact(df_ec_raw, target_attractions)

    # Ép kiểu int64 cho temps_id trên tất cả bảng Fact và Dim để tránh lỗi mismatch
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

    # Lọc chỉ giữ lại những temps_id đã thực sự tồn tại ở dim_temps (Khóa ngoại an toàn)
    valid_temps_ids = set(df_temps["temps_id"])
    df_weather_to_save = df_weather_to_save[
        df_weather_to_save["temps_id"].isin(valid_temps_ids)
    ]
    df_fact_to_save = df_fact_to_save[df_fact_to_save["temps_id"].isin(valid_temps_ids)]
    df_elec_to_save = df_elec_to_save[df_elec_to_save["temps_id"].isin(valid_temps_ids)]
    df_ec_to_save = df_ec_to_save[df_ec_to_save["temps_id"].isin(valid_temps_ids)]

    # Loại bỏ trùng lặp Primary Key/Unique key trước khi ghi vào Database
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

    # 7. NẠP VÀO DATABASE TRONG 1 TRANSACTION CHUẨN DÙNG ENGINE.BEGIN()
    try:
        with engine.begin() as conn:
            # A. Tạo các bản ghi dim_attraction nếu chưa có
            for attr_id in target_attractions:
                conn.execute(
                    text("""
                    INSERT INTO dim_attraction (id_attraction, name)
                    VALUES (:attr_id, :nom)
                    ON CONFLICT (id_attraction) DO NOTHING;
                """),
                    {"attr_id": attr_id, "nom": f"Attraction {attr_id}"},
                )

            # B. Truncate các bảng động
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

            # C. Nạp Dimension TRƯỚC
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

            # D. Nạp Fact SAU
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


# ==============================================================================
# 4. HÀM XUẤT MA TRẬN DỮ LIỆU FEATURE ML (GHÉP ĐỘNG + TĨNH)
# ==============================================================================
def get_ml_feature_matrix(
    engine: Engine, id_attraction_target: str = "H03"
) -> pd.DataFrame:
    """
    Truy vấn SQL kết hợp Dữ liệu Động + Dữ liệu Tĩnh (dim_cadence)
    để tự động tính toán 'duree', 'cycle_attraction_max', 'duty_cycle_max'
    theo đúng type_frequentation (BF/MF vs HF/THF).
    """
    query = text("""
    WITH elec_pivoted AS (
        SELECT 
            temps_id, id_attraction,
            MAX(CASE WHEN metric_name = 'elec_1' THEN value END) AS elec_1,
            MAX(CASE WHEN metric_name = 'elec_2' THEN value END) AS elec_2
        FROM fact_elec_hourly
        WHERE id_attraction = :id_attraction
        GROUP BY temps_id, id_attraction
    ),
    ec_pivoted AS (
        SELECT 
            temps_id, id_attraction,
            MAX(CASE WHEN metric_name = 'ec_value' THEN value END) AS ec_value
        FROM fact_ec_hourly
        WHERE id_attraction = :id_attraction
        GROUP BY temps_id, id_attraction
    )
    SELECT 
        ROW_NUMBER() OVER (ORDER BY f.datetime ASC) AS id,
        f.datetime AS date,
        f.id_attraction,
        t.annee AS year,
        t.mois AS month,
        t.jour AS day,
        t.heure AS hour,
        EXTRACT(WEEK FROM f.datetime) AS week,

        -- Bảng Động: Lịch hoạt động
        t.is_weekend,
        h.is_open,
        h.jf,
        h.type_frequentation,
        h.frequentation AS frequentation_park_daily,

        -- Bảng Tĩnh: Thông số công trình & Cadence
        cad.surface,
        cad.capacite_salle AS "capacite salle",
        cad.capacite_file_attente AS "capacite file d'attente",
        cad.capacite_pre_salle AS "capacite pre-salle",

        -- Logic chuyển đổi Duree & Cycle động theo type_frequentation của ngày đó
        CASE 
            WHEN h.type_frequentation IN ('BF', 'MF') THEN cad.duree_longue
            WHEN h.type_frequentation IN ('HF', 'THF') THEN cad.duree_courte
            ELSE NULL 
        END AS duree,

        CASE 
            WHEN h.type_frequentation IN ('BF', 'MF') THEN cad.cycle_max_vl
            WHEN h.type_frequentation IN ('HF', 'THF') THEN cad.cycle_max_vc
            ELSE NULL 
        END AS cycle_attraction_max,

        CASE 
            WHEN h.type_frequentation IN ('BF', 'MF') THEN cad.duty_cycle_max_vl
            WHEN h.type_frequentation IN ('HF', 'THF') THEN cad.duty_cycle_max_vc
            ELSE NULL 
        END AS duty_cycle_max,

        -- Bảng Động: Trạng thái & Visitor
        f.ouvert,
        f.interrompu,
        f.operation,
        f.visitor_count,

        -- Bảng Động: Thời tiết
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

        -- Bảng Động: Target Năng lượng
        e.elec_1,
        e.elec_2,
        c.ec_value

    FROM fact_attraction_hourly f
    LEFT JOIN dim_temps t ON f.temps_id = t.temps_id
    LEFT JOIN dim_weather w ON f.temps_id = w.temps_id
    LEFT JOIN dim_horaire h ON CAST(f.datetime AS DATE) = h.date
    LEFT JOIN dim_cadence cad ON f.id_attraction = cad.id_attraction
    LEFT JOIN elec_pivoted e ON f.temps_id = e.temps_id AND f.id_attraction = e.id_attraction
    LEFT JOIN ec_pivoted c ON f.temps_id = c.temps_id AND f.id_attraction = c.id_attraction
    WHERE f.id_attraction = :id_attraction
    ORDER BY f.datetime ASC;
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"id_attraction": id_attraction_target})

    return df


# ==============================================================================
# 5. CÁC HÀM PHÂN TÍCH & PROFILING DỮ LIỆU
# ==============================================================================
def generate_all_fact_shells_in_db(
    start_date: str,
    end_date: str,
    engine: Engine,
    target_attractions: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Tạo khung thời gian trống (Fact Shells) tự động quét và thích ứng
    với mọi biến thể metric_name thực tế của từng attraction.
    """

    filter_attraction_sql = ""
    if target_attractions:
        filter_attraction_sql = "AND a.id_attraction IN :attractions"

    # 1. Bảng fact_attraction_hourly (Cấu trúc cột cố định)
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

    # 2. Bảng fact_elec_hourly (Lấy động metric_name thực tế theo từng attraction)
    sql_elec = f"""
        WITH existing_metrics AS (
            -- Lấy danh sách cặp (id_attraction, metric_name) ĐÃ CÓ trong DB
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

    # 3. Bảng fact_ec_hourly (Lấy động tất cả các metric_name nóng/lạnh thực tế của từng attraction)
    sql_ec = f"""
        WITH existing_metrics AS (
            -- Lấy danh sách cặp (id_attraction, metric_name) ĐÃ CÓ trong DB
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

    # 4. Bảng dim_weather
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

    # Bind parameters
    params = {"start_date": start_date, "end_date": end_date}
    if target_attractions:
        params["attractions"] = target_attractions

    def prepare_stmt(query_str: str):
        stmt = text(query_str)
        if target_attractions:
            stmt = stmt.bindparams(bindparam("attractions", expanding=True))
        return stmt

    # Thực thi Transaction
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


def effectuer_collecte_analyse_globale(
    df: pd.DataFrame,
    nom_table: str = "fact_attraction_hourly",
    nb_lignes_apercu: Optional[int] = 10,
    id_attraction: Optional[str] = None,
) -> Dict[str, Any]:
    liste_attractions = []
    if "id_attraction" in df.columns:
        liste_attractions = sorted(
            df["id_attraction"].dropna().astype(str).unique().tolist()
        )

    # 2. Nếu người dùng chọn một attraction cụ thể
    if id_attraction and id_attraction != "ALL":
        # Lọc dòng theo id_attraction
        if "id_attraction" in df.columns:
            df = df[df["id_attraction"].astype(str) == str(id_attraction)].copy()

        # BƯỚC QUAN TRỌNG 1: Loại bỏ tất cả các cột bị rỗng hoàn toàn (100% NaN) sau khi lọc dòng
        df = df.dropna(how="all", axis=1)

        # BƯỚC QUAN TRỌNG 2: Loại bỏ các cột có tên chứa ID của attraction KHÁC (VD: H07_...)
        cols_to_keep = []
        for col in df.columns:
            # Tìm xem tên cột có chứa mã attraction dạng H01, H02, H03... không
            match = re.search(r"H\d{2}", col)
            if match:
                col_attr_id = match.group(0)
                # Chỉ giữ lại nếu trùng với id_attraction đang chọn
                if col_attr_id == id_attraction:
                    cols_to_keep.append(col)
            else:
                # Giữ lại các cột chung (datetime, temps_id, temperature,...)
                cols_to_keep.append(col)

        df = df[cols_to_keep]
    """Tính toán chỉ số Profiling, bộ nhớ và thống kê biến thu thập."""
    if df.empty:
        return {
            "total_lignes": 0,
            "total_colonnes": int(len(df.columns)),
            "nb_valeurs_manquantes": 0,
            "pourcentage_manquants": 0.0,
            "nb_outliers": 0,
            "pourcentage_outliers": 0.0,
            "nb_valeurs_negatives": 0,
            "pourcentage_negatifs": 0.0,
            "nb_doublons": 0,
            "pourcentage_dupliques": 0.0,
            "date_debut": "N/A",
            "date_fin": "N/A",
            "utilisation_memoire": [],
            "comparaison_colonnes": [],
            "apercu_donnees": [],
        }

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values(by="datetime", ascending=True).reset_index(drop=True)

    total_lignes = int(len(df))
    total_colonnes = int(len(df.columns))
    total_cellules = total_lignes * total_colonnes if total_colonnes > 0 else 1

    total_manquants = int(df.isnull().sum().sum())
    pourcentage_manquants = round((total_manquants / total_cellules) * 100, 2)
    lignes_dupliquees = int(df.duplicated().sum())
    pourcentage_dupliques = (
        round((lignes_dupliquees / total_lignes) * 100, 2) if total_lignes > 0 else 0.0
    )

    total_outliers = 0
    total_negatifs = 0
    comparaison_colonnes = []
    utilisation_memoire = []

    memoire_series = df.memory_usage(deep=True, index=False)
    total_memoire = memoire_series.sum()

    cols_ignore_outliers = {
        "heure",
        "jour",
        "mois",
        "annee",
        "is_weekend",
        "is_open",
        "jf",
    }

    for col in df.columns:
        serie = df[col]
        manquants_col = int(serie.isnull().sum())
        outliers_col = 0
        type_col = str(serie.dtype)

        moyenne, ecart_type, val_min, mediane, val_max = None, None, None, None, None

        if pd.api.types.is_numeric_dtype(serie):
            serie_valide = serie.dropna()
            if str(col) not in cols_ignore_outliers:
                total_negatifs += int((serie_valide < 0).sum())

                if len(serie_valide) > 0:
                    q1 = float(serie_valide.quantile(0.25))
                    q3 = float(serie_valide.quantile(0.75))
                    iqr = q3 - q1
                    borne_inf = q1 - 1.5 * iqr
                    borne_sup = q3 + 1.5 * iqr
                    outliers_col = int(
                        ((serie_valide < borne_inf) | (serie_valide > borne_sup)).sum()
                    )
                    total_outliers += outliers_col

            if not serie_valide.empty:
                moyenne = round(float(serie_valide.mean()), 2)
                ecart_type = round(float(serie_valide.std()), 2)
                val_min = round(float(serie_valide.min()), 2)
                mediane = round(float(serie_valide.median()), 2)
                val_max = round(float(serie_valide.max()), 2)

        mem_octets = int(memoire_series[col])
        pct_mem = (
            round((mem_octets / total_memoire) * 100, 2) if total_memoire > 0 else 0.0
        )

        utilisation_memoire.append(
            {"nom_colonne": str(col), "octets": mem_octets, "pourcentage": pct_mem}
        )
        comparaison_colonnes.append(
            {
                "nom_colonne": str(col),
                "type_donnees": type_col,
                "valeurs_manquantes": manquants_col,
                "valeurs_aberrantes": outliers_col,
                "moyenne": moyenne,
                "ecart_type": ecart_type,
                "val_min": val_min,
                "mediane": mediane,
                "val_max": val_max,
            }
        )

    date_debut, date_fin = "N/A", "N/A"
    if "datetime" in df.columns:
        try:
            serie_date = df["datetime"].dropna()
            if not serie_date.empty:
                date_debut = str(serie_date.min().strftime("%Y-%m-%d %H:%M:%S"))
                date_fin = str(serie_date.max().strftime("%Y-%m-%d %H:%M:%S"))
        except Exception:
            pass

    df_apercu = (
        df.head(nb_lignes_apercu).copy()
        if (nb_lignes_apercu and nb_lignes_apercu > 0)
        else df.copy()
    )
    for col in df_apercu.columns:
        if pd.api.types.is_datetime64_any_dtype(df_apercu[col]):
            df_apercu[col] = df_apercu[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "total_lignes": total_lignes,
        "total_colonnes": total_colonnes,
        "nb_valeurs_manquantes": total_manquants,
        "pourcentage_manquants": pourcentage_manquants,
        "nb_outliers": total_outliers,
        "pourcentage_outliers": round((total_outliers / total_cellules) * 100, 2),
        "nb_valeurs_negatives": total_negatifs,
        "pourcentage_negatifs": round((total_negatifs / total_cellules) * 100, 2),
        "nb_doublons": lignes_dupliquees,
        "pourcentage_dupliques": pourcentage_dupliques,
        "date_debut": date_debut,
        "date_fin": date_fin,
        "utilisation_memoire": utilisation_memoire,
        "comparaison_colonnes": comparaison_colonnes,
        "apercu_donnees": df_apercu.fillna("").astype(str).to_dict(orient="records"),
        "liste_attractions": liste_attractions,
    }


from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


def analyser_profil_colonne_seule(
    df: pd.DataFrame, col_name: str, id_attraction: Optional[str] = None
) -> Dict[str, Any]:
    """Phân tích chi tiết 1 cột (Single Column Profiling) với quy tắc gom nhóm linh hoạt."""
    if df.empty or col_name not in df.columns:
        return {"erreur": f"Cột '{col_name}' không tồn tại hoặc DataFrame rỗng."}

    df_target = df.copy()

    # 1. Lọc theo id_attraction (Nếu chọn từng attraction cụ thể)
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
            "erreur": f"Không có dữ liệu cho id_attraction = '{id_attraction}'"
        }

    serie = df_target[col_name]
    dtype_str = str(serie.dtype)

    nb_missing = int(serie.isnull().sum())
    pct_missing = round((nb_missing / total_rows) * 100, 2)
    nb_unique = int(serie.nunique(dropna=True))
    is_numeric = pd.api.types.is_numeric_dtype(serie)

    nb_zeros, pct_zeros, nb_negatifs, pct_negatifs, nb_outliers, pct_outliers = (
        0,
        0.0,
        0,
        0.0,
        0,
        0.0,
    )
    box_plot_data, histogram_data = None, None
    line_plots_data = {
        "par_heure": [],
        "par_jour": [],
        "par_mois": [],
        "par_annee": [],
    }

    if is_numeric:
        serie_valide = serie.dropna()

        if not serie_valide.empty:
            nb_zeros = int((serie_valide == 0).sum())
            pct_zeros = round((nb_zeros / total_rows) * 100, 2)
            nb_negatifs = int((serie_valide < 0).sum())
            pct_negatifs = round((nb_negatifs / total_rows) * 100, 2)

            q1, q3 = (
                float(serie_valide.quantile(0.25)),
                float(serie_valide.quantile(0.75)),
            )
            iqr = q3 - q1
            borne_inf, borne_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr

            outliers_mask = (serie_valide < borne_inf) | (
                serie_valide > borne_sup
            )
            nb_outliers = int(outliers_mask.sum())
            pct_outliers = round((nb_outliers / total_rows) * 100, 2)

            box_plot_data = {
                "min": round(float(serie_valide.min()), 2),
                "q25": round(q1, 2),
                "mediane": round(float(serie_valide.median()), 2),
                "q75": round(q3, 2),
                "max": round(float(serie_valide.max()), 2),
                "borne_inf": round(borne_inf, 2),
                "borne_sup": round(borne_sup, 2),
                "outliers_samples": [
                    round(float(x), 2)
                    for x in serie_valide[outliers_mask].head(50).tolist()
                ],
            }

            counts, bin_edges = np.histogram(serie_valide, bins=20)
            histogram_data = [
                {
                    "bin_range": (
                        f"{round(bin_edges[i], 1)} -"
                        f" {round(bin_edges[i + 1], 1)}"
                    ),
                    "count": int(counts[i]),
                }
                for i in range(len(counts))
            ]

            if "datetime" in df_target.columns:
                df_target["datetime"] = pd.to_datetime(df_target["datetime"])

                # 2. Nhận diện loại chỉ số để chọn phép gom nhóm (SUM cho visitor/năng lượng, MEAN cho còn lại)
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

                # 3. Gom nhóm theo chính xác từng GIỜ (khi chọn ALL sẽ cộng/trung bình tổng các attraction theo đúng từng giờ)
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
                        "valeur_moyenne": round(float(r[col_name]), 2),
                    }
                    for _, r in heure_agg.iterrows()
                ]

                # 4. Gom nhóm theo NGÀY
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
                        "valeur_moyenne": round(float(r[col_name]), 2),
                    }
                    for _, r in jour_agg.iterrows()
                ]

                # 5. Gom nhóm theo THÁNG
                df_target["year_month_str"] = df_target["datetime"].dt.strftime(
                    "%Y-%m"
                )
                mois_agg = (
                    df_target.groupby("year_month_str")[col_name]
                    .agg(agg_func)
                    .reset_index()
                    .sort_values("year_month_str")
                )
                line_plots_data["par_mois"] = [
                    {
                        "mois": str(r["year_month_str"]),
                        "valeur_moyenne": round(float(r[col_name]), 2),
                    }
                    for _, r in mois_agg.iterrows()
                ]

                # 6. Gom nhóm theo NĂM
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
                        "valeur_moyenne": round(float(r[col_name]), 2),
                    }
                    for _, r in annee_agg.iterrows()
                ]

    return {
        "nom_colonne": col_name,
        "id_attraction": id_attraction or "ALL",
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
