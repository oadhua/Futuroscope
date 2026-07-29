import pandas as pd
import numpy as np

# ==============================================================================
# 1. CHARGEMENT DU MASTER FILE BRUT CONSOLIDÉ
# ==============================================================================
master_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master\master_H03_elec.csv"
df_master = pd.read_csv(master_path)

# ------------------------------------------------------------------------------
# BƯỚC 1: KIỂM TRA TRƯỚC KHI DỌN DẸP (BEFORE CLEANING)
# ------------------------------------------------------------------------------
print("=" * 65)
print(f"1. BÁO CÁO DỮ LIỆU CÒN THIẾU BAN ĐẦU (Tổng số dòng: {len(df_master)})")
print("=" * 65)

null_before = df_master.isnull().sum()
percent_before = (null_before / len(df_master)) * 100

df_before_report = pd.DataFrame({
    'Cột': null_before.index,
    'Số dòng thiếu': null_before.values,
    'Tỷ lệ (%)': percent_before.round(2).values
}).sort_values(by='Số dòng thiếu', ascending=False).reset_index(drop=True)

# Chỉ hiển thị các cột thực sự có dữ liệu thiếu để dễ quan sát
missing_only_before = df_before_report[df_before_report['Số dòng thiếu'] > 0]

if missing_only_before.empty:
    print("-> Tuyệt vời! File gốc không có cột nào bị thiếu dữ liệu.")
else:
    print(missing_only_before.to_string(index=False))

print("=" * 65 + "\n")


# ==============================================================================
# 2. DÉFINITION DES HORAIRES D'OUVERTURE ET FERMETURE LOGIQUES (H03)
# ==============================================================================
df_master['h_ouv_h03'] = np.nan
df_master['h_ferm_h03'] = np.nan

# Règle BF : 10h - 19h
df_master.loc[df_master['type_frequentation'] == 'BF', ['h_ouv_h03', 'h_ferm_h03']] = [10, 19]

# Règle MF : 10h - 20h
df_master.loc[df_master['type_frequentation'] == 'MF', ['h_ouv_h03', 'h_ferm_h03']] = [10, 20]

# Règle HF et TF/THF : 10h - 21h
df_master.loc[df_master['type_frequentation'].isin(['HF', 'THF']), ['h_ouv_h03', 'h_ferm_h03']] = [10, 21]

# Fallback nếu type_frequentation bị khuyết
df_master['h_ouv_h03'] = df_master['h_ouv_h03'].fillna(10)
df_master['h_ferm_h03'] = df_master['h_ferm_h03'].fillna(18)


# ==============================================================================
# 3. NETTOYAGE ET IMPUTATION DES VALEURS MANQUANTES (NaN)
# ==============================================================================

# --- VARIABLE 1 : LE FLUX DE VISITEURS (visitor_count) ---
# if 'visitor_count' in df_master.columns:
#     cond_parc_ferme = (df_master['is_open'] == 0)
#     cond_hors_amplitude = (df_master['is_open'] == 1) & (
#         (df_master['hour'] < df_master['h_ouv_h03']) | (df_master['hour'] >= df_master['h_ferm_h03'])
#     )
#     df_master.loc[cond_parc_ferme | cond_hors_amplitude, 'visitor_count'] = df_master.loc[cond_parc_ferme | cond_hors_amplitude, 'visitor_count'].fillna(0)
#     df_master['visitor_count'] = df_master['visitor_count'].interpolate(method='linear')
#     df_master['visitor_count'] = df_master['visitor_count'].clip(lower=0).round().astype(int)

# --- VARIABLE 2 : DONNÉES MÉTÉOROLOGIQUES ---
colonnes_meteo = [
    'temperature', 'humidite', 'rayonnement_solaire', 
    'temp_max', 'temp_min', 'temp_moy', 
    'humidite_max', 'humidite_min', 'humidite_moy',
    'day_degree_cold', 'day_degree_hot',
]
for col in colonnes_meteo:
    if col in df_master.columns:
        df_master[col] = df_master[col].interpolate(method='linear').ffill().bfill()

# --- VARIABLE 3 : ÉNERGIE ELEC ---
if 'elec_1' in df_master.columns:
    df_master['elec_1'] = df_master['elec_1'].interpolate(method='linear').fillna(0)
if 'elec_2' in df_master.columns:
    df_master['elec_2'] = df_master['elec_2'].interpolate(method='linear').fillna(0)

# --- VARIABLE 4 : TRẠNG THÁI ETAT (interrompu, ouvert, operation) ---
for col in ['interrompu', 'ouvert', 'operation']:
    if col in df_master.columns:
        df_master[col] = df_master[col].fillna(0)


# ==============================================================================
# 4. RECONSTITUTION ET SÉCURISATION DES VARIABLES TEMPORELLES
# ==============================================================================
df_master['date'] = pd.to_datetime(df_master['date'])
df_master['year'] = df_master['date'].dt.year
df_master['month'] = df_master['date'].dt.month
df_master['day'] = df_master['date'].dt.day
df_master['hour'] = df_master['date'].dt.hour
df_master['min'] = df_master['date'].dt.minute
df_master['week'] = df_master['date'].dt.isocalendar().week
df_master['id'] = range(1, len(df_master) + 1)


# ==============================================================================
# 5. STRUCTURATION DES COLONNES ET EXPORTATION FINALE
# ==============================================================================
ordre_colonnes = [
    'date', 'visitor_count', 'frequentation', 'elec_1', 'elec_2',
    'interrompu', 'ouvert', 'operation',
    'cycle_attraction_max', 'duty_cycle_max', 'capacite salle', 'capacite file d\'attente', 'capacite pre-salle',
    'is_weekend', 'jf', 'is_open', 'type_frequentation',
    'temperature', 'temp_max', 'temp_min', 'temp_moy',
    'humidite', 'humidite_max', 'humidite_min', 'humidite_moy',
    'rayonnement_solaire',
    'day_degree_cold', 'day_degree_hot',
    'hour', 'min', 'day', 'month', 'year', 'week'
]

colonnes_finales = [c for c in ordre_colonnes if c in df_master.columns]
df_master_clean = df_master[colonnes_finales]

# ------------------------------------------------------------------------------
# BƯỚC 2: KIỂM TRA LẠI SAU KHI DỌN DẸP (AFTER CLEANING)
# ------------------------------------------------------------------------------
print("=" * 65)
print("2. BÁO CÁO DỮ LIỆU CÒN THIẾU SAU KHI XỬ LÝ (AFTER CLEANING)")
print("=" * 65)

null_after = df_master_clean.isnull().sum()
percent_after = (null_after / len(df_master_clean)) * 100

df_after_report = pd.DataFrame({
    'Cột': null_after.index,
    'Số dòng thiếu': null_after.values,
    'Tỷ lệ (%)': percent_after.round(2).values
}).sort_values(by='Số dòng thiếu', ascending=False).reset_index(drop=True)

print(df_after_report.to_string(index=False))
print("=" * 65 + "\n")

# Xuất file kết quả
output_clean_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_missing\master_H03_elec_missing.csv"
df_master_clean.to_csv(output_clean_path, index=False)

print(f"-> Xuất file thành công: {output_clean_path}")