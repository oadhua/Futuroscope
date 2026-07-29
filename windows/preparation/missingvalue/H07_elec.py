import pandas as pd
import numpy as np

# ==============================================================================
# 1. CHARGEMENT DES FILES BRUTS
# ==============================================================================
# Đường dẫn file của bạn
master_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master\master_H07_elec.csv"
horaire_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\cadence\horaire_all_years.csv"  # Đổi lại đường dẫn tuyệt đối nếu cần

df_master = pd.read_csv(master_path)
df_horaire = pd.read_csv(horaire_path)

# Đảm bảo cột date ở cả 2 dataframe đều ở định dạng datetime để merge chính xác
df_master['date'] = pd.to_datetime(df_master['date'])
df_horaire['date'] = pd.to_datetime(df_horaire['date'], format='%m/%d/%Y') # format DD/MM/YYYY của file horaire

# ==============================================================================
# 2. XỬ LÝ GIỜ MỞ/ĐÓNG CỬA TỪ FILE LỊCH CÔNG VIÊN
# ==============================================================================
# Chuyển đổi h_ouv và h_ferm từ dạng "HH:MM:SS" sang dạng giờ nguyên (Integer)
df_horaire['h_ouv_parsed'] = pd.to_datetime(df_horaire['h_ouv'], format='%H:%M:%S', errors='coerce').dt.hour
df_horaire['h_ferm_parsed'] = pd.to_datetime(df_horaire['h_ferm'], format='%H:%M:%S', errors='coerce').dt.hour

# Chọn các cột cần thiết để gộp vào master file
df_horaire_sub = df_horaire[['date', 'h_ouv_parsed', 'h_ferm_parsed']].copy()

# Gộp thông tin giờ mở/đóng vào master file theo cột 'date' (chỉ lấy phần ngày để khớp)
# Tạo cột phụ chỉ chứa ngày (YYYY-MM-DD) để merge
df_master['date_only'] = df_master['date'].dt.normalize()
df_horaire_sub['date_only'] = df_horaire_sub['date'].dt.normalize()

# Merge
df_master = df_master.merge(
    df_horaire_sub[['date_only', 'h_ouv_parsed', 'h_ferm_parsed']], 
    on='date_only', 
    how='left'
)

# Gán giá trị giờ mở/đóng thực tế của công viên cho H07
df_master['h_ouv_h07'] = df_master['h_ouv_parsed']
df_master['h_ferm_h07'] = df_master['h_ferm_parsed']

# Valeurs de secours (Fallback) nếu ngày đó công viên đóng cửa hoặc bị thiếu dữ liệu
# Sử dụng rule BF mặc định (10h - 19h) hoặc giá trị an toàn
df_master['h_ouv_h07'] = df_master['h_ouv_h07'].fillna(10)
df_master['h_ferm_h07'] = df_master['h_ferm_h07'].fillna(18)

# Xóa các cột phụ sau khi xử lý xong
df_master = df_master.drop(columns=['date_only', 'h_ouv_parsed', 'h_ferm_parsed'])

# ==============================================================================
# 3. NETTOYAGE ET IMPUTATION DES VALEURS MANQUANTES (NaN)
# ==============================================================================

# Đảm bảo cột 'hour' đã tồn tại trước khi dùng làm điều kiện lọc
df_master['hour'] = df_master['date'].dt.hour

# --- VARIABLE 1 : LE FLUX DE VISITEURS (visitor_count) ---
# # Cas 1 : Journées de fermeture complète du parc (is_open == 0)
# cond_parc_ferme = (df_master['is_open'] == 0)

# # Cas 2 : En dehors des horaires d'ouverture réels du parc
# cond_hors_amplitude = (df_master['is_open'] == 1) & (
#     (df_master['hour'] < df_master['h_ouv_h07']) | (df_master['hour'] >= df_master['h_ferm_h07'])
# )

# # Forçage à 0 visiteur pour les périodes d'inactivité de l'attraction
# df_master.loc[cond_parc_ferme | cond_hors_amplitude, 'visitor_count'] = df_master.loc[cond_parc_ferme | cond_hors_amplitude, 'visitor_count'].fillna(0)

# # Cas 3 : Interpolation linéaire pour les NaN restants (pendant l'ouverture)
# df_master['visitor_count'] = df_master['visitor_count'].interpolate(method='linear')
# df_master['visitor_count'] = df_master['visitor_count'].clip(lower=0).round().astype(int)


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


# --- VARIABLE 3 : ÉNERGIE CALORIFIQUE (ec_value) ---
for col in ['elec_1', 'elec_2', 'elec_3']:
    if col in df_master.columns:
        # Nếu cột đang ở dạng chuỗi, thay thế dấu phẩy ',' thành dấu chấm '.' (nếu có)
        if df_master[col].dtype == 'object':
            df_master[col] = df_master[col].astype(str).str.replace(',', '.', regex=False)
        
        # Chuyển đổi sang kiểu số thực (các giá trị lỗi chữ/khoảng trắng sẽ thành NaN)
        df_master[col] = pd.to_numeric(df_master[col], errors='coerce')
        
        # Gọi nội suy tuyến tính một cách an toàn trên cột dạng số thực
        df_master[col] = df_master[col].interpolate(method='linear').fillna(0)

# ==============================================================================
# 4. RECONSTITUTION ET SÉCURISATION DES VARIABLES TEMPORELLES
# ==============================================================================
df_master['year'] = df_master['date'].dt.year
df_master['month'] = df_master['date'].dt.month
df_master['day'] = df_master['date'].dt.day
df_master['min'] = df_master['date'].dt.minute
df_master['week'] = df_master['date'].dt.isocalendar().week

# Id unique séquentiel de 1 à 8760
df_master['id'] = range(1, len(df_master) + 1)

# ==============================================================================
# 5. STRUCTURATION DES COLONNES ET EXPORTATION FINALE
# ==============================================================================
ordre_colonnes = [
    'date', 'visitor_count', 'elec_1', 'elec_2', 'elec_3',
    'cycle_attraction_max', 'duty_cycle_max', 'capacite salle', 'capacite file d\'attente', 'capacite pre-salle',
    'is_weekend','jf', 'is_open', 'type_frequentation',
    'temperature', 'temp_max', 'temp_min', 'temp_moy',
    'humidite','humidite_max', 'humidite_min', 'humidite_moy',
    'rayonnement_solaire',
    'day_degree_cold', 'day_degree_hot',
    'hour', 'min', 'day', 'month', 'year', 'week'
]

colonnes_finales = [c for c in ordre_colonnes if c in df_master.columns]
df_master_clean = df_master[colonnes_finales]

# Sauvegarde du Master File propre pour l'entraînement ML
output_clean_path = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_missing\master_H07_elec_missing.csv"
df_master_clean.to_csv(output_clean_path, index=False)

print(f"Master File nettoyé avec succès ! Lignes : {len(df_master_clean)}")
print(f"Nombre de colonnes finales : {len(df_master_clean.columns)}")