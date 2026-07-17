import pandas as pd
import numpy as np

# ==============================================================================
# 1. CHARGEMENT DU MASTER FILE BRUT CONSOLIDÉ
# ==============================================================================
master_path = "D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_clean\\master\\master_H03_elec.csv"
df_master = pd.read_csv(master_path)

# ==============================================================================
# 2. DÉFINITION DES HORAIRES D'OUVERTURE ET FERMETURE LOGIQUES (H03)
# ==============================================================================
df_master['h_ouv_h03'] = np.nan
df_master['h_ferm_h03'] = np.nan

# Règle BF : 10h - 19h
df_master.loc[df_master['type_frequentation'] == 'BF', ['h_ouv_h03', 'h_ferm_h03']] = [10, 19]

# Règle MF : 10h - 20h
df_master.loc[df_master['type_frequentation'] == 'MF', ['h_ouv_h03', 'h_ferm_h03']] = [10, 20]

# Règle HF et TF : 10h - 21h (Correction du tag de TF)
df_master.loc[df_master['type_frequentation'].isin(['HF', 'TF']), ['h_ouv_h03', 'h_ferm_h03']] = [10, 21]

# Valeurs de secours (Fallback) au cas où type_frequentation est manquant
df_master['h_ouv_h03'] = df_master['h_ouv_h03'].fillna(10)
df_master['h_ferm_h03'] = df_master['h_ferm_h03'].fillna(18)

# ==============================================================================
# 3. NETTOYAGE ET IMPUTATION DES VALEURS MANQUANTES (NaN)
# ==============================================================================

# --- VARIABLE 1 : LE FLUX DE VISITEURS (visitor_count) ---
# Cas 1 : Journées de fermeture complète du parc
cond_parc_ferme = (df_master['is_open'] == 0)

# Cas 2 : En dehors des horaires d'ouverture spécifiques calculés pour H03
cond_hors_amplitude = (df_master['is_open'] == 1) & (
    (df_master['hour'] < df_master['h_ouv_h03']) | (df_master['hour'] >= df_master['h_ferm_h03'])
)

# Forçage à 0 visiteur pour les périodes d'inactivité de l'attraction
df_master.loc[cond_parc_ferme | cond_hors_amplitude, 'visitor_count'] = df_master.loc[cond_parc_ferme | cond_hors_amplitude, 'visitor_count'].fillna(0)

# Cas 3 : Interpolation linéaire pour les NaN restants (pendant l'ouverture)
df_master['visitor_count'] = df_master['visitor_count'].interpolate(method='linear')
df_master['visitor_count'] = df_master['visitor_count'].clip(lower=0).round().astype(int)


# --- VARIABLE 2 : DONNÉES MÉTÉOROLOGIQUES (temperature, humidite...) ---
# Imputation par interpolation linéaire continue pour la météo
colonnes_meteo = [
    'temperature', 'humidite', 'rayonnement_solaire', 
    'temp_max', 'temp_min', 'temp_moy', 
    'humidite_max', 'humidite_min', 'humidite_moy'
]
for col in colonnes_meteo:
    if col in df_master.columns:
        df_master[col] = df_master[col].interpolate(method='linear').ffill().bfill()


# --- VARIABLE 3 : ÉNERGIE CALORIFIQUE (ec_value) ---
# Lissage des coupures de données thermiques par interpolation
if 'elec_1' in df_master.columns:
    df_master['elec_1'] = df_master['elec_1'].interpolate(method='linear').fillna(0)
if 'elec_2' in df_master.columns:
    df_master['elec_2'] = df_master['elec_2'].interpolate(method='linear').fillna(0)

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

# Id unique séquentiel de 1 à 8760
df_master['id'] = range(1, len(df_master) + 1)

# ==============================================================================
# 5. STRUCTURATION DES COLONNES ET EXPORTATION FINALE
# ==============================================================================
# Organisation logique des Features pour l'apprentissage du modèle
ordre_colonnes = [
    'date', 'visitor_count', 'elec_1', 'elec_2',
    'cycle_attraction_max', 'duty_cycle_max', 'capacite salle', 'capacite file d\'attente', 'capacite pre-salle',
    'is_weekend','jf', 'is_open', 'type_frequentation',
    'temperature', 'temp_max', 'temp_min', 'temp_moy',
    'humidite','humidite_max', 'humidite_min', 'humidite_moy',
    'rayonnement_solaire',
    'hour', 'min', 'day', 'month', 'year', 'week'
]

# Sélection finale des colonnes valides
colonnes_finales = [c for c in ordre_colonnes if c in df_master.columns]
df_master_clean = df_master[colonnes_finales]

# Sauvegarde du Master File propre pour l'entraînement ML
output_clean_path = "D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_clean\\master_missing\\master_H03_elec_missing.csv"
df_master_clean.to_csv(output_clean_path, index=False)

print(f"Master File nettoyé avec succès ! Lignes : {len(df_master_clean)} (Attendu : 8760)")
print(f"Nombre de colonnes finales : {len(df_master_clean.columns)}")

# import pandas as pd
# import numpy as np

# # 1. Chargement des données CSV
# df_visit = pd.read_csv('D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_brut\\visitor\\visit_H03..csv')
# df_horaire = pd.read_csv('D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_brut\\cadence\\horaire2025_final_fr.csv')

# # 2. Standardisation des formats de date pour la fusion
# df_horaire['date_pure'] = pd.to_datetime(df_horaire['date'], format='%m/%d/%Y').dt.date

# df_visit['date'] = pd.to_datetime(df_visit['date'])
# df_visit['date_pure'] = df_visit['date'].dt.date

# # 3. Création du calendrier complet 2025 (365 jours × 24 heures)
# full_time_range = pd.date_range(start='2025-01-01 00:00:00', end='2025-12-31 23:00:00', freq='h')
# df_full = pd.DataFrame(index=full_time_range)
# df_full.index.name = 'date'
# df_full['date_pure'] = df_full.index.date
# df_full['hour'] = df_full.index.hour

# # 4. Fusion du calendrier global (avec type_frequentation)
# df_full = df_full.reset_index().merge(
#     df_horaire[['date_pure', 'is_open', 'type_frequentation']], 
#     on='date_pure', 
#     how='left'
# )

# # 5. Fusion des données d'affluence actuelles
# df_final = df_full.merge(df_visit[['date', 'visitor_count']], on='date', how='left')

# # 6. Définition des horaires d'ouverture (h_ouv_h03) et de fermeture (h_ferm_h03) pour H03
# df_final['h_ouv_h03'] = np.nan
# df_final['h_ferm_h03'] = np.nan

# # Application des règles d'ouverture H03 selon la fréquentation
# # BF : 10h - 19h
# df_final.loc[df_final['type_frequentation'] == 'BF', ['h_ouv_h03', 'h_ferm_h03']] = [10, 19]

# # MF : 10h - 20h
# df_final.loc[df_final['type_frequentation'] == 'MF', ['h_ouv_h03', 'h_ferm_h03']] = [10, 20]

# # HF et TF : 10h - 21h
# df_final.loc[df_final['type_frequentation'].isin(['HF', 'TF']), ['h_ouv_h03', 'h_ferm_h03']] = [10, 21]


# # 7. Imputation des valeurs manquantes (visitor_count)
# # Cas 1 : Jour de fermeture du parc (is_open == 0)
# is_closed_day = df_final['is_open'] == 0

# # Cas 2 : En dehors des horaires d'ouverture spécifiques de l'attraction H03
# is_outside_h03_hours = (df_final['is_open'] == 1) & (
#     (df_final['hour'] < df_final['h_ouv_h03']) | (df_final['hour'] >= df_final['h_ferm_h03'])
# )

# # Attribution de 0 visiteur pour les périodes de fermeture
# df_final.loc[is_closed_day | is_outside_h03_hours, 'visitor_count'] = df_final.loc[is_closed_day | is_outside_h03_hours, 'visitor_count'].fillna(0)

# # Cas 3 : Valeurs manquantes (NaN) pendant les heures d'ouverture -> Interpolation linéaire
# df_final['visitor_count'] = df_final['visitor_count'].interpolate(method='linear')

# # Arrondi et conversion en entiers non négatifs
# df_final['visitor_count'] = df_final['visitor_count'].clip(lower=0).round().astype(int)


# # 8. Reconstitution des variables temporelles et attributs d'origine
# df_final['year'] = df_final['date'].dt.year
# df_final['month'] = df_final['date'].dt.month
# df_final['day'] = df_final['date'].dt.day
# df_final['min'] = df_final['date'].dt.minute
# df_final['week'] = df_final['date'].dt.isocalendar().week

# df_final['id_attraction'] = "H03"
# df_final['attraction_name'] = "L'EXTRAORDINAIRE VOYAGE"
# df_final['id'] = range(1, len(df_final) + 1)

# # 9. Sélection des colonnes cibles et exportation
# output_cols = ['date', 'id_attraction', 'attraction_name', 'visitor_count', 'id', 'hour', 'min', 'week', 'day', 'month', 'year']
# df_output = df_final[output_cols]

# # Sauvegarde du fichier nettoyé
# df_output.to_csv('D:\\Stage SI\\Machine Learning\\Futuroscope\\windows\\donne_clean\\visitor\\visit_H03.csv', index=False)
# print(f"Traitement terminé ! Fichier synchronisé avec les règles de l'H03. Lignes totales : {len(df_output)}")