import pandas as pd
import numpy as np

def preparer_features_communes(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    if 'type_frequentation' in df.columns:
        df = pd.get_dummies(df, columns=['type_frequentation'], prefix='freq', drop_first=True)
        
    df['heure_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['heure_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['mois_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['mois_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # Thêm biến tương tác giữa Giờ và Loại ngày khách (HF, MF) để tăng sức mạnh bối cảnh
    if 'freq_HF' in df.columns:
        df['hour_x_freq_HF'] = df['hour'] * df['freq_HF']
    if 'freq_MF' in df.columns:
        df['hour_x_freq_MF'] = df['hour'] * df['freq_MF']
        
    colonnes_inutiles = [
        'cycle_attraction_max', 'duty_cycle_max', 'id', "id_attraction",
        'capacite salle', "capacite file d'attente", 'capacite pre-salle'
    ]
    return df.drop(columns=[c for c in colonnes_inutiles if c in df.columns])

def generer_dataset_pure_regression(df_base):
    df = df_base.copy()
    
    # LOẠI BỎ TOÀN BỘ LAG CỦA EC_VALUE
    # Chỉ bổ sung các biến bối cảnh thời tiết mở rộng (Trễ nhiệt độ và trung bình trượt nhiệt độ)
    # df['temp_moyenne_7j'] = df['temperature'].rolling(window=168, min_periods=1).mean()
    df['temp_decalage_1h'] = df['temperature'].shift(1)
    # df['temp_decalage_2h'] = df['temperature'].shift(2)
    # df['temp_deviation_7j'] = df['temperature'] - df['temp_moyenne_7j']
    
    # Không dùng dropna() diện rộng nữa, chỉ drop các dòng NaN do shift nhiệt độ (1-2 dòng đầu)
    return df.dropna(subset=['temp_decalage_1h']).reset_index(drop=True)
    # return df.dropna(subset=['temp_decalage_1h', 'temp_decalage_2h']).reset_index(drop=True)

if __name__ == "__main__":
    df_raw = pd.read_csv(r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_outliers\master_H03_elec_outliers.csv")
    df_commun = preparer_features_communes(df_raw)
    
    # Tạo tập dữ liệu hồi quy thuần túy không chứa biến lag điện
    df_ml = generer_dataset_pure_regression(df_commun)
    
    # Xuất ra file master dùng chung cho ML
    df_ml.to_csv(r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H03_predict_pure_elec.csv", index=False)
    print("Đã xuất file dữ liệu Pure Regression thành công!")
# import pandas as pd
# import numpy as np

# def preparer_features_communes(df):
#     df['date'] = pd.to_datetime(df['date'])
#     df = df.sort_values('date').reset_index(drop=True)
    
#     if 'type_frequentation' in df.columns:
#         df = pd.get_dummies(df, columns=['type_frequentation'], prefix='freq', drop_first=True)
        
#     df['heure_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
#     df['heure_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
#     df['mois_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
#     df['mois_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
#     colonnes_inutiles = [
#         'cycle_attraction_max', 'duty_cycle_max', 'id',
#         'capacite salle', "capacite file d'attente", 'capacite pre-salle'
#     ]
#     return df.drop(columns=[c for c in colonnes_inutiles if c in df.columns])

# def generer_dataset_1semaine(df_base):
#     df = df_base.copy()
#     # Horizon 1 semaine = Décalage minimum de 168h
#     df['ec_decalage_168h'] = df['ec_value'].shift(168)
#     df['ec_decalage_336h'] = df['ec_value'].shift(336)
#     df['ec_moyenne_7j'] = df['ec_value'].shift(168).rolling(window=168, min_periods=1).mean()
#     df['temp_moyenne_7j'] = df['temperature'].rolling(window=168, min_periods=1).mean()
#     return df.dropna().reset_index(drop=True)

# def generer_dataset_1mois(df_base):
#     df = df_base.copy()
#     # Horizon 1 mois = Décalage minimum de 720h (30 jours)
#     df['ec_decalage_720h'] = df['ec_value'].shift(720)
#     df['ec_decalage_1440h'] = df['ec_value'].shift(1440)
#     df['ec_moyenne_30j'] = df['ec_value'].shift(720).rolling(window=720, min_periods=1).mean()
#     df['temp_moyenne_30j'] = df['temperature'].rolling(window=720, min_periods=1).mean()
#     return df.dropna().reset_index(drop=True)

# if __name__ == "__main__":
#     df_raw = pd.read_csv("D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_outliers\master_H03_outliers.csv")
#     df_commun = preparer_features_communes(df_raw)
    
#     # Export du fichier pour prédiction à 1 semaine
#     df_1semaine = generer_dataset_1semaine(df_commun)
#     df_1semaine.to_csv("D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H03_predict_1week.csv", index=False)
    
#     # Export du fichier pour prédiction à 1 mois
#     df_1mois = generer_dataset_1mois(df_commun)
#     df_1mois.to_csv("D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\master_ml\master_H03_predict_1month.csv", index=False)