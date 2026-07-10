import pandas as pd
import openmeteo_requests
import requests_cache
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

DATA_DIR = r"D:\Stage SI\Machine Learning\Futuroscope\windows\donne_clean\meteo"
output_weather_file = DATA_DIR + r"\weather_hourly.csv"

print("Connexion à l'API Open-Meteo pour récupérer les données météo du Futuroscope...")

cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retries = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
cache_session.mount('https://', HTTPAdapter(max_retries=retries))

openmeteo = openmeteo_requests.Client(session=cache_session)

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 46.6692,
    "longitude": 0.3686,
    "start_date": "2024-12-31",
    "end_date": "2025-12-31",
    "hourly": ["temperature_2m", "relative_humidity_2m", "shortwave_radiation"]
}

responses = openmeteo.weather_api(url, params=params)
response = responses[0]

hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_solar_radiation = hourly.Variables(2).ValuesAsNumpy()

hourly_data = {
    "date_time": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
}

df_weather = pd.DataFrame(data=hourly_data)
df_weather['date_time'] = df_weather['date_time'].dt.tz_convert('Europe/Paris')

df_weather['date_key'] = df_weather['date_time'].dt.date.astype(str)
df_weather['heure'] = df_weather['date_time'].dt.hour.astype(int)
df_weather['temperature'] = hourly_temperature_2m
df_weather['humidite'] = hourly_relative_humidity_2m
df_weather['rayonnement_solaire'] = hourly_solar_radiation

print("Calcul des indicateurs Max, Min et Moyenne...")
df_daily_stats = df_weather.groupby('date_key').agg(
    temp_max=('temperature', 'max'),
    temp_min=('temperature', 'min'),
    temp_moy=('temperature', 'mean'),
    humidite_max=('humidite', 'max'),
    humidite_min=('humidite', 'min'),
    humidite_moy=('humidite', 'mean')
).reset_index()

df_final = pd.merge(df_weather, df_daily_stats, on='date_key', how='left')
df_final['date'] = df_final['date_time'].dt.strftime('%Y-%m-%d %H:%M:%S.000')

columns_to_keep = [
    'date', 'date_key', 'heure', 'temperature', 'humidite', 'rayonnement_solaire',
    'temp_max', 'temp_min', 'temp_moy', 
    'humidite_max', 'humidite_min', 'humidite_moy'
]
df_weather_clean = df_final[columns_to_keep]
df_weather_clean = df_weather_clean.rename(columns={'date': 'date'})

df_weather_clean.to_csv(output_weather_file, index=False)

print("\n[SUCCÈS] Le fichier de données météo a été généré avec succès !")
print(f"-> Nombre total de lignes : {df_weather_clean.shape[0]}")
print(f"-> Fichier enregistré à : {output_weather_file}")