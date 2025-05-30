import pandas as pd
import sqlite3
from datetime import datetime

# Cargar CSV
df = pd.read_csv("Muestra Tec Estudio de Precios Airbnb 2025_05_26 - 3 meses data.csv")  # cambia el nombre al correcto


columnas_metadata = [
    "Nombre", "#", "User", "Puntaje", "Profile", "Link",
    "Limpieza", "Veracidad", "Check-in", "Comunicación",
    "Ubicación", "Calidad", "Reviews", "Tarifa", "Cleaning",
    "Habs", "Jacuzzi", "Noches 05", "Noches 06", "Noches 07"
]

columnas_fecha = []
for col in [col for col in df.columns if col not in columnas_metadata]:

    datetime.strptime(col.strip(), "%d/%m/%Y")
    columnas_fecha.append(col)

columnas_metadata = [col for col in columnas_metadata if col in df.columns]

df["ID"] = df.index + 1  

df_metadata = df[["ID"] + columnas_metadata]

df_fechas = df[["ID"] + columnas_fecha].melt(
    id_vars="ID",
    var_name="Fecha",
    value_name="Valor"
)
df_fechas["Fecha"] = pd.to_datetime(df_fechas["Fecha"], format="%d/%m/%Y")


conn = sqlite3.connect("airbnb_data.sqlite")
df_metadata.to_sql("Inmueble", conn, if_exists="replace", index=False)
df_fechas.to_sql("Fecha", conn, if_exists="replace", index=False)
conn.close()

