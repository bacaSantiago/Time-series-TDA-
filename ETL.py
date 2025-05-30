import pandas as pd
import sqlite3
from datetime import datetime


try:
    df = pd.read_csv("Muestra Tec Estudio de Precios Airbnb 2025_05_26 - 3 meses data.csv")
    limit = datetime.strptime("1/8/2025", "%d/%m/%Y")
    filtered_columns = []
    for col in df.columns:
        try:
            fecha_col = datetime.strptime(col.strip(), "%d/%m/%Y")
            if fecha_col <= limit:
                filtered_columns.append(col)
        except ValueError:
            filtered_columns.append(col)
    df = df[filtered_columns]
    conn = sqlite3.connect("airbnb_sql.sqlite")
    df.to_sql("airbnb_db", conn, if_exists="replace", index=False)
    conn.close()
except Exception as E:
    print("Error al cargar los datos rápidos: ", E)
