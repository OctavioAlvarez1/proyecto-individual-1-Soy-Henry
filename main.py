from fastapi import FastAPI, HTTPException
import pandas as pd

app = FastAPI()

# Cargar los datos procesados desde el archivo .parquet
df_steam_games = pd.read_parquet('Dataset/api-dataset/processed_steam_games.parquet')

@app.get("/developer/")
def developer(desarrollador: str):
    # Filtrar el DataFrame por desarrollador
    df_filtered = df_steam_games[df_steam_games['developer'] == desarrollador]
    
    # Convertir la columna 'release_date' a tipo datetime
    df_filtered['release_date'] = pd.to_datetime(df_filtered['release_date'])
    
    # Extraer el año de lanzamiento
    df_filtered['release_year'] = df_filtered['release_date'].dt.year
    
    # Contar la cantidad de items por año
    items_por_anio = df_filtered.groupby('release_year').size().reset_index(name='Cantidad de Items')
    
    # Calcular el porcentaje de contenido Free por año
    contenido_free_por_anio = (df_filtered['price'] == 'Free To Play').groupby(df_filtered['release_year']).mean() * 100
    contenido_free_por_anio = contenido_free_por_anio.reset_index(name='Contenido Free')
    
    # Combinar los resultados en un DataFrame final
    resultado = pd.merge(items_por_anio, contenido_free_por_anio, on='release_year', how='left').fillna(0)
    
    # Convertir el porcentaje de contenido Free a string con el formato adecuado
    resultado['Contenido Free'] = resultado['Contenido Free'].astype(str) + '%'
    
    # Convertir el DataFrame a formato JSON y devolverlo como respuesta
    return resultado.to_dict(orient='records')
