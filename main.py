from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import logging


app = FastAPI()

# Cargo los datos procesados en formato .parquet
df_steam_games = pd.read_parquet('./Dataset/api-dataset/processed_steam_games.parquet')
df_users_reviews = pd.read_parquet('./Dataset/api-dataset/processed_user_reviews.parquet')
df_users_items = pd.read_parquet('./Dataset/api-dataset/processed_user_items.parquet')


#Primer endpoint
#Anda perfecto
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


#Segundo endpoint
#Le falta solo la parte de cantidad de items
@app.get("/userdata/")
def userdata(User_id: str):
    # Filtrar los items del usuario
    user_items = df_users_items[df_users_items['user_id'] == User_id]
    
    if user_items.empty:
        raise HTTPException(status_code=404, detail="User not found")

    # Obtener los IDs de los items que el usuario posee
    user_item_ids = user_items['item_id'].unique()
    
    # Filtrar los juegos correspondientes a esos item_ids
    user_games = df_steam_games[df_steam_games['id'].isin(user_item_ids)]
    
    # Calcular la cantidad de dinero gastado
    money_spent = user_games['price'].sum()
    
    # Filtrar las reviews del usuario
    user_reviews = df_users_reviews[df_users_reviews['user_id'] == User_id]
    
    # Calcular el porcentaje de recomendación
    if len(user_reviews) > 0:
        recommend_percentage = (user_reviews['sentiment_analysis'] == 2).mean() * 100
    else:
        recommend_percentage = 0.0
    
    # Calcular la cantidad de items
    items_count = user_items['item_id'].nunique()
    
    # Crear el diccionario de resultado
    result = {
        "Usuario X": User_id,
        "Dinero gastado": f"{money_spent:.2f} USD",
        "% de recomendación": f"{recommend_percentage:.2f}%",
        "cantidad de items": items_count
    }
    
    return result






#Tercer endpoint

@app.get("/UserForGenre/")
def UserForGenre(genero: str):
    try:
        # Filtrar los juegos por género
        genre_games = df_steam_games[df_steam_games['genres'].apply(lambda x: genero in x)]

        if genre_games.empty:
            raise HTTPException(status_code=404, detail="No se encontraron juegos para el género proporcionado")

        genre_game_ids = genre_games['id'].unique()

        # Filtrar los primeros 5000 registros de df_users_items
        subset_df_users_items = df_users_items.head(5000)

        # Filtrar los items de los usuarios que han jugado estos juegos
        genre_user_items = subset_df_users_items[subset_df_users_items['item_id'].isin(genre_game_ids)]

        if genre_user_items.empty:
            raise HTTPException(status_code=404, detail="No hay usuarios jugando juegos de este género")

        genre_games['release_year'] = pd.DatetimeIndex(genre_games['release_date']).year

        # Encontrar al usuario con más horas jugadas para este género
        top_user = genre_user_items.groupby('user_id')['playtime_forever'].sum().idxmax()

        # Calcular la acumulación de horas jugadas por año para el usuario con más horas jugadas
        user_hours_by_year = genre_user_items[genre_user_items['user_id'] == top_user].merge(genre_games[['id', 'release_year']], left_on='item_id', right_on='id')
        user_hours_by_year = user_hours_by_year.groupby('release_year').agg({'playtime_forever': 'sum'}).reset_index()
        user_hours_by_year = user_hours_by_year.rename(columns={'release_year': 'Año', 'playtime_forever': 'Horas'})

        result = {
            f"Usuario con más horas jugadas para {genero}": top_user,
            "Horas jugadas": user_hours_by_year.to_dict(orient='records')
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")



#Cuarto endpoint
@app.get("/best_developer_year/")
def best_developer_year(año: int):
    try:
        # Filtrar los juegos del año dado
        games_of_year = df_steam_games[df_steam_games['release_year'] == año]

        # Filtrar las reseñas positivas del año dado
        positive_reviews = df_users_reviews[(df_users_reviews['sentiment_analysis'] == 2) & (df_users_reviews['recommend'])]

        # Merge entre juegos y reseñas por el id del juego
        merged_data = games_of_year.merge(positive_reviews, left_on='id', right_on='item_id', how='inner')

        # Obtener el top 3 de desarrolladores más recomendados
        top_developers = merged_data.groupby('developer').size().nlargest(3).index.tolist()

        # Crear el mensaje con el año
        mensaje = f"Los 3 desarrolladores con juegos más recomendados para el año {año} son:"

        # Crear el resultado con la información de los 3 mejores desarrolladores
        result = [{
            'Puesto 1': top_developers[0],
            'Puesto 2': top_developers[1],
            'Puesto 3': top_developers[2]
        }]

        return {'Mensaje': mensaje, 'Desarrolladores': result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")


#Quinto endpoint
@app.get("/developer_reviews_analysis/")
def developer_reviews_analysis(desarrolladora: str):
    # Buscar el id del desarrollador en df_steam_games
    ids_desarrolladora = df_steam_games[df_steam_games['developer'] == desarrolladora]['id'].tolist()
    
    # Filtrar las reseñas según los juegos del desarrollador en df_users_reviews
    reseñas_desarrolladora = df_users_reviews[df_users_reviews['item_id'].isin(ids_desarrolladora)]
    
    # Contar la cantidad de reseñas positivas, negativas y neutrales
    positivas = reseñas_desarrolladora[reseñas_desarrolladora['sentiment_analysis'] == 2].shape[0]
    negativas = reseñas_desarrolladora[reseñas_desarrolladora['sentiment_analysis'] == 0].shape[0]
    
    # Crear diccionario con los resultados
    result = {desarrolladora: {'Negativas': negativas, 'Positivas': positivas}}
    
    return result



#MODELO MACHINE LEARNING  (Portal, Terraria, )
@app.get("/recomendacion_juego/")
def recomendacion_juego(item_name: str):
    cantidad_maxima = 10000  # Limitar a los primeros 10,000 registros
    
    try:
        logging.info(f"Solicitud recibida para recomendación de juego: {item_name}")
        
        clean_df = df_users_items.dropna(subset=['item_name']) 
        result = clean_df[clean_df['item_name'].str.contains(item_name, case=False)].head(cantidad_maxima)
        
        if not result.empty:
            genre = df_steam_games.loc[df_steam_games['app_name'] == item_name, 'genres'].iloc[0]
            
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(result['item_name'])
            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
            
            index = min(result.index[0], len(cosine_sim)-1)  # Índice asegurado dentro de los límites
            
            similar_games_indices = cosine_sim[index].argsort()[-6:-1][::-1]  # Índices de los juegos más similares
            top_juegos_recomendados = df_users_items.iloc[similar_games_indices]  # Juegos recomendados
            
            recommended_games_list = top_juegos_recomendados['item_name'].tolist()  # Lista de nombres de juegos recomendados
            
            return {"success": True, "message": f"Juegos similares recomendados para {item_name}", "juegos_recomendados": recommended_games_list}
        else:
            raise ValueError(f"{item_name} no encontrado en df_users_items")
        
    except Exception as e:
        logging.error(f"Error en la solicitud de recomendación de juego: {e}")
        return {"error": str(e)}, 500

