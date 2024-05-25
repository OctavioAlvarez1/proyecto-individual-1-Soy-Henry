from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import logging


app = FastAPI(title="Primer proyecto individual - Soy Henry")

# Cargo los datos procesados en formato .parquet
df_steam_games = pd.read_parquet('Dataset/api-dataset/processed_steam_games.parquet')
df_users_reviews = pd.read_parquet('Dataset/api-dataset/processed_user_reviews.parquet')
df_users_items = pd.read_parquet('Dataset/api-dataset/processed_user_items.parquet')


#Primer endpoint
@app.get("/developer/")
def developer(developer: str):
    # Filtro el DataFrame por desarrollador
    df_filtered = df_steam_games[df_steam_games['developer'] == developer]
    
    # Extraigo el año de lanzamiento
    df_filtered['release_year'] = df_filtered['release_date'].dt.year
    
    # Cuento la cantidad de elementos por año
    items_per_year = df_filtered.groupby('release_year').size().reset_index(name='Numero de items')
    
    # Calculo el porcentaje de contenido gratuito por año
    free_content_per_year = (df_filtered['price'] == 'Free To Play').groupby(df_filtered['release_year']).mean() * 100
    free_content_per_year = free_content_per_year.reset_index(name='Free Content')
    
    # Combino los resultados en un DataFrame final
    result = pd.merge(items_per_year, free_content_per_year, on='release_year', how='left').fillna(0)
    
    # Convierto el porcentaje de contenido gratuito a cadena con el formato apropiado
    result['Free Content'] = result['Free Content'].astype(str) + '%'
    
    # Convierto el DataFrame a formato JSON y lo devuelvo como respuesta
    return result.to_dict(orient='records')



#Segundo endpoint
@app.get("/userdata/")
def userdata(User_id: str):
    # Filtro los items del usuario
    user_items = df_users_items[df_users_items['user_id'] == User_id]
    
    if user_items.empty:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    # Obtengo los IDs de los items que el usuario posee
    user_item_ids = user_items['item_id'].unique()
    
    # Filtro los juegos correspondientes a esos item_ids
    user_games = df_steam_games[df_steam_games['id'].isin(user_item_ids)]
    
    # Calculo la cantidad de dinero gastado
    money_spent = user_games['price'].sum()
    
    # Filtro las reviews del usuario
    user_reviews = df_users_reviews[df_users_reviews['user_id'] == User_id]
    
    # Calculo el porcentaje de recomendación
    if len(user_reviews) > 0:
        recommend_percentage = (user_reviews['sentiment_analysis'] == 2).mean() * 100
    else:
        recommend_percentage = 0.0
    
    # Calculo la cantidad de items
    items_count = user_items['item_id'].nunique()
    
    # Creo el diccionario de resultado
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
        # Filtro los juegos por género
        genre_games = df_steam_games[df_steam_games['genres'].apply(lambda x: genero in x)]

        if genre_games.empty:
            raise HTTPException(status_code=404, detail="No se encontraron juegos para el género proporcionado")

        genre_game_ids = genre_games['id'].unique()

        # Filtro los primeros 5000 registros de df_users_items
        subset_df_users_items = df_users_items.head(5000)

        # Filtro los items de los usuarios que han jugado estos juegos
        genre_user_items = subset_df_users_items[subset_df_users_items['item_id'].isin(genre_game_ids)]

        if genre_user_items.empty:
            raise HTTPException(status_code=404, detail="No hay usuarios jugando juegos de este género")

        genre_games['release_year'] = pd.DatetimeIndex(genre_games['release_date']).year

        # Encuentro al usuario con más horas jugadas para este género
        top_user = genre_user_items.groupby('user_id')['playtime_forever'].sum().idxmax()

        # Calculo la acumulación de horas jugadas por año para el usuario con más horas jugadas
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
        # Filtro los juegos del año dado
        games_of_year = df_steam_games[df_steam_games['release_year'] == año]

        # Filtro las reseñas positivas del año dado
        positive_reviews = df_users_reviews[(df_users_reviews['sentiment_analysis'] == 2) & (df_users_reviews['recommend'])]

        # Merge entre juegos y reseñas por el id del juego
        merged_data = games_of_year.merge(positive_reviews, left_on='id', right_on='item_id', how='inner')

        # Obtengo el top 3 de desarrolladores más recomendados
        top_developers = merged_data.groupby('developer').size().nlargest(3).index.tolist()

        # Creo el mensaje con el año
        message = f"Los 3 desarrolladores con juegos más recomendados para el año {año} son:"

        # Creo el resultado con la información de los 3 mejores desarrolladores
        result = [{
            'Puesto 1': top_developers[0],
            'Puesto 2': top_developers[1],
            'Puesto 3': top_developers[2]
        }]

        return {'Mensaje': message, 'Desarrolladores': result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")


#Quinto endpoint
@app.get("/developer_reviews_analysis/")
def developer_reviews_analysis(developer: str):
    # Buscar el ID del desarrollador en df_steam_games
    developer_ids = df_steam_games[df_steam_games['developer'] == developer]['id'].tolist()
    
    # Filtrar las reseñas basadas en los juegos del desarrollador en df_users_reviews
    developer_reviews = df_users_reviews[df_users_reviews['item_id'].isin(developer_ids)]
    
    # Contar el número de reseñas positivas, negativas y neutrales
    positive_reviews = developer_reviews[developer_reviews['sentiment_analysis'] == 2].shape[0]
    negative_reviews = developer_reviews[developer_reviews['sentiment_analysis'] == 0].shape[0]
    
    # Crear un diccionario con los resultados
    result = {developer: {'Negativas': negative_reviews, 'Positivas': positive_reviews}}
    
    return result


#MODELO MACHINE LEARNING  

#Funcion para recomendación de juego
@app.get("/recomendacion_juego/")
def recomendacion_juego(item_name: str):
    cantidad_maxima = 10000  # Limito a los primeros 10,000 registros
    
    try:
        logging.info(f"Solicitud recibida para recomendación de juego: {item_name}")
        
        clean_df = df_users_items.dropna(subset=['item_name']) 
        result = clean_df[clean_df['item_name'].str.contains(item_name, case=False)].head(cantidad_maxima)
        
        if not result.empty:
            genre = df_steam_games.loc[df_steam_games['app_name'] == item_name, 'genres'].iloc[0]
            
            # Creo una matrix TF-IDF
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(result['item_name'])

            # Calculo la similitud del coseno
            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
            
            # Índice asegurado dentro de los límites
            index = min(result.index[0], len(cosine_sim)-1)  
            
            # Índices de los juegos más similares
            similar_games_indices = cosine_sim[index].argsort()[-6:-1][::-1] 

            # Juegos recomendados
            top_recommended_games = df_users_items.iloc[similar_games_indices]  
            
             # Lista de juegos recomendados
            recommended_games_list = top_recommended_games['item_name'].tolist() 
            
            return {"success": True, "message": f"Juegos similares recomendados para {item_name}", "juegos recomendados": recommended_games_list}
        else:
            raise ValueError(f"{item_name} no se encuentra en df_users_items")
        
    except Exception as e:
        logging.error(f"Error en la recomendación: {e}")
        return {"error": str(e)}, 500


# Función para recomendación de usuario
@app.get("/recomendacion_usuario/")
def recomendacion_usuario(user_id: str):
    maximum_quantity = 10000  # Limito a los primeros 10,000 registros
    
    try:
        logging.info(f"Solicitud recibida para recomendación de usuario: {user_id}")
        
        clean_df = df_users_reviews.dropna(subset=['user_id']) 
        user_reviews = clean_df[clean_df['user_id'] == user_id].head(maximum_quantity)
        
        if not user_reviews.empty:
            # Filtro usuarios similares
            similar_users = find_similar_users(user_id)
            
            # Obtengo juegos preferidos por usuarios similares
            recommended_games = get_recommended_games(similar_users)
            
            return {"success": True, "message": f"Juegos recomendados para el usuario {user_id}", "juegos_recomendados": recommended_games}
        else:
            raise ValueError(f"Las reseñas del usuario {user_id} no fueron encontradas en df_users_reviews")
        
    except Exception as e:
        logging.error(f"Error en la solicitud de recomendación de usuario: {e}")
        return {"error": str(e)}, 500
    
# Función para encontrar usuarios similares
def find_similar_users(user_id):
    # Obtengo reseñas del usuario
    user_reviews = df_users_reviews[df_users_reviews['user_id'] == user_id]
    
    # Filtro usuarios que también han revisado los mismos juegos
    similar_users = df_users_reviews[df_users_reviews['item_id'].isin(user_reviews['item_id'])]
    
    return similar_users['user_id'].unique()

# Función para obtener juegos recomendados para el usuario
def get_recommended_games(similar_users):
    # Obtengo juegos preferidos por usuarios similares
    recommended_games = df_users_reviews[df_users_reviews['user_id'].isin(similar_users)]
    
    # Cuento las ocurrencias de cada juego y ordenar por la cantidad de veces que fue revisado
    recommended_games = recommended_games.groupby('item_id').size().reset_index(name='count')
    recommended_games = recommended_games.sort_values(by='count', ascending=False)
    
    # Recomiendo los nombres de los juegos recomendados
    recommended_game_ids = recommended_games['item_id'].head(5)
    recommended_games_names = df_steam_games[df_steam_games['id'].isin(recommended_game_ids)]['title'].tolist()
    
    return recommended_games_names