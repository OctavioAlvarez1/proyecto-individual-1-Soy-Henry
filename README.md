![Banner](https://github.com/OctavioAlvarez1/proyecto-individual-1-Soy-Henry/blob/main/Images/henry.jfif)
<h1 align="center">DATA SCIENCE - PROYECTO INDIVIDUAL Nº1</h1>
<h1 align="center">Machine Learning Operations (MLOps)</h1>
Este proyecto abarca una serie de pasos para desarrollar un proceso de <strong>Data Engineering</strong> sobre un dataset de juegos de la plataforma Steam, para posteriormente tener a disposición un conjunto de endpoints y un modelo de recomendación de juegos utilizando <strong>Machine Learning</strong>, a través de una <strong>API</strong>.

![Imagen](https://github.com/OctavioAlvarez1/proyecto-individual-1-Soy-Henry/blob/main/Images/Mlops.png)

<h1>Contexto</h1>
La empresa (Steam) pide que se cree un sistema de recomendación de videojuegos para sus usuarios. Se plantea desde los depártamentos de Machine Learning y Analytics la necesidad de contar con los datos en una API para poder ser consumidos. Por otro, lado existe la necesidad de poder realizar las consultas al modelo de recomendación para lo cual resulta necesario hacer un deploy de la API.

<h1>Dataset</h1>
El proyecto cuenta con tres <a href="https://github.com/OctavioAlvarez1/proyecto-individual-1-Soy-Henry/tree/main/Dataset/api-dataset">datasets</a>:

  * <strong>processed_steam_games.parquet</strong>: reúne información acerca de los desarrolladores y sus juegos.
  
  * <strong>processed_user_items.parquet</strong>: reúne información sobre los usuarios, los juegos que juegan y la cantidad de horas que juegan.
    
  * <strong>processed_user_reviews.parquet</strong>: reúne información sobre las reviews que hacen los usuarios sobre los juegos.

<h1>Data Engineering</h1>

Para el trabajo de <strong>Data Engineering</strong> se procedió a efectuar una serie de transformaciones solicitadas sobre los datos, dentro de las cuales se encuentran, entre otras:

  * Se quitaron las columnas "url" y "reviews_url" de df_steam_games por considerarlas innecesarias para el análisis.
  
  * Se analizó la cantidad de datos nulos y se procedió a borrarlos para mejorar el análisis.
    
  * Se transformó la columna "release_date" a tipo date para un mejor análisis.

  * Se transformó la columna "id" de df_steam_games a tipo int para un mejor análisis.

  * Se desanidaron columnas en df_reviews para generar columnas nuevas que permitieran una mejora en el análisis.

  * Se reemplazo la columna "reviews" en df_reviews por la columna "sentiment_analysis" para lograr un mejor análisis de los datos.

  * Se generaron achivos con formato .parquet para que los datos sean más livianos para ser interpretados en la API.

Todas las transformaciones hechas pueden verificarse en el siguiente <a href="https://github.com/OctavioAlvarez1/proyecto-individual-1-Soy-Henry/blob/main/etl_data.ipynb">archivo</a>

<h1>API</h1>

Se solicitó efectuar la disponibilización de los siguientes endpoints a través del Framework <strong>FastAPI</strong>:

 * def developer( desarrollador : str ): Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora.  

 * def userdata( User_id : str ): Debe devolver cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.
   Ejemplo de retorno: {"Usuario X" : us213ndjss09sdf, "Dinero gastado": 200 USD, "% de recomendación": 20%, "cantidad de items": 5}

 * def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
   Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf, "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}

 * def best_developer_year( año : int ): Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos)
   Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

 * def developer_reviews_analysis( desarrolladora : str ): Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros
   de reseñas de usuarios que se encuentren categorizados con un análisis    de sentimiento como valor positivo o negativo.
   Ejemplo de retorno: {'Valve' : [Negative = 182, Positive = 278]}

El código para correr la API dentro de FastAPI se puede visualizar <a href="https://github.com/OctavioAlvarez1/proyecto-individual-1-Soy-Henry/blob/main/main.py">acá</a>

<h1>Análisis exploratorio de datos (EDA)</h1>

Para poder lograr una mejor comprensión en el análisis de los datos se procedió a realizar un análisi exploratorio de los mismos.

Entre las diversas acciones que se tomaron, se contaron los datos nulos y se procedió a borrar los mismos para tener un mejor análisis. Se procedió a ver el tipo de dato correspondiente
a cada columna y se chequeó que no existiesen valores duplicados.

Por otra parte, se analizaron los titulos de los juegos mediante una nube de palabras, se establecerieron dos matrices de correlaciones, y se llevaron adelante distintas combinaciones
de datos que permitieron arrojar gráficos que contuvieran datos interesantes como: los números de juegos lanzados por año, la cantidad de juegos por género, el top 10 de desarrolladores de juegos,
el top 10 de los juegos más jugados, y el top 10 de los juegos más vendidos

Para visualizar el EDA completo podes ingresar a este <a href="https://github.com/OctavioAlvarez1/proyecto-individual-1-Soy-Henry/blob/main/eda_data.ipynb">link</a>

<h1>Modelo de recomendación - Machine Learning</h1>

La consigna planteaba la idea de crear un sistema de recomendación basado en Machine Learning, siguiendo el modelo de similitud del coseno. 
Para ello, se planteaban dos propuestas de trabajo: En la primera, el modelo deberá tener una relación ítem-ítem, esto es se toma un item, en base a que tan similar esa ese ítem al resto, se recomiendan similares. Acá el input
es un juego y el output es una lista de juegos recomendados. 
La otra propuesta para el sistema de recomendación debe aplicar el filtro user-item, esto es tomar un usuario, se encuentran usuarios similares y se recomiendan ítems que a esos usuarios similares les gustaron. 
En este caso el input es un usuario y el output es una lista de juegos que se le recomienda a ese usuario, en general se explican como “A usuarios que son similares a tí también les gustó…”. 

Para el sistema de recomendación item-item se definía la siguiente función:

* def recomendacion_juego( id de producto ): Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.

Para el sistema de recomendación user-item se definía la siguiente función:

* def recomendacion_usuario( id de usuario ): Ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.

Para ver el desarrollo completo del modelo podés ingresar en este <a href="https://github.com/OctavioAlvarez1/proyecto-individual-1-Soy-Henry/blob/main/main.py">link</a>

<h1>Deployment</h1>

Para el deploy de la API, se utilizó la plataforma Render. Los datos están listos para ser consumidos y consultados a partir del siguiente link

<a href="https://proyecto-individual-1-soy-henry.onrender.com/docs">Link al deployment</a>
