import spotipy
import spotipy.util as util
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from collections import deque
from tabulate import tabulate
from art import *


class SpotipyClient():
    client = None
    client_id = None
    client_secret = None
    username = None
    redirect_uri = 'http://localhost:8080'

    def __init__(self, client_id, client_secret, username, redirect_uri, scope):
        '''definir el constructor
            Argumentos:
                    client_id : Recibe el identificador de la aplicación creada en spotify for developers
                    client_secret : Recibe la clave secreta  de la aplicación creada en spotify for developers
                    username :Recibe el usuario que usara la aplicacion
                    redirect_uri : recibe la dirección donde se redireccionara la autentificacion 
                    scope : Recibe los permisos que se van a solicitar
            Return:
                    None 
        '''
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.redirect_uri = redirect_uri
        self.scope = scope


    def client_auth(self):
        '''Realizar Autenticación del usuario en la API de Spotify
        Argumentos:
                self.username : Recibe el usuario que usara la aplicacion
                self.scope : Recibe los permisos que se van a solicitar 
                self.client_id : Recibe el identificador de la aplicación creada en 'spotify for developers' que solicita los permisos
                self.client_secret: Recibe la clave secreta  de la aplicación creada en spotify for developers que solicita los permisos
                self.redirect_uri :  recibe la dirección donde se redireccionara la autentificacion 
        Return:
                si el usuario valido ó no 
        '''
        token = util.prompt_for_user_token(self.username,self.scope,
            self.client_id,self.client_secret,self.redirect_uri)
        self.client = spotipy.Spotify(auth=token)


    def get_top_tracks(self):
        ''' Realiza la obtencion del listado de pistas más escuchadas recientemente por el usuario
        Argumentos:
                None 
        Return:
                top_tracks: Listado con las canciones escuchadas ultimamente por el usuario      
        '''
        top_tracks = self.client.current_user_top_tracks(time_range='short_term', limit=20)
        return top_tracks


    def create_tracks_dataframe(self, top_tracks):
        ''' Realiza la obtención "audio features" de las pistas más escuchadas recientemente por el usuario
        Argumentos:
                top_tracks : Listado con las canciones escuchadas ultimamente por el usuario 
        Return:
                top_tracks_df:Dataframe con los metadatos de la canción      
        '''
        tracks = top_tracks['items']
        tracks_ids = [track['id'] for track in tracks]
        audio_features = self.client.audio_features(tracks_ids)
        top_tracks_df = pd.DataFrame(audio_features)
        top_tracks_df = top_tracks_df[["id", "acousticness", "danceability", 
            "duration_ms", "energy", "instrumentalness",  "key", "liveness", 
            "loudness", "mode", "speechiness", "tempo", "valence"]]
        return top_tracks_df
    

    def get_artists_ids(self, top_tracks):
        ''' Realiza la obtención de ids de los artistas en "top_tracks" 
        Argumentos:
                top_tracks : Listado con las canciones escuchadas ultimamente por el usuario 
        Return:
                ids_artists:Listado de artistas en el  'top_tracks '     
        '''
        ids_artists = []
        for item in top_tracks['items']:
            artist_id = item['artists'][0]['id']
            artist_name = item['artists'][0]['name']
            ids_artists.append(artist_id)
        # Depurar lista para evitar repeticiones
        ids_artists = list(set(ids_artists))
        return ids_artists
    

    def get_similar_artists_ids(self, ids_artists):
        ''' Realiza la obtención de ids de  artistas parecidos a los del "top_tracks" 
        Argumentos:
                ids_artists : Listado de artistas en el  'top_tracks' 
        Return:
                ids_artists: Se agregan los artistas y se devuelve la misma lista de entrada    
        '''
        ids_similar_artists = []
        for artist_id in ids_artists:
            artists = self.client.artist_related_artists(artist_id)['artists']
            for item in artists:
                artist_id = item['id']
                artist_name = item['name']
                ids_similar_artists.append(artist_id)
        ids_artists.extend(ids_similar_artists)
        # Depurar lista para evitar repeticiones
        ids_artists = list(set(ids_artists))
        return ids_artists
    

    def get_new_releases_artists_ids(self, ids_artists):
        ''' Realiza la obtención de ids de  artistas con nuevos lanzamientos parecidos a los del "top_tracks" 
        Argumentos:
                ids_artists : Listado de artistas en el  'top_tracks' 
        Return:
                ids_artists: Se agregan los artistas y se devuelve la misma lista de entrada    
        '''
        new_releases = self.client.new_releases(limit=20)['albums']
        for item in new_releases['items']:
            artist_id = item['artists'][0]['id']
            ids_artists.append(artist_id)
        # Depurar lista para evitar repeticiones
        ids_artists = list(set(ids_artists))
        return ids_artists
    

    def get_albums_ids(self, ids_artists):
        ''' Realiza la obtención de albums de  artistas en 'ids_artists'
        Argumentos:
                ids_artists : Listado de artistas en el  'top_tracks' 
        Return:
                ids_albums: Lista de albums de los artistas en el 'top_tracks'  
        '''
        ids_albums = []
        for id_artist in ids_artists:
            album = self.client.artist_albums(id_artist, limit=5)['items'][0]
            ids_albums.append(album['id'])
        return ids_albums


    def get_albums_tracks(self, ids_albums):
        ''' Realiza la obtención de pistas de cada álbum en 'ids_albums'
        Argumentos:
            ids_albums : Listado de álbumes de artistas en 'ids_albums' 
        Return:
            tracks_info: Lista de diccionarios con información de las pistas extraídas
        '''

        tracks_info = []
        ids_tracks = [] 

        for id_album in ids_albums:
            album_tracks = self.client.album_tracks(id_album, limit=2)['items']
            
            for track in album_tracks:
                track_info = {
                    'id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name']
                }
                tracks_info.append(track_info)

            for track in album_tracks:
                ids_tracks.append(track['id'])

            ids_tracks = [af for af in ids_tracks if af is not None]

        return ids_tracks,tracks_info


    def get_tracks_features(self, ids_tracks):
        ''' Realiza la obtención de audio features de cada track en "ids_tracks" y almacenar resultado
        en un dataframe de Pandas
        Argumentos:
                ids_tracks  : Listado de pisatas extraidas de cada album de 'ids_albums'
        Return:
                candidates_df : Dataframe con los metadatos de cada pista que se consideran candidatas para la recomendacion   
        '''
        ntracks = len(ids_tracks)

        if ntracks > 100:
            # Crear lotes de 100 tracks (limitacion de audio_features)
            m = ntracks//100
            n = ntracks%100
            lotes = [None]*(m+1)
            for i in range(m):
                lotes[i] = ids_tracks[i*100:i*100+100]

            if n != 0:
                lotes[i+1] = ids_tracks[(i+1)*100:]
        else:
            lotes = [ids_tracks]

        # Iterar sobre "lotes" y agregar audio features
        audio_features = []
        # filtro para evitar elementos None
        lotes = [a_f for a_f in lotes if a_f is not None]
        for lote in lotes:
            features = self.client.audio_features(lote)
            audio_features.append(features)

        audio_features = [item for sublist in audio_features for item in sublist]
    
        # filtro para evitar elementos None
        audio_features = [a_f for a_f in audio_features if a_f is not None]
        # Crear dataframe
        candidates_df = pd.DataFrame(audio_features)
        candidates_df = candidates_df[["id", "acousticness", "danceability", "duration_ms",
            "energy", "instrumentalness",  "key", "liveness", "loudness", "mode", 
            "speechiness", "tempo", "valence"]]

        return candidates_df
    
   
    def custom_pca(self, dataframe, n_components):
        """
        Implementación del Análisis de Componentes Principales (PCA) en un DataFrame.

        Argumentos:
            - dataframe: Un DataFrame de pandas que contiene los datos.
            - n_components: El número de componentes principales que se deben retener.

        Return:
            - Los datos transformados con PCA como un DataFrame.
        """
        # Convertir el DataFrame a una matriz numpy
        data = dataframe.iloc[:, 1:].to_numpy()
        # Calcular la media de los datos
        mean_vector = np.mean(data, axis=0)
        # Centrar los datos restando la media
        centered_data = data - mean_vector
        # Calcular la matriz de covarianza
        covariance_matrix = np.cov(centered_data, rowvar=False)
        # Calcular los autovalores y autovectores de la matriz de covarianza
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # Ordenar los autovalores y autovectores en orden descendente
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        # Tomar los primeros n_components autovectores
        top_eigenvectors = eigenvectors[:, :n_components]
        # Proyectar los datos originales en el nuevo espacio
        transformed_data = np.dot(centered_data, top_eigenvectors)
        # Crear un DataFrame con los datos transformados
        transformed_dataframe = pd.DataFrame(transformed_data, columns=[f'PC{i+1}' for i in range(n_components)])
        
        return transformed_dataframe
    
    
    def custom_cossim(self, top_tracks_df, candidates_df):
        '''Calcula la similitud del coseno entre cada top_track y cada pista
        candidata en candidates_df. Retorna matriz de n_top_tracks x n_candidates_df'''
        print(text2art("3.  Metrica ", font="big"))

        top_tracks_mtx = top_tracks_df.iloc[:,1:].values
        candidates_mtx = candidates_df.iloc[:,1:].values

        # Estandarizar cada columna de features: mu = 0, sigma = 1
        scaler = StandardScaler()
        top_tracks_scaled = scaler.fit_transform(top_tracks_mtx)
        can_scaled = scaler.fit_transform(candidates_mtx)

        # Normalizar cada vector de características (magnitud resultante = 1)
        top_tracks_norm = np.sqrt((top_tracks_scaled*top_tracks_scaled).sum(axis=1))
        can_norm = np.sqrt((can_scaled*can_scaled).sum(axis=1))

        n_top_tracks = top_tracks_scaled.shape[0]
        n_candidates = can_scaled.shape[0]
        top_tracks = top_tracks_scaled/top_tracks_norm.reshape(n_top_tracks,1)
        candidates = can_scaled/can_norm.reshape(n_candidates,1)

        # Calcular similitudes del coseno
        cos_sim = linear_kernel(top_tracks,candidates)

        return cos_sim


 #                          aprendizaje no supervisado
    def cluster_tracks(self, top_tracks_df, n_clusters):
        ''' Realiza el clustering de las pistas de música basado en características.
        Argumentos:
            top_tracks_df : DataFrame con las características musicales de las pistas.
            n_clusters : El número de clústeres a crear.
        Return:
            cluster_labels : Una lista de etiquetas de clúster para cada pista.
        '''
        print(text2art("4.  Aprendizaje  ", font="big"))

        # Extraer las características relevantes para el clustering
        features = top_tracks_df.iloc[:, 1:].values
        # Crear una instancia del modelo K-Means con el número de clústeres deseado
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        # Entrenar el modelo K-Means
        kmeans.fit(features)
        # Obtener las etiquetas de clúster para cada pista
        cluster_labels = kmeans.labels_
        print(cluster_labels)
        return cluster_labels
    

    def busqueda_en_anchura_top_20(self, pca_top_canciones_df, pca_canciones_candidatas_df, n_clusters, umbral):
        '''Realizar una búsqueda en anchura (BFS) basada en similitud de coseno
           para encontrar una canción candidata en la lista de candidatas para cada canción del top 20
           que cumpla ciertas condiciones y comparar con las canciones candidatas.
        Argumentos:
            - pca_top_canciones_df: DataFrame con las características musicales del top 20.
            - pca_canciones_candidatas_df: DataFrame con las características de las canciones candidatas.
            - cos_sim: Matriz de similitud coseno.
            - n_clusters: Número de clústeres.
            - umbral: Umbral para seleccionar canciones candidatas.
        Return:
            -canciones_candidatas: Lista de índices de canciones candidatas para cada canción del top 20.
        '''

        # Obtener las etiquetas de clúster para todas las canciones del top 20 y candidatas
        Kmeans_top_20_df = self.cluster_tracks(pca_top_canciones_df, n_clusters)
        Kmeans_candidatas_df = self.cluster_tracks(pca_canciones_candidatas_df, n_clusters)
        cos_sim = self.custom_cossim(pca_top_canciones_df,pca_canciones_candidatas_df)
        
        # Inicializar la lista para almacenar las canciones candidatas para cada canción del top 20
        recomendaciones =[]

        # Iterar sobre cada canción del top 20
        for fila in range(pca_top_canciones_df.shape[0]): 

            # Obtener el índice de la canción actual en el top 20
            indice_cancion_actual = pca_top_canciones_df.index[fila]
            
            # Inicializar una cola de prioridad (ordenada por similitud de coseno)
            cola_prioridad = deque([(indice_cancion_actual, 1.0)])  # Tuplas (índice, similitud)
            visitados = set()  # Para evitar ciclos

            # Variable para almacenar la canción candidata que cumple las condiciones
            cancion_candidata_actual = None

            while cola_prioridad and not cancion_candidata_actual:
             #el bucle continuará mientras  la cola de prioridad no este vacia 
             # y mientras no se haya encontrado una canción candidata que cumple con las condiciones
                # Extraer el elemento de la cola con la mayor similitud
                indice_actual, similitud_actual = cola_prioridad.pop()
                # Marcar como visitado
                visitados.add(indice_actual)

                # Si la similitud es mayor que el umbral y la canción no es la canción de referencia
                if similitud_actual >= umbral and indice_actual != indice_cancion_actual:
                    cancion_candidata_actual = indice_actual

                # Obtener las canciones vecinas (todas las canciones en el mismo clúster)
                vecinos = [i for i, etiqueta in enumerate(Kmeans_candidatas_df) if etiqueta == Kmeans_top_20_df[indice_cancion_actual]]
                # Ordenar las canciones vecinas por similitud de coseno en orden descendente
                vecinos.sort(key=lambda x: cos_sim[fila][x], reverse=True)
                # Agregar las canciones vecinas no visitadas a la cola de prioridad
                for vecino in vecinos:
                    if vecino not in visitados:
                        cola_prioridad.appendleft((vecino, cos_sim[fila][vecino]))

            # Agregar la comparación de similitud con la canción candidata
            if cancion_candidata_actual is not None and cancion_candidata_actual  not in recomendaciones:
                similitud_candidata = cos_sim[indice_cancion_actual][cancion_candidata_actual]
                if similitud_candidata >= umbral:
                    recomendaciones.append(cancion_candidata_actual)
        return recomendaciones
