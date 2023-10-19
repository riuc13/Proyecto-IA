#frontend

from backend import*

# función para esperar
def Esperar():
    input("\nPresiona Enter para continuar...")


#Titulo 
proyecto= text2art("   Proyecto  ", font="epic")
ia = text2art(" IA ",font="block")
print(f'''{proyecto}  {ia}''')
Esperar()

#                       optencion de datos

print(text2art("1. Obtencion  de  Datos ", font="big"))
Esperar()

# Se crea una instancia de la clase SpotipyClient del modulo backend

#client_secret= '9542debab94c44aaba5d3ec2ceea10cd'
#client_id= '66a5b895bc934b77a8d17a4adce9a0d4'

#client_id ='31231b6ca182439694951d82e14ac175'
#client_secret = '8254e1922d824e9bbe8a0aa809460db6'

#client_id = 'a6b60f79f9424686946a5539e6ca540d'
#client_secret = 'c4be2a803a5d479e9ce6538c03cb429c'

#client_secret = 'b8d239e1122b4a28abb2785959d3ce72'
#client_id = '1c613bddfc8743aaab782e8693297c2a'


#client_id ='23ef65fb855c4abdb10232071110a7a9'
#client_secret = '6fb5390f1bc44c6b8b77969bf5a7082a'

client_secret='8ef6b8a6fdcd4846bb73cbcf9354a683'
client_id='efd3ef83d220473d849cce9f22835004'

user_name ='31j2eafa55r2ofbce6aiyqjrlmei'
redirect_uri = 'http://localhost:8080'
scope = 'playlist-modify-private,playlist-modify-public,user-top-read'

Spotify = SpotipyClient(client_id,client_secret,user_name,redirect_uri,scope)
####################################### obtencion de datos ########################################
#autenticamos al usuario 
Spotify.client_auth()
top_20 = Spotify.get_top_tracks()

#mostrar  las  20 caciones más escuchadas por el/la usuari@
print(text2art("  Top  20", font="bubble"))
for i, item in enumerate(top_20['items']):
    print(f"""      {i+1}: {item['name']} <--> {item['artists'][0]['name']} """)

#crear el dataframe con las ultimas 20 caciones
dataframe_top_20 = Spotify.create_tracks_dataframe(top_20)

#imprimir el dataframe con las ultimas 20 caciones
print(text2art("  DataFrame  top  20", font="bubble"))
print(tabulate(dataframe_top_20.head(), headers='keys', tablefmt='pretty'))

#extraer los artistas del top 20
ids_artistas = Spotify.get_artists_ids(top_20)

#extraer  artistas  similares a los del top 20
ids_artistas = Spotify.get_similar_artists_ids(ids_artistas)

#extraer  artistas con nuevos lanzamientos similares a los del top 20
ids_artistas = Spotify.get_new_releases_artists_ids(ids_artistas)
print(f" Num artistas: {len(ids_artistas)}")

#extraer albums de los  artistas en ids_artistas
ids_albums = Spotify.get_albums_ids(ids_artistas)

#extraer id e informacion(nombre,artista) de canciones de los albums en id_albums
ids_pista,pistas_candidatas_info = Spotify.get_albums_tracks(ids_albums)
print(f"Num pistas Candidatas: {len(ids_pista)}")

# extraer los metadatos de las pistas  el cual es  un dataframe
pistas_candidatas_df = Spotify.get_tracks_features(ids_pista)
print(text2art("  DataFrame pistas Candidatas: ", font="bubble"))
print(tabulate(pistas_candidatas_df.head(), headers='keys', tablefmt='pretty'))

#mostrar  las  caciones en la lista de candidatos 
print(" Nombres y artistas de las canciones candidatas:  ")
for i, item in enumerate(pistas_candidatas_info):
    print(f"""      {i+1}: {item['name']} <--> {item['artist']} """)
############################################## Fin ################################################

########################################## optimización ###########################################
print(text2art("2.  optimizacion  ", font="big"))
Esperar()
n_components = 5 # numero de componetes principales deseadas 
pca_top_20_transforme = Spotify.calculo_pca(dataframe_top_20,n_components ) #aplicar pca top_20
print('\npca_top_20_transforme\n',pca_top_20_transforme.head(),
      '\ntamaño:',pca_top_20_transforme.shape)

pca_candidatas_transforme = Spotify.calculo_pca(pistas_candidatas_df,n_components) #aplicar pca en candidatas
print('\npca_candidatos_transforme\n',pca_candidatas_transforme.head(),
      '\ntamaño:',pca_candidatas_transforme.shape)
############################################## Fin ################################################


################################### aprendizaje no supervisado ####################################
print(text2art("3.  Aprendizaje  ", font="big"))
Esperar()
n_clusters = 5 # numero de clusters 

Kmeans_top_20 = Spotify.cluster_tracks(pca_top_20_transforme, n_clusters)# aplicar kmeans a top_20
print(f'k-means top20:\n {Kmeans_top_20}')
Kmeans_candidatas = Spotify.cluster_tracks(pca_candidatas_transforme, n_clusters)# aplicar kmeans a candidatas
print(f'k-means candidatas:\n {Kmeans_candidatas}')

############################################## Fin ################################################


############################################# Metrica #############################################
print(text2art("4.  Metrica ", font="big"))
Esperar()
cos_sim = Spotify.simil_cos(pca_top_20_transforme,pca_candidatas_transforme)# calcular la matriz de similitud
print(" Matriz de similitud top_20 X candidatas")
print(cos_sim)
#Recorrer la matriz para mostrar en el mapa de color las 20 filas y solo 20 columnas
rows_to_show = 20 # filas a mostrar 
columns_to_show = 50 # columnas a mostrar 
subset_cos_sim = [cos_sim[i][:columns_to_show] for i in range(rows_to_show)]# sub matriz a mostrar

# se crea un mapa de calor (heatmap) para visualizar la similitud de coseno
sns.heatmap(subset_cos_sim, annot=False, cmap='viridis')
# Mostrar el título y el mapa de calor
plt.title('Similitud de Coseno top_20 y Candidatas')
plt.show()
############################################## Fin ################################################


############################################ Busqueda #############################################
print(text2art("5.  Busqueda ", font="big"))
Esperar()
umbral = 0.7 # unbral de similitud
i_recomendaciones = Spotify.BFS(pca_top_20_transforme, 
                                Kmeans_top_20, 
                                Kmeans_candidatas,
                                cos_sim, 
                                umbral)# realizar la busqueda de 1 cancion similar para cada cancion del top20 si la hay
print('i_recomendaciones\n',i_recomendaciones,'\ntamaño:',len(i_recomendaciones ))#mostrar

#se busacan los indices resultantes en el dataframe de pistas candidatas
canciones_seleccionadas = pistas_candidatas_df.iloc[i_recomendaciones]
print(text2art('recomendaciones\n',font="bubble"),canciones_seleccionadas,'\ntamaño:',canciones_seleccionadas.shape)#mostrar 

#se busaca la info de los indices  resultantes en  de pistas_candidatas_info
info_canciones_seleccionadas = [pistas_candidatas_info[i] for i in i_recomendaciones]
print(text2art('recomendaciones\n',font="bubble"),len(info_canciones_seleccionadas))
for i, item in enumerate(info_canciones_seleccionadas):
    print(f"{i + 1}: {item['name']} <--> {item['artist']}")#mostrar

############################################## Fin ################################################

############################################ playlist #############################################
print(text2art("6. Creacion Playlist ", font="big"))
Esperar()
ids_playlist = [] # lista para almacenar las ids de las canciones recomendadas
for indice in i_recomendaciones:
    # Obtiene la ID de la canción candidata correspondiente al índice
    id_cand = pistas_candidatas_df['id'][indice]
    # Agrega la ID de la canción a la lista de la playlist
    ids_playlist.append(id_cand)

# se crea la playlist en Spotify
playlist = Spotify.client.user_playlist_create(user=Spotify.username,
    name='Recomendados IA',
    description='Playlist creada con el sistema de recomendación')
Spotify.client.playlist_add_items(playlist['id'], ids_playlist)

print(text2art("  Playlist creada ", font="bubble"))
print(text2art("   Vuelve Pronto  ", font="epic"))

############################################## Fin ################################################
