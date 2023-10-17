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
#optencion de datos
datos = text2art("1. Optencion  de  Datos ", font="big")
print(datos)
Esperar()
###################################################
# Se crea una instancia de la clase SpotipyClient del modulo backend

#client_secret= '9542debab94c44aaba5d3ec2ceea10cd'
#client_id= '66a5b895bc934b77a8d17a4adce9a0d4'

#client_id ='31231b6ca182439694951d82e14ac175'
#client_secret = '8254e1922d824e9bbe8a0aa809460db6'

#client_id = 'a6b60f79f9424686946a5539e6ca540d'
#client_secret = 'c4be2a803a5d479e9ce6538c03cb429c'

client_secret = 'b8d239e1122b4a28abb2785959d3ce72'
client_id = '1c613bddfc8743aaab782e8693297c2a'


#client_id ='23ef65fb855c4abdb10232071110a7a9'
#client_secret = '6fb5390f1bc44c6b8b77969bf5a7082a'

#client_secret='8ef6b8a6fdcd4846bb73cbcf9354a683'
#client_id='efd3ef83d220473d849cce9f22835004'

user_name ='31j2eafa55r2ofbce6aiyqjrlmei'
redirect_uri = 'http://localhost:8080'
scope = 'playlist-modify-private,playlist-modify-public,user-top-read'

init = SpotipyClient(client_id,client_secret,user_name,redirect_uri,scope)

#autenticamos al usuario 
init.client_auth()
top_20 = init.get_top_tracks()

#mostrar  las ultimas 20 caciones más escuchadas por el/la usuari@
for i, item in enumerate(top_20['items']):
            print(f"""      {i+1}: {item['name']} <--> {item['artists'][0]['name']} """)

#Esperar()
#crear el dataframe con las ultimas 20 caciones
dataframe_top_20 = init.create_tracks_dataframe(top_20)

#imprimir el dataframe con las ultimas 20 caciones
print(tabulate(dataframe_top_20.head(), headers='keys', tablefmt='pretty'))

#Esperar()

#extraer los artistas del top 20
ids_artistas = init.get_artists_ids(top_20)
#extraer  artistas  similares a los del top 20
ids_artistas = init.get_similar_artists_ids(ids_artistas)
#extraer  artistas con nuevos lanzamientos similares a los del top 20
ids_artistas = init.get_new_releases_artists_ids(ids_artistas)
#print('1\n')
print('num ids artistas:',len(ids_artistas))
#Esperar()
ids_albums = init.get_albums_ids(ids_artistas)
#print('2\n')
#print(ids_albums)
#Esperar()
ids_pista,pistas_candidatas_info = init.get_albums_tracks(ids_albums)
#print('3\n')
print('num canciones candidatas: ',len(ids_pista))
#Esperar()
#                        pistas candidatas 
pistas_candidatas_df = init.get_tracks_features(ids_pista)
print('pistas candidatas:\n',tabulate(pistas_candidatas_df.head(), headers='keys', tablefmt='pretty'))

#mostrar  las  caciones en la lista de candidatos 
for i, item in enumerate(pistas_candidatas_info):
    print(f"{i + 1}: {item['name']} <--> {item['artist']}")

#                           optimización
print(text2art("2.  optimizacion  ", font="big"))
Esperar()
n_components = 5
pca_top_20_transforme = init.custom_pca(dataframe_top_20,n_components )
print('\npca_top_20_transforme\n',pca_top_20_transforme.head(),'\ntamaño:',pca_top_20_transforme.shape)
pca_candidatos_transforme = init.custom_pca(pistas_candidatas_df,n_components)
print('\npca_candidatos_transforme\n',pca_candidatos_transforme.head(),'\ntamaño:',pca_candidatos_transforme.shape)

#                              aprendizaje
#numero de clusters y humbral
n_clusters = 5
umbral = 0.7
#                              Busqueda
print(text2art("5.  Busqueda ", font="big"))
Esperar()
i_recomendaciones = init.busqueda_en_anchura_top_20(pca_top_20_transforme,pca_candidatos_transforme,n_clusters,umbral)
print(text2art("6.  PLaylist ", font="big"))
print('i_recomendaciones\n',i_recomendaciones,'\ntamaño:',len(i_recomendaciones ))
canciones_seleccionadas = pistas_candidatas_df.iloc[i_recomendaciones]
print(text2art('recomendaciones\n',font="bubble"),canciones_seleccionadas,'\ntamaño:',canciones_seleccionadas.shape)
info_canciones_seleccionadas = [pistas_candidatas_info[i] for i in i_recomendaciones]
print(text2art('recomendaciones\n',font="bubble"),len(info_canciones_seleccionadas))
for i, item in enumerate(info_canciones_seleccionadas):
    print(f"{i + 1}: {item['name']} <--> {item['artist']}")


ids_playlist = []
for j in i_recomendaciones:
    id_cand = pistas_candidatas_df['id'][j]
    ids_playlist.append(id_cand)
# Crear la playlist en Spotify
pl = init.client.user_playlist_create(user=init.username,
    name='Recomendados ia',
    description='Playlist creada con el sistema de recomendación')
init.client.playlist_add_items(pl['id'], ids_playlist)

print(text2art("  Playlist creada ", font="bubble"))
print(text2art("   Vuelve Pronto  ", font="epic"))


