import requests
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point

# URLs GIS de TransMilenio
TRONCAL_URL = (
    "https://gis.transmilenio.gov.co/arcgis/rest/services/Troncal/"
    "consulta_trazados_troncales/FeatureServer/0/query?"
    "where=1%3D1&outFields=id_trazado_troncal,"
    "nombre_trazado_troncal,longitud_trazado,origen_trazado,fin_trazado&"
    "outSR=4326&f=json"
)
ESTACIONES_URL = (
    "https://gis.transmilenio.gov.co/arcgis/rest/services/Troncal/"
    "consulta_estaciones_troncales/FeatureServer/0/query?"
    "where=1%3D1&outFields=numero_estacion,nombre_estacion,"
    "coordenada_x_estacion,coordenada_y_estacion,longitud_estacion&"
    "outSR=4326&f=json"
)

# Mapa de nombres de estación a su código numérico
CODIGOS = {
    'Portal 20 de Julio': 10000,
    'Portal Suba': 3000,
    'Portal El Dorado – C.C. NUESTRO BOGOTÁ': 6000,
    'Portal 80': 4000,
    'Portal Américas': 5000,
    'Portal Sur - JFK Coop. Financiera': 7000,
    'Portal Tunal': 8000,
    'Portal Usme': 9000,
    'AV Chile': 7103,
    'CAN - British Council': 6103,
    'Campin - UAN': 7106,
    'San Diego': 10008,
    'Granja - Kr 77': 4100,
    'Ricaurte - NQS': 7111,
    'Quirigua': 4001,
    'Calle 75 - Zona M': 7102,
    'AV. Boyacá': 4102,
    'Calle 40 S': 9100,
    'AV. Américas - AV. Boyacá': 5102,
    'Marsella': 5103,
    'Niza - Calle 127': 3006,
    'La Campiña': 3001,
    'Transversal 86': 5005,
    'Corferias': 6106,
    'Guatoque - Veraguas': 7201,
    'CDS - Carrera 32': 12002,
    'Ciudad Universitaria Loteria de Bogota': 6107,
    'Universidad Nacional': 7107,
    'Av. Roajs - Unisalesiana': 6100,
    'Tercer Milenio': 9109,
    'Concejo de Bogota': 6108,
    'NQS - Calle 38A S': 7007,
    '7 de Agosto': 7104,
    'AV. 68': 4104,
    'Distrito Grafiti': 5107,
    'Polo – FINCOMERCIO': 4108,
    'Tygua - San Jose': 7200,
    'Quinta Paredes': 6105,
    'Salitre - El Greco': 6102,
    'Zona Industrial': 12007,
    'León XIII': 7505,
    'Paloquemao': 7110,
    'Carrera 90': 4002,
    'Gobernación': 6104,
    'Terreros - Hospital Cardio Vascular': 7504,
    'Escuela Militar': 4107,
    'AV. Cali': 4003,
    'Suba - TV 91': 3002,
    'Ricaurte - CL 13': 12003,
    'Restrepo': 9104,
    'Prado': 2201,
    'Virrey': 2302,
    'Toberin - Foundever': 2101,
    'Pepe Sierra': 2204,
    'Calle 100 - Marketmedios': 2300,
    'Calle 142': 2105,
    'Calle 187': 2001,
    'Alcalá – Colegio S. Tomás Dominicos': 2200,
    'Calle 146': 2104,
    'Mandalay': 5101,
    'Calle 127 - LOréal Paris': 2202,
    'Héroes': 2304,
    'Calle 161': 2102,
    'Calle 106 - Maletas Explora': 2205,
    'Mazurén': 2103,
    'Calle 45 - American School Way': 9117,
    'Carrera 53': 4105,
    'La Castellana': 7101,
    'AV. 1 Mayo': 10002,
    'De La Sabana': 14001,
    'La Despensa': 7506,
    'Calle 72': 9122,
    'Centro Memoria': 6109,
    'Carrera 43 - Comapan': 12001,
    'Consuelo': 9002,
    'AV. Jiménez - Caracas': 9110,
    'Comuneros': 7112,
    'Calle 22': 9113,
    'El Tiempo Camara de Comercio de Bogota': 6101,
    'Biblioteca Tintal': 5002,
    'Ciudad Jardin - UAN': 10003,
    'Bicentenario': 10005,
    'Calle 57': 9119,
    'AV El Dorado': 7108,
    'Calle 26': 9114,
    'Alquería': 7005,
    'Bosa': 7010,
    'CAD': 7109,
    'Banderas': 5100,
    'Calle 19': 9111,
    'AV. Jiménez - CL 13': 14003,
    'Movistar Arena': 7105,
    'AV. 39': 9116,
    'Calle 76 - San Felipe': 9123,
    'Calle 85 – GATO DUMAS': 2303,
    'NQS - Calle 30 S': 7008,
    'Country Sur': 10001,
    'Carrera 47': 4106,
    'Olaya': 9103,
    'Molinos': 9001,
    'Nariño': 9106,
    'Gratamira': 3004,
    'Pradera': 5105,
    'Las Nieves': 10007,
    'Modelia': 6001,
    'Biblioteca': 8002,
    'San Facon Carrera 22': 12004,
    'Calle 63': 9120,
    'Flores – Areandina': 9121,
    'Museo del Oro': 14004,
    'General Santander': 7006,
    'Humedal Cordoba': 3007,
    'Parque': 8001,
    'Policarpa': 10004,
    'Museo Nacional - FNG': 10009,
    'Hospital': 9108,
    'Normandia': 6002,
    'Perdomo': 7001,
    'Hortua': 9107,
    'Ferias': 4103,
    'Marly': 9118,
    'Patio Bonito': 5001,
    'Fucha': 9105,
    'Minuto de Dios': 4101,
    'Las Aguas Centro Colombo Americano': 14005,
    'Suba AV Boyaca': 3005,
    'Santa Lucía': 9004,
    'San martin': 3014,
    'Universidades - CityU': 6111,
    'Santa Isabel': 7113,
    'Suba - Calle 95': 3012,
    'Puente Aranda': 12000,
    'Quiroga': 9101,
    'Puentelargo': 3010,
    'San Victorino': 10006,
    'CC Paseo Villa del Río - Madelena': 7002,
    'San Mateo - CC Unisur': 7503,
    'Sevillana': 7003,
    'Rionegro': 3013,
    'Suba - Calle 116': 3009,
    'San Bernardo': 10010,
    'Socorro': 9003,
    '21 Angeles': 3003,
    'Calle 34 – Fondo Nacional de Garantías': 9115,
    'Venecia': 7004,
    'Terminal': 2502,
    'Portal Norte – Unicervantes': 2000,
    'Suba - Calle 100100': 3011,
    'Danubio': 9005
 }



def fetch_troncal():
    """Descarga y retorna los trazados troncales como GeoSeries de LineString."""
    data = requests.get(TRONCAL_URL).json()
    lines = []
    for feat in data['features']:
        for path in feat['geometry']['paths']:
            lines.append(LineString(path))
    return gpd.GeoSeries(lines, crs="EPSG:4326")

def fetch_estaciones():
    """Descarga y retorna las estaciones como GeoDataFrame de Points."""
    data = requests.get(ESTACIONES_URL).json()
    records = []
    for feat in data['features']:
        x, y = feat['geometry']['x'], feat['geometry']['y']
        rec = feat['attributes'].copy()
        rec['geometry'] = Point(x, y)
        records.append(rec)
    return gpd.GeoDataFrame(records, crs="EPSG:4326")

def plot_map(lines_gs, stations_gdf, route=None, highlight=None):
    """
    Dibuja el mapa de TransMilenio:
     - trazados de troncal (gris),
     - estaciones (azul),
     - ruta opcional (rojo),
     - nodos highlight (amarillo) con sus etiquetas.
    
    route: lista de (x,y) para la ruta.
    highlight: dict {nombre_estación: (x,y)} para resaltar y anotar.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    # Base: troncales y estaciones
    lines_gs.plot(ax=ax, color='lightgray', linewidth=1, zorder=1)
    stations_gdf.plot(ax=ax, color='blue', markersize=20, zorder=2)

    # Ruta en rojo
    if route:
        gpd.GeoSeries([LineString(route)], crs="EPSG:4326") \
            .plot(ax=ax, color='red', linewidth=3, zorder=3)

    # Resaltar estaciones y anotarlas
    if highlight:
        for name, (x, y) in highlight.items():
            ax.scatter(x, y, s=120, facecolors='none',
                       edgecolors='gold', linewidths=2, zorder=4)
            ax.annotate(
                name,
                xy=(x, y),
                xytext=(3, 3),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold',
                color='darkgoldenrod',
                zorder=5
            )

    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("Mapa TransMilenio")
    plt.show()

def plot_map_with_route_and_highlight(lines_gs, stations_gdf,
                                      station_names, highlight_names):
    """
    Prepara coordenadas de ruta y highlight, luego grafica.
    """
    def coords_for(names):
        pts = []
        for name in names:
            if name not in CODIGOS:
                raise ValueError(f"Estación no mapeada: {name}")
            code = CODIGOS[name]
            row = stations_gdf[stations_gdf['numero_estacion'].astype(int) == code]
            if row.empty:
                raise ValueError(f"No encontrada en GIS: {name} (código {code})")
            p = row.geometry.iloc[0]
            pts.append((p.x, p.y))
        return pts

    route_coords = coords_for(station_names)
    highlight_coords = coords_for(highlight_names)
    highlight_dict = dict(zip(highlight_names, highlight_coords))

    plot_map(
        lines_gs,
        stations_gdf,
        route=route_coords,
        highlight=highlight_dict
    )

def plotBogota(mi_ruta, highlight):
    print("Cargando datos GIS de TransMilenio...")
    lines = fetch_troncal()
    stations = fetch_estaciones()

    print("Dibujando ruta y resaltando estaciones...")
    plot_map_with_route_and_highlight(lines, stations, mi_ruta, highlight)

if __name__ == "__main__":
    mi_ruta = ['La Despensa', 'Bosa', 'Portal Sur - JFK Coop. Financiera', 'Perdomo', 'CC Paseo Villa del Río - Madelena', 'Sevillana', 'Venecia', 'Alquería', 'General Santander', 'NQS - Calle 38A S', 'NQS - Calle 30 S', 'Santa Isabel', 'Comuneros', 'Guatoque - Veraguas', 'Tygua - San Jose', 'Tercer Milenio', 'AV. Jiménez - Caracas', 'Calle 19', 'Calle 22', 'Calle 26', 'Calle 34 – Fondo Nacional de Garantías', 'AV. 39', 'Calle 45 - American School Way']

    highlight = ['La Despensa', 'AV. Jiménez - Caracas', 'Venecia', 'Bosa', 'NQS - Calle 38A S', 'Calle 45 - American School Way']
    plotBogota(mi_ruta, highlight)
