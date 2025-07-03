import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from visual import plotBogota
from numerical_results import numerical_results

def build_matrices(data):
    stations = data['stations']
    edges = data['edges']
    populations = data['populations']
    
    n = len(stations)
    station_to_idx = {s: i for i, s in enumerate(stations)}
    
    dist_matrix = np.full((n, n), np.inf)
    np.fill_diagonal(dist_matrix, 0)
    for u, v, d in edges:
        i, j = station_to_idx[u], station_to_idx[v]
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d
    
    rewards = np.array([populations[s] for s in stations], dtype=float)
    return dist_matrix, rewards, station_to_idx

def build_graph(data):
    G = nx.Graph()
    for s in data['stations']:
        G.add_node(s)
    for u, v, d in data['edges']:
        G.add_edge(u, v, weight=d)
    return G

def heuristic(dist_matrix):
    with np.errstate(divide='ignore'):
        eta = 1.0 / (dist_matrix + 1e-6)
    eta[dist_matrix == np.inf] = 0
    return eta

def aco_orienteering_efficiency_forced_end(dist_matrix, rewards,
                                           start, end, budget,
                                           G, idx_to_station, station_to_idx,
                                           n_ants=20, n_iterations=100,
                                           alpha=1.0, beta=2.0, rho=0.1, Q=1.0):
    n = dist_matrix.shape[0]
    pheromone = np.ones((n, n))
    eta = heuristic(dist_matrix)
    
    best_eff = -np.inf
    best_path = None
    
    for _ in range(n_iterations):
        all_paths, all_effs = [], []
        
        for _ in range(n_ants):
            visited = {start}
            path    = [start]
            dist    = 0.0
            current = start
            
            # Construcción de ruta exploratoria
            while True:
                cand = [j for j in range(n)
                        if j not in visited and dist + dist_matrix[current, j] <= budget]
                if not cand:
                    break
                probs = np.array([(pheromone[current,j]**alpha)*(eta[current,j]**beta)
                                  for j in cand])
                probs /= probs.sum()
                nxt = np.random.choice(cand, p=probs)
                path.append(nxt)
                visited.add(nxt)
                dist += dist_matrix[current, nxt]
                current = nxt
                if current == end:
                    break
            
            # Fuerzo llegada a end en un subgrafo excluyendo repetidos
            if current != end:
                src_lbl = idx_to_station[current]
                tgt_lbl = idx_to_station[end]
                subG = G.copy()
                used = {idx_to_station[i] for i in visited}
                subG.remove_nodes_from([lbl for lbl in used
                                         if lbl not in (src_lbl, tgt_lbl)])
                try:
                    sp = nx.shortest_path(subG, src_lbl, tgt_lbl, weight='weight')
                    extra = sum(subG[u][v]['weight'] for u,v in zip(sp[:-1], sp[1:]))
                    # sólo aceptamos si cabe en presupuesto
                    if dist + extra <= budget:
                        for lbl in sp[1:]:
                            idx = station_to_idx[lbl]
                            path.append(idx)
                            visited.add(idx)
                        dist += extra
                        current = end
                    else:
                        continue  # descartar esta hormiga
                except nx.NetworkXNoPath:
                    continue  # descartar esta hormiga
            # si current == end ya cumplió
            
            # Calcular eficiencia y almacenar
            total_reward = sum(rewards[i] for i in path)
            eff = total_reward / len(path)
            all_paths.append(path)
            all_effs.append(eff)
            
            if eff > best_eff:
                best_eff = eff
                best_path = path
        
        # Actualizar feromonas
        pheromone *= (1 - rho)
        for path, eff in zip(all_paths, all_effs):
            for u, v in zip(path[:-1], path[1:]):
                pheromone[u, v] += Q * eff
                pheromone[v, u] += Q * eff
    
    return best_path, best_eff

def select_top_stops(best_path, rewards, start_idx, end_idx, top_n=4):
    inter = [i for i in best_path if i not in (start_idx, end_idx)]
    orden = sorted(inter, key=lambda i: rewards[i], reverse=True)
    top = orden[:top_n]
    return [start_idx] + top + [end_idx]


if __name__ == '__main__':
    example_data = {
        'stations': [
            "21 Angeles",
            "7 de Agosto",
            "AV. 1 Mayo",
            "AV Chile",
            "AV El Dorado",
            "Suba - Calle 116",
            "AV. 39",
            "AV. 68",
            "AV. Américas - AV. Boyacá",
            "AV. Boyacá",
            "AV. Cali",
            "AV. Jiménez - CL 13",
            "AV. Jiménez - Caracas",
            "Alcalá – Colegio S. Tomás Dominicos",
            "Alquería",
            "Av. Roajs - Unisalesiana",
            "Banderas",
            "Biblioteca",
            "Biblioteca Tintal",
            "Bicentenario",
            "Bosa",
            "CAD",
            "CAN - British Council",
            "CC Paseo Villa del Río - Madelena",
            "CDS - Carrera 32",
            "Calle 100 - Marketmedios",
            "Calle 106 - Maletas Explora",
            "Calle 127 - LOréal Paris",
            "Calle 142",
            "Calle 146",
            "Calle 161",
            "Calle 187",
            "Calle 19",
            "Calle 22",
            "Calle 26",
            "Calle 34 – Fondo Nacional de Garantías",
            "Calle 40 S",
            "Calle 45 - American School Way",
            "Calle 57",
            "Calle 63",
            "Calle 72",
            "Calle 76 - San Felipe",
            "Calle 85 – GATO DUMAS",
            "Campin - UAN",
            "Carrera 43 - Comapan",
            "Carrera 47",
            "Carrera 53",
            "Carrera 90",
            "Centro Memoria",
            "Ciudad Jardin - UAN",
            "Ciudad Universitaria Loteria de Bogota",
            "Comuneros",
            "Concejo de Bogota",
            "Consuelo",
            "Corferias",
            "Country Sur",
            "Danubio",
            "De La Sabana",
            "Distrito Grafiti",
            "El Tiempo Camara de Comercio de Bogota",
            "Escuela Militar",
            "Ferias",
            "Flores – Areandina",
            "Fucha",
            "General Santander",
            "Gobernación",
            "Granja - Kr 77",
            "Gratamira",
            "Guatoque - Veraguas",
            "Hortua",
            "Hospital",
            "Humedal Cordoba",
            "Héroes",
            "La Campiña",
            "La Castellana",
            "La Despensa",
            "Las Aguas Centro Colombo Americano",
            "Las Nieves",
            "León XIII",
            "Mandalay",
            "Marly",
            "Marsella",
            "Mazurén",
            "Minuto de Dios",
            "Modelia",
            "Molinos",
            "Movistar Arena",
            "Museo Nacional - FNG",
            "Museo del Oro",
            "NQS - Calle 30 S",
            "NQS - Calle 38A S",
            "Calle 75 - Zona M",
            "Nariño",
            "Niza - Calle 127",
            "Normandia",
            "Olaya",
            "Paloquemao",
            "Parque",
            "Patio Bonito",
            "Pepe Sierra",
            "Perdomo",
            "Policarpa",
            "Polo – FINCOMERCIO",
            "Portal 20 de Julio",
            "Portal 80",
            "Portal Américas",
            "Portal El Dorado – C.C. NUESTRO BOGOTÁ",
            "Portal Norte – Unicervantes",
            "Portal Suba",
            "Portal Sur - JFK Coop. Financiera",
            "Portal Tunal",
            "Portal Usme",
            "Pradera",
            "Prado",
            "Puente Aranda",
            "Puentelargo",
            "Quinta Paredes",
            "Quirigua",
            "Quiroga",
            "Restrepo",
            "Ricaurte - CL 13",
            "Ricaurte - NQS",
            "Rionegro",
            "Salitre - El Greco",
            "San Bernardo",
            "San Diego",
            "San Facon Carrera 22",
            "San Mateo - CC Unisur",
            "San Victorino",
            "San martin",
            "Santa Isabel",
            "Santa Lucía",
            "Sevillana",
            "Socorro",
            "Suba - TV 91",
            "Suba AV Boyaca",
            "Suba - Calle 100100",
            "Suba - Calle 95",
            "Tercer Milenio",
            "Terminal",
            "Terreros - Hospital Cardio Vascular",
            "Toberin - Foundever",
            "Transversal 86",
            "Tygua - San Jose",
            "Universidad Nacional",
            "Universidades - CityU",
            "Venecia",
            "Virrey",
            "Zona Industrial",
        ],
        'edges': [
            ('Terminal', 'Calle 187', 613.7737784031103),
            ('Calle 187', 'Portal Norte – Unicervantes', 979.8168547288823),
            ('Portal Norte – Unicervantes', 'Toberin - Foundever', 942.5443229295931),
            ('Toberin - Foundever', 'Calle 161', 479.4023637706919),
            ('Calle 161', 'Mazurén', 821.7516428712839),
            ('Mazurén', 'Calle 146', 413.70859358230393),
            ('Calle 146', 'Calle 142', 453.45407657469883),
            ('Calle 142', 'Alcalá – Colegio S. Tomás Dominicos', 638.3010944613851),
            ('Alcalá – Colegio S. Tomás Dominicos', 'Prado', 739.9320440288949),
            ('Prado', 'Calle 127 - LOréal Paris', 1112.302758473626),
            ('Calle 127 - LOréal Paris', 'Pepe Sierra', 730.1739670291689),
            ('Pepe Sierra', 'Calle 106 - Maletas Explora', 577.063066895741),
            ('Calle 106 - Maletas Explora',
            'Calle 100 - Marketmedios',
            1013.2211509068807),
            ('Calle 100 - Marketmedios', 'Virrey', 891.4532794187296),
            ('Virrey', 'Calle 85 – GATO DUMAS', 412.7343087643214),
            ('Calle 85 – GATO DUMAS', 'Héroes', 468.1493193683924),
            ('Calle 76 - San Felipe', 'Calle 72', 537.3951331612152),
            ('Calle 72', 'Flores – Areandina', 383.10479173944384),
            ('Flores – Areandina', 'Calle 63', 748.1750749032684),
            ('Calle 63', 'Calle 57', 608.9401911941986),
            ('Calle 57', 'Marly', 609.0261325589233),
            ('Marly', 'Calle 45 - American School Way', 686.0576300654972),
            ('Calle 45 - American School Way', 'AV. 39', 511.86199941499206),
            ('AV. 39', 'Calle 34 – Fondo Nacional de Garantías', 599.5190550180478),
            ('Calle 34 – Fondo Nacional de Garantías', 'Calle 26', 608.1504346047673),
            ('Calle 26', 'Calle 22', 615.3429450090298),
            ('Calle 22', 'Calle 19', 487.52984331847176),
            ('Calle 19', 'AV. Jiménez - Caracas', 679.0063836055799),
            ('AV. Jiménez - Caracas', 'Tercer Milenio', 647.0496142208113),
            ('Polo – FINCOMERCIO', 'Escuela Militar', 757.7861055488955),
            ('Escuela Militar', 'Carrera 47', 523.9900724751695),
            ('Carrera 47', 'Carrera 53', 592.8126774541992),
            ('Carrera 53', 'AV. 68', 549.7940483233496),
            ('AV. 68', 'Ferias', 732.0274020614077),
            ('Ferias', 'AV. Boyacá', 438.7516832045014),
            ('AV. Boyacá', 'Minuto de Dios', 569.0875296602867),
            ('Minuto de Dios', 'Granja - Kr 77', 573.3578892061603),
            ('Granja - Kr 77', 'AV. Cali', 600.629083590974),
            ('AV. Cali', 'Carrera 90', 532.4945684956679),
            ('Carrera 90', 'Quirigua', 515.13401040632),
            ('Quirigua', 'Portal 80', 394.2790923313795),
            ('Portal Américas', 'Patio Bonito', 1028.7913792463376),
            ('Patio Bonito', 'Biblioteca Tintal', 783.6625582200683),
            ('Biblioteca Tintal', 'Transversal 86', 905.7555802786974),
            ('Transversal 86', 'Banderas', 768.8768818611885),
            ('Banderas', 'Mandalay', 498.47647435591995),
            ('Mandalay', 'AV. Américas - AV. Boyacá', 789.1766884560835),
            ('AV. Américas - AV. Boyacá', 'Marsella', 453.20405537831317),
            ('Marsella', 'Pradera', 1281.3604038457784),
            ('Pradera', 'Distrito Grafiti', 790.0554906760718),
            ('Distrito Grafiti', 'Puente Aranda', 808.8111059523374),
            ('Puente Aranda', 'Carrera 43 - Comapan', 477.8459843507247),
            ('Carrera 43 - Comapan', 'Zona Industrial', 408.52052546145103),
            ('Zona Industrial', 'CDS - Carrera 32', 700.5102664917567),
            ('CDS - Carrera 32', 'Ricaurte - CL 13', 517.1178455736347),
            ('Ricaurte - CL 13', 'San Facon Carrera 22', 547.5784639764851),
            ('San Facon Carrera 22', 'De La Sabana', 721.8782522384112),
            ('San Mateo - CC Unisur',
            'Terreros - Hospital Cardio Vascular',
            736.6514824597666),
            ('Terreros - Hospital Cardio Vascular', 'León XIII', 792.7444016471367),
            ('León XIII', 'La Despensa', 618.316001517516),
            ('La Despensa', 'Bosa', 805.2191536558388),
            ('Bosa', 'Portal Sur - JFK Coop. Financiera', 1312.417779339756),
            ('Portal Sur - JFK Coop. Financiera', 'Perdomo', 509.7520122341789),
            ('Perdomo', 'CC Paseo Villa del Río - Madelena', 899.7785970453504),
            ('CC Paseo Villa del Río - Madelena', 'Sevillana', 1059.0690145615679),
            ('Sevillana', 'Venecia', 530.4405485129264),
            ('Venecia', 'Alquería', 941.1222123883987),
            ('Alquería', 'General Santander', 530.1752233072093),
            ('General Santander', 'NQS - Calle 38A S', 643.671682509168),
            ('NQS - Calle 38A S', 'NQS - Calle 30 S', 661.567297077083),
            ('NQS - Calle 30 S', 'Santa Isabel', 1856.8000967692753),
            ('Santa Isabel', 'Comuneros', 417.55035693614417),
            ('Portal Suba', 'La Campiña', 592.6031709507608),
            ('La Campiña', 'Suba - TV 91', 619.2679834994823),
            ('Suba - TV 91', '21 Angeles', 766.5838659053035),
            ('21 Angeles', 'Gratamira', 1124.408062836663),
            ('Gratamira', 'Suba AV Boyaca', 660.7910152486445),
            ('Suba AV Boyaca', 'Niza - Calle 127', 1078.3627168540156),
            ('Niza - Calle 127', 'Humedal Cordoba', 609.6365938272236),
            ('Humedal Cordoba', 'Suba - Calle 116', 819.6889524019698),
            ('Suba - Calle 116', 'Puentelargo', 722.7494748888088),
            ('Puentelargo', 'Suba - Calle 100100', 760.9277061201328),
            ('Suba - Calle 100100', 'Suba - Calle 95', 300.2421363875652),
            ('Rionegro', 'San martin', 510.0005478965067),
            ('Rionegro', 'Suba - Calle 95', 483.3516260427178),
            ('La Castellana', 'Calle 75 - Zona M', 973.3269991352154),
            ('Calle 75 - Zona M', 'AV Chile', 554.8009925777595),
            ('AV Chile', '7 de Agosto', 1060.8752139070957),
            ('7 de Agosto', 'Movistar Arena', 806.2418696434049),
            ('Movistar Arena', 'Campin - UAN', 515.1833951519112),
            ('Campin - UAN', 'Universidad Nacional', 917.6142472184517),
            ('Universidad Nacional', 'AV El Dorado', 716.3355399314464),
            ('AV El Dorado', 'CAD', 932.9648112265271),
            ('CAD', 'Paloquemao', 933.6270067722718),
            ('Paloquemao', 'Ricaurte - NQS', 751.5124824273972),
            ('Ricaurte - NQS', 'Guatoque - Veraguas', 837.8252879326365),
            ('Guatoque - Veraguas', 'Tygua - San Jose', 839.0502159708204),
            ('Museo del Oro', 'Las Aguas Centro Colombo Americano', 526.3469509683619),
            ('AV. Jiménez - CL 13', 'Museo del Oro', 719.3980896621787),
            ('Portal El Dorado – C.C. NUESTRO BOGOTÁ', 'Modelia', 866.9660442787672),
            ('Modelia', 'Normandia', 791.6161513131567),
            ('Normandia', 'Av. Roajs - Unisalesiana', 927.4210931765655),
            ('Av. Roajs - Unisalesiana',
            'El Tiempo Camara de Comercio de Bogota',
            641.4158885326369),
            ('El Tiempo Camara de Comercio de Bogota',
            'Salitre - El Greco',
            816.6489190015757),
            ('Salitre - El Greco', 'CAN - British Council', 526.7690953737257),
            ('CAN - British Council', 'Gobernación', 527.6090152718901),
            ('Gobernación', 'Quinta Paredes', 699.6269463336986),
            ('Quinta Paredes', 'Corferias', 516.1828362982548),
            ('Corferias', 'Ciudad Universitaria Loteria de Bogota', 784.4375785932996),
            ('Ciudad Universitaria Loteria de Bogota',
            'Concejo de Bogota',
            650.211537516562),
            ('Concejo de Bogota', 'Centro Memoria', 752.3080505764744),
            ('Centro Memoria', 'Universidades - CityU', 2011.6680280419664),
            ('Portal 20 de Julio', 'Country Sur', 651.6295389208622),
            ('Country Sur', 'AV. 1 Mayo', 819.50348511543),
            ('AV. 1 Mayo', 'Ciudad Jardin - UAN', 630.9728834692315),
            ('Ciudad Jardin - UAN', 'Policarpa', 645.7803823291272),
            ('Policarpa', 'San Bernardo', 521.9131876885481),
            ('San Bernardo', 'Bicentenario', 505.42655496786426),
            ('Bicentenario', 'San Victorino', 916.9809980838032),
            ('San Victorino', 'Las Nieves', 654.1717017592666),
            ('Las Nieves', 'San Diego', 619.7821137214136),
            ('Hospital', 'Hortua', 644.149950866518),
            ('Hortua', 'Nariño', 649.1422031615875),
            ('Nariño', 'Fucha', 709.0187356608656),
            ('Fucha', 'Restrepo', 256.9036416174972),
            ('Restrepo', 'Olaya', 733.4400192937634),
            ('Olaya', 'Quiroga', 812.5529050525861),
            ('Quiroga', 'Calle 40 S', 642.7073448888924),
            ('Calle 40 S', 'Santa Lucía', 741.8252902917268),
            ('Santa Lucía', 'Socorro', 701.3872199133357),
            ('Socorro', 'Consuelo', 526.3662876255275),
            ('Consuelo', 'Molinos', 416.1085608034215),
            ('Molinos', 'Danubio', 1485.2428925818401),
            ('Danubio', 'Portal Usme', 1329.9765973044327),
            ('Portal Tunal', 'Parque', 445.5608367098983),
            ('Parque', 'Biblioteca', 599.0618434385162),
            ('Biblioteca', 'Santa Lucía', 633.1047101151229),
            ('Calle 100 - Marketmedios', 'La Castellana', 1186.9885876802734),
            ('Héroes', 'Calle 76 - San Felipe', 574.3144616616626),
            ('Héroes', 'Polo – FINCOMERCIO', 567.9941515287359),
            ('Museo del Oro', 'Las Nieves', 566.9362292881569),
            ('Museo del Oro', 'San Victorino', 501.9523665241053),
            ('Museo Nacional - FNG', 'San Diego', 543.0650619524641),
            ('Centro Memoria', 'Calle 26', 627.2076688260946),
            ('Calle 26', 'Universidades - CityU', 1430.7692171784988),
            ('Calle 22', 'Universidades - CityU', 1131.8602002118676),
            ('Calle 22', 'Centro Memoria', 979.5293150414606),
            ('Las Aguas Centro Colombo Americano',
            'Universidades - CityU',
            258.1372949580651),
            ('AV. Jiménez - CL 13', 'Las Nieves', 628.5510065308383),
            ('AV. Jiménez - CL 13', 'San Victorino', 288.55811309415384),
            ('AV. Jiménez - CL 13', 'De La Sabana', 405.9728772880722),
            ('Tercer Milenio', 'Hospital', 431.2094747481956),
            ('Tercer Milenio', 'Tygua - San Jose', 574.5104657634419),
            ('Tygua - San Jose', 'Bicentenario', 1002.8913851341829),
            ('Tygua - San Jose', 'Hospital', 593.3231811047611),
            ('Comuneros', 'Ricaurte - NQS', 1057.3414523206075),
            ('Comuneros', 'Guatoque - Veraguas', 550.7882098783069),
            ('AV El Dorado', 'Concejo de Bogota', 517.3395823509678),
            ('AV El Dorado',
            'Ciudad Universitaria Loteria de Bogota',
            403.52783255648404),
            ('CAD', 'Concejo de Bogota', 519.5757275012026),
            ('CAD', 'Ciudad Universitaria Loteria de Bogota', 831.5792546005009),
            ('Calle 75 - Zona M', 'Escuela Militar', 611.1442721934594),
            ('Calle 75 - Zona M', 'Carrera 47', 901.4505792559347),
            ('San martin', 'Escuela Militar', 306.8317935686251),
            ('San martin', 'Calle 75 - Zona M', 847.9095003537213),
            ('San martin', 'Carrera 47', 737.5798887318725),
            ('Calle 76 - San Felipe', 'Polo – FINCOMERCIO', 932.0298496803925)
        ],

        'populations': {
            'Portal 20 de Julio': 22960.0,
            'Portal Suba': 37542.0,
            'Portal El Dorado – C.C. NUESTRO BOGOTÁ': 28334.0,
            'Portal 80': 38764.0,
            'Portal Américas': 55605.0,
            'Portal Sur - JFK Coop. Financiera': 0,
            'Portal Tunal': 33946.0,
            'Portal Usme': 25868.0,
            'AV Chile': 8378.0,
            'CAN - British Council': 2737.0,
            'Campin - UAN': 8333.0,
            'San Diego': 3558.0,
            'Granja - Kr 77': 4265.0,
            'Ricaurte - NQS': 8113.0,
            'Quirigua': 2660.0,
            'Calle 75 - Zona M': 3825.0,
            'AV. Boyacá': 7970.0,
            'Calle 40 S': 5795.0,
            'AV. Américas - AV. Boyacá': 3711.0,
            'Marsella': 4155.0,
            'Niza - Calle 127': 7142.0,
            'La Campiña': 3996.0,
            'Transversal 86': 4443.0,
            'Corferias': 4125.0,
            'Guatoque - Veraguas': 4234.0,
            'CDS - Carrera 32': 1724.0,
            'Ciudad Universitaria Loteria de Bogota': 3038.0,
            'Universidad Nacional': 6755.0,
            'Av. Roajs - Unisalesiana': 3233.0,
            'Tercer Milenio': 0.0,
            'Concejo de Bogota': 863.0,
            'NQS - Calle 38A S': 9012.0,
            '7 de Agosto': 7150.0,
            'AV. 68': 5927.0,
            'Distrito Grafiti': 6767.0,
            'Polo – FINCOMERCIO': 4317.0,
            'Tygua - San Jose': 5834.0,
            'Quinta Paredes': 1478.0,
            'Salitre - El Greco': 5142.0,
            'Zona Industrial': 3851.0,
            'León XIII': 8517.0,
            'Paloquemao': 9774.0,
            'Carrera 90': 4347.0,
            'Gobernación': 949.0,
            'Terreros - Hospital Cardio Vascular': 17752.0,
            'Escuela Militar': 727.0,
            'AV. Cali': 6110.0,
            'Suba - TV 91': 9389.0,
            'Ricaurte - CL 13': 3688.0,
            'Restrepo': 9219.0,
            'Prado': 6119.0,
            'Virrey': 5164.0,
            'Toberin - Foundever': 13923.0,
            'Pepe Sierra': 7058.0,
            'Calle 100 - Marketmedios': 15494.0,
            'Calle 142': 7150.0,
            'Calle 187': 5485.0,
            'Alcalá – Colegio S. Tomás Dominicos': 12693.0,
            'Calle 146': 0.0,
            'Mandalay': 1129.0,
            'Calle 127 - LOréal Paris': 11012.0,
            'Héroes': 7759.0,
            'Calle 161': 2751.0,
            'Calle 106 - Maletas Explora': 5865.0,
            'Mazurén': 8138.0,
            'Calle 45 - American School Way': 11818.0,
            'Carrera 53': 2333.0,
            'La Castellana': 2775.0,
            'AV. 1 Mayo': 13283.0,
            'De La Sabana': 5552.0,
            'La Despensa': 2527.0,
            'Calle 72': 0.0,
            'Centro Memoria': 2771.0,
            'Carrera 43 - Comapan': 2402.0,
            'Consuelo': 1219.0,
            'AV. Jiménez - Caracas': 28179.0,
            'Comuneros': 7670.0,
            'Calle 22': 6453.0,
            'El Tiempo Camara de Comercio de Bogota': 5804.0,
            'Biblioteca Tintal': 11317.0,
            'Ciudad Jardin - UAN': 2091.0,
            'Bicentenario': 3590.0,
            'Calle 57': 16668.0,
            'AV El Dorado': 2542.0,
            'Calle 26': 0.0,
            'Alquería': 7044.0,
            'Bosa': 9025.0,
            'CAD': 8477.0,
            'Banderas': 22586.0,
            'Calle 19': 0.0,
            'AV. Jiménez - CL 13': 28179.0,
            'Movistar Arena': 5061.0,
            'AV. 39': 4219.0,
            'Calle 76 - San Felipe': 11318.0,
            'Calle 85 – GATO DUMAS': 6338.0,
            'NQS - Calle 30 S': 4015.0,
            'Country Sur': 8610.0,
            'Carrera 47': 1815.0,
            'Olaya': 7842.0,
            'Molinos': 9763.0,
            'Nariño': 3417.0,
            'Gratamira': 596.0,
            'Pradera': 8084.0,
            'Las Nieves': 9455.0,
            'Modelia': 3276.0,
            'Biblioteca': 1922.0,
            'San Facon Carrera 22': 5979.0,
            'Calle 63': 0.0,
            'Flores – Areandina': 14735.0,
            'Museo del Oro': 7138.0,
            'General Santander': 5989.0,
            'Humedal Cordoba': 1189.0,
            'Parque': 1909.0,
            'Policarpa': 3113.0,
            'Museo Nacional - FNG': 7062.0,
            'Hospital': 2215.0,
            'Normandia': 3885.0,
            'Perdomo': 6840.0,
            'Hortua': 4429.0,
            'Ferias': 2044.0,
            'Marly': 0.0,
            'Patio Bonito': 0.0,
            'Fucha': 5719.0,
            'Minuto de Dios': 3864.0,
            'Las Aguas Centro Colombo Americano': 8178.0,
            'Suba AV Boyaca': 4784.0,
            'Santa Lucía': 8222.0,
            'San martin': 646.0,
            'Universidades - CityU': 5265.0,
            'Santa Isabel': 6573.0,
            'Suba - Calle 95': 5132.0,
            'Puente Aranda': 1471.0,
            'Quiroga': 2416.0,
            'Puentelargo': 3381.0,
            'San Victorino': 17791.0,
            'CC Paseo Villa del Río - Madelena': 8351.0,
            'San Mateo - CC Unisur': 29387.0,
            'Sevillana': 4078.0,
            'Rionegro': 2417.0,
            'Suba - Calle 116': 4073.0,
            'San Bernardo': 4062.0,
            'Socorro': 1691.0,
            '21 Angeles': 1614.0,
            'Calle 34 – Fondo Nacional de Garantías': 0.0,
            'Venecia': 9304.0,
            'Terminal': 6678.0,
            'Portal Norte – Unicervantes': 48888.0,
            'Suba - Calle 100100': 3518.0,
            'Danubio': 3935.0
        }
    }
    
    dist_matrix, rewards, station_to_idx = build_matrices(example_data)
    idx_to_station = {i: s for s, i in station_to_idx.items()}
    G = build_graph(example_data)
    
    start_idx = station_to_idx['La Despensa']
    end_idx   = station_to_idx['Calle 45 - American School Way']
    budget    = 20000
    n_ants    = 30
    n_iter    = 100
    
    best_path, best_eff = aco_orienteering_efficiency_forced_end(
        dist_matrix, rewards,
        start_idx, end_idx, budget,
        G, idx_to_station, station_to_idx,
        n_ants=n_ants, n_iterations=n_iter
    )
    
    best_route = [idx_to_station[i] for i in best_path]
    
    highlight = select_top_stops(best_path, rewards, start_idx, end_idx, top_n=4)
    labels = [idx_to_station[i] for i in highlight]

    numerical_results(best_route, labels, example_data['edges'], example_data['populations'])

    plotBogota(best_route, labels)
