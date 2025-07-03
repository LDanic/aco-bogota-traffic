import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

def plot_graph(data, best_route, highlight_idxs=None):
    G = build_graph(data)
    pos = nx.spring_layout(G, seed=42)
    pops = np.array([data['populations'][s] for s in data['stations']])
    sizes = 300 + (pops - pops.min())/(pops.max()-pops.min())*700
    
    plt.figure(figsize=(10,8))
    nx.draw_networkx_nodes(G, pos, node_size=sizes)
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_edge_labels(G, pos,
        edge_labels=nx.get_edge_attributes(G,'weight'), font_size=8)
    
    # ruta completa en rojo
    if best_route:
        edges = list(zip(best_route, best_route[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=3, edge_color='r')
    # nodos destacados en verde
    if highlight_idxs:
        labels = [data['stations'][i] for i in highlight_idxs]
        nx.draw_networkx_nodes(G, pos,
            nodelist=labels, node_color='green',
            node_size=500, alpha=0.9)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    example_data = {
        'stations': [
            "21 Angeles",
            "7 de Agosto",
            "AV. 1 Mayo",
            "AV Chile",
            "AV El Dorado",
            "AV Eldorado",
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
            "Ciudad Jardin - UA",
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
            "Gobernacion",
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
            "Santa Lucia",
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
            ('Terminal','Calle 187',1),
            ('Calle 187','Portal Norte – Unicervantes',1),
            ('Portal Norte – Unicervantes','Toberin - Foundever',1),
            ('Toberin - Foundever','Calle 161',1),
            ('Calle 161','Mazurén',1),
            ('Mazurén','Calle 146',1),
            ('Calle 146','Calle 142',1),
            ('Calle 142','Alcalá – Colegio S. Tomás Dominicos',1),
            ('Alcalá – Colegio S. Tomás Dominicos','Prado',1),
            ('Prado','Calle 127 - LOréal Paris',1),
            ('Calle 127 - LOréal Paris','Pepe Sierra',1),
            ('Pepe Sierra','Calle 106 - Maletas Explora',1),
            ('Calle 106 - Maletas Explora','Calle 100 - Marketmedios',1),
            ('Calle 100 - Marketmedios','Virrey',1),
            ('Virrey','Calle 85 – GATO DUMAS',1),
            ('Calle 85 – GATO DUMAS','Héroes',1),

#A
            ('Calle 76 - San Felipe','Calle 72',1),
            ('Calle 72','Flores – Areandina',1),
            ('Flores – Areandina','Calle 63',1),
            ('Calle 63','Calle 57',1),
            ('Calle 57','Marly',1),
            ('Marly','Calle 45 - American School Way',1),
            ('Calle 45 - American School Way','AV. 39',1),
            ('AV. 39','Calle 34 – Fondo Nacional de Garantías',1),
            ('Calle 34 – Fondo Nacional de Garantías','Calle 26',1),
            ('Calle 26','Calle 22',1),
            ('Calle 22','Calle 19',1),
            ('Calle 19','AV. Jiménez - Caracas',1),
            ('AV. Jiménez - Caracas','Tercer Milenio',1),

#D
            ('Polo – FINCOMERCIO','Escuela Militar',1),
            ('Escuela Militar','Carrera 47',1),
            ('Carrera 47','Carrera 53',1),
            ('Carrera 53','AV. 68',1),
            ('AV. 68','Ferias',1),
            ('Ferias','AV. Boyacá',1),
            ('AV. Boyacá','Minuto de Dios',1),
            ('Minuto de Dios','Granja - Kr 77',1),
            ('Granja - Kr 77','AV. Cali',1),
            ('AV. Cali','Carrera 90',1),
            ('Carrera 90','Quirigua',1),
            ('Quirigua','Portal 80',1),

#F
            ('Portal Américas','Patio Bonito',1),
            ('Patio Bonito','Biblioteca Tintal',1),
            ('Biblioteca Tintal','Transversal 86',1),
            ('Transversal 86','Banderas',1),
            ('Banderas','Mandalay',1),
            ('Mandalay','AV. Américas - AV. Boyacá',1),
            ('AV. Américas - AV. Boyacá','Marsella',1),
            ('Marsella','Pradera',1),
            ('Pradera','Distrito Grafiti',1),
            ('Distrito Grafiti','Puente Aranda',1),
            ('Puente Aranda','Carrera 43 - Comapan',1),
            ('Carrera 43 - Comapan','Zona Industrial',1),
            ('Zona Industrial','CDS - Carrera 32',1),
            ('CDS - Carrera 32','Ricaurte - CL 13',1),
            ('Ricaurte - CL 13','San Facon Carrera 22',1),
            ('San Facon Carrera 22','De La Sabana',1),

#G
            ('San Mateo - CC Unisur','Terreros - Hospital Cardio Vascular',1),
            ('Terreros - Hospital Cardio Vascular','León XIII',1),
            ('León XIII','La Despensa',1),
            ('La Despensa','Bosa',1),
            ('Bosa','Portal Sur - JFK Coop. Financiera',1),
            ('Portal Sur - JFK Coop. Financiera','Perdomo',1),
            ('Perdomo','CC Paseo Villa del Río - Madelena',1),
            ('CC Paseo Villa del Río - Madelena','Sevillana',1),
            ('Sevillana','Venecia',1),
            ('Venecia','Alquería',1),
            ('Alquería','General Santander',1),
            ('General Santander','NQS - Calle 38A S',1),
            ('NQS - Calle 38A S','NQS - Calle 30 S',1),
            ('NQS - Calle 30 S','Santa Isabel',1),
            ('Santa Isabel','Comuneros',1),

#C
            ('Portal Suba','La Campiña',1),
            ('La Campiña','Suba - TV 91',1),
            ('Suba - TV 91','21 Angeles',1),
            ('21 Angeles','Gratamira',1),
            ('Gratamira','Suba AV Boyaca',1),
            ('Suba AV Boyaca','Niza - Calle 127',1),
            ('Niza - Calle 127','Humedal Cordoba',1),
            ('Humedal Cordoba','Suba - Calle 116',1),
            ('Suba - Calle 116','Puentelargo',1),
            ('Puentelargo','Suba - Calle 100100',1),
            ('Suba - Calle 100100','Suba - Calle 95',1),
            ('Rionegro','San martin',1),

#E
            ('La Castellana','Calle 75 - Zona M',1),
            ('Calle 75 - Zona M','AV Chile',1),
            ('AV Chile','7 de Agosto',1),
            ('7 de Agosto','Movistar Arena',1),
            ('Movistar Arena','Campin - UAN',1),
            ('Campin - UAN','Universidad Nacional',1),
            ('Universidad Nacional','AV Eldorado',1),
            ('AV Eldorado','CAD',1),
            ('CAD','Paloquemao',1),
            ('Paloquemao','Ricaurte - NQS',1),
            ('Ricaurte - NQS','Guatoque - Veraguas',1),
            ('Guatoque - Veraguas','Tygua - San Jose',1),

#J
            ('Museo del Oro','Las Aguas Centro Colombo Americano',1),
            ('AV. Jiménez - CL 13','Museo del Oro',1),

#K
            ('Portal El Dorado – C.C. NUESTRO BOGOTÁ','Modelia',1),
            ('Modelia','Normandia',1),
            ('Normandia','Av. Roajs - Unisalesiana',1),
            ('Av. Roajs - Unisalesiana','El Tiempo Camara de Comercio de Bogota',1),
            ('El Tiempo Camara de Comercio de Bogota','Salitre - El Greco',1),
            ('Salitre - El Greco','CAN - British Council',1),
            ('CAN - British Council','Gobernacion',1),
            ('Gobernación','Quinta Paredes',1),
            ('Quinta Paredes','Corferias',1),
            ('Corferias','Ciudad Universitaria Loteria de Bogota',1),
            ('Ciudad Universitaria Loteria de Bogota','Concejo de Bogota',1),
            ('Concejo de Bogota','Centro Memoria',1),
            ('Centro Memoria','Universidades - CityU',1),

#L
            ('Portal 20 de Julio','Country Sur',1),
            ('Country Sur','AV. 1 Mayo',1),
            ('AV. 1 Mayo','Ciudad Jardin - UAN',1),
            ('Ciudad Jardin - UA','Policarpa',1),
            ('Policarpa','San Bernardo',1),
            ('San Bernardo','Bicentenario',1),
            ('Bicentenario','San Victorino',1),
            ('San Victorino','Las Nieves',1),
            ('Las Nieves','San Diego',1),

#M

#H
            ('Hospital','Hortua',1),
            ('Hortua','Nariño',1),
            ('Nariño','Fucha',1),
            ('Fucha','Restrepo',1),
            ('Restrepo','Olaya',1),
            ('Olaya','Quiroga',1),
            ('Quiroga','Calle 40 S',1),
            ('Calle 40 S','Santa Lucia',1),
            ('Santa Lucia','Socorro',1),
            ('Socorro','Consuelo',1),
            ('Consuelo','Molinos',1),
            ('Molinos','Danubio',1),
            ('Danubio','Portal Usme',1),
            ('Portal Tunal','Parque',1),
            ('Parque','Biblioteca',1),
            ('Biblioteca','Santa Lucia',1),

#Uniones
            ('Calle 100 - Marketmedios','La Castellana',1),
	    ('Héroes', 'Calle 76 - San Felipe',1),
	    ('Héroes', 'Polo – FINCOMERCIO',1),
	    ('Museo del Oro', 'Las Nieves',1),
	    ('Museo del Oro', 'San Victorino',1),
	    ('Museo Nacional - FNG', 'San Diego',1),
	    ('Centro Memoria', 'Calle 26',1),
	    ('Calle 26', 'Universidades - CityU',1),
	    ('Calle 22', 'Universidades - CityU',1),
	    ('Calle 22', 'Centro Memoria',1),
	    ('Las Aguas Centro Colombo Americano', 'Universidades - CityU',1),
	    ('AV. Jiménez - CL 13', 'Las Nieves',1),
	    ('AV. Jiménez - CL 13', 'San Victorino',1),
	    ('AV. Jiménez - CL 13', 'De La Sabana',1),
	    ('Tercer Milenio', 'Hospital',1),
	    ('Tercer Milenio', 'Tygua - San Jose',1),
	    ('Tercer Milenio', 'Bicentenario',1),
	    ('Tygua - San Jose', 'Bicentenario',1),
	    ('Tygua - San Jose', 'Hospital',1),
	    ('Comuneros', 'Ricaurte - NQS',1),
	    ('Comuneros', 'Guatoque - Veraguas',2),
	    ('AV El Dorado', 'Concejo de Bogota',1),
	    ('AV El Dorado', 'Ciudad Universitaria Loteria de Bogota',1),
	    ('CAD', 'Concejo de Bogota',1),
	    ('CAD', 'Ciudad Universitaria Loteria de Bogota',1),
	    ('Calle 75 - Zona M', 'Escuela Militar',1),
	    ('Calle 75 - Zona M', 'Carrera 47',1),
	    ('San martin', 'Escuela Militar',1),
	    ('San martin', 'Calle 75 - Zona M',1),
	    ('San martin', 'Carrera 47',1),
	    ('Calle 76 - San Felipe', 'Polo – FINCOMERCIO',1),        
],
        'populations': {
            "21 Angeles":1,
            "7 de Agosto":1,
            "AV. 1 Mayo":1,
            "AV Chile":1,
            "AV El Dorado":1,
            "AV Eldorado":1,
            "Suba - Calle 116":1,
            "AV. 39":1,
            "AV. 68":1,
            "AV. Américas - AV. Boyacá":1,
            "AV. Boyacá":1,
            "AV. Cali":1,
            "AV. Jiménez - CL 13":1,
            "AV. Jiménez - Caracas":1,
            "Alcalá – Colegio S. Tomás Dominicos":1,
            "Alquería":1,
            "Av. Roajs - Unisalesiana":1,
            "Banderas":1,
            "Biblioteca":1,
            "Biblioteca Tintal":1,
            "Bicentenario":1,
            "Bosa":1,
            "CAD":1,
            "CAN - British Council":1,
            "CC Paseo Villa del Río - Madelena":1,
            "CDS - Carrera 32":1,
            "Calle 100 - Marketmedios":1,
            "Calle 106 - Maletas Explora":1,
            "Calle 127 - LOréal Paris":1,
            "Calle 142":1,
            "Calle 146":1,
            "Calle 161":1,
            "Calle 187":1,
            "Calle 19":1,
            "Calle 22":1,
            "Calle 26":1,
            "Calle 34 – Fondo Nacional de Garantías":1,
            "Calle 40 S":1,
            "Calle 45 - American School Way":1,
            "Calle 57":1,
            "Calle 63":1,
            "Calle 72":1,
            "Calle 76 - San Felipe":1,
            "Calle 85 – GATO DUMAS":1,
            "Campin - UAN":1,
            "Carrera 43 - Comapan":1,
            "Carrera 47":1,
            "Carrera 53":1,
            "Carrera 90":1,
            "Centro Memoria":1,
            "Ciudad Jardin - UA":1,
            "Ciudad Jardin - UAN":1,
            "Ciudad Universitaria Loteria de Bogota":1,
            "Comuneros":1,
            "Concejo de Bogota":1,
            "Consuelo":1,
            "Corferias":1,
            "Country Sur":1,
            "Danubio":1,
            "De La Sabana":1,
            "Distrito Grafiti":1,
            "El Tiempo Camara de Comercio de Bogota":1,
            "Escuela Militar":1,
            "Ferias":1,
            "Flores – Areandina":1,
            "Fucha":1,
            "General Santander":1,
            "Gobernacion":1,
            "Gobernación":1,
            "Granja - Kr 77":1,
            "Gratamira":1,
            "Guatoque - Veraguas":1,
            "Hortua":1,
            "Hospital":1,
            "Humedal Cordoba":1,
            "Héroes":1,
            "La Campiña":1,
            "La Castellana":1,
            "La Despensa":1,
            "Las Aguas Centro Colombo Americano":1,
            "Las Nieves":1,
            "León XIII":1,
            "Mandalay":1,
            "Marly":1,
            "Marsella":1,
            "Mazurén":1,
            "Minuto de Dios":1,
            "Modelia":1,
            "Molinos":1,
            "Movistar Arena":1,
            "Museo Nacional - FNG":1,
            "Museo del Oro":1,
            "NQS - Calle 30 S":1,
            "NQS - Calle 38A S":1,
            "Calle 75 - Zona M":1,
            "Nariño":1,
            "Niza - Calle 127":1,
            "Normandia":1,
            "Olaya":1,
            "Paloquemao":1,
            "Parque":1,
            "Patio Bonito":1,
            "Pepe Sierra":1,
            "Perdomo":1,
            "Policarpa":1,
            "Polo – FINCOMERCIO":1,
            "Portal 20 de Julio":1,
            "Portal 80":1,
            "Portal Américas":1,
            "Portal El Dorado – C.C. NUESTRO BOGOTÁ":1,
            "Portal Norte – Unicervantes":1,
            "Portal Suba":1,
            "Portal Sur - JFK Coop. Financiera":1,
            "Portal Tunal":1,
            "Portal Usme":1,
            "Pradera":1,
            "Prado":1,
            "Puente Aranda":1,
            "Puentelargo":1,
            "Quinta Paredes":1,
            "Quirigua":1,
            "Quiroga":1,
            "Restrepo":1,
            "Ricaurte - CL 13":1,
            "Ricaurte - NQS":1,
            "Rionegro":1,
            "Salitre - El Greco":1,
            "San Bernardo":1,
            "San Diego":1,
            "San Facon Carrera 22":1,
            "San Mateo - CC Unisur":1,
            "San Victorino":1,
            "San martin":1,
            "Santa Isabel":1,
            "Santa Lucia":1,
            "Sevillana":1,
            "Socorro":1,
            "Suba - TV 91":1,
            "Suba AV Boyaca":1,
            "Suba - Calle 100100":1,
            "Suba - Calle 95":1,
            "Tercer Milenio":1,
            "Terminal":1,
            "Terreros - Hospital Cardio Vascular":1,
            "Toberin - Foundever":1,
            "Transversal 86":1,
            "Tygua - San Jose":1,
            "Universidad Nacional":1,
            "Universidades - CityU":1,
            "Venecia":1,
            "Virrey":1,
            "Zona Industrial":1,
        }
    }
    
    dist_matrix, rewards, station_to_idx = build_matrices(example_data)
    idx_to_station = {i: s for s, i in station_to_idx.items()}
    G = build_graph(example_data)
    
    start_idx = station_to_idx['Bosa']
    end_idx   = station_to_idx['Comuneros']
    budget    = 200
    n_ants    = 30
    n_iter    = 100
    
    best_path, best_eff = aco_orienteering_efficiency_forced_end(
        dist_matrix, rewards,
        start_idx, end_idx, budget,
        G, idx_to_station, station_to_idx,
        n_ants=n_ants, n_iterations=n_iter
    )
    
    best_route = [idx_to_station[i] for i in best_path]
    print(f"Ruta (termina en L): {' → '.join(best_route)}")
    print(f"Eficiencia: {best_eff:.2f}")
    
    highlight = select_top_stops(best_path, rewards, start_idx, end_idx, top_n=4)
    labels = [idx_to_station[i] for i in highlight]
    print(f"Nodos destacados: {labels}")
    
    plot_graph(example_data, best_route, highlight)
