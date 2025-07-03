def sum_distance_mts(path, distance):

    # Construimos un diccionario para búsqueda rápida, con clave como frozenset de dos nodos
    peso_por_par = {
        frozenset((u, v)): w
        for u, v, w in distance
    }

    total = 0.0
    # Recorremos pares consecutivos de path
    for a, b in zip(path, path[1:]):
        clave = frozenset((a, b))
        if clave in peso_por_par:
            total += peso_por_par[clave]
        else:
            print(f"Advertencia: no se encontró arista entre '{a}' y '{b}'")

    return total

def sum_population(stops, populations):
    total = 0.0
    for station in stops[:-1]:
        if station in populations:
            total += populations[station]
        else:
            # Advertencia si la estación no está en el diccionario
            print(f"Advertencia: no se encontró población para '{station}'")
    return total

def estimate_travel_time(distance_meters, stops, penalty_per_stop_minutes=2):
    speed_kmh = 50.0

    distance_km = distance_meters / 1000.0

    time_hours = distance_km / speed_kmh
    time_minutes = time_hours * 60.0

    total_penalty = (len(stops)-1) * penalty_per_stop_minutes

    total_time_minutes = time_minutes + total_penalty
    return total_time_minutes


def numerical_results(path, stops, distances, populations, penalty_per_stop_minutes=2):
    distance = sum_distance_mts(path, distances)
    print(f"Suma total de distancias: {distance}")

    population = sum_population(stops, populations)
    print(f"Suma total de población: {population}")

    time_minutes = estimate_travel_time(distance, stops, penalty_per_stop_minutes)
    print(f"Tiempo estimado de viaje: {time_minutes}")
