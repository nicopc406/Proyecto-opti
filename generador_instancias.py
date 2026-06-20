import numpy as np
import json
import os
from math import ceil

# Rangos de la Tabla 1 del enunciado
# P = plantas, CD = centros de distribucion, ZD = zonas de demanda
RANGOS = {
    "small":  {"plants": (3,  10), "dc": (6,  12), "zones": (8,  15)},
    "medium": {"plants": (11, 20), "dc": (12, 24), "zones": (16, 30)},
    "large":  {"plants": (21, 35), "dc": (25, 40), "zones": (31, 45)},
}


def generate_instance(n_plants, n_dc, n_zones, seed):
    """
    Genera una instancia aleatoria del problema logistico.
    
    Para que sea factible hay que cuidar varias cosas:
    - La capacidad total de las plantas tiene que alcanzar para cubrir toda la demanda
    - Los centros tienen que poder manejar suficiente producto (considerando que solo
      P_max de ellos van a estar abiertos)
    - P_max tiene que ser menor que n_dc (sino seria trivial abrir todos)
    - Los costos fijos tienen que ser lo suficientemente altos para que no convenga
      abrir todos los centros
    """
    rng = np.random.default_rng(seed)

    # Primero: generar la demanda de cada zona
    R = rng.integers(50, 201, size=n_zones).tolist()
    total_demand = sum(R)

    # P_max: entre 35% y n_dc-1 de los centros disponibles
    # (si P_max = n_dc la restriccion no hace nada, seria trivial)
    p_min = max(1, ceil(n_dc * 0.35))
    p_max_bound = n_dc - 1
    P_max = int(rng.integers(p_min, p_max_bound + 1))

    # Capacidades de plantas: que sumen al menos 1.2 veces la demanda total
    lo = total_demand / n_plants * 0.60
    hi = total_demand / n_plants * 2.00
    S = rng.uniform(lo, hi, size=n_plants)
    # si no alcanza, escalar para que sume 1.2 * demanda
    if S.sum() < 1.20 * total_demand:
        S = S * (1.20 * total_demand / S.sum())
    S = np.maximum(1, np.round(S)).astype(int).tolist()

    # Capacidades de centros de distribucion
    # Los P_max centros mas grandes tienen que poder cubrir la demanda
    lo_h = total_demand / (n_dc * 1.5)
    hi_h = 2.5 * total_demand / P_max
    H = rng.uniform(lo_h, hi_h, size=n_dc)
    # verificar que los P_max mejores centros cubran la demanda
    top_cap = sorted(H, reverse=True)[:P_max]
    if sum(top_cap) < 1.10 * total_demand:
        factor = 1.10 * total_demand / sum(top_cap)
        H = H * factor
    H = np.maximum(1, np.round(H)).astype(int).tolist()

    # Costos de transporte (valores entre 1 y 50)
    C = rng.integers(1, 51, size=(n_plants, n_dc)).tolist()   # planta -> centro
    D = rng.integers(1, 51, size=(n_dc, n_zones)).tolist()    # centro -> zona

    # Costos fijos de operacion de los centros
    # Tienen que ser suficientemente altos para que no convenga abrir todos
    avg_dc_flow = total_demand / P_max
    avg_transport = 25.0   # promedio de los costos de transporte (rango 1-50)
    lo_f = avg_dc_flow * avg_transport * 0.30
    hi_f = avg_dc_flow * avg_transport * 1.20
    F = np.round(rng.uniform(lo_f, hi_f, size=n_dc)).astype(int).tolist()

    return {
        "size":         None,   # se llena despues
        "id":           None,
        "n_plants":     n_plants,
        "n_dc":         n_dc,
        "n_zones":      n_zones,
        "P_max":        P_max,
        "total_demand": total_demand,
        "R": R,
        "S": S,
        "H": H,
        "C": C,
        "D": D,
        "F": F,
    }


def main():
    os.makedirs("instances", exist_ok=True)

    print("Generando 15 instancias...")
    print()

    for size, bounds in RANGOS.items():
        for rep in range(1, 6):
            # Semilla fija para que sea reproducible
            master_seed = abs(hash(f"{size}_{rep}")) % (2**30)
            rng_dim = np.random.default_rng(master_seed)

            # Dimensiones aleatorias dentro de los rangos de la Tabla 1
            n_plants = int(rng_dim.integers(bounds["plants"][0], bounds["plants"][1] + 1))
            n_dc     = int(rng_dim.integers(bounds["dc"][0],     bounds["dc"][1]     + 1))
            n_zones  = int(rng_dim.integers(bounds["zones"][0],  bounds["zones"][1]  + 1))

            inst = generate_instance(n_plants, n_dc, n_zones, seed=master_seed + 9999)
            inst["size"] = size
            inst["id"]   = rep

            fname = f"instances/{size}_{rep}.json"
            with open(fname, "w") as f:
                json.dump(inst, f, indent=2)
            print(f"  {fname}  ->  P={n_plants}, CD={n_dc}, ZD={n_zones}, P_max={inst['P_max']}")

    print()
    print("Listo! Se generaron 15 instancias en instances/")


if __name__ == "__main__":
    main()
