
# solver.py - Resolver las instancias del proyecto con lp_solve
# INF292 Optimizacion - Proyecto 2026-S1
#
# Nota: lp_solve se instala con  sudo apt install lp-solve
# El formato .lp lo saque de aca: http://lpsolve.sourceforge.net/5.5/lp_format.htm

import json
import os
import sys
import subprocess
import time
import csv
import matplotlib
matplotlib.use("Agg")   # para que no necesite pantalla (me tiraba error en el server)
import matplotlib.pyplot as plt


# ==========================================================
#  Generar archivo .lp para lp_solve
# ==========================================================

def generar_archivo_lp(inst, filepath):
    """
    Arma el archivo .lp con el modelo PLEM del problema logistico.
    
    El formato que usa lp_solve es:
        min: <funcion objetivo>;
        <restriccion1>;
        <restriccion2>;
        ...
        bin <variables binarias>;
    
    Las variables continuas son >= 0 por defecto en lp_solve,
    asi que no hay que declarar x_ij >= 0 ni y_jk >= 0.
    """
    n_p = inst["n_plants"]
    n_dc = inst["n_dc"]
    n_z = inst["n_zones"]
    P_max = inst["P_max"]
    R = inst["R"]
    S = inst["S"]
    H = inst["H"]
    C = inst["C"]
    D = inst["D"]
    F = inst["F"]

    lines = []

    # -- Funcion objetivo --
    # min Z = sum(Fj*zj) + sum(Cij*xij) + sum(Djk*yjk)
    obj_parts = []
    for j in range(n_dc):
        obj_parts.append(f"{F[j]} z_{j}")
    for i in range(n_p):
        for j in range(n_dc):
            obj_parts.append(f"{C[i][j]} x_{i}_{j}")
    for j in range(n_dc):
        for k in range(n_z):
            obj_parts.append(f"{D[j][k]} y_{j}_{k}")

    lines.append("min: " + " + ".join(obj_parts) + ";")
    lines.append("")

    # -- Restriccion 1: Capacidad de plantas --
    # sum_j(x_ij) <= S_i   para cada planta i
    for i in range(n_p):
        terms = " + ".join(f"x_{i}_{j}" for j in range(n_dc))
        lines.append(f"{terms} <= {S[i]};")
    lines.append("")

    # -- Restriccion 2: Conservacion de flujo en centros --
    # sum_i(x_ij) - sum_k(y_jk) = 0   para cada centro j
    # (lo que entra al centro = lo que sale)
    for j in range(n_dc):
        entrada = " + ".join(f"x_{i}_{j}" for i in range(n_p))
        salida = " - ".join(f"y_{j}_{k}" for k in range(n_z))
        lines.append(f"{entrada} - {salida} = 0;")
    lines.append("")

    # -- Restriccion 3: Capacidad de centros + habilitacion --
    # sum_k(y_jk) - Hj * zj <= 0
    # (solo puede despachar si esta abierto, y hasta su capacidad)
    for j in range(n_dc):
        terms = " + ".join(f"y_{j}_{k}" for k in range(n_z))
        lines.append(f"{terms} - {H[j]} z_{j} <= 0;")
    lines.append("")

    # -- Restriccion 4: Satisfaccion de demanda --
    # sum_j(y_jk) = Rk   para cada zona k
    for k in range(n_z):
        terms = " + ".join(f"y_{j}_{k}" for j in range(n_dc))
        lines.append(f"{terms} = {R[k]};")
    lines.append("")

    # -- Restriccion 5: Limite maximo de centros abiertos --
    # sum_j(zj) <= P_max
    terms = " + ".join(f"z_{j}" for j in range(n_dc))
    lines.append(f"{terms} <= {P_max};")
    lines.append("")

    # -- Declarar variables binarias --
    bin_vars = ", ".join(f"z_{j}" for j in range(n_dc))
    lines.append(f"bin {bin_vars};")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


# ==========================================================
#  Llamar a lp_solve y parsear la salida
# ==========================================================

def ejecutar_lp_solve(lp_path):
    """
    Ejecuta lp_solve por linea de comandos y parsea el output.
    
    La salida de lp_solve se ve asi:
    
        Value of objective function: 37988.00000000
        
        Actual values of the variables:
        x_0_0                           185
        x_0_1                           0
        z_0                             1
        ...
    
    Retorna (status, valor_objetivo, diccionario_variables)
    """
    try:
        result = subprocess.run(
            ["lp_solve", lp_path],
            capture_output=True, text=True, timeout=300
        )
    except FileNotFoundError:
        print("ERROR: no se encontro lp_solve.")
        print("Instalar con:  sudo apt install lp-solve")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT al resolver {lp_path}")
        return "Timeout", None, {}

    salida = result.stdout
    # print(salida)  # descomentar si algo falla

    # si lp_solve falla retorna codigo != 0
    if result.returncode != 0:
        # a veces imprime "This problem is infeasible" en stdout
        if "infeasible" in salida.lower():
            return "Infeasible", None, {}
        else:
            print(f"  lp_solve error (exit code {result.returncode})")
            #print(result.stderr)
            return "Error", None, {}

    # Parsear el valor de la funcion objetivo
    obj_value = None
    variables = {}
    leyendo_vars = False

    for linea in salida.split("\n"):
        linea = linea.strip()

        # La linea del objetivo se ve asi:
        #   Value of objective function: 37988.00000000
        if linea.startswith("Value of objective function:"):
            try:
                obj_value = float(linea.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
            continue

        # Despues viene "Actual values of the variables:" y luego las vars
        if "Actual values" in linea:
            leyendo_vars = True
            continue

        # Cada variable es:  nombre_var       valor
        if leyendo_vars and linea:
            partes = linea.split()
            if len(partes) >= 2:
                try:
                    variables[partes[0]] = float(partes[1])
                except ValueError:
                    pass

    if obj_value is not None:
        return "Optimal", obj_value, variables
    else:
        return "Error", None, {}


# ==========================================================
#  Resolver una instancia completa
# ==========================================================

def solve_instance(inst):
    """Genera el .lp, llama a lp_solve, y extrae los resultados"""
    size = inst.get("size", "?")
    iid = inst.get("id", "?")
    label = f"{size}_{iid}"

    print(f"\n--- {label} ---")
    print(f"  Plantas={inst['n_plants']}, CDs={inst['n_dc']}, Zonas={inst['n_zones']}, P_max={inst['P_max']}")
    print(f"  Demanda total: {inst['total_demand']}")

    # contar dimensiones del modelo
    n_vars = (inst["n_plants"] * inst["n_dc"]     # variables x_ij
              + inst["n_dc"] * inst["n_zones"]     # variables y_jk
              + inst["n_dc"])                       # variables z_j
    n_rest = (inst["n_plants"]                      # cap plantas
              + inst["n_dc"]                        # flujo
              + inst["n_dc"]                        # cap centros
              + inst["n_zones"]                     # demanda
              + 1)                                  # limite centros
    print(f"  Variables: {n_vars}, Restricciones: {n_rest}")

    # generar el archivo .lp
    os.makedirs("lp_files", exist_ok=True)
    lp_path = f"lp_files/{label}.lp"
    generar_archivo_lp(inst, lp_path)

    # resolver
    t0 = time.time()
    status, obj_value, variables = ejecutar_lp_solve(lp_path)
    elapsed = time.time() - t0

    print(f"  Estado: {status}  |  Tiempo: {elapsed:.4f}s")

    result = {
        "label":          label,
        "size":           size,
        "id":             iid,
        "status":         status,
        "n_plants":       inst["n_plants"],
        "n_dc":           inst["n_dc"],
        "n_zones":        inst["n_zones"],
        "P_max":          inst["P_max"],
        "total_demand":   inst["total_demand"],
        "n_vars":         n_vars,
        "n_constraints":  n_rest,
        "time_sec":       round(elapsed, 4),
        "objective":      None,
        "open_dcs":       None,
        "fixed_cost":     None,
        "transport_cost": None,
    }

    if status != "Optimal":
        print(f"  >> No se obtuvo solucion optima")
        return result

    result["objective"] = round(obj_value, 2)

    # ver cuales centros quedaron abiertos (z_j > 0.5 porque es binaria)
    open_dcs = []
    for j in range(inst["n_dc"]):
        val = variables.get(f"z_{j}", 0)
        if val > 0.5:
            open_dcs.append(j)

    fixed_cost = sum(inst["F"][j] for j in open_dcs)
    transport_cost = obj_value - fixed_cost

    result["open_dcs"] = len(open_dcs)
    result["fixed_cost"] = round(fixed_cost, 2)
    result["transport_cost"] = round(transport_cost, 2)

    print(f"  Costo optimo: {result['objective']}")
    print(f"  Costo fijo:   {result['fixed_cost']}  |  Transporte: {result['transport_cost']}")
    print(f"  CDs abiertos: {len(open_dcs)} de {inst['P_max']} permitidos")

    return result


# ==========================================================
#  Guardar resultados y hacer graficos
# ==========================================================

def guardar_csv(results, path="output/results.csv"):
    """Guarda todos los resultados en un CSV"""
    if not results:
        return
    os.makedirs("output", exist_ok=True)
    keys = list(results[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f"\nResultados guardados en {path}")


def imprimir_tabla(results):
    """Imprime una tabla resumen por consola"""
    print("\n" + "=" * 90)
    print("RESUMEN DE RESULTADOS")
    print("=" * 90)
    print(f"{'Instancia':<12} {'Vars':>6} {'Rest':>6} {'CDs':>8} {'Costo':>12} {'Tiempo':>10}")
    print("-" * 90)
    for r in results:
        obj_str = f"{r['objective']}" if r["objective"] is not None else "N/A"
        dc_str = f"{r['open_dcs']}/{r['P_max']}" if r["open_dcs"] is not None else "N/A"
        print(f"{r['label']:<12} {r['n_vars']:>6} {r['n_constraints']:>6} {dc_str:>8} {obj_str:>12} {r['time_sec']:>9.4f}s")
    print("=" * 90)


def hacer_graficos(results):
    """
    Genera los 4 graficos que pide el enunciado:
    1. Costo total por instancia
    2. Dimensionalidad (variables) vs tiempo de ejecucion
    3. CDs abiertos vs P_max
    4. Descomposicion del costo (fijo vs transporte)
    """
    resueltos = [r for r in results if r["objective"] is not None]
    if not resueltos:
        print("No hay instancias resueltas para graficar")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    labels = [r["label"] for r in resueltos]
    x_pos = range(len(resueltos))

    # colores por tamaño para que se distinga mejor
    colores = []
    for r in resueltos:
        if r["size"] == "small":
            colores.append("#4CAF50")
        elif r["size"] == "medium":
            colores.append("#2196F3")
        else:
            colores.append("#F44336")

    # --- Grafico 1: Costo total ---
    ax = axes[0, 0]
    ax.bar(x_pos, [r["objective"] for r in resueltos], color=colores)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title("Costo total óptimo por instancia")
    ax.set_ylabel("Costo ($)")
    ax.ticklabel_format(style='plain', axis='y')

    # --- Grafico 2: Variables vs Tiempo ---
    ax = axes[0, 1]
    for r in resueltos:
        c = "#4CAF50" if r["size"] == "small" else ("#2196F3" if r["size"] == "medium" else "#F44336")
        ax.scatter(r["n_vars"], r["time_sec"], color=c, s=50, edgecolors="black", linewidths=0.5)
    ax.set_xlabel("Número de variables")
    ax.set_ylabel("Tiempo (s)")
    ax.set_title("Dimensionalidad vs tiempo de resolución")
    ax.grid(True, alpha=0.3)
    # leyenda manual
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50', markersize=8, label='Small'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=8, label='Medium'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336', markersize=8, label='Large'),
    ]
    ax.legend(handles=legend_items, fontsize=8)

    # --- Grafico 3: CDs abiertos vs P_max ---
    ax = axes[1, 0]
    ancho = 0.35
    x_arr = list(x_pos)
    ax.bar([p - ancho/2 for p in x_arr], [r["P_max"] for r in resueltos],
           width=ancho, label="P_max", color="#90CAF9")
    ax.bar([p + ancho/2 for p in x_arr], [r["open_dcs"] for r in resueltos],
           width=ancho, label="Abiertos", color="#E65100")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title("CDs abiertos vs límite máximo (P_max)")
    ax.set_ylabel("Cantidad de CDs")
    ax.legend(fontsize=8)

    # --- Grafico 4: Descomposicion del costo ---
    ax = axes[1, 1]
    fijos = [r["fixed_cost"] for r in resueltos]
    transp = [r["transport_cost"] for r in resueltos]
    ax.bar(x_pos, fijos, label="Costo fijo", color="#7E57C2")
    ax.bar(x_pos, transp, bottom=fijos, label="Costo transporte", color="#FFB74D")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title("Descomposición del costo total")
    ax.set_ylabel("Costo ($)")
    ax.legend(fontsize=8)
    ax.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/resultados.png", dpi=150)
    print("Graficos guardados en output/resultados.png")
    plt.close()


# ==========================================================
#  Main
# ==========================================================

if __name__ == "__main__":
    inst_dir = "instances"

    if not os.path.isdir(inst_dir):
        print(f"No se encontro la carpeta '{inst_dir}'. Ejecutar primero generate_instances.py")
        sys.exit(1)

    archivos = sorted([f for f in os.listdir(inst_dir) if f.endswith(".json")])
    print(f"Se encontraron {len(archivos)} instancias en '{inst_dir}/'")

    results = []
    for fname in archivos:
        with open(os.path.join(inst_dir, fname)) as f:
            inst = json.load(f)
        result = solve_instance(inst)
        results.append(result)

    imprimir_tabla(results)
    guardar_csv(results)
    hacer_graficos(results)

    print("\nListo!")
