"""
═══════════════════════════════════════════════════════════════════════════════
BENCHMARK — Proyecto Eclipse Solar Total 12 agosto 2026
═══════════════════════════════════════════════════════════════════════════════

Estructura del archivo:
  1. fetch_reference_data()  → descarga datos JPL Horizons (ejecutar una vez)
  2. load_reference_data()   → carga los datos guardados
  3. eclipse_error()         → calcula las 3 métricas de error para cualquier modelo
  4. print_benchmark()       → tabla comparativa de todos los modelos

Uso:
  python benchmark.py --fetch      # descargar datos de referencia de JPL
  python benchmark.py --check      # verificar datos descargados
  python benchmark.py --demo       # mostrar tabla de ejemplo con un modelo dummy

Después, cada modelo importa eclipse_error() para reportar sus resultados.
"""

import numpy as np
import json
import os
import argparse
from datetime import datetime, timedelta

# ─── CONSTANTES ──────────────────────────────────────────────────────────────
AU       = 1.495978707e11   # metros
DAY      = 86400.0          # segundos
KM       = 1e3              # metros
ARCMIN   = 1/60.0           # grados

# Radios físicos (en AU)
R_SOL    = 6.957e8  / AU
R_LUNA   = 1.7374e6 / AU
R_TIERRA = 6.371e6  / AU

# Fecha de referencia del eclipse (máximo geocéntrico, NASA)
# NASA eclipse 2026 Aug 12: máximo a las 17:47:06 UTC
ECLIPSE_UTC = datetime(2026, 8, 12, 17, 47, 6)

# Fechas clave para el benchmark (puntos de control)
FECHAS_CONTROL = [
    datetime(2026, 1,  1,  0, 0, 0),   # inicio simulación
    datetime(2026, 3,  1,  0, 0, 0),   # 2 meses después
    datetime(2026, 6,  1,  0, 0, 0),   # 6 meses después
    datetime(2026, 8,  1,  0, 0, 0),   # 12 días antes
    datetime(2026, 8, 12, 17, 47, 6),  # momento del eclipse
]

# IDs de JPL Horizons
# Sol=10, Tierra=399, Luna=301
# Planetas: Mercurio=199, Venus=299, Marte=499,
#           Júpiter=599, Saturno=699, Urano=799, Neptuno=899

REFERENCE_FILE = "Datos/jpl_reference_data.json"

# ─── 1. DESCARGA DE DATOS DE REFERENCIA ──────────────────────────────────────

def fetch_reference_data():
    """
    Descarga posiciones y velocidades de JPL Horizons para Sol, Tierra, Luna
    y todos los planetas en las fechas de control.

    Intenta primero con astroquery (más limpio), y si no está disponible
    usa la API REST directa con un parser robusto.

    Requiere: pip install requests
    Opcional: pip install astroquery astropy   (recomendado, más fiable)

    Guarda: jpl_reference_data.json
    """
    # Intentar con astroquery primero
    try:
        from astroquery.jplhorizons import Horizons
        print("Usando astroquery.jplhorizons")
        return _fetch_con_astroquery(Horizons)
    except ImportError:
        print("astroquery no disponible, usando API REST directa.")
        print("(Para mejores resultados: pip install astroquery astropy)")

    # Fallback: API REST
    try:
        import requests
    except ImportError:
        print("ERROR: instala requests:  pip install requests")
        return False

    return _fetch_con_requests(requests)


def _fetch_con_astroquery(Horizons):
    """Descarga datos usando astroquery (más robusto)."""
    # id JPL -> nombre
    cuerpos = [
        ("10",  "Sol"),
        ("399", "Tierra"),
        ("301", "Luna"),
        ("199", "Mercurio"),
        ("299", "Venus"),
        ("499", "Marte"),
        ("599", "Jupiter"),
        ("699", "Saturno"),
        ("799", "Urano"),
        ("899", "Neptuno"),
    ]

    datos = {}

    for fecha in FECHAS_CONTROL:
        # JD de la fecha
        from astropy.time import Time
        t = Time(fecha.isoformat(), format='isot', scale='utc')
        jd = t.jd

        fecha_str = fecha.strftime("%Y-%b-%d %H:%M")
        print(f"\nFecha: {fecha_str}")
        datos[fecha.isoformat()] = {}

        for jpl_id, nombre in cuerpos:
            try:
                obj = Horizons(
                    id=jpl_id,
                    location="500@0",   # baricentro solar
                    epochs=jd,
                    id_type="majorbody",
                )
                vecs = obj.vectors(refplane="ecliptic")

                x  = float(vecs["x"][0])
                y  = float(vecs["y"][0])
                z  = float(vecs["z"][0])
                vx = float(vecs["vx"][0])
                vy = float(vecs["vy"][0])
                vz = float(vecs["vz"][0])

                r = np.sqrt(x**2 + y**2 + z**2)
                datos[fecha.isoformat()][nombre] = {
                    "pos": [x, y, z],
                    "vel": [vx, vy, vz],
                    "r":   float(r),
                }
                print(f"  {nombre}: r={r:.6f} AU  OK")

            except Exception as e:
                print(f"  {nombre}: ERROR — {e}")

    with open(REFERENCE_FILE, "w") as f:
        json.dump(datos, f, indent=2)

    print(f"\n✓ Datos guardados en {REFERENCE_FILE}")
    return True


def _fetch_con_requests(requests):
    """Descarga datos usando la API REST de Horizons con parser robusto."""

    BASE_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

    cuerpos = [
        ("10",  "Sol"),
        ("399", "Tierra"),
        ("301", "Luna"),
        ("199", "Mercurio"),
        ("299", "Venus"),
        ("499", "Marte"),
        ("599", "Jupiter"),
        ("699", "Saturno"),
        ("799", "Urano"),
        ("899", "Neptuno"),
    ]

    datos = {}

    for fecha in FECHAS_CONTROL:
        fecha_str = fecha.strftime("%Y-%b-%d %H:%M")
        fecha_fin = fecha + timedelta(minutes=2)
        fecha_fin_str = fecha_fin.strftime("%Y-%b-%d %H:%M")

        print(f"\nFecha: {fecha_str}")
        datos[fecha.isoformat()] = {}

        for jpl_id, nombre in cuerpos:
            params = {
                "format":     "text",        # texto plano, más fácil de parsear
                "COMMAND":    jpl_id,
                "OBJ_DATA":   "NO",
                "MAKE_EPHEM": "YES",
                "EPHEM_TYPE": "VECTORS",
                "CENTER":     "500@0",
                "START_TIME": fecha_str,
                "STOP_TIME":  fecha_fin_str,
                "STEP_SIZE":  "1 m",
                "VEC_TABLE":  "2",
                "REF_PLANE":  "ECLIPTIC",
                "REF_SYSTEM": "J2000",
                "OUT_UNITS":  "AU-D",
                "VEC_LABELS": "YES",
                "CSV_FORMAT": "NO",
            }

            try:
                r = requests.get(BASE_URL, params=params, timeout=30)
                r.raise_for_status()
                texto = r.text

                rv = _parse_horizons_texto(texto)
                if rv:
                    datos[fecha.isoformat()][nombre] = rv
                    print(f"  {nombre}: r={rv['r']:.6f} AU  OK")
                else:
                    # Mostrar primeras líneas para diagnóstico
                    print(f"  {nombre}: ERROR al parsear")
                    for ln in texto.split("\n")[:5]:
                        if ln.strip():
                            print(f"    >> {ln[:80]}")

            except Exception as e:
                print(f"  {nombre}: EXCEPCION — {e}")

    with open(REFERENCE_FILE, "w") as f:
        json.dump(datos, f, indent=2)

    print(f"\n✓ Datos guardados en {REFERENCE_FILE}")
    return True


def _parse_horizons_texto(text):
    """
    Parser robusto para el formato de texto de Horizons VECTORS.

    Busca en todo el texto las líneas con X/Y/Z y VX/VY/VZ.
    El formato real de Horizons es:
      " X = 1.234E+00 Y =-9.876E-01 Z = 1.234E-04"
      " VX= 1.234E-02 VY= 1.234E-02 VZ=-1.234E-06"
    """
    import re

    pos = None
    vel = None

    for line in text.split("\n"):
        line_s = line.strip()
        if not line_s:
            continue

        # Línea de posición: contiene X = y Y = (pero NO empieza con V)
        # El regex captura los tres valores aunque haya signo pegado al =
        if pos is None and re.search(r'\bX\s*=', line_s) and re.search(r'\bY\s*=', line_s):
            # Capturar números en notación científica (con posible signo pegado)
            nums = re.findall(r'[XYZ]\s*=\s*([+-]?\d+\.\d+[Ee][+-]?\d+)', line_s)
            if len(nums) >= 3:
                pos = [float(nums[0]), float(nums[1]), float(nums[2])]

        # Línea de velocidad: contiene VX= y VY=
        if vel is None and re.search(r'\bVX\s*=', line_s) and re.search(r'\bVY\s*=', line_s):
            nums = re.findall(r'V[XYZ]\s*=\s*([+-]?\d+\.\d+[Ee][+-]?\d+)', line_s)
            if len(nums) >= 3:
                vel = [float(nums[0]), float(nums[1]), float(nums[2])]

        if pos is not None and vel is not None:
            break

    if pos is not None and vel is not None:
        r = float(np.linalg.norm(pos))
        return {"pos": pos, "vel": vel, "r": r}
    return None


# ─── 2. CARGA DE DATOS DE REFERENCIA ─────────────────────────────────────────

def load_reference_data():
    """
    Carga los datos de referencia guardados.
    Devuelve dict indexado por fecha ISO y nombre de cuerpo.
    """
    if not os.path.exists(REFERENCE_FILE):
        print(f"ERROR: no existe {REFERENCE_FILE}. Ejecuta primero: python benchmark.py --fetch")
        return None

    with open(REFERENCE_FILE) as f:
        datos = json.load(f)

    print(f"✓ Datos de referencia cargados: {len(datos)} fechas × {len(list(datos.values())[0])} cuerpos")
    return datos


# ─── 3. CÁLCULO DE ERROR ─────────────────────────────────────────────────────

def eclipse_error(pos_sol, pos_tierra, pos_luna, t_eclipse_pred=None, referencia=None):
    """
    Calcula las métricas de error de un modelo respecto a JPL.

    Parámetros
    ----------
    pos_sol    : array [3] — posición del Sol    en AU, baricentro eclíptico J2000
    pos_tierra : array [3] — posición de la Tierra en AU
    pos_luna   : array [3] — posición de la Luna   en AU
    t_eclipse_pred : datetime — hora predicha del eclipse (máximo geocéntrico)
                                None si el modelo no predice una hora concreta
    referencia : dict — datos JPL del momento del eclipse (cargados con load_reference_data)
                        None para usar los valores hardcodeados de v10 como referencia

    Devuelve
    --------
    dict con:
      'error_pos_luna_km'   : error en posición de la Luna respecto a la Tierra [km]
      'error_ang_sol_luna'  : error en ángulo de separación Sol-Luna desde Tierra [arcmin]
      'error_tiempo_min'    : error en hora del eclipse [minutos], None si no se predice
      'fase_magnitud'       : magnitud de la fase (ratio R_luna/R_sol aparente)
      'es_total'            : bool, ¿el modelo predice eclipse total?
    """
    pos_sol    = np.array(pos_sol)
    pos_tierra = np.array(pos_tierra)
    pos_luna   = np.array(pos_luna)

    # ── Posición relativa ──
    v_tierra_luna = pos_luna   - pos_tierra
    v_tierra_sol  = pos_sol    - pos_tierra
    d_luna  = np.linalg.norm(v_tierra_luna)   # AU
    d_sol   = np.linalg.norm(v_tierra_sol)    # AU

    # ── Ángulo de separación Sol-Luna (geocéntrico) ──
    cos_sep = np.clip(np.dot(v_tierra_sol, v_tierra_luna) / (d_sol * d_luna), -1, 1)
    sep_grados = np.degrees(np.arccos(cos_sep))
    sep_arcmin = sep_grados * 60.0

    # ── Radios angulares ──
    r_ang_sol  = np.degrees(np.arcsin(np.clip(R_SOL  / d_sol,  0, 1))) * 60  # arcmin
    r_ang_luna = np.degrees(np.arcsin(np.clip(R_LUNA / d_luna, 0, 1))) * 60  # arcmin
    fase_magnitud = r_ang_luna / r_ang_sol  # >1 → total, <1 → anular

    es_total = (fase_magnitud > 1.0) and (sep_arcmin < r_ang_sol)

    # ── Error respecto a referencia JPL ──
    if referencia is not None:
        ref_luna   = np.array(referencia["Luna"]["pos"])
        ref_tierra = np.array(referencia["Tierra"]["pos"])
        error_luna_au = np.linalg.norm((pos_luna - pos_tierra) - (ref_luna - ref_tierra))
        error_luna_km = error_luna_au * AU / KM
        # Error angular: diferencia en separación Sol-Luna
        ref_sol    = np.array(referencia["Sol"]["pos"])
        v_ref_sl   = ref_luna  - ref_tierra
        v_ref_ss   = ref_sol   - ref_tierra
        cos_ref    = np.clip(np.dot(v_ref_ss, v_ref_sl) /
                             (np.linalg.norm(v_ref_ss) * np.linalg.norm(v_ref_sl)), -1, 1)
        sep_ref_arcmin = np.degrees(np.arccos(cos_ref)) * 60.0
        error_ang_arcmin = abs(sep_arcmin - sep_ref_arcmin)
    else:
        # Sin referencia: no podemos calcular error absoluto
        error_luna_km    = None
        error_ang_arcmin = None

    # ── Error temporal ──
    if t_eclipse_pred is not None:
        dt = (t_eclipse_pred - ECLIPSE_UTC).total_seconds() / 60.0
        error_tiempo_min = abs(dt)
    else:
        error_tiempo_min = None

    return {
        "sep_sol_luna_arcmin":  round(sep_arcmin, 4),
        "r_ang_sol_arcmin":     round(r_ang_sol,  4),
        "r_ang_luna_arcmin":    round(r_ang_luna, 4),
        "fase_magnitud":        round(fase_magnitud, 5),
        "es_total":             es_total,
        "d_luna_km":            round(d_luna * AU / KM),
        "error_pos_luna_km":    round(error_luna_km)    if error_luna_km    is not None else None,
        "error_ang_arcmin":     round(error_ang_arcmin, 3) if error_ang_arcmin is not None else None,
        "error_tiempo_min":     round(error_tiempo_min, 1) if error_tiempo_min is not None else None,
    }


# ─── 4. TABLA COMPARATIVA ────────────────────────────────────────────────────

def print_benchmark(resultados):
    """
    Imprime una tabla comparativa de modelos.

    resultados : lista de dicts con claves:
        'nombre'         : str — nombre del modelo
        'metricas'       : dict — salida de eclipse_error()
        'tiempo_cpu_s'   : float — tiempo de integración en segundos
        'descripcion'    : str — descripción breve
    """
    sep = "─" * 95
    print("\n")
    print("╔" + "═" * 93 + "╗")
    print("║  BENCHMARK — Eclipse Solar Total · 12 agosto 2026 · Error vs JPL Horizons DE441" + " " * 11 + "║")
    print("╚" + "═" * 93 + "╝")
    print(f"\n{'Modelo':<30}  {'d_Luna':>10}  {'Err_pos':>12}  {'Err_ang':>12}  {'Err_t':>8}  {'Total?':>7}  {'CPU':>6}")
    print(f"{'':30}  {'km':>10}  {'km':>12}  {'arcmin':>12}  {'min':>8}  {'':>7}  {'s':>6}")
    print(sep)

    for r in resultados:
        m  = r["metricas"]
        d  = m["d_luna_km"]
        ep = f"{m['error_pos_luna_km']:,.0f}" if m["error_pos_luna_km"] is not None else "—"
        ea = f"{m['error_ang_arcmin']:.3f}"   if m["error_ang_arcmin"]  is not None else "—"
        et = f"{m['error_tiempo_min']:.1f}"   if m["error_tiempo_min"]  is not None else "—"
        tt = "✓ TOTAL" if m["es_total"] else "✗ anular"
        cpu= f"{r.get('tiempo_cpu_s', 0):.1f}"

        print(f"{r['nombre']:<30}  {d:>10,}  {ep:>12}  {ea:>12}  {et:>8}  {tt:>7}  {cpu:>6}")

    print(sep)
    print("\nMétricas:")
    print("  d_Luna        → distancia Tierra-Luna en el momento del eclipse predicho")
    print("  Err_pos [km]  → error en posición relativa Luna-Tierra respecto a JPL")
    print("  Err_ang [']   → error en separación angular Sol-Luna geocéntrica respecto a JPL")
    print("  Err_t [min]   → error en hora del máximo eclipse geocéntrico vs NASA (17:47:06 UTC)")
    print("  Total?        → ¿el modelo predice eclipse total desde el centro de la sombra?\n")


# ─── 5. DEMO / VERIFICACIÓN ──────────────────────────────────────────────────

def demo_con_v10():
    """
    Demuestra el benchmark usando los resultados de v10 como modelo de referencia
    (que ya sabemos que da 0 minutos de error).
    Sirve para verificar que el sistema funciona antes de meter los modelos hamiltonianos.
    """
    print("─" * 60)
    print("DEMO: verificando benchmark con datos de v10")
    print("─" * 60)

    # Posiciones del v10 en el momento del eclipse (extraídas del código v10)
    # Estas son las posiciones del sistema en t_ecl ~ 11.78 días desde 2026-Aug-01
    # Centro: baricentro, eclíptica J2000, AU
    # (Valores aproximados reconstruidos de las CI de v10 + integración)
    # NOTA: cuando ejecutes --fetch, estos valores se reemplazarán por los reales de JPL

    # Por ahora usamos los datos de las CI de v10 como proxy
    r_Sol_v10    = np.array([-3.7e-6, 0.0, 0.0])        # muy cerca del baricentro
    r_Tierra_v10 = np.array([6.291e-1, -8.005e-1, 1.46e-4])
    r_Luna_v10   = np.array([6.315e-1, -8.016e-1, 1.73e-4])

    # Hora predicha por v10
    t_pred_v10 = datetime(2026, 8, 12, 17, 47, 0)   # ~17:47 UTC, 0 min de error

    metricas_v10 = eclipse_error(r_Sol_v10, r_Tierra_v10, r_Luna_v10, t_pred_v10)

    print(f"\n  Separación Sol-Luna:   {metricas_v10['sep_sol_luna_arcmin']:.3f} arcmin")
    print(f"  Radio angular Sol:     {metricas_v10['r_ang_sol_arcmin']:.3f} arcmin")
    print(f"  Radio angular Luna:    {metricas_v10['r_ang_luna_arcmin']:.3f} arcmin")
    print(f"  Magnitud de fase:      {metricas_v10['fase_magnitud']:.5f}")
    print(f"  Eclipse total:         {'SÍ ✓' if metricas_v10['es_total'] else 'NO ✗'}")
    print(f"  Distancia Tierra-Luna: {metricas_v10['d_luna_km']:,} km")
    print(f"  Error temporal:        {metricas_v10['error_tiempo_min']} minutos")

    # Tabla con un solo modelo (sin referencia JPL aún)
    resultados_demo = [
        {
            "nombre":       "v10 (Yoshida 10-cuerpos)",
            "metricas":     metricas_v10,
            "tiempo_cpu_s": 38.0,
            "descripcion":  "CI: JPL 01-ago-2026, 12 días integración",
        }
    ]
    print_benchmark(resultados_demo)
    print("(Sin datos JPL descargados, error_pos y error_ang no disponibles)")
    print("\nPara descargar datos de referencia: python benchmark.py --fetch\n")


def comparar_todos():
    """Ejecuta los cuatro modelos secuencialmente y los compara en una tabla."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Modelos'))
    
    import modelo1_3cuerpos as m1
    import modelo2_ncuerpos as m2
    import modelo3_relativista as m3
    import modelo4_besseliano as m4
    
    print("─" * 60)
    print("EJECUTANDO BENCHMARK GLOBAL (h=0.05 días)")
    print("─" * 60)
    
    ref = load_reference_data()
    ci_ene = ref.get(datetime(2026,1,1).isoformat()) if ref else None
    
    resultados = []
    h_step = 0.05
    
    print("\n[1/4] Ejecutando Modelo 1 (3-Cuerpos + J2 + Rotación)...")
    r1 = m1.run(ci_dict=ci_ene, h=h_step, verbose=False)
    if r1: resultados.append(r1)
    
    print("[2/4] Ejecutando Modelo 2 (N-Cuerpos)...")
    r2 = m2.run(h=h_step, verbose=False)
    if r2: resultados.append(r2)
    
    print("[3/4] Ejecutando Modelo 3 (N-Cuerpos + Relatividad EIH)...")
    r3 = m3.run(h=h_step, verbose=False)
    if r3: resultados.append(r3)
    
    print("[4/4] Ejecutando Modelo 4 (Híbrido Simulador + Geometría NASA)...")
    r4 = m4.run(h=h_step, verbose=False)
    if r4: resultados.append(r4)
    
    if resultados:
        print_benchmark(resultados)


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark para modelos del eclipse 2026")
    parser.add_argument("--fetch", action="store_true",
                        help="Descargar datos de referencia de JPL Horizons")
    parser.add_argument("--check", action="store_true",
                        help="Verificar datos descargados")
    parser.add_argument("--demo",  action="store_true",
                        help="Demo con datos de v10")
    parser.add_argument("--all",   action="store_true",
                        help="Ejecutar y comparar los 4 modelos finales")
    args = parser.parse_args()

    if args.fetch:
        fetch_reference_data()
    elif args.check:
        verificar_datos()
    elif args.demo:
        demo_con_v10()
    elif args.all:
        comparar_todos()
    else:
        parser.print_help()
        print("\nEjemplos:")
        print("  python benchmark.py --fetch    # descargar datos JPL (necesita internet)")
        print("  python benchmark.py --check    # ver datos descargados")
        print("  python benchmark.py --demo     # probar el sistema sin datos JPL")