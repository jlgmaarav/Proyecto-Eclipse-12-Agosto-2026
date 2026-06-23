"""
═══════════════════════════════════════════════════════════════════════════════
CÓMO USAR EL BENCHMARK EN CADA MODELO
═══════════════════════════════════════════════════════════════════════════════

Este archivo muestra el patrón que seguirá cada modelo hamiltoniano para
reportar sus resultados al benchmark. Cópialo y adapta la parte marcada.
"""

import numpy as np
import time
from datetime import datetime
from benchmark import eclipse_error, print_benchmark, load_reference_data

# ─── CONSTANTES (mismas que en benchmark.py) ─────────────────────────────────
AU  = 1.495978707e11
DAY = 86400.0
G   = 6.6743e-11 * DAY**2 / AU**3

# ─── PARÁMETROS COMUNES A TODOS LOS MODELOS ──────────────────────────────────
T_INICIO   = datetime(2026, 1, 1)    # todos parten del 1 enero
T_ECLIPSE  = datetime(2026, 8, 12, 17, 47, 6)
DIAS_TOTAL = (T_ECLIPSE - T_INICIO).days + 1  # ~223 días

# ─────────────────────────────────────────────────────────────────────────────
# CONDICIONES INICIALES COMPARTIDAS (JPL DE441, 2026-Jan-01 00:00 TDB)
# Plano eclíptico J2000, baricentro solar, AU y AU/día
#
# INSTRUCCIONES: ejecuta benchmark.py --fetch para obtener estos valores exactos.
# Por ahora hay valores placeholder para que el código ejecute.
# ─────────────────────────────────────────────────────────────────────────────

CI = {
    # ── REEMPLAZAR CON VALORES REALES DE JPL ──────────────────────
    # Formato: {"pos": [x, y, z], "vel": [vx, vy, vz]}   (AU y AU/día)
    "Sol":      {"pos": [0.0, 0.0, 0.0],                   "vel": [0.0, 0.0, 0.0]},
    "Tierra":   {"pos": [-0.1777, 0.9672, 0.0000],         "vel": [-0.01724, -0.00308, 0.0]},
    "Luna":     {"pos": [-0.1804, 0.9659, 0.0002],         "vel": [-0.01700, -0.00358, 0.00005]},
    "Mercurio": {"pos": [-0.3863, -0.1419, 0.0254],        "vel": [0.00532, -0.02816, -0.00284]},
    "Venus":    {"pos": [-0.6910, -0.2889, 0.0362],        "vel": [0.00762, -0.01936, -0.000496]},
    "Marte":    {"pos": [1.3923,  0.0371, -0.0310],        "vel": [-0.000817, 0.01523, 0.000337]},
    "Jupiter":  {"pos": [-3.8048, 3.4625, 0.0689],         "vel": [-0.00556, -0.00490, 0.000143]},
    "Saturno":  {"pos": [7.1888, 5.7124, -0.3730],         "vel": [-0.00399, 0.00437, -0.0000177]},
    "Urano":    {"pos": [14.290, -13.244, -0.2265],        "vel": [0.002686, 0.002532, -0.0000306]},
    "Neptuno":  {"pos": [29.816, -1.7680, -0.6468],        "vel": [0.000182, 0.003144, -0.0000712]},
    # ── FIN PLACEHOLDER ───────────────────────────────────────────
}

MASAS = {
    "Sol":      1.98892e30,
    "Mercurio": 3.30200e23,
    "Venus":    4.86850e24,
    "Tierra":   5.97219e24,
    "Luna":     7.34581e22,
    "Marte":    6.41712e23,
    "Jupiter":  1.89813e27,
    "Saturno":  5.68319e26,
    "Urano":    8.68103e25,
    "Neptuno":  1.02410e26,
}


# ─────────────────────────────────────────────────────────────────────────────
# PLANTILLA DE MODELO — COPIA Y ADAPTA ESTO PARA CADA VERSIÓN
# ─────────────────────────────────────────────────────────────────────────────

def modelo_placeholder():
    """
    Plantilla que muestra dónde conectar el modelo con el benchmark.
    Sustituye el contenido de run_integracion() con tu modelo real.
    """

    def run_integracion():
        """
        ── AQUÍ VA TU INTEGRADOR ──────────────────────────────────────────────
        Debe devolver:
          pos_sol    : array [3] — posición Sol    en AU al momento del eclipse
          pos_tierra : array [3] — posición Tierra en AU
          pos_luna   : array [3] — posición Luna   en AU
          t_pred     : datetime  — hora predicha del eclipse
        ──────────────────────────────────────────────────────────────────────
        """
        # Aquí irá tu código de integración (Modelo 1, 2, 3...)
        pos_sol    = np.array(CI["Sol"]["pos"])
        pos_tierra = np.array(CI["Tierra"]["pos"])
        pos_luna   = np.array(CI["Luna"]["pos"])
        t_pred     = None   # el modelo aún no predice una hora

        return pos_sol, pos_tierra, pos_luna, t_pred

    # Ejecutar y cronometrar
    t0 = time.time()
    pos_sol, pos_tierra, pos_luna, t_pred = run_integracion()
    cpu = time.time() - t0

    # Cargar referencia JPL (None si no se ha descargado aún)
    ref_datos = load_reference_data()
    ref_eclipse = None
    if ref_datos:
        clave_eclipse = T_ECLIPSE.isoformat()
        if clave_eclipse in ref_datos:
            ref_eclipse = ref_datos[clave_eclipse]

    # Calcular métricas
    metricas = eclipse_error(pos_sol, pos_tierra, pos_luna, t_pred, ref_eclipse)

    return {
        "nombre":       "NOMBRE DEL MODELO",
        "metricas":     metricas,
        "tiempo_cpu_s": cpu,
        "descripcion":  "Descripción breve",
    }


# ─────────────────────────────────────────────────────────────────────────────
# EJECUTAR TODOS LOS MODELOS Y COMPARAR
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nEjecutando modelos...")

    resultados = []

    # Cuando cada modelo esté listo, añádelo aquí:
    # from modelo1_3cuerpos import run as run_m1
    # resultados.append(run_m1())

    # Por ahora, placeholder:
    resultados.append(modelo_placeholder())

    print_benchmark(resultados)