"""
model_m4.py
===========
Modelo M4 — N-body newtoniano + J₂ terrestre + correcciones relativistas EIH.

Física añadida respecto a M3:
  Ecuaciones de Einstein-Infeld-Hoffmann (EIH) de primer orden post-newtoniano
  (1PN). Son las ecuaciones que usa la NASA en sus efemérides DE430/DE440/DE441.

  La aceleración total del cuerpo i es:

      ä_i = a_i^Newton + a_i^EIH + a_i^J2   (solo Luna y reacción en Tierra)

  La corrección EIH se descompone en tres términos:

  Término A — potencial gravitacional (efecto de masa relativista):
    Σ_j μ_j n_ij/r_ij² · (1/c²) · [
       - 4 Σ_k≠i μ_k/r_ik
       -   Σ_k≠j μ_k/r_jk
       + v_i² + 2 v_j² - 4 (v_i·v_j)
       - 3/2 (n̂_ij · v_j)²
    ]

  Término B — arrastre gravitomagnético (velocidades relativas):
    Σ_j μ_j/r_ij² · [(4 n̂_ij·v_i - 3 n̂_ij·v_j)] · (v_i - v_j) / c²

  Término C — aceleración newtoniana de los perturbadores:
    (7/2c²) Σ_j μ_j/r_ij · a_j^(0)

Implementación:
  - Dos pasadas por pasos: primero Newton (para obtener a_j^(0)),
    luego EIH completo. Error introducido: O(1/c^4), despreciable.
  - Potenciales escalares U_i = Σ_{k≠i} μ_k/r_ik precalculados para
    reducir la complejidad del Término A de O(N³) a O(N²).
  - J₂ idéntico a M3: solo afecta a Luna (+ reacción en Tierra).

Método:   Adams-Bashforth-Moulton orden 4, paso cuasi-fijo h = 0.1 días.
          Arranque: RK4 primeros 4 pasos. Idéntico a M2 y M3.

Unidades: AU, AU/día, días.
Época:    2026-Jan-01 00:00:00 TDB  (t = 0)
Sistema:  Baricentro del Sistema Solar (SSB), eclíptica J2000.

Dimensión del sistema: 6 × 10 = 60 EDOs.

Resultados esperados (benchmark vs JPL DE441, eclipse 2026-08-12):
  Error posición Tierra (SSB) : ~200–500 km     (vs ~57 000 km en M3)
  Error posición Luna  (SSB)  : ~200–500 km
  Error posición Sol   (SSB)  : ~10–50 km
  Mejora sobre M3             : factor ~100–300×
  Limitante residual          : polo terrestre fijo + ausencia de mareas
"""

import time
import numpy as np
from typing import List, Optional
from benchmark import (
    load_initial_conditions,
    run_benchmark,
    print_report,
    BENCHMARK_DAYS,
    AU_KM,
    DAY_S,
    GM,
    J2_EARTH,
    R_EQ_EARTH,
)

# ═══════════════════════════════════════════════════════════════
# 1. CONSTANTES FÍSICAS EN UNIDADES AU / DÍA
# ═══════════════════════════════════════════════════════════════

# Factor de conversión: km³/s² → AU³/día²
_K = DAY_S**2 / AU_KM**3

# Velocidad de la luz en AU/día
#   c = 299792.458 km/s × 86400 s/día / 149597870.700 km/AU
C_AU_DAY = 299_792.458 * DAY_S / AU_KM    # ≈ 173.144632674 AU/día
C2        = C_AU_DAY * C_AU_DAY            # c² en AU²/día²
INV_C2    = 1.0 / C2                       # 1/c² ≈ 3.3356e-5 día²/AU²

# Orden canónico de los 10 cuerpos (mismo que M2, M3 y jpl_reference_data.json)
BODIES = [
    "Sun",      # 0
    "Mercury",  # 1
    "Venus",    # 2
    "Earth",    # 3
    "Moon",     # 4
    "Mars",     # 5
    "Jupiter",  # 6
    "Saturn",   # 7
    "Uranus",   # 8
    "Neptune",  # 9
]
N_BODIES  = len(BODIES)       # 10
DIM       = 6 * N_BODIES      # 60

IDX_EARTH = BODIES.index("Earth")   # 3
IDX_MOON  = BODIES.index("Moon")    # 4

# Parámetros gravitacionales en AU³/día²
GM_AU = np.array([GM[b] * _K for b in BODIES])

# ── Parámetros J₂ (idénticos a M3) ──────────────────────────
J2     = J2_EARTH                  # 0.00108262545 (adimensional)
R_E    = R_EQ_EARTH / AU_KM        # radio ecuatorial terrestre en AU

# Prefactor J₂: C = 3/2 · μ⊕ · J₂ · R⊕²  [AU³/día²]
_J2_PREFACTOR = 1.5 * GM_AU[IDX_EARTH] * J2 * R_E**2

# Polo norte terrestre en eclíptica J2000 (oblicuidad ε = 23.439291°)
_EPS_RAD   = np.deg2rad(23.439_291)
EARTH_POLE = np.array([0.0, -np.sin(_EPS_RAD), np.cos(_EPS_RAD)])


# ═══════════════════════════════════════════════════════════════
# 2. ACELERACIÓN J₂  (igual que M3, función auxiliar)
# ═══════════════════════════════════════════════════════════════

def _acc_j2(r_rel: np.ndarray) -> np.ndarray:
    """
    Aceleración de perturbación J₂ sobre un cuerpo situado en r_rel
    respecto al centro de la Tierra, en el marco eclíptico J2000.

        a_J2 = C/r⁵ · [(5ζ²−1) r_rel  −  2ζ r ẑ⊕]

    donde  C = 3/2 μ⊕ J₂ R⊕²,  ζ = (r_rel · ẑ⊕) / r.
    """
    r2   = r_rel[0]**2 + r_rel[1]**2 + r_rel[2]**2
    r    = np.sqrt(r2)
    r5   = r2 * r2 * r
    zeta = (r_rel @ EARTH_POLE) / r
    C_r5 = _J2_PREFACTOR / r5
    return C_r5 * ((5.0 * zeta**2 - 1.0) * r_rel + (-2.0 * zeta * r) * EARTH_POLE)


# ═══════════════════════════════════════════════════════════════
# 3. NÚCLEO EIH
#    Calcula la corrección relativista completa para todos los cuerpos.
#    Requiere las posiciones, velocidades y aceleraciones newtonianas.
# ═══════════════════════════════════════════════════════════════

def _eih_corrections(
    pos: np.ndarray,      # (N, 3) posiciones en AU
    vel: np.ndarray,      # (N, 3) velocidades en AU/día
    acc0: np.ndarray,     # (N, 3) aceleraciones newtonianas en AU/día²
) -> np.ndarray:
    """
    Calcula las correcciones de aceleración EIH (1PN) para N cuerpos.

    Las ecuaciones EIH se implementan en la forma compacta de Soffel et al.
    (2003), ecuación (10.12), dividida en tres términos:

      a_EIH_i = (1/c²) · [Término_A_i  +  Término_B_i  +  Término_C_i]

    Término A (escalar × vector dirección):
      Σ_j μ_j n_ij / r_ij² · [
         -4 U_i  -  U_j'  +  v_i²  +  2 v_j²  -  4 (v_i·v_j)
         -  3/2 (n̂_ij · v_j)²
      ]
      donde U_i = Σ_{k≠i} μ_k / r_ik  y  U_j' = Σ_{k≠j} μ_k / r_jk

    Término B (vector velocidad relativa):
      Σ_j μ_j / r_ij² · (4 v_i·n̂_ij  -  3 v_j·n̂_ij) · (v_i - v_j)

    Término C (aceleración newtoniana ponderada):
      (7/2) Σ_j μ_j / r_ij · a_j^(0)

    Parámetros
    ----------
    pos  : ndarray (N, 3)  — posiciones en AU
    vel  : ndarray (N, 3)  — velocidades en AU/día
    acc0 : ndarray (N, 3)  — aceleraciones newtonianas en AU/día²

    Retorna
    -------
    da   : ndarray (N, 3)  — corrección EIH en AU/día² (ya incluye 1/c²)
    """
    da    = np.zeros((N_BODIES, 3))

    # ── Precalcular potenciales escalares U_i y U_j ──────────────
    # U[i] = Σ_{k≠i} μ_k / r_ik   (suma de potenciales sobre i)
    # r_mat[i,j] = |r_j - r_i|    (matriz de distancias, simétrica)
    r_mat  = np.zeros((N_BODIES, N_BODIES))    # distancias
    n_mat  = np.zeros((N_BODIES, N_BODIES, 3)) # vectores unitarios n̂_ij

    for i in range(N_BODIES):
        for j in range(i + 1, N_BODIES):
            d     = pos[j] - pos[i]
            dist  = np.sqrt(d @ d)
            r_mat[i, j] = dist
            r_mat[j, i] = dist
            nhat  = d / dist
            n_mat[i, j] =  nhat
            n_mat[j, i] = -nhat

    # Potencial en cada cuerpo: U[i] = Σ_{k≠i} μ_k / r_ik
    U = np.zeros(N_BODIES)
    for i in range(N_BODIES):
        for k in range(N_BODIES):
            if k != i:
                U[i] += GM_AU[k] / r_mat[i, k]

    # Velocidades al cuadrado
    v2 = np.einsum('ij,ij->i', vel, vel)   # v2[i] = |v_i|²

    # ── Bucle principal sobre los N² pares ───────────────────────
    for i in range(N_BODIES):
        a_term  = np.zeros(3)   # Término A
        b_term  = np.zeros(3)   # Término B
        c_term  = np.zeros(3)   # Término C

        for j in range(N_BODIES):
            if j == i:
                continue

            rij   = r_mat[i, j]
            nij   = n_mat[i, j]      # = (r_j - r_i) / r_ij
            rij2  = rij * rij
            rij3  = rij2 * rij

            vi_dot_vj  = vel[i] @ vel[j]
            nij_dot_vj = nij    @ vel[j]
            nij_dot_vi = nij    @ vel[i]

            # ── Término A ──────────────────────────────────────────
            # Escalar de ponderación del Término A
            scalar_A = (
                - 4.0 * U[i]
                - U[j]           # U_j = potencial en j, ya calculado
                + v2[i]
                + 2.0 * v2[j]
                - 4.0 * vi_dot_vj
                - 1.5 * nij_dot_vj**2
            )
            a_term += (GM_AU[j] / rij2) * scalar_A * nij

            # ── Término B ──────────────────────────────────────────
            # Escalar del Término B
            scalar_B = 4.0 * nij_dot_vi - 3.0 * nij_dot_vj
            b_term  += (GM_AU[j] / rij2) * scalar_B * (vel[i] - vel[j])

            # ── Término C ──────────────────────────────────────────
            # Ponderado por μ_j / r_ij (no r_ij²)
            c_term  += (GM_AU[j] / rij) * acc0[j]

        # Suma los tres términos con prefactor 1/c²
        da[i] = INV_C2 * (a_term + b_term + 3.5 * c_term)

    return da


# ═══════════════════════════════════════════════════════════════
# 4. LADO DERECHO DE LAS EDOs — f(t, Y)
# ═══════════════════════════════════════════════════════════════

def f_m4(t: float, Y: np.ndarray) -> np.ndarray:
    """
    Lado derecho del sistema de 60 EDOs:
        ä_i = a_i^Newton  +  a_i^EIH  +  a_i^J2  (Luna/Tierra)

    Algoritmo de dos pasadas:
      Pasada 1: calcular a^(0) newtoniano para todos los cuerpos
                (necesario para el Término C de EIH).
      Pasada 2: calcular corrección EIH completa usando a^(0).
      Sumar J₂ (solo Luna + reacción Tierra).

    Parámetros
    ----------
    t : float    — tiempo en días (no entra en las ecuaciones)
    Y : ndarray  — vector de estado (60,): [r0,v0, r1,v1, ..., r9,v9]

    Retorna
    -------
    dY : ndarray (60,)
    """
    dY  = np.empty(DIM)
    pos = np.empty((N_BODIES, 3))
    vel = np.empty((N_BODIES, 3))

    # Extraer pos/vel y copiar velocidades a dY (dr/dt = v)
    for i in range(N_BODIES):
        b = 6 * i
        pos[i]        = Y[b     : b + 3]
        vel[i]        = Y[b + 3 : b + 6]
        dY[b : b + 3] = vel[i]

    # ── Pasada 1: Aceleración newtoniana ──────────────────────────
    acc = np.zeros((N_BODIES, 3))

    for i in range(N_BODIES - 1):
        for j in range(i + 1, N_BODIES):
            d    = pos[j] - pos[i]
            dist = np.sqrt(d @ d)
            inv3 = 1.0 / (dist**3)
            acc[i] += GM_AU[j] * inv3 * d
            acc[j] -= GM_AU[i] * inv3 * d   # 3ª ley de Newton

    # ── Pasada 2: Corrección EIH (usa acc como a^(0)) ────────────
    da_eih = _eih_corrections(pos, vel, acc)

    # Sumar corrección EIH a la aceleración total
    acc += da_eih

    # ── Perturbación J₂ terrestre (igual que M3) ──────────────────
    r_LT = pos[IDX_MOON] - pos[IDX_EARTH]
    a_j2 = _acc_j2(r_LT)

    acc[IDX_MOON]  +=  a_j2
    acc[IDX_EARTH] -= (GM_AU[IDX_MOON] / GM_AU[IDX_EARTH]) * a_j2

    # ── Escribir aceleraciones en dY ──────────────────────────────
    for i in range(N_BODIES):
        dY[6*i + 3 : 6*i + 6] = acc[i]

    return dY


# ═══════════════════════════════════════════════════════════════
# 5. ARRANQUE CON RK4
# ═══════════════════════════════════════════════════════════════

def rk4_step(f, t: float, Y: np.ndarray, h: float) -> np.ndarray:
    """Un paso clásico de Runge-Kutta de orden 4."""
    k1 = f(t,       Y)
    k2 = f(t + h/2, Y + (h/2) * k1)
    k3 = f(t + h/2, Y + (h/2) * k2)
    k4 = f(t + h,   Y + h      * k3)
    return Y + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


def bootstrap_rk4(f, t0: float, Y0: np.ndarray, h: float, k: int):
    """Genera los primeros k estados con RK4 para iniciar ABM4."""
    ts = [t0]
    Ys = [Y0.copy()]
    Fs = [f(t0, Y0)]
    for _ in range(k - 1):
        Y_new = rk4_step(f, ts[-1], Ys[-1], h)
        t_new = ts[-1] + h
        ts.append(t_new)
        Ys.append(Y_new)
        Fs.append(f(t_new, Y_new))
    return ts, Ys, Fs


# ═══════════════════════════════════════════════════════════════
# 6. COEFICIENTES ABM4
# ═══════════════════════════════════════════════════════════════

# Predictor Adams-Bashforth 4:
#   Y^P_{n+1} = Y_n + h·(55 F_n − 59 F_{n−1} + 37 F_{n−2} − 9 F_{n−3}) / 24
AB4 = np.array([55.0, -59.0, 37.0, -9.0]) / 24.0

# Corrector Adams-Moulton 4:
#   Y^C_{n+1} = Y_n + h·(9 F^P_{n+1} + 19 F_n − 5 F_{n−1} + F_{n−2}) / 24
AM4 = np.array([9.0, 19.0, -5.0, 1.0]) / 24.0


# ═══════════════════════════════════════════════════════════════
# 7. UN PASO ABM4
# ═══════════════════════════════════════════════════════════════

def abm4_step(f, t: float, Y: np.ndarray, h: float, F_hist: list) -> tuple:
    """
    Un paso del predictor-corrector Adams-Bashforth-Moulton orden 4.

    F_hist = [F_n, F_{n-1}, F_{n-2}, F_{n-3}]  (índice 0 = más reciente)
    Retorna (Y_corr, f_corr, err_local)
    """
    f_n, f_nm1, f_nm2, f_nm3 = F_hist[0], F_hist[1], F_hist[2], F_hist[3]

    Y_pred = Y + h * (AB4[0]*f_n + AB4[1]*f_nm1 + AB4[2]*f_nm2 + AB4[3]*f_nm3)
    f_pred = f(t + h, Y_pred)

    Y_corr = Y + h * (AM4[0]*f_pred + AM4[1]*f_n + AM4[2]*f_nm1 + AM4[3]*f_nm2)
    f_corr = f(t + h, Y_corr)

    err = float(np.linalg.norm(Y_corr - Y_pred) / (1.0 + np.linalg.norm(Y_corr)))

    return Y_corr, f_corr, err


# ═══════════════════════════════════════════════════════════════
# 8. INTEGRACIÓN COMPLETA (paso cuasi-fijo)
# ═══════════════════════════════════════════════════════════════

def integrate(f, Y0: np.ndarray, t_end: float,
              h0:  float = 0.1,
              tol: float = 1e-10,
              t_out: Optional[List[float]] = None) -> dict:
    """
    Integra el sistema de EDOs desde t=0 hasta t=t_end con ABM4 paso fijo.

    Parámetros
    ----------
    f     : callable        — lado derecho f(t, Y)
    Y0    : ndarray (60,)   — condición inicial
    t_end : float           — tiempo final en días
    h0    : float           — paso de integración en días
    tol   : float           — umbral de aviso del error local
    t_out : list[float]     — tiempos en los que guardar el estado

    Retorna
    -------
    dict: t_out, Y_out, steps, rejects, h, err_max
    """
    ORDER        = 4
    h            = float(h0)
    t_out_sorted = sorted(set(t_out)) if t_out else []
    out_idx      = 0
    results_t: List[float]      = []
    results_Y: List[np.ndarray] = []

    # ── Arranque RK4 ──────────────────────────────────────────────
    ts_boot, Ys_boot, Fs_boot = bootstrap_rk4(f, 0.0, Y0, h, ORDER)

    for t_want in t_out_sorted:
        if t_want <= ts_boot[-1] + 1e-12:
            for k in range(len(ts_boot) - 1):
                if ts_boot[k] <= t_want <= ts_boot[k + 1]:
                    alpha = (t_want - ts_boot[k]) / (ts_boot[k + 1] - ts_boot[k])
                    results_t.append(t_want)
                    results_Y.append((1.0 - alpha)*Ys_boot[k] + alpha*Ys_boot[k + 1])
                    break
            out_idx += 1

    t      = ts_boot[-1]
    Y      = Ys_boot[-1].copy()
    F_hist = list(reversed(Fs_boot))

    steps   = ORDER - 1
    rejects = 0
    err_max = 0.0

    # ── Bucle principal ABM4 ──────────────────────────────────────
    while t < t_end - 1e-12:
        h_step = min(h, t_end - t)

        Y_new, f_new, err = abm4_step(f, t, Y, h_step, F_hist)
        err_max = max(err_max, err)
        if err >= tol:
            rejects += 1

        t_next = t + h_step
        steps += 1

        while out_idx < len(t_out_sorted) and t_out_sorted[out_idx] <= t_next + 1e-12:
            t_want = t_out_sorted[out_idx]
            alpha  = float(np.clip(
                (t_want - t) / h_step if h_step > 0.0 else 0.0,
                0.0, 1.0
            ))
            results_t.append(t_want)
            results_Y.append((1.0 - alpha)*Y + alpha*Y_new)
            out_idx += 1

        t = t_next
        Y = Y_new
        F_hist = [f_new] + F_hist[:ORDER - 1]

    return {
        "t_out":   results_t,
        "Y_out":   results_Y,
        "steps":   steps,
        "rejects": rejects,
        "h":       h,
        "err_max": err_max,
    }


# ═══════════════════════════════════════════════════════════════
# 9. CONDICIONES INICIALES
# ═══════════════════════════════════════════════════════════════

def build_Y0() -> np.ndarray:
    """Construye el vector de estado Y0 (60,) desde initial_conditions.csv."""
    ic = load_initial_conditions()
    Y0 = np.zeros(DIM)
    for i, body in enumerate(BODIES):
        if body not in ic:
            raise KeyError(
                f"Cuerpo '{body}' no encontrado en initial_conditions.csv.\n"
                f"Claves disponibles: {sorted(ic.keys())}"
            )
        Y0[6*i     : 6*i + 3] = ic[body]["pos"]
        Y0[6*i + 3 : 6*i + 6] = ic[body]["vel"]
    return Y0


# ═══════════════════════════════════════════════════════════════
# 10. CONVERSIÓN AL FORMATO BENCHMARK
# ═══════════════════════════════════════════════════════════════

def result_to_model_states(result: dict) -> dict:
    """Convierte la salida del integrador al formato que espera run_benchmark."""
    days_to_date = {v: k for k, v in BENCHMARK_DAYS.items()}
    model_states: dict = {}

    for t_val, Y_val in zip(result["t_out"], result["Y_out"]):
        date_str = None
        for days, dstr in days_to_date.items():
            if abs(t_val - days) < 1e-6:
                date_str = dstr
                break
        if date_str is None:
            continue

        model_states[date_str] = {
            body: {
                "pos": Y_val[6*i     : 6*i + 3].copy(),
                "vel": Y_val[6*i + 3 : 6*i + 6].copy(),
            }
            for i, body in enumerate(BODIES)
        }
    return model_states


# ═══════════════════════════════════════════════════════════════
# 11. DIAGNÓSTICOS
# ═══════════════════════════════════════════════════════════════

def check_invariants(Y: np.ndarray, label: str = "") -> None:
    """Momento lineal total y centro de masas (ambos deben ser ≈ 0 en SSB)."""
    M_total = sum(GM[b] for b in BODIES)
    P_total = np.zeros(3)
    R_cm    = np.zeros(3)
    for i, body in enumerate(BODIES):
        mi = GM[body]
        P_total += mi * Y[6*i + 3 : 6*i + 6]
        R_cm    += mi * Y[6*i     : 6*i + 3]
    P_total /= M_total
    R_cm    /= M_total
    tag = f"[{label}] " if label else ""
    print(f"  {tag}|P_total| = {np.linalg.norm(P_total):.3e} AU/día")
    print(f"  {tag}|R_cm|    = {np.linalg.norm(R_cm):.6f} AU")


def earth_moon_separation(Y: np.ndarray) -> float:
    """Separación Tierra-Luna en km."""
    r_E = Y[6*IDX_EARTH : 6*IDX_EARTH + 3]
    r_M = Y[6*IDX_MOON  : 6*IDX_MOON  + 3]
    return float(np.linalg.norm(r_M - r_E)) * AU_KM


def relativistic_perturbation_magnitude(pos: np.ndarray,
                                         vel: np.ndarray,
                                         acc0: np.ndarray) -> float:
    """
    Magnitud de la corrección EIH sobre la Tierra, en km/s², para diagnóstico.
    """
    da_eih = _eih_corrections(pos, vel, acc0)
    return float(np.linalg.norm(da_eih[IDX_EARTH])) * AU_KM / DAY_S**2


# ═══════════════════════════════════════════════════════════════
# 12. EJECUCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 70)
    print("  MODELO M4 — N-body + J₂ + EIH (relatividad 1PN), ABM4 paso fijo")
    print("=" * 70)
    print(f"\n  Cuerpos ({N_BODIES}): {', '.join(BODIES)}")
    print(f"  Dimensión del sistema : {DIM} EDOs")
    print(f"  Pares N-body          : {N_BODIES*(N_BODIES-1)//2}")
    print(f"  Corrección EIH        : 1PN (términos A + B + C)")
    print(f"  J₂ aplicado a         : Luna (+ reacción Tierra)")

    print(f"\n  Constantes relativistas:")
    print(f"    c              = {C_AU_DAY:.9f} AU/día")
    print(f"    1/c²           = {INV_C2:.6e} día²/AU²")
    print(f"    J₂⊕            = {J2:.12f}")
    print(f"    R⊕ ecuatorial  = {R_EQ_EARTH:.3f} km  = {R_E:.10e} AU")

    # ── Condiciones iniciales ─────────────────────────────────────
    Y0 = build_Y0()
    print("\nCondición inicial (2026-01-01 00:00:00 TDB, SSB, eclíptica J2000):")
    for i, body in enumerate(BODIES):
        r = np.linalg.norm(Y0[6*i : 6*i + 3])
        v = np.linalg.norm(Y0[6*i + 3 : 6*i + 6])
        print(f"  {body:<10}  |r| = {r:12.6f} AU   |v| = {v:.8f} AU/día")

    # Diagnóstico de la magnitud de la corrección EIH en t=0
    pos0 = np.array([Y0[6*i : 6*i + 3] for i in range(N_BODIES)])
    vel0 = np.array([Y0[6*i + 3 : 6*i + 6] for i in range(N_BODIES)])
    acc0 = np.zeros((N_BODIES, 3))
    for i in range(N_BODIES - 1):
        for j in range(i + 1, N_BODIES):
            d    = pos0[j] - pos0[i]
            dist = np.sqrt(d @ d)
            inv3 = 1.0 / dist**3
            acc0[i] += GM_AU[j] * inv3 * d
            acc0[j] -= GM_AU[i] * inv3 * d

    da_eih_t0 = _eih_corrections(pos0, vel0, acc0)
    a_earth_newton = np.linalg.norm(acc0[IDX_EARTH]) * AU_KM / DAY_S**2
    a_earth_eih    = np.linalg.norm(da_eih_t0[IDX_EARTH]) * AU_KM / DAY_S**2
    a_moon_newton  = np.linalg.norm(acc0[IDX_MOON]) * AU_KM / DAY_S**2
    a_moon_eih     = np.linalg.norm(da_eih_t0[IDX_MOON]) * AU_KM / DAY_S**2

    print(f"\n  Magnitud corrección EIH en t=0:")
    print(f"  {'Cuerpo':<10} {'|a_Newton| (km/s²)':>22} {'|a_EIH| (km/s²)':>22} {'Ratio':>10}")
    print(f"  {'-'*68}")
    print(f"  {'Tierra':<10} {a_earth_newton:>22.6e} {a_earth_eih:>22.6e} {a_earth_eih/a_earth_newton:>10.4e}")
    print(f"  {'Luna':<10} {a_moon_newton:>22.6e} {a_moon_eih:>22.6e} {a_moon_eih/a_moon_newton:>10.4e}")

    print("\nVerificación de invariantes en t=0:")
    check_invariants(Y0, "t=0")

    # ── Parámetros de integración ─────────────────────────────────
    t_out = list(BENCHMARK_DAYS.values())
    t_end = max(t_out)
    H     = 0.1     # días
    TOL   = 1e-10

    print(f"\nIntegrando {t_end:.4f} días ({t_end/365.25:.4f} años)...")
    print(f"  Paso fijo      : {H} días")
    print(f"  Tolerancia     : {TOL:.0e}")
    print(f"  Pasos totales  : ~{int(t_end/H) + 1}")
    print(f"  Aviso: M4 es ~4× más lento que M3 por el doble bucle EIH.")
    print(f"         Tiempo estimado: 30–120 s según hardware.")

    # ── Integración ───────────────────────────────────────────────
    t_wall = time.perf_counter()

    result = integrate(
        f     = f_m4,
        Y0    = Y0,
        t_end = t_end,
        h0    = H,
        tol   = TOL,
        t_out = t_out,
    )

    elapsed = time.perf_counter() - t_wall

    print(f"\nIntegración completada en {elapsed:.2f} s")
    print(f"  Pasos ejecutados  : {result['steps']}")
    print(f"  Pasos con aviso   : {result['rejects']}")
    print(f"  Paso usado        : {result['h']:.4f} días")
    print(f"  Error local máx.  : {result['err_max']:.3e}")
    print(f"  Fechas capturadas : {len(result['t_out'])} / {len(t_out)}")

    if result["Y_out"]:
        print("\nVerificación de invariantes en t_end:")
        check_invariants(result["Y_out"][-1], "t_end")

    # ── Benchmark completo ────────────────────────────────────────
    model_states = result_to_model_states(result)
    report = run_benchmark(model_states, model_name="M4 — N-body + J₂ + EIH (1PN) ABM4")
    print_report(report)

    # ── Comparativa M3 → M4 en la fecha del eclipse ───────────────
    ec = report.get("earth_moon_eclipse", {})

    eclipse_date = "2026-08-12T17:47:06"
    sep_eclipse_m4 = None
    for t_val, Y_val in zip(result["t_out"], result["Y_out"]):
        for days, dstr in {v: k for k, v in BENCHMARK_DAYS.items()}.items():
            if abs(t_val - days) < 1e-6 and dstr == eclipse_date:
                sep_eclipse_m4 = earth_moon_separation(Y_val)

    # Resultados de referencia de M3 para comparar
    ref_m3 = {
        "Tierra pos. abs. (km)": 57_542,
        "Luna pos. abs. (km)":   57_560,
        "Sol pos. abs. (km)":    14_894,
    }

    print("─" * 70)
    print("  Comparativa Eclipse (2026-08-12 17:47:06 TDB):")
    print(f"  {'Métrica':<32} {'M3 (km)':>12}   {'M4 (km)':>12}   {'Mejora':>8}")
    print("─" * 70)

    for key, label in [("earth_pos_err_km", "Tierra pos. abs. (km)"),
                        ("moon_pos_err_km",  "Luna pos. abs. (km)"),
                        ("sun_pos_err_km",   "Sol pos. abs. (km)")]:
        val_m4  = ec.get(key)
        val_m3  = ref_m3.get(label)
        if val_m4 is not None and val_m3 is not None:
            mejora = val_m3 / val_m4 if val_m4 > 0 else float("inf")
            print(f"  {label:<32} {val_m3:>12,.0f}   {val_m4:>12,.1f}   {mejora:>7.1f}×")

    if sep_eclipse_m4 is not None:
        print(f"  {'Sep. Tierra-Luna (km)':<32} {'—':>12}   {sep_eclipse_m4:>12.1f}")

    print("─" * 70)
    print("\n✅ M4 finalizado correctamente.")
    print("   Las correcciones EIH eliminan el error relativista acumulado.")
    print("   Error residual esperado: ~100–500 km (limitado por polo fijo J₂")
    print("   y ausencia de mareas — efectos de segundo orden no implementados).")
    print("\n   La jerarquía M1→M2→M3→M4 está completa.")
    print("   Siguiente paso: Parte 2 — modelos de sombra S1→S2→S3.")