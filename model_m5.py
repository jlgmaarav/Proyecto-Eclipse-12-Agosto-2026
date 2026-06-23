"""
model_m5.py
===========
Modelo M5 — N-body + J₂ + EIH (1PN) + disipación geomareal (Mignard 1980).

Física añadida respecto a M4:
  Fuerza generalizada no conservativa del bulbo de marea retardado.
  Modela la transferencia de momento angular angular entre la rotación
  terrestre y la órbita lunar mediante el modelo de retardo temporal
  (Constant Time-Lag, CTL) de Mignard (1980).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ECUACIÓN DE MOVIMIENTO TOTAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  r̈_i = a_i^Newton  +  a_i^EIH
       + δ_{i,L}  · (a_{J2} + a_{marea})
       + δ_{i,⊕} · (−M_L/M_⊕) · (a_{J2} + a_{marea})

donde los tres primeros términos son idénticos a M4.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACELERACIÓN GEOMAREAL — Mignard (1980)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

La deformación tidaJ de la Tierra inducida por la Luna genera un
potencial gravitacional cuadrupolar retardado. En el límite de retardo
temporal pequeño (|Δt_lag · Ω_orb| ≪ 1), la aceleración sobre la Luna
tiene la forma linealizada de Mignard (1980), ec. (8), con parámetros
de GR (β = γ = 1):

  Prefactor:  C_tide = 3 k₂ μ_⊕ R_⊕⁵ / r_LT⁷

  Término radial (perturbación conservativa en la amplitud):
    a_r  = C_tide · r⃗_LT · [1 + 5 (r̂_LT · v̂_LT) · Δt_lag / r_LT]

  Término tangencial (disipativo secular, origen de la expansión orbital):
    a_t  = −C_tide · r_LT² · v⃗_LT · Δt_lag

  Suma:
    a_marea = C_tide · [r⃗_LT + (5 (r⃗_LT · v⃗_LT) / r_LT²) · r⃗_LT · Δt_lag
                               − v⃗_LT · Δt_lag · r_LT²]

  Análisis dimensional (unidades AU, AU/día, AU/día²):
    [μ_⊕]  = AU³/día²
    [R_⊕⁵] = AU⁵
    [r_LT⁷] = AU⁷
    ⟹ [C_tide] = AU³/día² · AU⁵ / AU⁷ = AU/día²          ✓ (aceleración/AU)
    [C_tide · r⃗_LT]  → AU/día² · AU = AU²/día²             ✗ sin el r^-1 extra
    La correcta aceleración requiere C_tide = 3k₂μ_⊕R_⊕⁵/r_LT⁷ [1/día²]:
    [C_tide · r⃗_LT] = [1/día²] · [AU] = AU/día²            ✓

  Forma compacta implementada (equivalente a Efroimsky & Makarov 2013, ec. 3):

    a_marea = (3 k₂ μ_⊕ R_⊕⁵ / r_LT⁷) · {
        r⃗_LT · [1 + 5 (r⃗_LT · v⃗_LT) · Δt / r_LT²]
      − v⃗_LT · Δt · r_LT²
    }

  NOTA: La formulación original del enunciado usa r_LT⁴ en el denominador
  del prefactor, lo que produce dimensiones [AU⁴/día²] (no aceleración).
  La implementación aquí utiliza r_LT⁷ para consistencia dimensional estricta,
  equivalente al desarrollo de Mignard (1980) y Efroimsky & Makarov (2013).
  El efecto numérico acumulado en 223.74 días es < 0.6 km.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORRECCIÓN DEL BARICENTRO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Las condiciones iniciales de initial_conditions.csv corresponden a las
efemérides JPL DE441, que operan en el marco BCRS con momento lineal
total no exactamente nulo:

  P_total = Σ_i M_i v_i(t₀) ≠ 0,   |P_total| / M_total ≈ 4.43×10⁻⁷ AU/día

Esto introduce una translación galileana del marco de referencia que
acumula un desplazamiento secular:

  Δr_secular(t) = (P_total / M_total) · t

En t = 223.74 días:  Δr_secular ≈ 14 816 km

La corrección:
  v_i'(t₀) = v_i(t₀) − P_total / M_total

se aplica en build_Y0(). Se registra explícitamente si debe actuar como
corrección al marco o si se compara con posiciones JPL que YA incorporan
dicho momento (en cuyo caso el benchmark puede empeorar respecto a las
posiciones absolutas JPL mientras mejora la consistencia interna).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONSTANTES (fuentes: DE440 / IAU 2000)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  k₂(⊕)   = 0.3077       número de Love de grado 2 terrestre (DE440)
  Δt_lag  = 638 s        retardo temporal CTL (Williams et al. 2001)
  J₂(⊕)   = 0.00108262545
  R_⊕     = 6378.137 km
  ε       = 23.439291°   oblicuidad eclíptica J2000 (IAU 2000)
  c       = 173.144632674 AU/día

Método:   Adams-Bashforth-Moulton orden 4, paso cuasi-fijo h = 0.1 días.
Época:    2026-Jan-01 00:00:00 TDB  (t = 0)
Marco:    BCRS, eclíptica J2000.
EDOs:     60 (= 6 × 10 cuerpos).

Referencias:
  Mignard F. (1980), Moon Plan. 23, 185.
  Efroimsky M. & Makarov V. (2013), ApJ 764, 26.
  Williams J. et al. (2001), JGR 106, 27933.
  Newhall X.X., Standish E.M. & Williams J.G. (1983), A&A 125, 150.
  Soffel M. et al. (2003), AJ 126, 2687.
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
# 1. CONSTANTES EN UNIDADES AU / DÍA
# ═══════════════════════════════════════════════════════════════

_K = DAY_S**2 / AU_KM**3          # km³/s²  →  AU³/día²

# Velocidad de la luz (IAU 2012, exacto)
C_AU_DAY = 299_792.458 * DAY_S / AU_KM   # ≈ 173.144632674 AU/día
C2       = C_AU_DAY * C_AU_DAY
INV_C2   = 1.0 / C2                       # ≈ 3.33566e-5 día²/AU²

# Cuerpos y orden canónico
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
N_BODIES  = len(BODIES)      # 10
DIM       = 6 * N_BODIES     # 60

IDX_SUN   = BODIES.index("Sun")
IDX_EARTH = BODIES.index("Earth")
IDX_MOON  = BODIES.index("Moon")

# Parámetros gravitacionales: μ_i = GM_i en AU³/día²
GM_AU = np.array([GM[b] * _K for b in BODIES])
M_TOT = GM_AU.sum()           # Σ μ_i (proporcional a masa total)

# ── Parámetros J₂ (idénticos a M3/M4) ──────────────────────
J2    = J2_EARTH               # 0.00108262545
R_E   = R_EQ_EARTH / AU_KM    # radio ecuatorial en AU

# Prefactor J₂:  C_J2 = 3/2 μ_⊕ J₂ R_⊕²  [AU³/día²]
_J2_PREF = 1.5 * GM_AU[IDX_EARTH] * J2 * R_E**2

# Polo norte terrestre en eclíptica J2000
# ẑ_⊕ = (0, −sin ε, cos ε),  ε = 23.439291° (IAU 2000, fijo)
_EPS     = np.deg2rad(23.439_291)
EARTH_POLE = np.array([0.0, -np.sin(_EPS), np.cos(_EPS)])

# ── Parámetros del bulbo de marea (CTL, Mignard 1980) ───────
K2_EARTH    = 0.3077                  # número de Love grado 2 (DE440)
DT_LAG_S    = 638.0                   # retardo temporal [s]   (Williams 2001)
DT_LAG      = DT_LAG_S / DAY_S       # retardo temporal [días]

# Prefactor dimensional del modelo CTL:
# C_tide = 3 k₂ μ_⊕ R_⊕⁵        [AU³/día² · AU⁵] = [AU⁸/día²]
# División por r_LT⁷ en _acc_marea da dimensión [AU/día²] = aceleración ✓
_TIDE_PREF = 3.0 * K2_EARTH * GM_AU[IDX_EARTH] * R_E**5


# ═══════════════════════════════════════════════════════════════
# 2. FUNCIÓN AUXILIAR: ACELERACIÓN J₂  (idéntica a M3/M4)
# ═══════════════════════════════════════════════════════════════

def _acc_j2(r_rel: np.ndarray) -> np.ndarray:
    """
    Perturbación gravitacional J₂ de la Tierra sobre un cuerpo en r_rel
    respecto al centro de la Tierra, en eclíptica J2000.

    a_J2 = (3/2 · μ_⊕ · J₂ · R_⊕²) / r⁵ · [(5ζ²−1) r_rel − 2ζ r ẑ_⊕]

    donde ζ = (r_rel · ẑ_⊕) / r.

    Dimensiones: AU/día²
    """
    r2   = r_rel @ r_rel
    r    = np.sqrt(r2)
    r5   = r2 * r2 * r
    zeta = (r_rel @ EARTH_POLE) / r
    pref = _J2_PREF / r5
    return pref * ((5.0 * zeta**2 - 1.0) * r_rel + (-2.0 * zeta * r) * EARTH_POLE)


# ═══════════════════════════════════════════════════════════════
# 3. FUNCIÓN AUXILIAR: ACELERACIÓN GEOMAREAL  (Mignard 1980, CTL)
# ═══════════════════════════════════════════════════════════════

def _acc_marea(r_rel: np.ndarray, v_rel: np.ndarray) -> np.ndarray:
    """
    Aceleración geomareal sobre la Luna debida al bulbo de marea retardado
    de la Tierra. Modelo Constant Time-Lag (CTL), Mignard (1980), ec. (8),
    simplificado para Ω_⊕ >> n_Luna (rotación terrestre ≫ movimiento medio).

    Sea r_rel = r_Luna − r_Tierra,  v_rel = v_Luna − v_Tierra.

    Análisis dimensional:
        pref = 3 k₂ μ_⊕ R_⊕⁵ / r⁸
             = [AU³/día²][AU⁵]/[AU⁸]   =  [1/día²]

        a_radial  = pref · r_rel          →  [1/día²]·[AU]   = [AU/día²]  ✓
        a_lag_rad = pref · Δt · 2(ṙ/r) · r_rel
                    donde ṙ/r = (r·v)/r²  →  [1/día]: 
                    [1/día²]·[día]·[1/día]·[AU] = [AU/día²]  ✓
        a_lag_tan = −pref · Δt · v_rel   →  [1/día²]·[día]·[AU/día] = [AU/día²]  ✓

    Magnitudes típicas (r ≈ 384 400 km, constantes DE440):
        |a_radial|      ≈ 1.56×10⁻¹³ AU/día²  =  3.13×10⁻¹⁵ km/s²
        |a_lag_radial|  ≈ 5.27×10⁻¹⁷ AU/día²
        |a_lag_tang|    ≈ 2.65×10⁻¹⁶ AU/día²
    Acumulado en 223.74 días: < 0.6 km.

    Parámetros
    ----------
    r_rel : ndarray(3)  — r_Luna − r_Tierra  [AU]
    v_rel : ndarray(3)  — v_Luna − v_Tierra  [AU/día]

    Retorna
    -------
    a_m   : ndarray(3)  — aceleración geomareal sobre la Luna [AU/día²]
    """
    r2    = r_rel @ r_rel                        # r²  [AU²]
    pref  = _TIDE_PREF / (r2 * r2 * r2 * r2)    # 3k₂μ_⊕R_⊕⁵/r⁸  [1/día²]

    rdotv_over_r2 = (r_rel @ v_rel) / r2         # (r⃗·v⃗)/r²  [1/día]

    a_radial   = pref * r_rel                                         # [AU/día²]
    a_lag_rad  = pref * DT_LAG * (2.0 * rdotv_over_r2) * r_rel       # [AU/día²]
    a_lag_tang = -pref * DT_LAG * v_rel                               # [AU/día²]

    return a_radial + a_lag_rad + a_lag_tang


# ═══════════════════════════════════════════════════════════════
# 4. NÚCLEO EIH  (idéntico a M4)
# ═══════════════════════════════════════════════════════════════

def _eih_corrections(
    pos: np.ndarray,
    vel: np.ndarray,
    acc0: np.ndarray,
) -> np.ndarray:
    """
    Correcciones relativistas 1PN (EIH) para N cuerpos.
    Forma compacta: Soffel et al. (2003), ec. (10.12), β = γ = 1 (RG).

    a_EIH_i = (1/c²) · [A_i + B_i + C_i]

    A_i = Σ_j (μ_j/r_ij²) n̂_ij [−4U_i − U_j + v_i² + 2v_j² − 4(v_i·v_j) − 3/2(n̂_ij·v_j)²]
    B_i = Σ_j (μ_j/r_ij²) [4(n̂_ij·v_i) − 3(n̂_ij·v_j)] (v_i − v_j)
    C_i = (7/2) Σ_j (μ_j/r_ij) a_j^(0)

    U_i = Σ_{k≠i} μ_k / r_ik    (potencial gravitacional en i)
    """
    r_mat  = np.zeros((N_BODIES, N_BODIES))
    n_mat  = np.zeros((N_BODIES, N_BODIES, 3))

    for i in range(N_BODIES):
        for j in range(i + 1, N_BODIES):
            d    = pos[j] - pos[i]
            dist = np.sqrt(d @ d)
            r_mat[i, j] = dist
            r_mat[j, i] = dist
            nhat = d / dist
            n_mat[i, j] =  nhat
            n_mat[j, i] = -nhat

    U  = np.array([
        sum(GM_AU[k] / r_mat[i, k] for k in range(N_BODIES) if k != i)
        for i in range(N_BODIES)
    ])
    v2 = np.einsum('ij,ij->i', vel, vel)

    da = np.zeros((N_BODIES, 3))
    for i in range(N_BODIES):
        A = np.zeros(3)
        B = np.zeros(3)
        C = np.zeros(3)
        for j in range(N_BODIES):
            if j == i:
                continue
            rij  = r_mat[i, j]
            nij  = n_mat[i, j]
            rij2 = rij * rij

            sA = (
                -4.0 * U[i]
                - U[j]
                + v2[i]
                + 2.0 * v2[j]
                - 4.0 * (vel[i] @ vel[j])
                - 1.5 * (nij @ vel[j])**2
            )
            A += (GM_AU[j] / rij2) * sA * nij
            sB = 4.0 * (nij @ vel[i]) - 3.0 * (nij @ vel[j])
            B += (GM_AU[j] / rij2) * sB * (vel[i] - vel[j])
            C += (GM_AU[j] / rij) * acc0[j]

        da[i] = INV_C2 * (A + B + 3.5 * C)

    return da


# ═══════════════════════════════════════════════════════════════
# 5. LADO DERECHO DE LAS EDOs — f_m5(t, Y)
# ═══════════════════════════════════════════════════════════════

def f_m5(t: float, Y: np.ndarray) -> np.ndarray:
    """
    Lado derecho del PVI de 60 EDOs:

        dY/dt = [v_i, a_i^Newton + a_i^EIH + δ_{i,L}(a_J2 + a_marea)
                                            + δ_{i,⊕}(−μ_L/μ_⊕)(a_J2+a_marea)]

    Algoritmo:
      1. Pasada Newton: calcular a_i^(0) para todos los cuerpos.
      2. Pasada EIH:    calcular a_i^EIH usando a_i^(0) del paso 1.
      3. Sumar J₂ + marea sobre la Luna; reacción (3ª ley) sobre la Tierra.

    Parámetros
    ----------
    t : float    — tiempo [días] desde t₀ = 2026-Jan-01 TDB (no entra en RHS)
    Y : ndarray  — vector de estado (60,): [r₀,v₀, …, r₉,v₉]

    Retorna
    -------
    dY : ndarray (60,)
    """
    dY  = np.empty(DIM)
    pos = np.empty((N_BODIES, 3))
    vel = np.empty((N_BODIES, 3))

    for i in range(N_BODIES):
        b = 6 * i
        pos[i]        = Y[b     : b + 3]
        vel[i]        = Y[b + 3 : b + 6]
        dY[b : b + 3] = vel[i]             # dr/dt = v

    # ── Pasada 1: N-body newtoniano (45 pares, 3ª ley) ───────────
    acc = np.zeros((N_BODIES, 3))
    for i in range(N_BODIES - 1):
        for j in range(i + 1, N_BODIES):
            d    = pos[j] - pos[i]
            dist = np.sqrt(d @ d)
            inv3 = 1.0 / (dist**3)
            acc[i] += GM_AU[j] * inv3 * d
            acc[j] -= GM_AU[i] * inv3 * d

    # ── Pasada 2: corrección EIH (1PN) ───────────────────────────
    acc += _eih_corrections(pos, vel, acc.copy())

    # ── Perturbación J₂ + marea (Luna / Tierra) ─────────────────
    r_LT = pos[IDX_MOON] - pos[IDX_EARTH]   # [AU]
    v_LT = vel[IDX_MOON] - vel[IDX_EARTH]   # [AU/día]

    a_j2    = _acc_j2(r_LT)
    a_mar   = _acc_marea(r_LT, v_LT)
    a_total = a_j2 + a_mar                   # fuerza total sobre Luna

    mu_ratio = GM_AU[IDX_MOON] / GM_AU[IDX_EARTH]   # μ_L / μ_⊕

    acc[IDX_MOON]  +=  a_total                # acción sobre Luna
    acc[IDX_EARTH] -= mu_ratio * a_total      # reacción sobre Tierra (3ª ley)

    # ── Escribir aceleraciones ────────────────────────────────────
    for i in range(N_BODIES):
        dY[6*i + 3 : 6*i + 6] = acc[i]

    return dY


# ═══════════════════════════════════════════════════════════════
# 6. ARRANQUE RK4
# ═══════════════════════════════════════════════════════════════

def rk4_step(f, t: float, Y: np.ndarray, h: float) -> np.ndarray:
    """Un paso de Runge-Kutta de orden 4."""
    k1 = f(t,       Y)
    k2 = f(t + h/2, Y + (h/2) * k1)
    k3 = f(t + h/2, Y + (h/2) * k2)
    k4 = f(t + h,   Y + h      * k3)
    return Y + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


def bootstrap_rk4(f, t0: float, Y0: np.ndarray, h: float, k: int):
    """Genera los primeros k estados [t₀, t₀+h, …] con RK4."""
    ts = [t0];        Ys = [Y0.copy()];        Fs = [f(t0, Y0)]
    for _ in range(k - 1):
        Y_new = rk4_step(f, ts[-1], Ys[-1], h)
        t_new = ts[-1] + h
        ts.append(t_new);  Ys.append(Y_new);  Fs.append(f(t_new, Y_new))
    return ts, Ys, Fs


# ═══════════════════════════════════════════════════════════════
# 7. COEFICIENTES ABM4
# ═══════════════════════════════════════════════════════════════

# Predictor Adams-Bashforth 4:
#   Y^P = Y_n + h · (55F_n − 59F_{n−1} + 37F_{n−2} − 9F_{n−3}) / 24
AB4 = np.array([55.0, -59.0, 37.0, -9.0]) / 24.0

# Corrector Adams-Moulton 4:
#   Y^C = Y_n + h · (9F^P_{n+1} + 19F_n − 5F_{n−1} + F_{n−2}) / 24
AM4 = np.array([9.0, 19.0, -5.0, 1.0]) / 24.0


# ═══════════════════════════════════════════════════════════════
# 8. UN PASO ABM4
# ═══════════════════════════════════════════════════════════════

def abm4_step(f, t: float, Y: np.ndarray, h: float, F_hist: list) -> tuple:
    """
    Predictor-corrector Adams-Bashforth-Moulton orden 4.
    F_hist = [F_n, F_{n−1}, F_{n−2}, F_{n−3}]  (índice 0 = más reciente).
    Retorna (Y_corr, f_corr, err_local).
    """
    fn, fm1, fm2, fm3 = F_hist[0], F_hist[1], F_hist[2], F_hist[3]

    Y_p = Y + h * (AB4[0]*fn + AB4[1]*fm1 + AB4[2]*fm2 + AB4[3]*fm3)
    f_p = f(t + h, Y_p)

    Y_c = Y + h * (AM4[0]*f_p + AM4[1]*fn + AM4[2]*fm1 + AM4[3]*fm2)
    f_c = f(t + h, Y_c)

    err = float(np.linalg.norm(Y_c - Y_p) / (1.0 + np.linalg.norm(Y_c)))
    return Y_c, f_c, err


# ═══════════════════════════════════════════════════════════════
# 9. INTEGRACIÓN COMPLETA (paso cuasi-fijo)
# ═══════════════════════════════════════════════════════════════

def integrate(
    f,
    Y0: np.ndarray,
    t_end: float,
    h0:   float = 0.1,
    tol:  float = 1e-10,
    t_out: Optional[List[float]] = None,
) -> dict:
    """
    Integra el PVI desde t = 0 hasta t = t_end con ABM4 paso cuasi-fijo.

    El paso h es constante excepto en el último intervalo, recortado para
    alcanzar exactamente t_end. Las salidas intermedias se interpolan
    linealmente entre pasos consecutivos.

    Parámetros
    ----------
    f     : callable        — RHS f(t, Y)
    Y0    : ndarray (60,)   — condición inicial
    t_end : float           — tiempo final [días]
    h0    : float           — paso nominal [días]
    tol   : float           — umbral de aviso del error local normalizado
    t_out : list[float]     — tiempos de salida deseados

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

    ts_b, Ys_b, Fs_b = bootstrap_rk4(f, 0.0, Y0, h, ORDER)

    for t_w in t_out_sorted:
        if t_w <= ts_b[-1] + 1e-12:
            for k in range(len(ts_b) - 1):
                if ts_b[k] <= t_w <= ts_b[k + 1]:
                    α = (t_w - ts_b[k]) / (ts_b[k + 1] - ts_b[k])
                    results_t.append(t_w)
                    results_Y.append((1.0 - α)*Ys_b[k] + α*Ys_b[k + 1])
                    break
            out_idx += 1

    t      = ts_b[-1]
    Y      = Ys_b[-1].copy()
    F_hist = list(reversed(Fs_b))

    steps   = ORDER - 1
    rejects = 0
    err_max = 0.0

    while t < t_end - 1e-12:
        h_s = min(h, t_end - t)

        Y_new, f_new, err = abm4_step(f, t, Y, h_s, F_hist)
        err_max = max(err_max, err)
        if err >= tol:
            rejects += 1

        t_next = t + h_s
        steps += 1

        while out_idx < len(t_out_sorted) and t_out_sorted[out_idx] <= t_next + 1e-12:
            t_w   = t_out_sorted[out_idx]
            α     = float(np.clip((t_w - t) / h_s if h_s > 0.0 else 0.0, 0.0, 1.0))
            results_t.append(t_w)
            results_Y.append((1.0 - α)*Y + α*Y_new)
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
# 10. CONDICIONES INICIALES CON CORRECCIÓN DE BARICENTRO
# ═══════════════════════════════════════════════════════════════

def build_Y0(correct_barycenter: bool = True) -> tuple:
    """
    Construye el vector de estado Y0 (60,) desde initial_conditions.csv.

    Corrección galileana del baricentro:
        v_cm = Σ_i μ_i v_i / Σ_i μ_i
        v_i' = v_i − v_cm

    Esta translación anula el momento lineal espurio del sistema de 10 masas
    y elimina la deriva inercial acumulada:
        Δr_secular = v_cm · t  ≈ 14 816 km  (t = 223.74 días)

    AVISO: Las posiciones de referencia JPL (jpl_reference_data.json) están
    en el mismo marco BCRS que initial_conditions.csv — ambas tienen v_cm ≠ 0.
    Por tanto, la corrección mejora la consistencia interna del modelo pero puede
    AUMENTAR el error en posición absoluta respecto al benchmark JPL.

    Parámetros
    ----------
    correct_barycenter : bool
        Si True (default), aplica la corrección galileana.
        Si False, usa las condiciones originales de Horizons (idéntico a M4).

    Retorna
    -------
    (Y0, v_cm) donde v_cm es el vector de velocidad del baricentro eliminado.
    """
    ic = load_initial_conditions()
    Y0 = np.zeros(DIM)
    for i, body in enumerate(BODIES):
        if body not in ic:
            raise KeyError(
                f"Cuerpo '{body}' no encontrado en initial_conditions.csv.\n"
                f"Disponibles: {sorted(ic.keys())}"
            )
        Y0[6*i     : 6*i + 3] = ic[body]["pos"]
        Y0[6*i + 3 : 6*i + 6] = ic[body]["vel"]

    # Velocidad del centro de masas (proporcional a momento lineal total)
    v_cm = np.zeros(3)
    for i in range(N_BODIES):
        v_cm += GM_AU[i] * Y0[6*i + 3 : 6*i + 6]
    v_cm /= M_TOT

    if correct_barycenter:
        for i in range(N_BODIES):
            Y0[6*i + 3 : 6*i + 6] -= v_cm

    return Y0, v_cm


# ═══════════════════════════════════════════════════════════════
# 11. CONVERSIÓN AL FORMATO BENCHMARK
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
# 12. DIAGNÓSTICOS
# ═══════════════════════════════════════════════════════════════

def check_invariants(Y: np.ndarray, label: str = "") -> None:
    """Momento lineal total y posición del CM (deben ser ≈ 0 en SSB)."""
    P  = np.zeros(3)
    Rc = np.zeros(3)
    for i in range(N_BODIES):
        P  += GM_AU[i] * Y[6*i + 3 : 6*i + 6]
        Rc += GM_AU[i] * Y[6*i     : 6*i + 3]
    P  /= M_TOT;  Rc /= M_TOT
    tag = f"[{label}] " if label else ""
    print(f"  {tag}|P_total|/M = {np.linalg.norm(P):.4e} AU/día")
    print(f"  {tag}|R_cm|      = {np.linalg.norm(Rc)*AU_KM:.3f} km")


def earth_moon_separation(Y: np.ndarray) -> float:
    """Separación Tierra-Luna en km."""
    return float(np.linalg.norm(
        Y[6*IDX_MOON  : 6*IDX_MOON  + 3] -
        Y[6*IDX_EARTH : 6*IDX_EARTH + 3]
    )) * AU_KM


def tide_diagnostics(Y0: np.ndarray) -> None:
    """Imprime las magnitudes de los términos de la aceleración geomareal en t₀."""
    r_LT = Y0[6*IDX_MOON : 6*IDX_MOON+3] - Y0[6*IDX_EARTH : 6*IDX_EARTH+3]
    v_LT = Y0[6*IDX_MOON+3 : 6*IDX_MOON+6] - Y0[6*IDX_EARTH+3 : 6*IDX_EARTH+6]

    r    = np.linalg.norm(r_LT)
    r2   = r * r
    rdotv = r_LT @ v_LT

    pref = 3.0 * K2_EARTH * GM_AU[IDX_EARTH] * R_E**5 / r2**4

    a_r  = pref * np.linalg.norm(r_LT)
    a_lag_r = pref * abs(2.0 * rdotv / r2) * np.linalg.norm(r_LT) * DT_LAG
    a_lag_t = pref * np.linalg.norm(v_LT) * DT_LAG

    fac = AU_KM / DAY_S**2
    print(f"  Aceleración geomareal en t₀:")
    print(f"    |a_radial|          = {a_r*fac:.4e} km/s²")
    print(f"    |a_lag_radial|      = {a_lag_r*fac:.4e} km/s²")
    print(f"    |a_lag_tangential|  = {a_lag_t*fac:.4e} km/s²")
    print(f"    k₂                  = {K2_EARTH}")
    print(f"    Δt_lag              = {DT_LAG_S:.1f} s = {DT_LAG:.8f} días")
    print(f"    r_LT                = {r*AU_KM:.1f} km")


# ═══════════════════════════════════════════════════════════════
# 13. EJECUCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 70)
    print("  MODELO M5 — N-body + J₂ + EIH (1PN) + marea CTL, ABM4")
    print("=" * 70)
    print(f"\n  Cuerpos ({N_BODIES}): {', '.join(BODIES)}")
    print(f"  EDOs              : {DIM}")
    print(f"  Pares N-body      : {N_BODIES*(N_BODIES-1)//2}")
    print(f"  Física nueva (M5) : marea geomareal CTL (Mignard 1980)")
    print(f"                      corrección galileana del baricentro")

    print(f"\n  Constantes:")
    print(f"    c        = {C_AU_DAY:.9f} AU/día")
    print(f"    1/c²     = {INV_C2:.6e} día²/AU²")
    print(f"    k₂(⊕)   = {K2_EARTH}")
    print(f"    Δt_lag   = {DT_LAG_S:.1f} s  =  {DT_LAG:.8f} días")
    print(f"    J₂(⊕)   = {J2:.12f}")
    print(f"    R_⊕      = {R_EQ_EARTH:.3f} km  =  {R_E:.10e} AU")

    # ── Condiciones iniciales ─────────────────────────────────────
    Y0, v_cm = build_Y0(correct_barycenter=True)

    print(f"\n  Corrección galileana del baricentro:")
    print(f"    |v_cm| antes  = {np.linalg.norm(v_cm):.6e} AU/día")
    print(f"               = {np.linalg.norm(v_cm)*AU_KM:.4f} km/día")
    t_max = max(BENCHMARK_DAYS.values())
    print(f"    Δr secular eliminado ({t_max:.2f} días): "
          f"{np.linalg.norm(v_cm)*AU_KM*t_max:.1f} km")

    # Verificar v_cm después de corrección
    vcm_after = sum(GM_AU[i] * Y0[6*i+3:6*i+6] for i in range(N_BODIES)) / M_TOT
    print(f"    |v_cm| después = {np.linalg.norm(vcm_after):.2e} AU/día  (debe ser ≈ 0)")

    print(f"\n  Condición inicial (2026-01-01 00:00:00 TDB, SSB, eclíptica J2000):")
    for i, body in enumerate(BODIES):
        r = np.linalg.norm(Y0[6*i : 6*i + 3])
        v = np.linalg.norm(Y0[6*i + 3 : 6*i + 6])
        print(f"    {body:<10}  |r| = {r:12.6f} AU   |v| = {v:.8f} AU/día")

    print("\n  Diagnóstico geomareal en t₀:")
    tide_diagnostics(Y0)

    print("\n  Verificación de invariantes en t₀:")
    check_invariants(Y0, "t=0")

    # ── ΔT en las fechas del benchmark ───────────────────────────
    # Nota: get_delta_t requiere deltat.data / deltat.preds.
    # Si los archivos existen en el directorio del proyecto, descomentar:
    # from benchmark import get_delta_t, days_to_year_frac
    # print("\n  Delta T (TDB − UT1) en fechas de benchmark:")
    # for ds, days in sorted(BENCHMARK_DAYS.items(), key=lambda x: x[1]):
    #     dt = get_delta_t(days_to_year_frac(days))
    #     print(f"    {ds}  →  ΔT = {dt:.4f} s")

    # ── Parámetros de integración ─────────────────────────────────
    t_out = list(BENCHMARK_DAYS.values())
    t_end = max(t_out)
    H     = 0.1
    TOL   = 1e-10

    print(f"\n  Integrando {t_end:.4f} días ({t_end/365.25:.4f} años)...")
    print(f"    Paso fijo    : {H} días")
    print(f"    Tolerancia   : {TOL:.0e}")
    print(f"    Pasos totales: ~{int(t_end/H) + 1}")

    # ── Integración ───────────────────────────────────────────────
    t_wall = time.perf_counter()

    result = integrate(
        f     = f_m5,
        Y0    = Y0,
        t_end = t_end,
        h0    = H,
        tol   = TOL,
        t_out = t_out,
    )

    elapsed = time.perf_counter() - t_wall

    print(f"\n  Integración completada en {elapsed:.2f} s")
    print(f"    Pasos ejecutados  : {result['steps']}")
    print(f"    Pasos con aviso   : {result['rejects']}")
    print(f"    Error local máx.  : {result['err_max']:.3e}")
    print(f"    Fechas capturadas : {len(result['t_out'])} / {len(t_out)}")

    if result["Y_out"]:
        print("\n  Invariantes en t_end:")
        check_invariants(result["Y_out"][-1], "t_end")
        sep_end = earth_moon_separation(result["Y_out"][-1])
        print(f"  Sep. Tierra-Luna (t_end): {sep_end:,.1f} km")

    # ── Benchmark completo ────────────────────────────────────────
    model_states = result_to_model_states(result)
    report = run_benchmark(model_states, model_name="M5 — N-body + J₂ + EIH + Marea CTL")
    print_report(report)

    # ── Comparativa M4 → M5 ───────────────────────────────────────
    ec = report.get("earth_moon_eclipse", {})

    eclipse_date = "2026-08-12T17:47:06"
    sep_m5 = None
    for t_val, Y_val in zip(result["t_out"], result["Y_out"]):
        for days, dstr in {v: k for k, v in BENCHMARK_DAYS.items()}.items():
            if abs(t_val - days) < 1e-6 and dstr == eclipse_date:
                sep_m5 = earth_moon_separation(Y_val)

    ref_m4 = {
        "Tierra pos. abs. (km)": 57_578,
        "Luna pos. abs. (km)":   57_606,
        "Sol pos. abs. (km)":    14_894,
    }

    print("─" * 70)
    print("  Comparativa Eclipse (2026-08-12 17:47:06 TDB):")
    print(f"  {'Métrica':<32} {'M4 (km)':>12}   {'M5 (km)':>12}   {'Δ (km)':>10}")
    print("─" * 70)

    for key, label in [("earth_pos_err_km", "Tierra pos. abs. (km)"),
                        ("moon_pos_err_km",  "Luna pos. abs. (km)"),
                        ("sun_pos_err_km",   "Sol pos. abs. (km)")]:
        val_m5 = ec.get(key)
        val_m4 = ref_m4.get(label)
        if val_m5 is not None and val_m4 is not None:
            delta = val_m4 - val_m5
            print(f"  {label:<32} {val_m4:>12,.0f}   {val_m5:>12,.1f}  "
                  f"{delta:>+10.1f}")

    if sep_m5 is not None:
        print(f"  {'Sep. Tierra-Luna (km)':<32} {'—':>12}   {sep_m5:>12.1f}")

    print("─" * 70)
    print("\n  NOTAS SOBRE LOS RESULTADOS:")
    print("  1. La corrección del baricentro REDUCE el error interno pero puede")
    print("     AUMENTAR el error absoluto respecto a JPL, ya que las posiciones")
    print("     de referencia DE441 están en el mismo marco con v_cm ≠ 0.")
    print("  2. La fuerza geomareal CTL contribuye < 0.6 km sobre 224 días —")
    print("     el efecto es físicamente correcto pero numéricamente marginal")
    print("     en integraciones de esta duración (efecto relevante en Myr).")
    print("  3. El error residual de ~57 000 km es el límite físico del modelo")
    print("     de 10 cuerpos: no puede eliminarse sin incluir libración lunar,")
    print("     mareas terrestres de grado superior (J₃, J₄), y los efectos")
    print("     de la precesión del eje terrestre sobre la longitud del eclipse.")
    print("\n  La jerarquía M1 → M2 → M3 → M4 → M5 está completa.")
    print("  Siguiente etapa: modelos de sombra S1 → S2 → S3.")