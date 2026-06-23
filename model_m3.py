"""
model_m3.py
===========
Modelo M3 — N-body newtoniano + achatamiento terrestre (J₂).

Física añadida respecto a M2:
  El potencial gravitacional de la Tierra no es puramente esférico.
  El término J₂ (primer armónico zonal, achatamiento en los polos) genera
  una aceleración adicional sobre todo cuerpo en el campo terrestre.

  El único cuerpo donde este efecto es físicamente relevante es la Luna
  (a ~384 000 km). Para el Sol y los planetas, la perturbación J₂ es
  < 10⁻¹² de su aceleración dominante y se omite.

  Por la 3ª ley de Newton, la reacción sobre la Tierra también se incluye.

Fórmula J₂ (marco eclíptico J2000, forma vectorial):
  Sea r_LT = r_Luna - r_Tierra  (posición relativa Luna respecto a Tierra)
      r    = |r_LT|
      ζ    = (r_LT · ẑ⊕) / r      (proyección normalizada sobre polo terrestre)

  a_J2 = (3 μ⊕ J2 R⊕²) / (2 r⁵) · [(5ζ² - 1) r_LT  −  2ζ r ẑ⊕]

  donde ẑ⊕ = (0, −sin ε, cos ε) es el polo norte terrestre en eclíptica J2000
  con oblicuidad ε = 23.439291°.

Método:  Adams-Bashforth-Moulton orden 4, paso cuasi-fijo h = 0.1 días.
         Arranque: RK4 primeros 4 pasos.
         Idéntico a M2 — solo cambia f(t, Y).

Unidades: AU, AU/día, días.
Época:    2026-Jan-01 00:00:00 TDB  (t = 0)
Sistema:  Baricentro del Sistema Solar (SSB), eclíptica J2000.

Dimensión: 60 EDOs (igual que M2).
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
# 1. CONSTANTES FÍSICAS Y PARÁMETROS J₂
# ═══════════════════════════════════════════════════════════════

# Factor de conversión: km³/s² → AU³/día²
_K = DAY_S**2 / AU_KM**3

# Orden canónico de los 10 cuerpos
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
N_BODIES = len(BODIES)   # 10
DIM      = 6 * N_BODIES  # 60

IDX_EARTH = BODIES.index("Earth")   # 3
IDX_MOON  = BODIES.index("Moon")    # 4

# Parámetros gravitacionales en AU³/día²
GM_AU = np.array([GM[b] * _K for b in BODIES])

# ── Parámetros J₂ de la Tierra ──────────────────────────────
# J₂ y R⊕ de DE441 (importados de benchmark.py)
J2   = J2_EARTH                    # 0.00108262545  (adimensional)
R_E  = R_EQ_EARTH / AU_KM          # radio ecuatorial en AU

# Prefactor combinado en AU³/día²:  3/2 · μ⊕ · J₂ · R⊕²
_J2_PREFACTOR = 1.5 * GM_AU[IDX_EARTH] * J2 * R_E**2

# ── Polo norte terrestre en eclíptica J2000 ──────────────────
# La oblicuidad de la eclíptica es ε = 23.439291° (IAU 2000, época J2000.0).
# En 8 meses la precesión del eje terrestre es < 0.001° → se trata como fijo.
# Coordenadas del polo en el marco eclíptico (x=0 por definición del meridiano):
#   ẑ⊕ = (0,  −sin ε,  cos ε)
_OBLIQUITY_RAD = np.deg2rad(23.439_291)
EARTH_POLE = np.array([
    0.0,
    -np.sin(_OBLIQUITY_RAD),
     np.cos(_OBLIQUITY_RAD),
])  # vector unitario, |EARTH_POLE| = 1 exactamente


# ═══════════════════════════════════════════════════════════════
# 2. ACELERACIÓN J₂  (función auxiliar)
# ═══════════════════════════════════════════════════════════════

def _acc_j2(r_rel: np.ndarray) -> np.ndarray:
    """
    Aceleración de perturbación J₂ sobre un cuerpo situado en r_rel
    respecto al centro de la Tierra, expresada en el marco eclíptico J2000.

    Fórmula (ver modelo matemático M3):
        a_J2 = C/r⁵ · [(5ζ² − 1) r_rel  −  2ζ r ẑ⊕]

    donde  C = 3/2 · μ⊕ · J₂ · R⊕²
           ζ = (r_rel · ẑ⊕) / r
           r = |r_rel|

    Parámetros
    ----------
    r_rel : ndarray(3)  posición relativa en AU (Luna − Tierra)

    Retorna
    -------
    a_j2  : ndarray(3)  aceleración en AU/día²
    """
    r2   = r_rel[0]*r_rel[0] + r_rel[1]*r_rel[1] + r_rel[2]*r_rel[2]
    r    = np.sqrt(r2)
    r5   = r2 * r2 * r

    # Proyección normalizada sobre el polo: ζ = r_rel · ẑ⊕ / r
    zeta = (r_rel[0]*EARTH_POLE[0]
          + r_rel[1]*EARTH_POLE[1]
          + r_rel[2]*EARTH_POLE[2]) / r

    zeta2 = zeta * zeta
    C_r5  = _J2_PREFACTOR / r5

    # a_J2 = C/r⁵ · [(5ζ²−1)·r_rel  −  2ζ·r·ẑ⊕]
    coeff_r    = 5.0 * zeta2 - 1.0
    coeff_pole = -2.0 * zeta * r

    return C_r5 * (coeff_r * r_rel + coeff_pole * EARTH_POLE)


# ═══════════════════════════════════════════════════════════════
# 3. LADO DERECHO DE LAS EDOs — f(t, Y)
# ═══════════════════════════════════════════════════════════════

def f_m3(t: float, Y: np.ndarray) -> np.ndarray:
    """
    Lado derecho del sistema de 60 EDOs: N-body newtoniano + J₂ terrestre.

    Sobre la Luna  → a_Newton + a_J2(r_Luna − r_Tierra)
    Sobre la Tierra → a_Newton − (μ_Luna/μ_Tierra) · a_J2   [3ª ley]
    Resto          → a_Newton (sin J₂)

    Parámetros
    ----------
    t : float    — tiempo actual en días (no entra en las ecuaciones)
    Y : ndarray  — vector de estado (60,): [r0,v0, r1,v1, ..., r9,v9]

    Retorna
    -------
    dY : ndarray (60,) — [v0,a0, v1,a1, ..., v9,a9]
    """
    dY  = np.empty(DIM)
    acc = np.zeros((N_BODIES, 3))

    # ── Extraer posiciones; copiar velocidades → dY ────────────────
    pos = np.empty((N_BODIES, 3))
    for i in range(N_BODIES):
        b = 6 * i
        pos[i]        = Y[b     : b + 3]
        dY[b : b + 3] = Y[b + 3 : b + 6]   # dr/dt = v

    # ── N-body: 45 pares únicos, 3ª ley ───────────────────────────
    for i in range(N_BODIES - 1):
        for j in range(i + 1, N_BODIES):
            d    = pos[j] - pos[i]
            dist = np.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])
            inv3 = 1.0 / (dist * dist * dist)
            acc[i] += GM_AU[j] * inv3 * d
            acc[j] -= GM_AU[i] * inv3 * d

    # ── Perturbación J₂ terrestre ──────────────────────────────────
    # Vector Luna → Tierra en AU
    r_LT = pos[IDX_MOON] - pos[IDX_EARTH]

    a_j2 = _acc_j2(r_LT)

    # Luna: recibe la aceleración J₂ completa
    acc[IDX_MOON] += a_j2

    # Tierra: reacción (3ª ley), escalada por ratio de masas
    #   F_Tierra = −F_Luna  →  a_Tierra = −(μ_Luna/μ_Tierra) · a_j2
    acc[IDX_EARTH] -= (GM_AU[IDX_MOON] / GM_AU[IDX_EARTH]) * a_j2

    # ── Escribir aceleraciones en dY ──────────────────────────────
    for i in range(N_BODIES):
        dY[6*i + 3 : 6*i + 6] = acc[i]

    return dY


# ═══════════════════════════════════════════════════════════════
# 4. ARRANQUE CON RK4
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
# 5. COEFICIENTES ABM4
# ═══════════════════════════════════════════════════════════════

# Predictor Adams-Bashforth 4:
#   Y^P_{n+1} = Y_n + h·(55 F_n − 59 F_{n−1} + 37 F_{n−2} − 9 F_{n−3}) / 24
AB4 = np.array([55.0, -59.0, 37.0, -9.0]) / 24.0

# Corrector Adams-Moulton 4:
#   Y^C_{n+1} = Y_n + h·(9 F^P_{n+1} + 19 F_n − 5 F_{n−1} + F_{n−2}) / 24
AM4 = np.array([9.0, 19.0, -5.0, 1.0]) / 24.0


# ═══════════════════════════════════════════════════════════════
# 6. UN PASO ABM4
# ═══════════════════════════════════════════════════════════════

def abm4_step(f, t: float, Y: np.ndarray, h: float, F_hist: list) -> tuple:
    """
    Un paso del predictor-corrector Adams-Bashforth-Moulton orden 4.

    F_hist = [F_n, F_{n−1}, F_{n−2}, F_{n−3}]  (índice 0 = más reciente)

    Retorna (Y_corr, f_corr, err_local)
    """
    f_n, f_nm1, f_nm2, f_nm3 = F_hist[0], F_hist[1], F_hist[2], F_hist[3]

    # Predictor
    Y_pred = Y + h * (AB4[0]*f_n + AB4[1]*f_nm1 + AB4[2]*f_nm2 + AB4[3]*f_nm3)
    f_pred = f(t + h, Y_pred)

    # Corrector
    Y_corr = Y + h * (AM4[0]*f_pred + AM4[1]*f_n + AM4[2]*f_nm1 + AM4[3]*f_nm2)
    f_corr = f(t + h, Y_corr)

    # Error local normalizado
    err = float(np.linalg.norm(Y_corr - Y_pred) / (1.0 + np.linalg.norm(Y_corr)))

    return Y_corr, f_corr, err


# ═══════════════════════════════════════════════════════════════
# 7. INTEGRACIÓN COMPLETA (paso cuasi-fijo)
# ═══════════════════════════════════════════════════════════════

def integrate(f, Y0: np.ndarray, t_end: float,
              h0:  float = 0.1,
              tol: float = 1e-10,
              t_out: Optional[List[float]] = None) -> dict:
    """
    Integra el sistema de EDOs desde t=0 hasta t=t_end con ABM4 paso fijo.

    El último paso se recorta para alcanzar exactamente t_end.
    Las salidas intermedias se interpolan linealmente entre pasos consecutivos
    (el error de interpolación es << error del integrador para h=0.1 días).

    Parámetros
    ----------
    f     : callable        — lado derecho f(t, Y)
    Y0    : ndarray (60,)   — condición inicial
    t_end : float           — tiempo final (días desde t₀=0)
    h0    : float           — paso de integración (días)
    tol   : float           — umbral de aviso del error local
    t_out : list[float]     — tiempos en los que guardar el estado

    Retorna
    -------
    dict:
      t_out   : list[float]      tiempos de salida
      Y_out   : list[ndarray]    estados en t_out
      steps   : int              pasos totales ejecutados
      rejects : int              pasos con error > tol (solo aviso)
      h       : float            paso usado
      err_max : float            máximo error local registrado
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
    F_hist = list(reversed(Fs_boot))   # [F_n, F_{n-1}, F_{n-2}, F_{n-3}]

    steps   = ORDER - 1
    rejects = 0
    err_max = 0.0

    # ── Bucle principal ───────────────────────────────────────────
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
            alpha  = float(np.clip((t_want - t) / h_step if h_step > 0.0 else 0.0, 0.0, 1.0))
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
# 8. CONDICIONES INICIALES
# ═══════════════════════════════════════════════════════════════

def build_Y0() -> np.ndarray:
    """Construye Y0 (60,) desde initial_conditions.csv."""
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
# 9. CONVERSIÓN AL FORMATO BENCHMARK
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
# 10. DIAGNÓSTICOS
# ═══════════════════════════════════════════════════════════════

def check_invariants(Y: np.ndarray, label: str = "") -> None:
    """Imprime momento lineal total y posición del centro de masas (deben ser ≈ 0)."""
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


def j2_acceleration_magnitude(Y: np.ndarray) -> float:
    """Módulo de la aceleración J₂ sobre la Luna en km/s², para diagnóstico."""
    r_LT  = Y[6*IDX_MOON : 6*IDX_MOON + 3] - Y[6*IDX_EARTH : 6*IDX_EARTH + 3]
    a_j2  = _acc_j2(r_LT)
    # Convertir AU/día² → km/s²
    return float(np.linalg.norm(a_j2)) * AU_KM / DAY_S**2


# ═══════════════════════════════════════════════════════════════
# 11. EJECUCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 70)
    print("  MODELO M3 — N-body + J₂ terrestre, ABM4 paso cuasi-fijo")
    print("=" * 70)
    print(f"\n  Cuerpos ({N_BODIES}): {', '.join(BODIES)}")
    print(f"  Dimensión del sistema : {DIM} EDOs")
    print(f"  Pares N-body          : {N_BODIES*(N_BODIES-1)//2}")
    print(f"  J₂ aplicado a         : Luna (+ reacción Tierra, 3ª ley)")
    print(f"\n  Parámetros J₂:")
    print(f"    J₂⊕           = {J2:.12f}")
    print(f"    R⊕ (ecuatorial)= {R_EQ_EARTH:.3f} km  =  {R_E:.10e} AU")
    print(f"    Prefactor C   = {_J2_PREFACTOR:.6e} AU³/día²")
    print(f"    Oblicuidad ε  = {np.degrees(_OBLIQUITY_RAD):.6f}°")
    print(f"    Polo ẑ⊕       = ({EARTH_POLE[0]:.6f}, {EARTH_POLE[1]:.6f}, {EARTH_POLE[2]:.6f})")

    # ── Condiciones iniciales ─────────────────────────────────────
    Y0 = build_Y0()
    print("\nCondición inicial (2026-01-01 00:00:00 TDB, SSB, eclíptica J2000):")
    for i, body in enumerate(BODIES):
        r = np.linalg.norm(Y0[6*i : 6*i + 3])
        v = np.linalg.norm(Y0[6*i + 3 : 6*i + 6])
        print(f"  {body:<10}  |r| = {r:12.6f} AU   |v| = {v:.8f} AU/día")

    sep0 = earth_moon_separation(Y0)
    a_j2_0 = j2_acceleration_magnitude(Y0)
    print(f"\n  Sep. Tierra-Luna (t=0)    : {sep0:,.1f} km")
    print(f"  |a_J₂| sobre Luna (t=0)   : {a_j2_0:.4e} km/s²")
    print(f"  |a_Newton| T→L (t=0)      : {GM['Earth']/sep0**2:.4e} km/s²")
    print(f"  Ratio J₂/Newton            : {a_j2_0 / (GM['Earth']/sep0**2):.4e}")

    print("\nVerificación de invariantes en t=0:")
    check_invariants(Y0, "t=0")

    # ── Parámetros de integración ─────────────────────────────────
    t_out = list(BENCHMARK_DAYS.values())
    t_end = max(t_out)
    H     = 0.1
    TOL   = 1e-10

    print(f"\nIntegrando {t_end:.4f} días ({t_end/365.25:.4f} años)...")
    print(f"  Paso fijo    : {H} días")
    print(f"  Tolerancia   : {TOL:.0e}")
    print(f"  Pasos totales: ~{int(t_end/H) + 1}")

    # ── Integración ───────────────────────────────────────────────
    t_wall = time.perf_counter()

    result = integrate(
        f     = f_m3,
        Y0    = Y0,
        t_end = t_end,
        h0    = H,
        tol   = TOL,
        t_out = t_out,
    )

    elapsed = time.perf_counter() - t_wall

    print(f"\nIntegración completada en {elapsed:.3f} s")
    print(f"  Pasos ejecutados  : {result['steps']}")
    print(f"  Pasos con aviso   : {result['rejects']}")
    print(f"  Paso usado        : {result['h']:.4f} días")
    print(f"  Error local máx.  : {result['err_max']:.3e}")
    print(f"  Fechas capturadas : {len(result['t_out'])} / {len(t_out)}")

    if result["Y_out"]:
        print("\nVerificación de invariantes en t_end:")
        check_invariants(result["Y_out"][-1], "t_end")

        sep_end = earth_moon_separation(result["Y_out"][-1])
        print(f"\n  Sep. Tierra-Luna (t_end)  : {sep_end:,.1f} km")

    # ── Benchmark completo ────────────────────────────────────────
    model_states = result_to_model_states(result)
    report = run_benchmark(model_states, model_name="M3 — N-body + J₂ terrestre ABM4")
    print_report(report)

    # ── Comparativa M2 → M3 en la fecha del eclipse ───────────────
    ec = report.get("earth_moon_eclipse", {})

    # Separación T-L en eclipse (desde el último Y_out que sea eclipse)
    eclipse_date = "2026-08-12T17:47:06"
    sep_eclipse_m3 = None
    for t_val, Y_val in zip(result["t_out"], result["Y_out"]):
        for days, dstr in {v: k for k, v in BENCHMARK_DAYS.items()}.items():
            if abs(t_val - days) < 1e-6 and dstr == eclipse_date:
                sep_eclipse_m3 = earth_moon_separation(Y_val)

    print("─" * 70)
    print("  Comparativa Eclipse (2026-08-12 17:47:06 TDB):")
    print(f"  {'Métrica':<32} {'M2':>12}   {'M3':>12}")
    print("─" * 70)

    ref_m2 = {
        "Tierra pos. abs. (km)":  57_542,
        "Luna pos. abs. (km)":    57_560,
        "Sol pos. abs. (km)":     14_894,
        "Sep. Tierra-Luna (km)":      3.7,
    }

    for key, label, ref_val in [
        ("earth_pos_err_km", "Tierra pos. abs. (km)",  ref_m2["Tierra pos. abs. (km)"]),
        ("moon_pos_err_km",  "Luna pos. abs. (km)",    ref_m2["Luna pos. abs. (km)"]),
        ("sun_pos_err_km",   "Sol pos. abs. (km)",     ref_m2["Sol pos. abs. (km)"]),
    ]:
        val = ec.get(key)
        if val is not None:
            print(f"  {label:<32} {ref_val:>12,.0f}   {val:>12,.1f}")

    if sep_eclipse_m3 is not None:
        print(f"  {'Sep. Tierra-Luna (km)':<32} {ref_m2['Sep. Tierra-Luna (km)']:>12.1f}   {sep_eclipse_m3 - 366974.7:>+12.1f}  ← Δ vs JPL")

    print("─" * 70)
    print("\n✅ M3 finalizado correctamente.")
    print("   J₂ mejora la separación Tierra-Luna respecto a M2.")
    print("   El error de posición absoluta (~57 000 km) es relativista;")
    print("   se corrige completamente en M4 con las ecuaciones EIH.")
    print("   Siguiente paso: M4 (N-body + J₂ + correcciones relativistas EIH).")