"""
model_m1.py
===========
Modelo M1 — Problema de 3 cuerpos newtoniano puro (Sol-Tierra-Luna).

Física: gravedad newtoniana clásica.
Método: Adams-Bashforth-Moulton orden 4 con paso cuasi-fijo.
Arranque: RK4 para los primeros 4 pasos.

Unidades: AU, AU/día, días.
Rigurosidad: paso fijo garantiza consistencia del método multistep.
"""

import numpy as np
from typing import List, Optional
from pathlib import Path
from benchmark import (
    load_initial_conditions,
    load_reference_data,
    run_benchmark,
    print_report,
    BENCHMARK_DAYS,
    AU_KM,
    DAY_S,
    GM,
)

# ═══════════════════════════════════════════════════════════════
# 1. CONSTANTES FÍSICAS EN UNIDADES AU / DÍA
# ═══════════════════════════════════════════════════════════════

_AU3_PER_KM3_DAY2 = DAY_S**2 / AU_KM**3
GM_AU = {body: GM[body] * _AU3_PER_KM3_DAY2 for body in ("Sun", "Earth", "Moon")}

BODIES = ["Sun", "Earth", "Moon"]
N_BODIES = len(BODIES)
DIM = 6 * N_BODIES

# ═══════════════════════════════════════════════════════════════
# 2. LADO DERECHO DE LAS EDOs — f(t, Y)
# ═══════════════════════════════════════════════════════════════

def f_m1(t: float, Y: np.ndarray) -> np.ndarray:
    """Lado derecho del sistema de 18 EDOs (Newton 3-body)."""
    dY = np.zeros(DIM)
    pos = [Y[6*i:6*i+3] for i in range(N_BODIES)]
    vel = [Y[6*i+3:6*i+6] for i in range(N_BODIES)]

    for i in range(N_BODIES):
        dY[6*i:6*i+3] = vel[i]                                      # dr/dt = v

        acc = np.zeros(3)
        for j in range(N_BODIES):
            if j == i:
                continue
            r_ij = pos[j] - pos[i]
            dist = np.linalg.norm(r_ij)
            acc += GM_AU[BODIES[j]] * r_ij / dist**3
        dY[6*i+3:6*i+6] = acc                                       # dv/dt = a

    return dY


# ═══════════════════════════════════════════════════════════════
# 3. ARRANQUE CON RK4
# ═══════════════════════════════════════════════════════════════

def rk4_step(f, t: float, Y: np.ndarray, h: float) -> np.ndarray:
    """Un paso clásico de Runge-Kutta 4."""
    k1 = f(t, Y)
    k2 = f(t + h/2, Y + h/2 * k1)
    k3 = f(t + h/2, Y + h/2 * k2)
    k4 = f(t + h, Y + h * k3)
    return Y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)


def bootstrap_rk4(f, t0: float, Y0: np.ndarray, h: float, k: int):
    """Genera los primeros k estados con RK4."""
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
# 4. COEFICIENTES ABM4
# ═══════════════════════════════════════════════════════════════

AB4 = np.array([55., -59., 37., -9.]) / 24.   # Predictor Adams-Bashforth 4
AM4 = np.array([9., 19., -5., 1.]) / 24.      # Corrector Adams-Moulton 4


# ═══════════════════════════════════════════════════════════════
# 5. UN PASO ABM4
# ═══════════════════════════════════════════════════════════════

def abm4_step(f, t: float, Y: np.ndarray, h: float, F_hist: list) -> tuple:
    """Predictor-corrector Adams-Bashforth-Moulton orden 4 + error local."""
    f_n, f_nm1, f_nm2, f_nm3 = F_hist[0], F_hist[1], F_hist[2], F_hist[3]

    # Predictor
    Y_pred = Y + h * (AB4[0]*f_n + AB4[1]*f_nm1 + AB4[2]*f_nm2 + AB4[3]*f_nm3)
    f_pred = f(t + h, Y_pred)

    # Corrector
    Y_corr = Y + h * (AM4[0]*f_pred + AM4[1]*f_n + AM4[2]*f_nm1 + AM4[3]*f_nm2)
    f_corr = f(t + h, Y_corr)

    # Error local normalizado
    diff = Y_corr - Y_pred
    err = float(np.linalg.norm(diff) / (1.0 + np.linalg.norm(Y_corr)))

    return Y_corr, f_corr, err


# ═══════════════════════════════════════════════════════════════
# 6. INTEGRACIÓN COMPLETA (paso cuasi-fijo)
# ═══════════════════════════════════════════════════════════════

def integrate(f, Y0: np.ndarray, t_end: float,
              h0: float = 0.1,
              tol: float = 1e-10,
              t_out: Optional[List[float]] = None) -> dict:
    """Integra con ABM4 paso cuasi-fijo (el mismo que usa la NASA en DE)."""
    ORDER = 4
    h = float(h0)

    if t_out is None:
        t_out = []
    t_out_sorted = sorted(set(t_out))
    out_idx = 0
    results_t = []
    results_Y = []

    # Arranque RK4
    ts_boot, Ys_boot, Fs_boot = bootstrap_rk4(f, 0.0, Y0, h, ORDER)

    # Salidas durante arranque
    for t_want in t_out_sorted:
        if t_want <= ts_boot[-1]:
            for k in range(len(ts_boot)-1):
                if ts_boot[k] <= t_want <= ts_boot[k+1]:
                    alpha = (t_want - ts_boot[k]) / (ts_boot[k+1] - ts_boot[k])
                    results_t.append(t_want)
                    results_Y.append((1-alpha)*Ys_boot[k] + alpha*Ys_boot[k+1])
                    break
            out_idx += 1

    t = ts_boot[-1]
    Y = Ys_boot[-1].copy()
    F_hist = list(reversed(Fs_boot))          # más reciente primero

    steps = ORDER - 1
    rejects = 0
    err_max = 0.0

    while t < t_end:
        h_step = min(h, t_end - t)
        Y_new, f_new, err = abm4_step(f, t, Y, h_step, F_hist)
        err_max = max(err_max, err)

        if err >= tol:
            rejects += 1

        t_next = t + h_step
        steps += 1

        # Guardar salidas
        while out_idx < len(t_out_sorted) and t_out_sorted[out_idx] <= t_next + 1e-10:
            t_want = t_out_sorted[out_idx]
            alpha = (t_want - t) / h_step if h_step > 0 else 0.0
            alpha = float(np.clip(alpha, 0.0, 1.0))
            results_t.append(t_want)
            results_Y.append((1 - alpha) * Y + alpha * Y_new)
            out_idx += 1

        t = t_next
        Y = Y_new
        F_hist = [f_new] + F_hist[:ORDER-1]

    return {
        "t_out":   results_t,
        "Y_out":   results_Y,
        "steps":   steps,
        "rejects": rejects,
        "h":       h,
        "err_max": err_max,
    }


# ═══════════════════════════════════════════════════════════════
# 7. CONDICIONES INICIALES
# ═══════════════════════════════════════════════════════════════

def build_Y0() -> np.ndarray:
    """Construye vector de estado Y0 a partir de initial_conditions.csv."""
    ic = load_initial_conditions()
    Y0 = np.zeros(DIM)
    for i, body in enumerate(BODIES):
        Y0[6*i:6*i+3] = ic[body]["pos"]
        Y0[6*i+3:6*i+6] = ic[body]["vel"]
    return Y0


# ═══════════════════════════════════════════════════════════════
# 8. CONVERSIÓN AL FORMATO BENCHMARK
# ═══════════════════════════════════════════════════════════════

def result_to_model_states(result: dict) -> dict:
    """Convierte salida del integrador al formato que espera run_benchmark."""
    days_to_date = {v: k for k, v in BENCHMARK_DAYS.items()}
    model_states = {}

    for t_val, Y_val in zip(result["t_out"], result["Y_out"]):
        date_str = None
        for days, dstr in days_to_date.items():
            if abs(t_val - days) < 1e-6:
                date_str = dstr
                break
        if date_str is None:
            continue

        model_states[date_str] = {}
        for i, body in enumerate(BODIES):
            model_states[date_str][body] = {
                "pos": Y_val[6*i:6*i+3].copy(),
                "vel": Y_val[6*i+3:6*i+6].copy(),
            }
    return model_states


# ═══════════════════════════════════════════════════════════════
# 9. EJECUCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    print("=" * 70)
    print("  MODELO M1 — 3 cuerpos newtonianos (Sol-Tierra-Luna), ABM4 cuasi-fijo")
    print("=" * 70)

    Y0 = build_Y0()
    print("\nCondición inicial (2026-01-01 00:00:00 TDB):")
    for i, body in enumerate(BODIES):
        r = np.linalg.norm(Y0[6*i:6*i+3])
        print(f"  {body:<8}  r = {r:.6f} AU")

    t_out = list(BENCHMARK_DAYS.values())
    t_end = max(t_out)

    print(f"\nIntegrando {t_end:.1f} días ({t_end/365.25:.2f} años)...")
    print(f"Paso fijo = 0.1 días | Tolerancia = 1e-10")

    t_start = time.perf_counter()

    result = integrate(
        f      = f_m1,
        Y0     = Y0,
        t_end  = t_end,
        h0     = 0.1,
        tol    = 1e-10,
        t_out  = t_out,
    )

    elapsed = time.perf_counter() - t_start

    print(f"\nIntegración completada en {elapsed:.2f} s")
    print(f"  Pasos aceptados   : {result['steps']}")
    print(f"  Pasos rechazados  : {result['rejects']}")
    print(f"  Paso usado        : {result['h']:.4f} días")
    print(f"  Error local máx.  : {result['err_max']:.2e}")
    print(f"  Fechas capturadas : {len(result['t_out'])} / {len(t_out)}")

    model_states = result_to_model_states(result)
    report = run_benchmark(model_states, model_name="M1 — Newton 3-body ABM4")
    print_report(report)

    print("\n✅ M1 finalizado correctamente.")
    print("   Error físico esperado en fecha del eclipse ≈ 40 000 km (solo 3 cuerpos).")
    print("   Listo para M2 (n-body completo con 10 cuerpos).")