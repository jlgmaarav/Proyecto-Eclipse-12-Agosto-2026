"""
benchmark.py
============
Módulo de referencia para el Proyecto Eclipse 2026.

Fuentes de datos (todos en la misma carpeta que este archivo):
  - initial_conditions.csv       condiciones iniciales 1 ene 2026 (DE441)
  - jpl_reference_data.json      posiciones JPL en 5 fechas (verdad absoluta)
  - deltat.data                  ΔT histórico
  - deltat.preds                 ΔT predicciones

Autor: Proyecto Eclipse 2026
"""

import csv
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# Rutas — todos los datos en la misma carpeta que este script
# ─────────────────────────────────────────────
_BASE     = Path(__file__).parent
_IC_FILE  = _BASE / "initial_conditions.csv"
_REF_FILE = _BASE / "jpl_reference_data.json"
_DT_DATA  = _BASE / "deltat.data"
_DT_PREDS = _BASE / "deltat.preds"

# ─────────────────────────────────────────────
# Constantes físicas
# ─────────────────────────────────────────────
AU_KM = 149_597_870.700
DAY_S = 86_400.0

GM = {
    "Sun":     132_712_440_018.0,
    "Mercury":         22_031.868_551,
    "Venus":        3_248_585.920_790,
    "Earth":          398_600.435_436,
    "Moon":             4_902.800_066,
    "Mars":            42_828.375_214,
    "Jupiter":    126_712_764.100,
    "Saturn":      37_940_584.841_800,
    "Uranus":       5_794_556.400,
    "Neptune":      6_836_527.100_580,
}

R_EQ_EARTH  = 6_378.137
R_POL_EARTH = 6_356.752
J2_EARTH    = 0.001_082_625_45
R_SUN       = 696_000.0
R_MOON      = 1_737.4

# ─────────────────────────────────────────────
# Normalización de nombres
# ─────────────────────────────────────────────
_NAME_MAP = {
    "Sun": "Sun", "Earth": "Earth", "Moon": "Moon",
    "Mercury Barycenter": "Mercury", "Venus Barycenter": "Venus",
    "Mars Barycenter": "Mars",       "Jupiter Barycenter": "Jupiter",
    "Saturn Barycenter": "Saturn",   "Uranus Barycenter": "Uranus",
    "Neptune Barycenter": "Neptune",
    "Sol": "Sun",   "Tierra": "Earth", "Luna": "Moon",
    "Mercurio": "Mercury", "Marte": "Mars", "Saturno": "Saturn",
    "Urano": "Uranus", "Neptuno": "Neptune",
    "Jupiter": "Jupiter", "Venus": "Venus",
}

def _normalize(name: str) -> str:
    return _NAME_MAP.get(name.strip(), name.strip())


# ═══════════════════════════════════════════════════════════════
# Fechas de benchmark
# ═══════════════════════════════════════════════════════════════
BENCHMARK_DATES = [
    "2026-01-01T00:00:00",
    "2026-03-01T00:00:00",
    "2026-06-01T00:00:00",
    "2026-08-01T00:00:00",
    "2026-08-12T17:47:06",
]

def _date_to_days(date_str: str) -> float:
    t0 = datetime(2026, 1, 1, 0, 0, 0)
    t  = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
    return (t - t0).total_seconds() / DAY_S

BENCHMARK_DAYS = {d: _date_to_days(d) for d in BENCHMARK_DATES}


# ═══════════════════════════════════════════════════════════════
# 1. Condiciones iniciales
# ═══════════════════════════════════════════════════════════════
def load_initial_conditions() -> dict:
    ic = {}
    with open(_IC_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = _normalize(row["body"])
            ic[name] = {
                "pos": np.array([float(row["x_au"]),
                                 float(row["y_au"]),
                                 float(row["z_au"])]),
                "vel": np.array([float(row["vx_au_d"]),
                                 float(row["vy_au_d"]),
                                 float(row["vz_au_d"])]),
            }
    return ic


# ═══════════════════════════════════════════════════════════════
# 2. Datos de referencia JPL (verdad absoluta)
# ═══════════════════════════════════════════════════════════════
def load_reference_data() -> dict:
    with open(_REF_FILE, encoding="utf-8") as f:
        raw = json.load(f)

    ref = {}
    for date_str, bodies in raw.items():
        ref[date_str] = {}
        for bname, data in bodies.items():
            name = _normalize(bname)
            ref[date_str][name] = {
                "pos": np.array(data["pos"]),
                "vel": np.array(data["vel"]),
                "r":   float(data["r"]),
            }
    return ref


# ═══════════════════════════════════════════════════════════════
# 3. Delta T
# ═══════════════════════════════════════════════════════════════
def _load_delta_t_table() -> tuple:
    years, dts = [], []
    with open(_DT_DATA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                y, m = int(parts[0]), int(parts[1])
                years.append(y + (m - 0.5) / 12.0)
                dts.append(float(parts[3]))

    last = years[-1] if years else 0.0
    with open(_DT_PREDS, encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                yr = float(parts[1])
                dt = float(parts[2])
            except ValueError:
                continue
            if yr > last:
                years.append(yr)
                dts.append(dt)

    return np.array(years), np.array(dts)

_DT_YEARS, _DT_SECONDS = _load_delta_t_table()

def get_delta_t(year_frac: float) -> float:
    return float(np.interp(year_frac, _DT_YEARS, _DT_SECONDS))

def days_to_year_frac(days: float) -> float:
    return 2026.0 + days / 365.25


# ═══════════════════════════════════════════════════════════════
# 4. Métricas de error
# ═══════════════════════════════════════════════════════════════
def position_error(pos_model: np.ndarray, pos_ref: np.ndarray) -> dict:
    diff   = pos_model - pos_ref
    err_au = float(np.linalg.norm(diff))
    return {"error_au": err_au, "error_km": err_au * AU_KM, "error_vec": diff}

def velocity_error(vel_model: np.ndarray, vel_ref: np.ndarray) -> dict:
    diff     = vel_model - vel_ref
    err_au_d = float(np.linalg.norm(diff))
    return {"error_au_d": err_au_d, "error_km_s": err_au_d * AU_KM / DAY_S}


# ═══════════════════════════════════════════════════════════════
# 5. Evaluación completa de un modelo
# ═══════════════════════════════════════════════════════════════
def run_benchmark(model_states: dict, model_name: str = "Modelo") -> dict:
    ref    = load_reference_data()
    report = {
        "model":              model_name,
        "dates":              {},
        "summary":            {},
        "earth_moon_eclipse": {},
    }
    all_errors = {}

    for date_str in BENCHMARK_DATES:
        if date_str not in model_states:
            continue
        if date_str not in ref:
            continue

        report["dates"][date_str] = {}
        model_at_t = model_states[date_str]
        ref_at_t   = ref[date_str]

        for body, state in model_at_t.items():
            if body not in ref_at_t:
                continue
            pe = position_error(state["pos"], ref_at_t[body]["pos"])
            ve = velocity_error(state["vel"], ref_at_t[body]["vel"])
            report["dates"][date_str][body] = {
                "pos_err_km":   pe["error_km"],
                "vel_err_km_s": ve["error_km_s"],
                "pos_err_au":   pe["error_au"],
            }
            all_errors.setdefault(body, []).append(pe["error_km"])

    for body, errors in all_errors.items():
        report["summary"][body] = {
            "mean_pos_err_km": float(np.mean(errors)),
            "max_pos_err_km":  float(np.max(errors)),
            "n_dates":         len(errors),
        }

    eclipse_date = "2026-08-12T17:47:06"
    if eclipse_date in report["dates"]:
        ed = report["dates"][eclipse_date]
        report["earth_moon_eclipse"] = {
            "earth_pos_err_km": ed.get("Earth", {}).get("pos_err_km"),
            "moon_pos_err_km":  ed.get("Moon",  {}).get("pos_err_km"),
            "sun_pos_err_km":   ed.get("Sun",   {}).get("pos_err_km"),
        }
    return report


# ═══════════════════════════════════════════════════════════════
# 6. Impresión del informe
# ═══════════════════════════════════════════════════════════════
def print_report(report: dict) -> None:
    sep = "─" * 70
    print(f"\n{'='*70}")
    print(f"  BENCHMARK — {report['model']}")
    print(f"{'='*70}")

    for date_str, bodies in report.get("dates", {}).items():
        print(f"\n  {date_str}")
        print(f"  {'Cuerpo':<18} {'Error pos (km)':>18} {'Error vel (km/s)':>18}")
        print(f"  {sep}")
        for body, errs in sorted(bodies.items()):
            print(f"  {body:<18} {errs['pos_err_km']:>18.3f} {errs['vel_err_km_s']:>18.6f}")

    print(f"\n  {sep}")
    print(f"  RESUMEN POR CUERPO (todas las fechas)")
    print(f"  {'Cuerpo':<18} {'Media error (km)':>18} {'Max error (km)':>18}")
    print(f"  {sep}")
    for body, s in sorted(report.get("summary", {}).items()):
        print(f"  {body:<18} {s['mean_pos_err_km']:>18.3f} {s['max_pos_err_km']:>18.3f}")

    ec = report.get("earth_moon_eclipse", {})
    if any(v is not None for v in ec.values()):
        print(f"\n  Eclipse (2026-08-12 17:47:06 TDB):")
        for key, label in [("earth_pos_err_km", "Tierra"),
                            ("moon_pos_err_km",  "Luna"),
                            ("sun_pos_err_km",   "Sol")]:
            val = ec.get(key)
            if val is not None:
                print(f"     {label:<10}: {val:>12.3f} km")
    print(f"\n{'='*70}\n")


# ═══════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Cargando condiciones iniciales...")
    ic = load_initial_conditions()
    print(f"  Cuerpos: {sorted(ic.keys())}")

    print("\nCargando datos de referencia JPL...")
    ref = load_reference_data()
    print(f"  Fechas: {sorted(ref.keys())}")

    print("\nDelta T:")
    for date_str, days in BENCHMARK_DAYS.items():
        dt = get_delta_t(days_to_year_frac(days))
        print(f"  {date_str}  ->  DT = {dt:.4f} s")

    print("\nSelf-test (error debe ser 0 en todo)...")
    model_states = {
        date_str: {body: {"pos": d["pos"], "vel": d["vel"]}
                   for body, d in bodies.items()}
        for date_str, bodies in ref.items()
    }
    report = run_benchmark(model_states, "Self-test JPL vs JPL")
    print_report(report)