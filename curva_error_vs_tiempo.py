"""
CURVA DE ERROR vs. DÍAS DE INTEGRACIÓN
═══════════════════════════════════════════════════════════════════════════════

Mide cómo crece el error en la posición de la Luna en función de cuántos
días antes del eclipse se inician las condiciones iniciales de JPL.

Fechas de inicio disponibles en jpl_reference_data.json:
  2026-08-12 17:47  →  ~0 días  (momento del eclipse, error = 0 por definición)
  2026-08-01        →  11 días
  2026-06-01        →  72 días
  2026-03-01        → 163 días
  2026-01-01        → 223 días

Para cada fecha de inicio:
  1. Tomar CI de JPL para esa fecha
  2. Integrar con N-cuerpos newtoniano hasta el 12 agosto 17:47 UTC
  3. Comparar posición de la Luna con JPL en el momento del eclipse
  4. Registrar error_pos, error_ang, error_tiempo

Modelo usado: N-cuerpos newtoniano puro (10 cuerpos, sin rotación ni J2)
Esto aisla el crecimiento de error de la física faltante (caos, relatividad, etc.)
"""

import numpy as np
import time
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Modelos'))
from benchmark import eclipse_error, load_reference_data, ECLIPSE_UTC, print_benchmark
from experimento_sin_rotacion import (yoshida4_sin_rot, energia_sin_rot,
                                       buscar_eclipse)
from constantes import AU, MASAS, GM, N, I_SOL, I_TIE, I_LUN
from modelo2_ncuerpos import (construir_estado, ir, ip, ITH, IPPS)

# Fechas de inicio disponibles en el JSON (las que tenemos de JPL)
FECHAS_INICIO = [
    datetime(2026, 1, 1),
    datetime(2026, 3, 1),
    datetime(2026, 6, 1),
    datetime(2026, 8, 1),
    # 2026-08-12 17:47 es el eclipse mismo — no integramos desde ahí
]

def run_desde_fecha(fecha_inicio, h=0.005, verbose=False):
    """
    Integra desde fecha_inicio hasta el eclipse y mide el error.
    Usa N-cuerpos newtoniano puro (sin rotación).
    """
    ref = load_reference_data()
    if ref is None:
        return None

    # Buscar CI en el JSON para esa fecha
    clave = fecha_inicio.isoformat()
    if clave not in ref:
        print(f"  No hay CI para {fecha_inicio.strftime('%d %b %Y')}")
        return None
    ci = ref[clave]

    # Construir estado inicial sin rotación
    s = construir_estado(ci)
    s[ITH:IPPS+1] = 0.0

    E0 = energia_sin_rot(s)
    dias_total = (datetime(2026, 8, 12) - fecha_inicio).days
    N_pasos = int(dias_total / h)

    t0c = time.time()
    t = 0.0
    for _ in range(N_pasos):
        s = yoshida4_sin_rot(s, h)
        t += h

    # Buscar eclipse con paso fino
    s_ecl, t_ecl, sep = buscar_eclipse(s, t, h/10, fecha_inicio)

    # Calcular error de energía
    E_fin = energia_sin_rot(s)
    dE = abs((E_fin - E0) / E0) if E0 != 0 else 0

    t1c = time.time()

    # Métricas
    ps = s_ecl[ir(I_SOL):ir(I_SOL)+3]
    pt = s_ecl[ir(I_TIE):ir(I_TIE)+3]
    pl = s_ecl[ir(I_LUN):ir(I_LUN)+3]
    ref_ecl = ref.get(ECLIPSE_UTC.isoformat())
    m = eclipse_error(ps, pt, pl, t_ecl, ref_ecl)

    return {
        "fecha_inicio":  fecha_inicio,
        "dias":          dias_total,
        "metricas":      m,
        "dE_E":          dE,
        "cpu_s":         t1c - t0c,
        "t_ecl":         t_ecl,
        "sep_deg":       sep,
    }


def imprimir_tabla(resultados):
    sep = "─" * 82
    print("\n")
    print("╔" + "═" * 80 + "╗")
    print("║  CURVA DE ERROR vs. DÍAS DE INTEGRACIÓN · Eclipse 12 ago 2026            " + " "*5 + "║")
    print("║  Modelo: N-cuerpos newtoniano puro (10 cuerpos, sin rotación)             " + " "*5 + "║")
    print("╚" + "═" * 80 + "╝")
    print(f"\n  {'Inicio CI':<16} {'Días':>5}  {'Err_pos':>10}  {'Err_ang':>10}  {'Err_t':>8}  {'ΔE/E':>10}  {'Total?':>7}")
    print(f"  {'':16} {'':5}  {'km':>10}  {'arcmin':>10}  {'min':>8}  {'':>10}  {'':>7}")
    print("  " + sep)

    for r in resultados:
        m = r["metricas"]
        f = r["fecha_inicio"].strftime("%d %b %Y")
        d = r["dias"]
        ep = f"{m['error_pos_luna_km']:>10,.0f}" if m['error_pos_luna_km'] is not None else f"{'—':>10}"
        ea = f"{m['error_ang_arcmin']:>10.3f}"   if m['error_ang_arcmin']  is not None else f"{'—':>10}"
        et = f"{m['error_tiempo_min']:>8.1f}"    if m['error_tiempo_min']  is not None else f"{'—':>8}"
        de = f"{r['dE_E']:>10.2e}"
        tt = "✓ TOTAL" if m['es_total'] else "✗ anular"
        t_pred = r['t_ecl'].strftime("%H:%M UTC")
        print(f"  {f:<16} {d:>5}  {ep}  {ea}  {et}  {de}  {tt:>7}  ({t_pred})")

    print("  " + sep)

    # Análisis del crecimiento
    if len(resultados) >= 2:
        print("\n  Análisis de crecimiento del error:")
        errs = [(r["dias"], r["metricas"]["error_pos_luna_km"])
                for r in resultados if r["metricas"]["error_pos_luna_km"] is not None]
        errs.sort()

        for i in range(1, len(errs)):
            d1, e1 = errs[i-1]
            d2, e2 = errs[i]
            if e1 and e2 and e1 > 0:
                ratio = e2 / e1
                dd = d2 - d1
                # Si el crecimiento es exponencial: e = e0 * exp(lambda * t)
                # ln(e2/e1) = lambda * (d2-d1)
                if ratio > 0:
                    lam = np.log(ratio) / max(dd, 1)
                    print(f"  {d1}d → {d2}d: error {e1:.0f} → {e2:.0f} km  "
                          f"(×{ratio:.2f}, λ={lam:.4f}/día)")

        # Tiempo de duplicación (si hay patrón exponencial)
        if len(errs) >= 3:
            dias_arr = np.array([e[0] for e in errs], dtype=float)
            errs_arr = np.array([e[1] for e in errs], dtype=float)
            # Ajuste log-lineal
            log_e = np.log(errs_arr)
            # Regresión lineal: log(e) = a + lambda*d
            A = np.vstack([np.ones_like(dias_arr), dias_arr]).T
            try:
                coefs, _, _, _ = np.linalg.lstsq(A, log_e, rcond=None)
                lam_fit = coefs[1]
                t_doble = np.log(2) / lam_fit if lam_fit > 0 else float('inf')
                e0_fit = np.exp(coefs[0])
                print(f"\n  Ajuste exponencial: error ≈ {e0_fit:.1f} · exp({lam_fit:.4f}·días)")
                print(f"  Tiempo de duplicación: {t_doble:.1f} días")
                print(f"  (Lyapunov-like: λ ≈ {lam_fit:.4f} día⁻¹ = {lam_fit*365:.2f} año⁻¹)")
            except Exception:
                pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Curva de error vs. días de integración")
    parser.add_argument("--h", type=float, default=0.005,
                        help="Paso de integración en días")
    parser.add_argument("--rapido", action="store_true",
                        help="h=0.05 para prueba rápida")
    args = parser.parse_args()

    h = 0.05 if args.rapido else args.h
    if args.rapido:
        print("Modo rápido: h=0.05 días (precisión reducida)")

    print(f"\nMidiendo error en función de la fecha de inicio de integración...")
    print(f"Paso h={h} días | Modelo: N-cuerpos newtoniano puro\n")

    resultados = []
    for fecha in FECHAS_INICIO:
        dias = (datetime(2026, 8, 12) - fecha).days
        print(f"  Integrando desde {fecha.strftime('%d %b %Y')} ({dias} días)...",
              end=" ", flush=True)
        t0 = time.time()
        r = run_desde_fecha(fecha, h=h)
        if r:
            resultados.append(r)
            m = r["metricas"]
            ep = f"{m['error_pos_luna_km']:.0f} km" if m['error_pos_luna_km'] else "—"
            et = f"{m['error_tiempo_min']:.1f} min" if m['error_tiempo_min'] else "—"
            print(f"Err_pos={ep}  Err_t={et}  ({time.time()-t0:.0f}s)")
        else:
            print("FALLO")

    if resultados:
        imprimir_tabla(resultados)