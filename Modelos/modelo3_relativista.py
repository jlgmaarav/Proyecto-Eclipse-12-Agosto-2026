"""
MODELO 3 — N-cuerpos + correcciones relativistas EIH
═══════════════════════════════════════════════════════════════════════════════

Añade las correcciones post-newtonianas de Einstein-Infeld-Hoffmann (EIH)
al hamiltoniano del Modelo 2.

La corrección EIH a la aceleración del cuerpo i es (en unidades c=1 reducidas):

  a_i^EIH = (1/c²) · Σⱼ (GM[j]/rᵢⱼ²) · {
      [vᵢ² - 4·vᵢ·vⱼ + 2·vⱼ² - 3/2·(n̂ᵢⱼ·vⱼ)² - 4·Σₖ≠ᵢ GM[k]/rᵢₖ - Σₖ≠ⱼ GM[k]/rⱼₖ] · n̂ᵢⱼ
    + [4·(n̂ᵢⱼ·vᵢ) - 3·(n̂ᵢⱼ·vⱼ)] · (vᵢ - vⱼ)
  } + (7/2c²) · Σⱼ (GM[j]/rᵢⱼ²) · aⱼ^Newton

donde n̂ᵢⱼ = (rⱼ - rᵢ)/|rⱼ - rᵢ| es el vector unitario de i a j.

Referencia: Soffel et al. 2003, AJ 126, 2687 (ecuación estándar EIH)

Estrategia de implementación:
  - Primero calcular todas las aceleraciones newtonianas
  - Luego calcular la corrección EIH usando esas aceleraciones
  - El integrador sigue siendo Yoshida O(4) (igual que Modelos 1 y 2)

Nota sobre magnitudes esperadas:
  - Corrección EIH / aceleración newtoniana ~ v²/c² ~ (30 km/s)²/(3e5 km/s)² ~ 1e-8
  - En la Luna: aceleración ~ 2.7e-3 m/s² → corrección ~ 2.7e-11 m/s²
  - Acumulado en 7 meses: Δv ~ 2.7e-11 * 7*30*86400 ~ 5e-4 m/s → Δr ~ 100 km
  - → Esperamos una reducción del error sistemático de orden 100-200 km
    pero no más, por el límite del caos (λ=0.0145/día)
"""

import numpy as np
import time
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from benchmark import eclipse_error, print_benchmark, load_reference_data, ECLIPSE_UTC
from constantes import (AU, DAY, MASAS, GM, N, I_SOL, I_TIE, I_LUN,
                        I_perp, I_z, dI, C_YOSHIDA as _C, D_YOSHIDA as _D)
from modelo2_ncuerpos import (construir_estado, ir, ip, N_STATE,
                              ITH, IPH, IPS, IPTH, IPPH, IPPS, _IQ, _IP)

# Velocidad de la luz en AU/día
C_LUZ = 299792.458e3 / (AU / DAY)   # m/s → AU/día
C2    = C_LUZ**2                    # (AU/día)²

# ─── ACELERACIÓN NEWTONIANA (sin masa, en AU/día²) ────────────────────────────
def acels_newton(r):
    """
    Calcula aceleraciones newtonianas para todos los cuerpos.
    a[i] = Σⱼ≠ᵢ GM[j]/rᵢⱼ³ · (rⱼ - rᵢ)   [AU/día²]
    """
    a = np.zeros((N, 3))
    for i in range(N):
        for j in range(i+1, N):
            dr = r[j] - r[i]
            d2 = np.dot(dr, dr)
            d3 = d2 * np.sqrt(d2)
            f  = dr / d3
            a[i] += GM[j] * f
            a[j] -= GM[i] * f
    return a


# ─── CORRECCIÓN EIH ───────────────────────────────────────────────────────────
def correccion_eih(r, v, a_newton):
    """
    Corrección post-newtoniana EIH a la aceleración de cada cuerpo.
    Devuelve da[i] en AU/día².

    Fórmula (Will 1993, Misner-Thorne-Wheeler adaptada para N cuerpos):

    da_i = (1/c²) Σⱼ≠ᵢ (GM[j]/rᵢⱼ) {
        [-v_i² - 2v_j² + 4(v_i·v_j) + 3/2(n̂ᵢⱼ·v_j)²
         + 4 Σₖ≠ᵢ GM[k]/rᵢₖ + Σₖ≠ⱼ GM[k]/rⱼₖ] · (r_j - r_i)/rᵢⱼ²
        + [4(v_i - v_j) · (r_j - r_i)/rᵢⱼ²] · (v_i - v_j)  ... término mixto
        + [4(n̂ᵢⱼ·v_i) - 3(n̂ᵢⱼ·v_j)] · (v_i - v_j) / rᵢⱼ
    }
    + (7/(2c²)) Σⱼ≠ᵢ (GM[j]/rᵢⱼ) · a_j^Newton

    Implementamos la forma compacta estándar de Newhall et al. (1983) / DE405 documentation.
    """
    da = np.zeros((N, 3))

    # Precalcular potenciales gravitatorios en cada cuerpo
    # Phi[i] = Σⱼ≠ᵢ GM[j]/rᵢⱼ
    Phi = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i != j:
                d = np.linalg.norm(r[j] - r[i])
                Phi[i] += GM[j] / d

    for i in range(N):
        vi2 = np.dot(v[i], v[i])

        for j in range(N):
            if i == j:
                continue

            dr  = r[j] - r[i]
            d   = np.linalg.norm(dr)
            d2  = d * d
            nij = dr / d          # vector unitario i→j

            vj2  = np.dot(v[j], v[j])
            vivj = np.dot(v[i], v[j])
            nijvj = np.dot(nij, v[j])
            nijvi = np.dot(nij, v[i])
            dv   = v[i] - v[j]

            # Término escalar (coeficiente de n̂ᵢⱼ)
            A = (-vi2
                 - 2.0 * vj2
                 + 4.0 * vivj
                 + 1.5 * nijvj**2
                 + 4.0 * Phi[i]        # potencial en i (suma sobre k≠i)
                 - Phi[j]              # potencial en j (suma sobre k≠j, con signo)
                 )

            # Término vectorial (coeficiente de (vᵢ - vⱼ))
            B = 4.0 * nijvi - 3.0 * nijvj

            # Contribución al cuerpo i por j
            da[i] += (GM[j] / (C2 * d2)) * (A * nij + B * dv)

        # Término de aceleración cruzada: (7/2c²) Σⱼ GM[j]/rᵢⱼ · a_j^Newton
        for j in range(N):
            if i == j:
                continue
            d = np.linalg.norm(r[j] - r[i])
            da[i] += (3.5 / C2) * (GM[j] / d) * a_newton[j]

    return da


# ─── DERIVADAS COMPLETAS (Newton + EIH) ──────────────────────────────────────
def derivadas_eih(s):
    """
    dstate/dt con correcciones relativistas EIH.
    Sin rotación terrestre ni J2 (para aislar el efecto relativista puro).
    """
    ds = np.zeros(N_STATE)

    r = np.array([s[ir(i):ir(i)+3] for i in range(N)])
    p = np.array([s[ip(i):ip(i)+3] for i in range(N)])
    v = np.array([p[i] / MASAS[i] for i in range(N)])  # velocidades AU/día

    # Cinemática: ṙ = v = p/m
    for i in range(N):
        ds[ir(i):ir(i)+3] = v[i]

    # Aceleración newtoniana
    a_n = acels_newton(r)

    # Corrección EIH
    da_eih = correccion_eih(r, v, a_n)

    # Dinámica: ṗᵢ = mᵢ · (aᵢ_Newton + aᵢ_EIH)
    for i in range(N):
        ds[ip(i):ip(i)+3] = MASAS[i] * (a_n[i] + da_eih[i])

    return ds


# ─── INTEGRADOR YOSHIDA ───────────────────────────────────────────────────────
def yoshida4(s, h):
    s = s.copy()
    for k in range(3):
        ds = derivadas_eih(s)
        s[_IQ] += _C[k] * h * ds[_IQ]
        ds = derivadas_eih(s)
        s[_IP] += _D[k] * h * ds[_IP]
    ds = derivadas_eih(s)
    s[_IQ] += _C[3] * h * ds[_IQ]
    return s


# ─── ENERGÍA (diagnóstico) ────────────────────────────────────────────────────
def energia(s):
    r = np.array([s[ir(i):ir(i)+3] for i in range(N)])
    p = np.array([s[ip(i):ip(i)+3] for i in range(N)])
    T = sum(np.dot(p[i], p[i]) / (2*MASAS[i]) for i in range(N))
    V = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(r[j]-r[i])
            V -= GM[i] * MASAS[j] / d
    # Nota: la energía EIH no se conserva exactamente con Yoshida estándar
    # porque el hamiltoniano EIH no es separable en T+V. El ΔE/E será mayor
    # que en el modelo newtoniano puro, pero sigue siendo pequeño.
    return T + V


# ─── BÚSQUEDA DEL ECLIPSE ─────────────────────────────────────────────────────
def buscar_eclipse(s, t, h, t0):
    sep_min=999.0; s_ecl=None; t_ecl=t
    while True:
        vES = s[ir(I_SOL):ir(I_SOL)+3] - s[ir(I_TIE):ir(I_TIE)+3]
        vEL = s[ir(I_LUN):ir(I_LUN)+3] - s[ir(I_TIE):ir(I_TIE)+3]
        c = np.clip(np.dot(vES,vEL) / (np.linalg.norm(vES)*np.linalg.norm(vEL)), -1, 1)
        sep = np.degrees(np.arccos(c))
        if sep < sep_min:
            sep_min=sep; s_ecl=s.copy(); t_ecl=t
        s = yoshida4(s, h); t += h
        if sep_min < 1.0 and sep > sep_min + 0.1: break
        if t > (datetime(2026,8,20)-t0).days: break
    return s_ecl, t0+timedelta(days=float(t_ecl)), sep_min


# ─── FUNCIÓN PRINCIPAL ────────────────────────────────────────────────────────
def run(h=0.005, verbose=True):
    T0=datetime(2026,1,1); DIAS=223

    ref = load_reference_data()
    ci  = ref.get(T0.isoformat()) if ref else None
    if ci is None:
        print("ERROR: ejecuta benchmark.py --fetch"); return None

    # Estado inicial: igual que experimento sin rotación
    s = construir_estado(ci)
    s[ITH:IPPS+1] = 0.0   # sin rotación terrestre

    E0 = energia(s)

    # Magnitud de la corrección EIH en el paso inicial (diagnóstico)
    r0 = np.array([s[ir(i):ir(i)+3] for i in range(N)])
    p0 = np.array([s[ip(i):ip(i)+3] for i in range(N)])
    v0 = np.array([p0[i]/MASAS[i] for i in range(N)])
    a_n0 = acels_newton(r0)
    da_eih0 = correccion_eih(r0, v0, a_n0)
    ratio_tierra = np.linalg.norm(da_eih0[I_TIE]) / np.linalg.norm(a_n0[I_TIE])
    ratio_luna   = np.linalg.norm(da_eih0[I_LUN]) / np.linalg.norm(a_n0[I_LUN])

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  MODELO 3 — N-cuerpos + correcciones EIH relativistas")
        print(f"{'═'*60}")
        print(f"  h={h}d={h*DAY:.0f}s  |  {DIAS} días  |  {int(DIAS/h):,} pasos")
        print(f"  E₀={E0:.6e}  |  {N_STATE} variables")
        print(f"  c_luz={C_LUZ:.4f} AU/día")
        print(f"  Corrección EIH/Newton: Tierra={ratio_tierra:.2e}  Luna={ratio_luna:.2e}\n")

    t0c=time.time(); t=0.0; N_p=int(DIAS/h); Nr=max(1,N_p//10); Em=0.0

    for i in range(N_p):
        s = yoshida4(s, h); t += h
        if verbose and i % Nr == 0:
            E   = energia(s); dE = abs((E-E0)/E0); Em = max(Em, dE)
            d   = np.linalg.norm(s[ir(I_LUN):ir(I_LUN)+3]-s[ir(I_TIE):ir(I_TIE)+3])*AU/1e3
            fecha = T0 + timedelta(days=float(t))
            print(f"  {fecha.strftime('%d %b %Y')}  |  ΔE/E={dE:.2e}  |  d_Luna={d:.0f} km")

    t1c = time.time()
    if verbose:
        Ef = energia(s)
        print(f"\n  CPU={t1c-t0c:.1f}s  |  ΔE/E_fin={abs((Ef-E0)/E0):.2e}  |  max={Em:.2e}")
        print(f"  Buscando eclipse...")

    s_ecl, t_ecl, sep = buscar_eclipse(s, t, h/10, T0)
    if verbose:
        print(f"  Máximo: {t_ecl.strftime('%d %b %Y  %H:%M UTC')}  θ={sep:.4f}°")

    ps = s_ecl[ir(I_SOL):ir(I_SOL)+3]
    pt = s_ecl[ir(I_TIE):ir(I_TIE)+3]
    pl = s_ecl[ir(I_LUN):ir(I_LUN)+3]
    ref_ecl = ref.get(ECLIPSE_UTC.isoformat()) if ref else None
    m = eclipse_error(ps, pt, pl, t_ecl, ref_ecl)

    if verbose:
        print(f"\n  ── Resultados ──")
        print(f"  d_Luna={m['d_luna_km']:,} km  |  Total={'SÍ ✓' if m['es_total'] else 'NO ✗'}")
        print(f"  Magnitud fase: {m['fase_magnitud']:.5f}")
        if m['error_pos_luna_km'] is not None: print(f"  Err_pos={m['error_pos_luna_km']:,} km")
        if m['error_ang_arcmin']  is not None: print(f"  Err_ang={m['error_ang_arcmin']:.3f} arcmin")
        if m['error_tiempo_min']  is not None: print(f"  Err_t  ={m['error_tiempo_min']:.1f} min\n")

    return {
        "nombre":       "Modelo 3: N-cuerpos + EIH",
        "metricas":     m,
        "tiempo_cpu_s": t1c - t0c,
        "descripcion":  f"10 cuerpos + correcciones relativistas EIH, h={h}d",
        "_ratio_eih_tierra": ratio_tierra,
        "_ratio_eih_luna":   ratio_luna,
        "_pos_sol":    ps,
        "_pos_tierra": pt,
        "_pos_luna":   pl,
        "_t_ecl":      t_ecl,
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Modelo 3 — EIH relativista")
    parser.add_argument("--h",        type=float, default=0.005)
    parser.add_argument("--rapido",   action="store_true")
    parser.add_argument("--benchmark",action="store_true",
                        help="Tabla comparativa Modelo 1, 2, exp, 3")
    args = parser.parse_args()

    h = 0.05 if args.rapido else args.h
    if args.rapido: print("Modo rápido: h=0.05 días")

    r3 = run(h=h, verbose=True)

    if args.benchmark and r3:
        import modelo1_3cuerpos as m1
        import modelo2_ncuerpos as m2
        from experimento_sin_rotacion import run_sin_rot

        ref     = load_reference_data()
        ci_ene  = ref.get(datetime(2026,1,1).isoformat()) if ref else None

        print("Ejecutando Modelo 1...")
        r1 = m1.run(ci_dict=ci_ene, h=h, verbose=False)
        print("Ejecutando Modelo 2...")
        r2 = m2.run(h=h, verbose=False)
        print("Ejecutando Experimento sin rotación...")
        r_exp = run_sin_rot(h=h, verbose=False)

        print_benchmark([r1, r2, r_exp, r3])