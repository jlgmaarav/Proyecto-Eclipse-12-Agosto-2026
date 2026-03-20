"""
EXPERIMENTO: Modelo 2 sin rotación terrestre
═══════════════════════════════════════════════════════════════════════════════

Hipótesis: el error de ~960 km del Modelo 2 viene principalmente de los
ángulos de Euler (rotación terrestre + J2), no de los planetas que faltan.

Para verificarlo, desactivamos la rotación terrestre:
  - Theta, phi, psi → 0 (eje polar fijo, sin precesión)
  - J2 → 0 (dI = 0, Tierra esférica)
  - El hamiltoniano se reduce a pura gravitación newtoniana N-cuerpos

Si el error baja drásticamente → los ángulos de Euler son el error dominante.
Si el error no baja → el error viene de física que falta (relatividad, etc.)
"""

import numpy as np
import time
import sys
import os
from datetime import datetime, timedelta
from copy import deepcopy

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Modelos'))
from benchmark import eclipse_error, print_benchmark, load_reference_data, ECLIPSE_UTC
import modelo2_ncuerpos as m2

from constantes import (AU, DAY, G, MASAS, GM, N, I_SOL, I_TIE, I_LUN,
                        C_YOSHIDA as _C, D_YOSHIDA as _D)

ir    = m2.ir
ip    = m2.ip
N_STATE = m2.N_STATE
ITH=m2.ITH; IPH=m2.IPH; IPS=m2.IPS
IPTH=m2.IPTH; IPPH=m2.IPPH; IPPS=m2.IPPS
_IQ=m2._IQ; _IP=m2._IP


# ─── DERIVADAS SIN ROTACIÓN ───────────────────────────────────────────────────
def derivadas_sin_rot(s):
    """
    Solo gravitación newtoniana entre los 10 cuerpos.
    Sin rotación terrestre, sin J2.
    Los 6 últimos componentes (ángulos de Euler y sus momentos) son cero
    y se mantienen en cero toda la integración.
    """
    ds = np.zeros(N_STATE)
    r = np.array([s[ir(i):ir(i)+3] for i in range(N)])
    p = np.array([s[ip(i):ip(i)+3] for i in range(N)])

    # Cinemática traslacional
    for i in range(N):
        ds[ir(i):ir(i)+3] = p[i] / MASAS[i]

    # Dinámica: gravedad newtoniana pura
    for i in range(N):
        for j in range(i+1, N):
            dr = r[j] - r[i]
            d2 = np.dot(dr, dr)
            d3 = d2 * np.sqrt(d2)
            f  = dr / d3
            ds[ip(i):ip(i)+3] += MASAS[i] * GM[j] * f
            ds[ip(j):ip(j)+3] -= MASAS[j] * GM[i] * f

    # Ángulos de Euler: quedan en cero (ds[ITH..IPPS] = 0)
    return ds


def yoshida4_sin_rot(s, h):
    s = s.copy()
    for k in range(3):
        ds = derivadas_sin_rot(s)
        s[_IQ] += _C[k] * h * ds[_IQ]
        ds = derivadas_sin_rot(s)
        s[_IP] += _D[k] * h * ds[_IP]
    ds = derivadas_sin_rot(s)
    s[_IQ] += _C[3] * h * ds[_IQ]
    return s


def energia_sin_rot(s):
    r = np.array([s[ir(i):ir(i)+3] for i in range(N)])
    p = np.array([s[ip(i):ip(i)+3] for i in range(N)])
    T = sum(np.dot(p[i], p[i]) / (2*MASAS[i]) for i in range(N))
    V = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(r[j]-r[i])
            V -= GM[i] * MASAS[j] / d
    return T + V


def buscar_eclipse(s, t, h, t0):
    sep_min=999.0; s_ecl=None; t_ecl=t
    while True:
        vES = s[ir(I_SOL):ir(I_SOL)+3] - s[ir(I_TIE):ir(I_TIE)+3]
        vEL = s[ir(I_LUN):ir(I_LUN)+3] - s[ir(I_TIE):ir(I_TIE)+3]
        c = np.clip(np.dot(vES,vEL)/(np.linalg.norm(vES)*np.linalg.norm(vEL)),-1,1)
        sep = np.degrees(np.arccos(c))
        if sep < sep_min:
            sep_min=sep; s_ecl=s.copy(); t_ecl=t
        s = yoshida4_sin_rot(s, h); t += h
        if sep_min < 1.0 and sep > sep_min + 0.1: break
        if t > (datetime(2026,8,20)-t0).days: break
    return s_ecl, t0+timedelta(days=float(t_ecl)), sep_min


def run_sin_rot(h=0.005, verbose=True):
    T0=datetime(2026,1,1); DIAS=223
    ref=load_reference_data()
    ci=ref.get(T0.isoformat()) if ref else None
    if ci is None:
        print("ERROR: ejecuta benchmark.py --fetch"); return None

    # Estado inicial: igual que Modelo 2 pero con rotación a cero
    s = m2.construir_estado(ci)
    # Desactivar rotación: poner ángulos y momentos rotacionales a cero
    s[ITH]=0.0; s[IPH]=0.0; s[IPS]=0.0
    s[IPTH]=0.0; s[IPPH]=0.0; s[IPPS]=0.0

    E0 = energia_sin_rot(s)

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  EXPERIMENTO — N-cuerpos SIN rotación terrestre")
        print(f"{'═'*60}")
        print(f"  h={h}d  |  {DIAS} días  |  {int(DIAS/h):,} pasos")
        print(f"  E₀={E0:.6e}  |  Solo gravedad newtoniana pura\n")

    t0c=time.time(); t=0.0; N_p=int(DIAS/h); Nr=max(1,N_p//10); Em=0.0

    for i in range(N_p):
        s=yoshida4_sin_rot(s,h); t+=h
        if verbose and i%Nr==0:
            E=energia_sin_rot(s); dE=abs((E-E0)/E0); Em=max(Em,dE)
            d=np.linalg.norm(s[ir(I_LUN):ir(I_LUN)+3]-s[ir(I_TIE):ir(I_TIE)+3])*AU/1e3
            fecha=T0+timedelta(days=float(t))
            print(f"  {fecha.strftime('%d %b %Y')}  |  ΔE/E={dE:.2e}  |  d_Luna={d:.0f} km")

    t1c=time.time()
    if verbose:
        Ef=energia_sin_rot(s)
        print(f"\n  CPU={t1c-t0c:.1f}s  |  ΔE/E_fin={abs((Ef-E0)/E0):.2e}  |  max={Em:.2e}")
        print(f"  Buscando eclipse...")

    s_ecl, t_ecl, sep = buscar_eclipse(s, t, h/10, T0)
    if verbose:
        print(f"  Máximo: {t_ecl.strftime('%d %b %Y  %H:%M UTC')}  θ={sep:.4f}°")

    ps=s_ecl[ir(I_SOL):ir(I_SOL)+3]
    pt=s_ecl[ir(I_TIE):ir(I_TIE)+3]
    pl=s_ecl[ir(I_LUN):ir(I_LUN)+3]
    ref_ecl = ref.get(ECLIPSE_UTC.isoformat()) if ref else None
    m = eclipse_error(ps, pt, pl, t_ecl, ref_ecl)

    if verbose:
        print(f"\n  ── Resultados ──")
        print(f"  d_Luna={m['d_luna_km']:,} km  |  Total={'SÍ ✓' if m['es_total'] else 'NO ✗'}  |  fase={m['fase_magnitud']:.5f}")
        if m['error_pos_luna_km'] is not None: print(f"  Err_pos={m['error_pos_luna_km']:,} km")
        if m['error_ang_arcmin']  is not None: print(f"  Err_ang={m['error_ang_arcmin']:.3f} arcmin")
        if m['error_tiempo_min']  is not None: print(f"  Err_t  ={m['error_tiempo_min']:.1f} min\n")

    return {"nombre":"Experimento: N-cuerpos sin rotación","metricas":m,
            "tiempo_cpu_s":t1c-t0c,"descripcion":f"10 cuerpos newtonianos puros, h={h}d"}


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--h",type=float,default=0.005)
    parser.add_argument("--rapido",action="store_true")
    parser.add_argument("--comparar",action="store_true",
                        help="Comparar con Modelo 1 y Modelo 2 en tabla")
    args=parser.parse_args()
    h=0.05 if args.rapido else args.h
    if args.rapido: print("Modo rápido: h=0.05")

    r_exp=run_sin_rot(h=h, verbose=True)

    if args.comparar and r_exp:
        import modelo1_3cuerpos as m1
        ref=load_reference_data()
        ci_enero=ref.get(datetime(2026,1,1).isoformat()) if ref else None
        print("Ejecutando Modelo 1...")
        r1=m1.run(ci_dict=ci_enero, h=h, verbose=False)
        print("Ejecutando Modelo 2...")
        r2=m2.run(h=h, verbose=False)
        print_benchmark([r1, r2, r_exp])