"""
MODELO 2 — Hamiltoniano N-cuerpos completo
Eclipse Solar Total · 12 agosto 2026

Extiende Modelo 1 añadiendo los 7 planetas restantes como masas puntuales.
La rotación terrestre y el J2 se mantienen igual que en Modelo 1.

Cuerpos (índices):
  0=Sol  1=Tierra  2=Luna  3=Mercurio  4=Venus  5=Marte
  6=Jupiter  7=Saturno  8=Urano  9=Neptuno

Estado: 66 variables
  [r_Sol(3), p_Sol(3), r_Tie(3), p_Tie(3), ..., r_Nep(3), p_Nep(3),
   theta, phi, psi, ptheta, pphi, ppsi]

Unidades: AU, kg, día
  - Posición: AU
  - Momento:  kg·AU/día
  - Energía:  kg·AU²/día²

La clave de las unidades:
  GM[i] = G·mᵢ  [AU³/día²]
  Fuerza sobre i por j: Fᵢⱼ = G·mᵢ·mⱼ/r² · r̂ = mᵢ·GM[j]/r² · r̂
  → dp[i]/dt = mᵢ · GM[j]/r³ · dr  (en kg·AU/día²)
"""

import numpy as np
import time
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from benchmark import eclipse_error, print_benchmark, load_reference_data, ECLIPSE_UTC

# ─── CONSTANTES (importadas) ─────────────────────────────────────────────────
from constantes import (AU, DAY, G, NOMBRES, MASAS, N, GM, 
                        I_SOL, I_TIE, I_LUN, I_perp, I_z, dI,
                        theta_0, phi_0, psi_0, ptheta_0, pphi_0, ppsi_0,
                        C_YOSHIDA, D_YOSHIDA)

# ─── LAYOUT DEL VECTOR DE ESTADO ──────────────────────────────────────────────
# Cuerpo i: posición en [6i:6i+3], momento en [6i+3:6i+6]
def ir(i): return 6*i
def ip(i): return 6*i+3

N_TRAS  = 6*N      # 60
N_STATE = N_TRAS+6 # 66

ITH=60; IPH=61; IPS=62; IPTH=63; IPPH=64; IPPS=65

# ─── CONDICIONES INICIALES DE ROTACIÓN ────────────────────────────────────────
# Importadas de constantes.py

# ─── DERIVADAS ────────────────────────────────────────────────────────────────
def derivadas(s):
    """
    dstate/dt: vector de 66 componentes.

    Cinemática:
      ṙᵢ = pᵢ/mᵢ
      θ̇  = pθ/I⊥
      φ̇  = (pφ−pψcosθ)/(I⊥sin²θ)
      ψ̇  = pψ/Iz − (pφ−pψcosθ)cosθ/(I⊥sin²θ)

    Dinámica traslacional (ṗᵢ = Fᵢ en kg·AU/día²):
      Fᵢ = Σⱼ≠ᵢ mᵢ·GM[j]/rᵢⱼ³ · (rⱼ−rᵢ)  +  corrección J2 (solo Tierra↔Luna,Sol)

    Dinámica rotacional (idéntica a Modelo 1):
      ṗθ = término centrífugo + 2CL·uL·uθL + 2CS·uS·uθS
      ṗφ = 2CL·uL·uφL + 2CS·uS·uφS
      ṗψ = 0
    """
    ds = np.zeros(N_STATE)

    # Extraer posiciones y momentos
    r = np.array([s[ir(i):ir(i)+3] for i in range(N)])
    p = np.array([s[ip(i):ip(i)+3] for i in range(N)])

    th  = s[ITH]; ph  = s[IPH]
    pth = s[IPTH]; pph = s[IPPH]; pps = s[IPPS]

    sin_th = np.sin(th); cos_th = np.cos(th)
    sin_ph = np.sin(ph); cos_ph = np.cos(ph)
    sin_th_s = sin_th if abs(sin_th)>1e-10 else np.sign(sin_th+1e-20)*1e-10
    pph_m = pph - pps*cos_th
    up = np.array([sin_th*sin_ph, -sin_th*cos_ph, cos_th])

    # ── Cinemática traslacional ───────────────────────────────────────────────
    for i in range(N):
        ds[ir(i):ir(i)+3] = p[i]/MASAS[i]

    # ── Cinemática rotacional ─────────────────────────────────────────────────
    ds[ITH] = pth/I_perp
    ds[IPH] = pph_m/(I_perp*sin_th_s**2)
    ds[IPS] = pps/I_z - pph_m*cos_th/(I_perp*sin_th_s**2)

    # ── Dinámica traslacional: gravedad newtoniana ────────────────────────────
    # ṗᵢ = Fᵢ = mᵢ · Σⱼ GM[j]/r³ · dr  (kg·AU/día²)
    for i in range(N):
        for j in range(i+1, N):
            dr  = r[j] - r[i]
            d2  = np.dot(dr,dr)
            d3  = d2*np.sqrt(d2)
            f   = dr/d3
            ds[ip(i):ip(i)+3] += MASAS[i]*GM[j]*f
            ds[ip(j):ip(j)+3] -= MASAS[j]*GM[i]*f

    # ── J2 terrestre (pares Tierra-Luna y Tierra-Sol) ─────────────────────────
    # Usamos exactamente la misma convención que Modelo 1:
    # g = C*(fJ2*dr - 2*u*up) se aplica directamente a dp (sin multiplicar masa)
    # C = 3*GM[ext]*dI/(2*r^5) ya lleva las unidades correctas del Modelo 1.
    for ext in [I_LUN, I_SOL]:
        dr  = r[ext] - r[I_TIE]
        d2  = np.dot(dr,dr)
        d   = np.sqrt(d2)
        d5  = d2*d2*d
        u   = np.dot(dr, up)
        C   = 3.0*GM[ext]*dI/(2.0*d5)
        fJ2 = 5.0*u**2/d2 - 1.0
        g   = C*(fJ2*dr - 2.0*u*up)
        ds[ip(ext):ip(ext)+3]      -= g
        ds[ip(I_TIE):ip(I_TIE)+3] += g

    # ── Dinámica rotacional Tierra ────────────────────────────────────────────
    r_EL = r[I_LUN]-r[I_TIE]; r_ES = r[I_SOL]-r[I_TIE]
    d_EL2=np.dot(r_EL,r_EL); d_EL=np.sqrt(d_EL2); d_EL5=d_EL2**2*d_EL
    d_ES2=np.dot(r_ES,r_ES); d_ES=np.sqrt(d_ES2); d_ES5=d_ES2**2*d_ES
    uL=np.dot(r_EL,up); uS=np.dot(r_ES,up)
    uTL=r_EL[0]*cos_th*sin_ph-r_EL[1]*cos_th*cos_ph-r_EL[2]*sin_th
    uFL=r_EL[0]*sin_th*cos_ph+r_EL[1]*sin_th*sin_ph
    uTS=r_ES[0]*cos_th*sin_ph-r_ES[1]*cos_th*cos_ph-r_ES[2]*sin_th
    uFS=r_ES[0]*sin_th*cos_ph+r_ES[1]*sin_th*sin_ph
    CL=3.0*GM[I_LUN]*dI/(2.0*d_EL5)
    CS=3.0*GM[I_SOL]*dI/(2.0*d_ES5)

    ds[IPTH]=(pph_m**2*cos_th/(I_perp*sin_th_s**3)
              -pph_m*pps/(I_perp*sin_th_s)
              +2.0*CL*uL*uTL+2.0*CS*uS*uTS)
    ds[IPPH]=2.0*CL*uL*uFL+2.0*CS*uS*uFS
    ds[IPPS]=0.0

    return ds


# ─── INTEGRADOR YOSHIDA O(4) ──────────────────────────────────────────────────
_C = C_YOSHIDA; _D = D_YOSHIDA

_IQ = np.array([ir(i)+k for i in range(N) for k in range(3)]+[ITH,IPH,IPS])
_IP = np.array([ip(i)+k for i in range(N) for k in range(3)]+[IPTH,IPPH,IPPS])

def yoshida4(s, h):
    s = s.copy()
    for k in range(3):
        ds = derivadas(s)
        s[_IQ] += _C[k]*h*ds[_IQ]
        ds = derivadas(s)
        s[_IP] += _D[k]*h*ds[_IP]
    ds = derivadas(s)
    s[_IQ] += _C[3]*h*ds[_IQ]
    return s


# ─── ENERGÍA (diagnóstico) ────────────────────────────────────────────────────
def energia(s):
    r = np.array([s[ir(i):ir(i)+3] for i in range(N)])
    p = np.array([s[ip(i):ip(i)+3] for i in range(N)])
    th=s[ITH]; ph=s[IPH]; pth=s[IPTH]; pph=s[IPPH]; pps=s[IPPS]
    sin_th=np.sin(th); cos_th=np.cos(th)
    sin_ph=np.sin(ph); cos_ph=np.cos(ph)
    sin_s=max(abs(sin_th),1e-10)*np.sign(sin_th+1e-20)
    pph_m=pph-pps*cos_th
    up=np.array([sin_th*sin_ph,-sin_th*cos_ph,cos_th])

    # Cinética traslacional: T = p²/(2m)
    T = sum(np.dot(p[i],p[i])/(2*MASAS[i]) for i in range(N))
    # Cinética rotacional
    T += pth**2/(2*I_perp)+pph_m**2/(2*I_perp*sin_s**2)+pps**2/(2*I_z)

    # Potencial newtoniano: V = -G·mᵢ·mⱼ/r = -GM[i]·MASAS[j]/r
    V = 0.0
    for i in range(N):
        for j in range(i+1,N):
            d=np.linalg.norm(r[j]-r[i])
            V -= GM[i]*MASAS[j]/d

    # Potencial J2
    for ext in [I_LUN,I_SOL]:
        dr=r[ext]-r[I_TIE]; d=np.linalg.norm(dr)
        u=np.dot(dr,up)
        V -= GM[ext]*dI/(2*d**3)*(3*u**2/d**2-1)

    return T+V


# ─── ESTADO INICIAL ───────────────────────────────────────────────────────────
NOMBRE_IDX = {'Sol':0,'Tierra':1,'Luna':2,'Mercurio':3,'Venus':4,
              'Marte':5,'Jupiter':6,'Saturno':7,'Urano':8,'Neptuno':9}

def construir_estado(ci_dict):
    s = np.zeros(N_STATE)
    for nombre, idx in NOMBRE_IDX.items():
        if nombre not in ci_dict:
            print(f"  AVISO: {nombre} no en CI")
            continue
        pos = np.array(ci_dict[nombre]['pos'])
        vel = np.array(ci_dict[nombre]['vel'])
        s[ir(idx):ir(idx)+3] = pos
        s[ip(idx):ip(idx)+3] = MASAS[idx]*vel
    s[ITH]=theta_0; s[IPH]=phi_0; s[IPS]=psi_0
    s[IPTH]=ptheta_0; s[IPPH]=pphi_0; s[IPPS]=ppsi_0
    return s


# ─── BÚSQUEDA DEL ECLIPSE ─────────────────────────────────────────────────────
def buscar_eclipse(s, t_dias, h, t0):
    sep_min=999.0; s_ecl=None; t_ecl=t_dias
    while True:
        vES=s[ir(I_SOL):ir(I_SOL)+3]-s[ir(I_TIE):ir(I_TIE)+3]
        vEL=s[ir(I_LUN):ir(I_LUN)+3]-s[ir(I_TIE):ir(I_TIE)+3]
        c=np.clip(np.dot(vES,vEL)/(np.linalg.norm(vES)*np.linalg.norm(vEL)),-1,1)
        sep=np.degrees(np.arccos(c))
        if sep<sep_min:
            sep_min=sep; s_ecl=s.copy(); t_ecl=t_dias
        s=yoshida4(s,h); t_dias+=h
        if sep_min<1.0 and sep>sep_min+0.1: break
        if t_dias>(datetime(2026,8,20)-t0).days: break
    return s_ecl, t0+timedelta(days=float(t_ecl)), sep_min


# ─── FUNCIÓN PRINCIPAL ────────────────────────────────────────────────────────
def run(h=0.005, verbose=True):
    T0=datetime(2026,1,1); DIAS=223

    ref=load_reference_data()
    ci=ref.get(T0.isoformat()) if ref else None
    if ci is None:
        print("ERROR: ejecuta benchmark.py --fetch primero"); return None

    s=construir_estado(ci)
    E0=energia(s)

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  MODELO 2 — Hamiltoniano N-cuerpos (10 cuerpos + J2 + Euler)")
        print(f"{'═'*60}")
        print(f"  h={h} días={h*DAY:.0f}s  |  {DIAS} días  |  {int(DIAS/h):,} pasos")
        print(f"  E₀={E0:.6e}  |  {N_STATE} variables\n")

    t0c=time.time(); t=0.0; N_p=int(DIAS/h); Nr=max(1,N_p//10); Em=0.0

    for i in range(N_p):
        s=yoshida4(s,h); t+=h
        if verbose and i%Nr==0:
            E=energia(s); dE=abs((E-E0)/E0) if E0 else 0; Em=max(Em,dE)
            d=np.linalg.norm(s[ir(I_LUN):ir(I_LUN)+3]-s[ir(I_TIE):ir(I_TIE)+3])*AU/1e3
            fecha=T0+timedelta(days=float(t))
            print(f"  {fecha.strftime('%d %b %Y')}  |  ΔE/E={dE:.2e}  |  d_Luna={d:.0f} km")

    t1c=time.time()
    if verbose:
        Ef=energia(s); print(f"\n  CPU={t1c-t0c:.1f}s  |  ΔE/E_fin={abs((Ef-E0)/E0):.2e}  |  ΔE/E_max={Em:.2e}")
        print(f"  Buscando eclipse...")

    s_ecl,t_ecl,sep=buscar_eclipse(s,t,h/10,T0)
    if verbose:
        print(f"  Máximo: {t_ecl.strftime('%d %b %Y  %H:%M UTC')}  θ={sep:.4f}°")

    ps=s_ecl[ir(I_SOL):ir(I_SOL)+3]
    pt=s_ecl[ir(I_TIE):ir(I_TIE)+3]
    pl=s_ecl[ir(I_LUN):ir(I_LUN)+3]

    ref_ecl=None
    if ref:
        ref_ecl=ref.get(ECLIPSE_UTC.isoformat())

    m=eclipse_error(ps,pt,pl,t_ecl,ref_ecl)

    if verbose:
        print(f"\n  ── Resultados ──")
        print(f"  d_Luna={m['d_luna_km']:,} km  |  Total={'SÍ ✓' if m['es_total'] else 'NO ✗'}  |  fase={m['fase_magnitud']:.5f}")
        if m['error_pos_luna_km'] is not None: print(f"  Err_pos={m['error_pos_luna_km']:,} km")
        if m['error_ang_arcmin']  is not None: print(f"  Err_ang={m['error_ang_arcmin']:.3f} arcmin")
        if m['error_tiempo_min']  is not None: print(f"  Err_t  ={m['error_tiempo_min']:.1f} min\n")

    return {"nombre":"Modelo 2: H N-cuerpos","metricas":m,
            "tiempo_cpu_s":t1c-t0c,"descripcion":f"10 cuerpos + J2 + Euler, h={h}d"}


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--h",type=float,default=0.005)
    p.add_argument("--rapido",action="store_true")
    p.add_argument("--benchmark",action="store_true")
    args=p.parse_args()

    h=0.05 if args.rapido else args.h
    if args.rapido: print("Modo rápido: h=0.05 días")

    r2=run(h=h,verbose=True)

    if args.benchmark and r2:
        import modelo1_3cuerpos as m1
        print("Ejecutando Modelo 1 para comparar...")
        ref = load_reference_data()
        ci_enero = ref.get(datetime(2026,1,1).isoformat()) if ref else None
        r1 = m1.run(ci_dict=ci_enero, h=h, verbose=False)
        print_benchmark([r1, r2])