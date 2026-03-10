"""
═══════════════════════════════════════════════════════════════════════════════
MODELO 1 — Hamiltoniano Sol-Tierra-Luna
Eclipse Solar Total · 12 agosto 2026
═══════════════════════════════════════════════════════════════════════════════

Implementación directa de las ecuaciones del diario (páginas 1-8 del PDF).

El Hamiltoniano completo es:

  H = T_tras(S) + T_tras(⊕) + T_tras(L) + H_rot(⊕) + V_SL + V_⊕L + V_⊕S

donde:
  T_tras(i)  = (px²+py²+pz²) / 2mᵢ          (energía cinética traslacional)

  H_rot(⊕)   = pθ²/(2I⊥)
              + (pϕ − pψ cosθ)² / (2I⊥ sin²θ)
              + pψ² / (2Iz)                   (rotación Tierra, ángulos Euler)

  V_SL       = −G·mS·mL / rSL                (potencial Sol-Luna, puntual)

  V_⊕L       = −G·mE·mL / rEL
              − G·mL·(Iz−I⊥) / (2·rEL⁵) · (3·uL² − rEL²)
                                              (Luna-Tierra, con J2 terrestre)

  V_⊕S       = −G·mE·mS / rES
              − G·mS·(Iz−I⊥) / (2·rES⁵) · (3·uS² − rES²)
                                              (Sol-Tierra, con J2 terrestre)

Variables auxiliares (pág. 6 del PDF):
  xEL, yEL, zEL = xL−xE, yL−yE, zL−zE        (vector Tierra→Luna)
  xES, yES, zES = xS−xE, yS−yE, zS−zE        (vector Tierra→Sol)
  xSL, ySL, zSL = xL−xS, yL−yS, zL−zS          (vector Sol→Luna)

  uL = xEL·sinθ·sinϕ − yEL·sinθ·cosϕ + zEL·cosθ   (proyección eje polar)
  uS = xES·sinθ·sinϕ − yES·sinθ·cosϕ + zES·cosθ

  CL = 3·G·mL·(Iz−I⊥) / (2·rEL⁵)
  CS = 3·G·mS·(Iz−I⊥) / (2·rES⁵)

Ecuaciones canónicas implementadas (18 + 4 = 22 variables):
  Traslación Sol    : ẋS, ẏS, żS, ṗxS, ṗyS, ṗzS   (6 variables)
  Traslación Tierra : ẋ⊕, ẏ⊕, ż⊕, ṗxE, ṗyE, ṗzE   (6 variables)
  Traslación Luna   : ẋL, ẏL, żL, ṗxL, ṗyL, ṗzL   (6 variables)
  Rotación Tierra   : θ̇, ϕ̇, ψ̇, ṗθ, ṗϕ             (4 variables, ṗψ=0)

Estado del sistema: vector de 22 variables
  [xS, yS, zS, pxS, pyS, pzS,
   xE, yE, zE, pxE, pyE, pzE,
   xL, yL, zL, pxL, pyL, pzL,
   θ, ϕ, ψ, pθ, pϕ, pψ]         ← índices 0-23 (24 en total, ψ no evoluciona)
"""

import numpy as np
import time
import sys
import os
from datetime import datetime, timedelta

# Para importar benchmark.py desde el mismo directorio
sys.path.insert(0, os.path.dirname(__file__))
from benchmark import eclipse_error, print_benchmark, load_reference_data, ECLIPSE_UTC

# ─── CONSTANTES FÍSICAS ───────────────────────────────────────────────────────
AU  = 1.495978707e11   # m
DAY = 86400.0          # s
G   = 6.6743e-11 * DAY**2 / AU**3   # AU³ / (kg·día²)

# Masas (kg)
mS = 1.98892e30
mE = 5.97219e24
mL = 7.34581e22

# Momentos de inercia de la Tierra (kg·m², convertidos a kg·AU²)
# Tierra real: geoide achatado
# I⊥ = 8.0096e37 kg·m²  (momento ecuatorial)
# Iz = 8.0358e37 kg·m²  (momento polar, mayor por achatamiento)
_m2_to_AU2 = 1.0 / AU**2
I_perp = 8.0096e37 * _m2_to_AU2   # kg·AU²
I_z    = 8.0358e37 * _m2_to_AU2   # kg·AU²
dI     = I_z - I_perp              # Iz - I⊥ > 0

# GM precalculados en AU³/día²
GmS = G * mS
GmE = G * mE
GmL = G * mL

# ─── CONDICIONES INICIALES DE ROTACIÓN TERRESTRE ─────────────────────────────
# La Tierra rota una vez cada ~0.9973 días sidéreos.
# Condición inicial: eje polar apuntando hacia el polo norte eclíptico (aprox).
# En J2000 eclíptico, el eje terrestre tiene oblicuidad ε = 23.44°.
# θ = oblicuidad respecto al polo eclíptico = 23.44°
# ψ̇ = velocidad de rotación propia = 2π / 0.9973 rad/día ≈ 6.3003 rad/día
# ϕ̇ = precesión lunisolar ≈ −2π / (25772 años) → despreciable en 7 meses

eps       = np.radians(23.4393)     # oblicuidad J2000
theta_0   = eps                     # θ inicial
phi_0     = 0.0                     # ϕ inicial (longitud del nodo ascendente)
psi_0     = 0.0                     # ψ inicial (ángulo de rotación propia)

omega_rot = 2 * np.pi / 0.9972697   # rad/día (período sidéreo)
phi_dot_0 = 0.0                     # precesión despreciable
psi_dot_0 = omega_rot - phi_dot_0 * np.cos(theta_0)  # ψ̇ + ϕ̇cosθ = Ω_rot

# Momentos conjugados iniciales de la rotación
ptheta_0 = I_perp * 0.0             # θ̇ = 0 (eje polar estable)
ppsi_0   = I_z * psi_dot_0          # pψ = Iz·(ψ̇ + ϕ̇cosθ)
pphi_0   = I_perp * phi_dot_0 * np.sin(theta_0)**2 + ppsi_0 * np.cos(theta_0)

# ─── ÍNDICES EN EL VECTOR DE ESTADO ──────────────────────────────────────────
# Sol
IxS=0;  IyS=1;  IzS=2;  IpxS=3; IpyS=4; IpzS=5
# Tierra
IxE=6;  IyE=7;  IzE=8;  IpxE=9; IpyE=10; IpzE=11
# Luna
IxL=12; IyL=13; IzL=14; IpxL=15; IpyL=16; IpzL=17
# Rotación Tierra
Ith=18; Iph=19; Ips=20; Ipth=21; Ipph=22; Ipps=23

N_STATE = 24

# ─── DERIVADAS DEL HAMILTONIANO ───────────────────────────────────────────────

def derivadas(state):
    """
    Ecuaciones canónicas de Hamilton:  q̇ = ∂H/∂p,  ṗ = −∂H/∂q
    Implementación directa de las ecuaciones de las páginas 5-8 del PDF.

    Devuelve dstate/dt (vector de 24 componentes).
    """
    # ── Desempaquetar estado ──────────────────────────────────────────────────
    xS, yS, zS = state[IxS], state[IyS], state[IzS]
    pxS,pyS,pzS= state[IpxS],state[IpyS],state[IpzS]

    xE, yE, zE = state[IxE], state[IyE], state[IzE]
    pxE,pyE,pzE= state[IpxE],state[IpyE],state[IpzE]

    xL, yL, zL = state[IxL], state[IyL], state[IzL]
    pxL,pyL,pzL= state[IpxL],state[IpyL],state[IpzL]

    th, ph, ps  = state[Ith], state[Iph], state[Ips]
    pth, pph, pps = state[Ipth], state[Ipph], state[Ipps]

    # ── Variables auxiliares (pág. 6 del PDF) ────────────────────────────────
    # Vectores relativos
    x_EL = xL - xE;  y_EL = yL - yE;  z_EL = zL - zE   # Tierra→Luna
    x_ES = xS - xE;  y_ES = yS - yE;  z_ES = zS - zE   # Tierra→Sol
    x_SL = xL - xS;  y_SL = yL - yS;  z_SL = zL - zS   # Sol→Luna

    r_EL2 = x_EL**2 + y_EL**2 + z_EL**2
    r_ES2 = x_ES**2 + y_ES**2 + z_ES**2
    r_SL2 = x_SL**2 + y_SL**2 + z_SL**2

    r_EL  = np.sqrt(r_EL2)
    r_ES  = np.sqrt(r_ES2)
    r_SL  = np.sqrt(r_SL2)

    r_EL3 = r_EL2 * r_EL
    r_ES3 = r_ES2 * r_ES
    r_SL3 = r_SL2 * r_SL

    r_EL5 = r_EL3 * r_EL2
    r_ES5 = r_ES3 * r_ES2

    # Funciones trigonométricas de los ángulos de Euler
    sin_th = np.sin(th);  cos_th = np.cos(th)
    sin_ph = np.sin(ph);  cos_ph = np.cos(ph)

    # Vector unitario del polo terrestre: û_p = (sinθ sinϕ, −sinθ cosϕ, cosθ)
    # (pág. 4 del PDF)
    up_x = sin_th * sin_ph
    up_y = -sin_th * cos_ph
    up_z = cos_th

    # Proyecciones del eje polar sobre los vectores Tierra→cuerpo
    # uL = xEL·sinθ·sinϕ − yEL·sinθ·cosϕ + zEL·cosθ  (pág. 6)
    uL = x_EL * up_x + y_EL * up_y + z_EL * up_z
    uS = x_ES * up_x + y_ES * up_y + z_ES * up_z

    # Derivadas de uL y uS respecto a θ y ϕ (pág. 6)
    uθL = x_EL * cos_th * sin_ph - y_EL * cos_th * cos_ph - z_EL * sin_th
    uϕL = x_EL * sin_th * cos_ph + y_EL * sin_th * sin_ph
    uθS = x_ES * cos_th * sin_ph - y_ES * cos_th * cos_ph - z_ES * sin_th
    uϕS = x_ES * sin_th * cos_ph + y_ES * sin_th * sin_ph

    # Constantes de acoplamiento multipolar (pág. 6)
    CL = 3.0 * GmL * dI / (2.0 * r_EL5)
    CS = 3.0 * GmS * dI / (2.0 * r_ES5)   # nota: en el potencial V_⊕S la masa
                                             # perturbadora es mS, no mE

    # Factores del término multipolar: (5u²/r² − 1)
    fL = 5.0 * uL**2 / r_EL2 - 1.0
    fS = 5.0 * uS**2 / r_ES2 - 1.0

    # ── Ángulo de Euler: cantidad auxiliar para la rotación ──────────────────
    sin2_th = sin_th**2
    # Proteger contra sin(θ) → 0 (singularidad de Euler en θ=0,π)
    if abs(sin_th) < 1e-10:
        sin_th_safe = 1e-10 * np.sign(sin_th + 1e-20)
    else:
        sin_th_safe = sin_th

    # (pϕ − pψ cosθ)
    pph_minus_pps_costh = pph - pps * cos_th

    # ─────────────────────────────────────────────────────────────────────────
    # ECUACIONES CINEMÁTICAS: q̇ = ∂H/∂p
    # (páginas 5 y 7 del PDF)
    # ─────────────────────────────────────────────────────────────────────────

    # Traslación Sol
    dx_S  = pxS / mS
    dy_S  = pyS / mS
    dz_S  = pzS / mS

    # Traslación Tierra
    dx_E  = pxE / mE
    dy_E  = pyE / mE
    dz_E  = pzE / mE

    # Traslación Luna
    dx_L  = pxL / mL
    dy_L  = pyL / mL
    dz_L  = pzL / mL

    # Rotación Tierra (pág. 5 y 3 del PDF)
    dth   = pth / I_perp
    dph   = pph_minus_pps_costh / (I_perp * sin_th_safe**2)
    dps   = pps / I_z - pph_minus_pps_costh * cos_th / (I_perp * sin_th_safe**2)

    # ─────────────────────────────────────────────────────────────────────────
    # ECUACIONES DINÁMICAS: ṗ = −∂H/∂q
    # (páginas 7-8 del PDF)
    # ─────────────────────────────────────────────────────────────────────────

    # ── Dinámica del Sol (pág. 7-8) ──────────────────────────────────────────
    # ṗxS = GmSmL/rSL³·xSL − GmEmS/rES³·xES − CS·[(5uS²/rES²−1)·xES − 2uS·sinθ·sinϕ]
    # Nota: en el PDF, xES = xS − xE = −x_ES (apunta de ⊕ a S, pero con signo)
    # Cuidado con los signos: la fuerza sobre S por ⊕ apunta de S hacia ⊕,
    # es decir, en la dirección −x_ES = x_ES·(−1). El PDF usa xES = xS−xE = −x_ES
    # y la fórmula tiene −GmEmS/r³·xES = +GmEmS/r³·x_ES (coherente: atracción)

    dpxS = ( GmL*mS/r_SL3 * x_SL
            - GmE*mS/r_ES3 * (-x_ES)
            - CS * (fS * (-x_ES) - 2.0 * uS * up_x) )

    dpyS = ( GmL*mS/r_SL3 * y_SL
            - GmE*mS/r_ES3 * (-y_ES)
            - CS * (fS * (-y_ES) + 2.0 * uS * sin_th * cos_ph) )

    dpzS = ( GmL*mS/r_SL3 * z_SL
            - GmE*mS/r_ES3 * (-z_ES)
            - CS * (fS * (-z_ES) - 2.0 * uS * cos_th) )

    # ── Dinámica de la Tierra (pág. 8) ───────────────────────────────────────
    dpxE = ( GmE*mL/r_EL3 * x_EL
            + CL * (fL * x_EL - 2.0 * uL * up_x)
            + GmE*mS/r_ES3 * x_ES
            + CS * (fS * x_ES - 2.0 * uS * up_x) )

    dpyE = ( GmE*mL/r_EL3 * y_EL
            + CL * (fL * y_EL + 2.0 * uL * sin_th * cos_ph)
            + GmE*mS/r_ES3 * y_ES
            + CS * (fS * y_ES + 2.0 * uS * sin_th * cos_ph) )

    dpzE = ( GmE*mL/r_EL3 * z_EL
            + CL * (fL * z_EL - 2.0 * uL * cos_th)
            + GmE*mS/r_ES3 * z_ES
            + CS * (fS * z_ES - 2.0 * uS * cos_th) )

    # ── Dinámica de la Luna (pág. 8) ─────────────────────────────────────────
    dpxL = ( -GmL*mS/r_SL3 * x_SL
             - GmE*mL/r_EL3 * x_EL
             - CL * (fL * x_EL - 2.0 * uL * up_x) )

    dpyL = ( -GmL*mS/r_SL3 * y_SL
             - GmE*mL/r_EL3 * y_EL
             - CL * (fL * y_EL + 2.0 * uL * sin_th * cos_ph) )

    dpzL = ( -GmL*mS/r_SL3 * z_SL
             - GmE*mL/r_EL3 * z_EL
             - CL * (fL * z_EL - 2.0 * uL * cos_th) )

    # ── Dinámica de la rotación terrestre (pág. 8) ───────────────────────────
    # ṗθ = (pϕ−pψcosθ)²·cosθ / (I⊥·sin³θ) − (pϕ−pψcosθ)·pψ / (I⊥·sinθ)
    #      + 2CL·uL·uθL + 2CS·uS·uθS
    dpth = ( pph_minus_pps_costh**2 * cos_th / (I_perp * sin_th_safe**3)
            - pph_minus_pps_costh * pps / (I_perp * sin_th_safe)
            + 2.0 * CL * uL * uθL
            + 2.0 * CS * uS * uθS )

    # ṗϕ = 2CL·uL·uϕL + 2CS·uS·uϕS
    dpph = 2.0 * CL * uL * uϕL + 2.0 * CS * uS * uϕS

    # ṗψ = 0  (pψ es constante de movimiento)
    dpps = 0.0

    return np.array([
        dx_S, dy_S, dz_S, dpxS, dpyS, dpzS,
        dx_E, dy_E, dz_E, dpxE, dpyE, dpzE,
        dx_L, dy_L, dz_L, dpxL, dpyL, dpzL,
        dth,  dph,  dps,  dpth, dpph, dpps
    ])


# ─── INTEGRADOR YOSHIDA O(4) SIMPLÉCTICO ─────────────────────────────────────
# Mismo integrador que v10. Coeficientes de Yoshida (1990).

_w1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
_w0 = -2.0**(1.0/3.0) * _w1
_C  = np.array([_w1/2, (_w0+_w1)/2, (_w0+_w1)/2, _w1/2])
_D  = np.array([_w1, _w0, _w1])

# Índices de posiciones y momentos en el vector de estado
_IDX_Q = np.array([IxS,IyS,IzS, IxE,IyE,IzE, IxL,IyL,IzL, Ith,Iph,Ips])
_IDX_P = np.array([IpxS,IpyS,IpzS, IpxE,IpyE,IpzE, IpxL,IpyL,IpzL, Ipth,Ipph,Ipps])

def yoshida4(state, h):
    """
    Un paso del integrador simpléctico de Yoshida orden 4.
    Opera sobre el vector de estado completo de 24 componentes.
    """
    s = state.copy()
    for k in range(3):
        # Paso de posición
        s[_IDX_Q] += _C[k] * h * derivadas(s)[_IDX_Q]
        # Recomputar derivadas con posiciones actualizadas para el paso de momento
        ds = derivadas(s)
        s[_IDX_P] += _D[k] * h * ds[_IDX_P]
    # Último medio paso de posición
    s[_IDX_Q] += _C[3] * h * derivadas(s)[_IDX_Q]
    return s


# ─── ENERGÍA DEL HAMILTONIANO (para diagnóstico) ─────────────────────────────

def energia(state):
    """Calcula H para verificar conservación durante la integración."""
    xS,yS,zS = state[IxS],state[IyS],state[IzS]
    pxS,pyS,pzS = state[IpxS],state[IpyS],state[IpzS]
    xE,yE,zE = state[IxE],state[IyE],state[IzE]
    pxE,pyE,pzE = state[IpxE],state[IpyE],state[IpzE]
    xL,yL,zL = state[IxL],state[IyL],state[IzL]
    pxL,pyL,pzL = state[IpxL],state[IpyL],state[IpzL]
    th,ph = state[Ith],state[Iph]
    pth,pph,pps = state[Ipth],state[Ipph],state[Ipps]

    # Cinética traslacional
    T_tras = ((pxS**2+pyS**2+pzS**2)/(2*mS) +
              (pxE**2+pyE**2+pzE**2)/(2*mE) +
              (pxL**2+pyL**2+pzL**2)/(2*mL))

    # Cinética rotacional
    sin_th = np.sin(th); cos_th = np.cos(th)
    sin_th_safe = max(abs(sin_th), 1e-10) * np.sign(sin_th + 1e-20)
    pph_m = pph - pps * cos_th
    T_rot = (pth**2/(2*I_perp) +
             pph_m**2/(2*I_perp*sin_th_safe**2) +
             pps**2/(2*I_z))

    # Potenciales
    x_EL=xL-xE; y_EL=yL-yE; z_EL=zL-zE
    x_ES=xS-xE; y_ES=yS-yE; z_ES=zS-zE
    x_SL=xL-xS; y_SL=yL-yS; z_SL=zL-zS

    r_EL=np.sqrt(x_EL**2+y_EL**2+z_EL**2)
    r_ES=np.sqrt(x_ES**2+y_ES**2+z_ES**2)
    r_SL=np.sqrt(x_SL**2+y_SL**2+z_SL**2)

    up_x = np.sin(th)*np.sin(ph)
    up_y = -np.sin(th)*np.cos(ph)
    up_z = np.cos(th)
    uL = x_EL*up_x + y_EL*up_y + z_EL*up_z
    uS = x_ES*up_x + y_ES*up_y + z_ES*up_z

    V_SL = -GmL*mS/r_SL
    V_EL = (-GmE*mL/r_EL
            - GmL*dI/(2*r_EL**3)*(3*uL**2/r_EL**2 - 1))
    V_ES = (-GmE*mS/r_ES
            - GmS*dI/(2*r_ES**3)*(3*uS**2/r_ES**2 - 1))

    return T_tras + T_rot + V_SL + V_EL + V_ES


# ─── CONSTRUCCIÓN DEL ESTADO INICIAL ─────────────────────────────────────────

def construir_estado_inicial(ci_dict):
    """
    Construye el vector de estado inicial de 24 componentes a partir de
    las condiciones iniciales del benchmark.

    ci_dict: dict con claves 'Sol', 'Tierra', 'Luna' y subcampos 'pos', 'vel'
    Los momentos son p = m·v (en unidades AU/día → p en kg·AU/día)
    """
    def pos_vel(nombre, masa):
        pos = np.array(ci_dict[nombre]["pos"])
        vel = np.array(ci_dict[nombre]["vel"])
        p   = masa * vel
        return pos, p

    rS, pS = pos_vel("Sol",    mS)
    rE, pE = pos_vel("Tierra", mE)
    rL, pL = pos_vel("Luna",   mL)

    state = np.zeros(N_STATE)
    state[IxS:IzS+1]   = rS
    state[IpxS:IpzS+1] = pS
    state[IxE:IzE+1]   = rE
    state[IpxE:IpzE+1] = pE
    state[IxL:IzL+1]   = rL
    state[IpxL:IpzL+1] = pL

    # Rotación terrestre
    state[Ith]  = theta_0
    state[Iph]  = phi_0
    state[Ips]  = psi_0
    state[Ipth] = ptheta_0
    state[Ipph] = pphi_0
    state[Ipps] = ppsi_0

    return state


# ─── BÚSQUEDA DEL ECLIPSE ────────────────────────────────────────────────────

def buscar_eclipse(state, t_dias, h, t_inicio):
    """
    Integra desde el estado actual en pasos de h días,
    buscando el mínimo de la separación angular Sol-Luna geocéntrica.
    Devuelve (estado_en_mínimo, t_eclipse_datetime, sep_min_grados).
    """
    sep_min = 999.0
    state_ecl = None
    t_ecl_dias = t_dias

    while True:
        vES = state[IxS:IzS+1] - state[IxE:IzE+1]
        vEL = state[IxL:IzL+1] - state[IxE:IzE+1]
        dES = np.linalg.norm(vES)
        dEL = np.linalg.norm(vEL)
        cos_ = np.clip(np.dot(vES, vEL) / (dES * dEL), -1, 1)
        sep  = np.degrees(np.arccos(cos_))

        if sep < sep_min:
            sep_min   = sep
            state_ecl = state.copy()
            t_ecl_dias = t_dias

        # Avanzar un paso
        state = yoshida4(state, h)
        t_dias += h

        # Parar si la separación aumenta consistentemente (ya pasó el mínimo)
        if sep_min < 1.0 and sep > sep_min + 0.1:
            break

        # Parar si hemos pasado el 20 de agosto (límite de seguridad)
        if t_dias > (datetime(2026, 8, 20) - t_inicio).days:
            break

    t_ecl = t_inicio + timedelta(days=float(t_ecl_dias))
    return state_ecl, t_ecl, sep_min


# ─── FUNCIÓN PRINCIPAL DE INTEGRACIÓN ────────────────────────────────────────

def run(ci_dict=None, h=0.005, verbose=True):
    """
    Integra el Modelo 1 desde el 1 de enero de 2026 hasta el eclipse.

    Parámetros
    ----------
    ci_dict : dict con condiciones iniciales de JPL (de benchmark.load_reference_data)
              Si None, usa placeholders del benchmark
    h       : paso de integración en días (por defecto 0.005 = 7.2 minutos)
    verbose : mostrar progreso

    Devuelve el dict estándar del benchmark.
    """
    from plantilla_modelo import CI as CI_DEFAULT

    if ci_dict is None:
        if verbose:
            print("AVISO: usando condiciones iniciales placeholder. "
                  "Ejecuta benchmark.py --fetch para datos reales de JPL.")
        ci_dict = CI_DEFAULT

    T_INICIO = datetime(2026, 1, 1)
    T_ECLIPSE_APROX = datetime(2026, 8, 12)
    DIAS_INTEGRACION = (T_ECLIPSE_APROX - T_INICIO).days   # ~223 días

    # ── Estado inicial ────────────────────────────────────────────────────────
    state = construir_estado_inicial(ci_dict)
    E0    = energia(state)

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  MODELO 1 — Hamiltoniano Sol-Tierra-Luna")
        print(f"{'═'*60}")
        print(f"  Paso h = {h} días = {h*DAY:.0f} s")
        print(f"  Integración: {DIAS_INTEGRACION} días ({DIAS_INTEGRACION/h:.0f} pasos)")
        print(f"  Energía inicial: {E0:.6e}")
        print(f"  Variables de estado: {N_STATE}")
        print(f"  (9 traslacionales × 3 cuerpos + 6 Euler Tierra)")
        print()

    # ── Integración principal ─────────────────────────────────────────────────
    t0_cpu = time.time()
    t_dias = 0.0
    N_pasos = int(DIAS_INTEGRACION / h)
    N_report = max(1, N_pasos // 10)

    E_max_error = 0.0

    for i in range(N_pasos):
        state = yoshida4(state, h)
        t_dias += h

        # Diagnóstico de energía cada 10%
        if verbose and (i % N_report == 0):
            E = energia(state)
            dE = abs((E - E0) / E0) if E0 != 0 else 0
            E_max_error = max(E_max_error, dE)
            fecha = T_INICIO + timedelta(days=float(t_dias))
            # Posición de la Luna respecto a la Tierra
            d_EL = np.linalg.norm(state[IxL:IzL+1] - state[IxE:IzE+1])
            print(f"  {fecha.strftime('%d %b %Y')}  |  ΔE/E={dE:.2e}  "
                  f"|  d_Luna={d_EL*AU/1e3:.0f} km")

    t1_cpu = time.time()

    if verbose:
        print(f"\n  Integración completada en {t1_cpu-t0_cpu:.1f}s")
        E_final = energia(state)
        dE_final = abs((E_final - E0) / E0) if E0 != 0 else 0
        print(f"  Error energía final: ΔE/E = {dE_final:.2e}")
        print(f"  Error energía máximo: ΔE/E = {E_max_error:.2e}")

    # ── Búsqueda del eclipse (paso fino alrededor del 12 agosto) ─────────────
    if verbose:
        print(f"\n  Buscando mínimo de separación Sol-Luna...")

    # En este punto estamos en t_dias ≈ 223 días = ~1 agosto
    # Integrar con paso fino (h/10) para encontrar el mínimo
    h_fino = h / 10.0
    state_ecl, t_ecl, sep_min = buscar_eclipse(state, t_dias, h_fino, T_INICIO)

    if verbose:
        print(f"  Máximo geocéntrico: {t_ecl.strftime('%d %b %Y  %H:%M UTC')}")
        print(f"  Separación Sol-Luna: {sep_min:.4f}°")

    # ── Métricas de error ─────────────────────────────────────────────────────
    pos_sol    = state_ecl[IxS:IzS+1]
    pos_tierra = state_ecl[IxE:IzE+1]
    pos_luna   = state_ecl[IxL:IzL+1]

    # Cargar referencia JPL si existe
    ref_datos  = load_reference_data()
    ref_eclipse = None
    if ref_datos:
        clave = ECLIPSE_UTC.isoformat()
        if clave in ref_datos:
            ref_eclipse = ref_datos[clave]

    metricas = eclipse_error(pos_sol, pos_tierra, pos_luna, t_ecl, ref_eclipse)

    if verbose:
        print(f"\n  ── Resultados ──────────────────────────────────────────")
        print(f"  d_Luna:         {metricas['d_luna_km']:,} km")
        print(f"  Eclipse total:  {'SÍ ✓' if metricas['es_total'] else 'NO ✗'}")
        print(f"  Magnitud fase:  {metricas['fase_magnitud']:.5f}")
        if metricas['error_pos_luna_km'] is not None:
            print(f"  Error posición: {metricas['error_pos_luna_km']:,} km")
        if metricas['error_ang_arcmin'] is not None:
            print(f"  Error angular:  {metricas['error_ang_arcmin']:.3f} arcmin")
        if metricas['error_tiempo_min'] is not None:
            print(f"  Error tiempo:   {metricas['error_tiempo_min']:.1f} minutos")
        print()

    return {
        "nombre":       "Modelo 1: H Sol-Tierra-Luna",
        "metricas":     metricas,
        "tiempo_cpu_s": t1_cpu - t0_cpu,
        "descripcion":  f"Hamiltoniano completo 3 cuerpos + J2 + Euler, h={h}d",
        # Extras para diagnóstico
        "_E0":          E0,
        "_E_max_error": E_max_error,
        "_t_ecl":       t_ecl,
        "_sep_min_deg": sep_min,
        "_state_ecl":   state_ecl,
    }


# ─── EJECUCIÓN DIRECTA ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Modelo 1 — Hamiltoniano Sol-Tierra-Luna")
    parser.add_argument("--h",       type=float, default=0.005,
                        help="Paso de integración en días (default: 0.005 ≈ 7 min)")
    parser.add_argument("--rapido",  action="store_true",
                        help="Paso h=0.05 para prueba rápida (~10× más rápido)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Mostrar tabla de benchmark al final")
    args = parser.parse_args()

    if args.rapido:
        h = 0.05
        print("Modo rápido: h=0.05 días (precisión reducida, solo para prueba)")
    else:
        h = args.h

    # Intentar cargar condiciones iniciales reales
    ref_datos = load_reference_data()
    ci_dict = None
    if ref_datos:
        clave_enero = datetime(2026, 1, 1).isoformat()
        if clave_enero in ref_datos:
            ci_dict = ref_datos[clave_enero]
            print("✓ Usando condiciones iniciales JPL DE441 del 1 enero 2026")

    resultado = run(ci_dict=ci_dict, h=h, verbose=True)

    if args.benchmark:
        print_benchmark([resultado])