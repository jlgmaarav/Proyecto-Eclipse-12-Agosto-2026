"""
CONSTANTES FÍSICAS Y ASTRONÓMICAS
Base para todos los modelos del Eclipse 2026.
"""
import numpy as np

# ─── CONSTANTES FUNDAMENTALES ────────────────────────────────────────────────
AU  = 1.495978707e11   # m
DAY = 86400.0          # s
G   = 6.6743e-11 * DAY**2 / AU**3   # AU³ / (kg·día²)

# ─── CUERPOS Y MASAS ─────────────────────────────────────────────────────────
NOMBRES = ['Sol', 'Tierra', 'Luna', 'Mercurio', 'Venus', 'Marte',
           'Jupiter', 'Saturno', 'Urano', 'Neptuno']

MASAS = np.array([
    1.98892e30,  # Sol
    5.97219e24,  # Tierra
    7.34581e22,  # Luna
    3.30200e23,  # Mercurio
    4.86850e24,  # Venus
    6.41712e23,  # Marte
    1.89813e27,  # Jupiter
    5.68319e26,  # Saturno
    8.68103e25,  # Urano
    1.02410e26,  # Neptuno
])
N = len(MASAS)
GM = G * MASAS  # AU³/día²

# Índices base
I_SOL = 0
I_TIE = 1
I_LUN = 2

# Masas específicas (legacy variable names para compatibilidad)
mS = MASAS[I_SOL]
mE = MASAS[I_TIE]
mL = MASAS[I_LUN]
GmS = GM[I_SOL]
GmE = GM[I_TIE]
GmL = GM[I_LUN]

# ─── MOMENTOS DE INERCIA (Tierra) ────────────────────────────────────────────
_m2AU2 = 1.0 / AU**2
I_perp = 8.0096e37 * _m2AU2
I_z    = 8.0358e37 * _m2AU2
dI     = I_z - I_perp

# ─── CONDICIONES INICIALES DE ROTACIÓN TERRESTRE ──────────────────────────────
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

# ─── COEFICIENTES INTEGRADOR YOSHIDA O(4) ────────────────────────────────────
_w1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
_w0 = -2.0**(1.0/3.0) * _w1
C_YOSHIDA = np.array([_w1/2.0, (_w0+_w1)/2.0, (_w0+_w1)/2.0, _w1/2.0])
D_YOSHIDA = np.array([_w1, _w0, _w1])
