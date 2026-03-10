"""
Sistema Sol-Tierra-Luna con Rotación Terrestre — RESPA
======================================================

Arquitectura RESPA (Reference System Propagator Algorithm):

    H = H_fast + H_slow

    H_fast = H_tras(q_tras, p_tras) + H_rot_libre(angles, p_angles)
           = [p²/2m + V_grav(q)] + [p_θ²/2I⊥ + A²/(2I⊥sin²θ) + p_ψ²/2Iz]

    H_slow = V_multi(q_tras, θ, φ)   (acoplamiento orbital-rotacional)

    V_multi/V_grav ~ 5e-13  →  kick lento actualizable cada N pasos rápidos.

Integración:
    · H_tras  →  SV estándar con h_orb = 0.5 días
    · H_rot   →  SV estándar con h_rot = 60 s  (72×144 substeps/paso orbital)
    · V_multi →  kick cada paso orbital (es despreciable a corto plazo)
    · Composición Yoshida Y4 aplicada a cada bloque por separado.

Esto es exactamente simpléctico porque cada sub-integrador lo es y la
composición RESPA preserva la forma simpléctica.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

G      = 6.67430e-11
mS     = 1.98892e30
mE     = 5.97219e24
mL     = 7.34200e22
AU     = 1.49597870700e11
DIA    = 86400.0
GM_S   = 1.32712440018e20
I_perp = 8.0080e37
I_z    = 8.0340e37
dI     = I_z - I_perp

inv_m_tras = np.array([1/mS,1/mS,1/mS, 1/mE,1/mE,1/mE, 1/mL,1/mL,1/mL])
m_tras     = np.array([mS,mS,mS, mE,mE,mE, mL,mL,mL])

# ══════════════════════════════════════════════════════════════════════════════
# 2. CONDICIONES INICIALES
# ══════════════════════════════════════════════════════════════════════════════

a_E = AU; e_E = 0.0167
rE0 = np.array([a_E*(1-e_E), 0.0, 0.0])
vE0 = np.array([0.0, np.sqrt(GM_S/a_E*(1+e_E)/(1-e_E)), 0.0])

a_L = 3.844e8; e_L = 0.0549; i_L = np.radians(5.145)
rL0 = rE0 + np.array([a_L*(1-e_L), 0.0, 0.0])
vL0 = vE0 + np.array([0.0,
                       np.sqrt(G*mE/a_L*(1+e_L)/(1-e_L))*np.cos(i_L),
                       np.sqrt(G*mE/a_L*(1+e_L)/(1-e_L))*np.sin(i_L)])

rS0 = -(mE*rE0 + mL*rL0)/mS
vS0 = -(mE*vE0 + mL*vL0)/mS

# Ángulos de Euler
theta0     = np.radians(23.4393)
phi0       = 0.0
psi0       = 0.0
omega_spin = 2*np.pi/86164.1      # rad/s

p_psi0   = I_z * omega_spin
p_phi0   = I_z * omega_spin * np.cos(theta0)
p_theta0 = 0.0

# Estado completo: q=(xS,yS,zS,xE,yE,zE,xL,yL,zL,θ,φ,ψ)  p=(pS,pE,pL,pθ,pφ,pψ)
q0 = np.array([*rS0, *rE0, *rL0, theta0, phi0, psi0])
p0 = np.array([*(mS*vS0), *(mE*vE0), *(mL*vL0), p_theta0, p_phi0, p_psi0])

# ══════════════════════════════════════════════════════════════════════════════
# 3. GRADIENTES DE CADA PARTE DEL HAMILTONIANO
# ══════════════════════════════════════════════════════════════════════════════

# ── 3a. H_tras: gravitación puntual ─────────────────────────────────────────

def grad_V_grav(q_tras: np.ndarray) -> np.ndarray:
    """
    Gradiente de V_grav = -GmSmL/r_SL - GmEmL/r_EL - GmEmS/r_ES
    respecto a q_tras = (rS, rE, rL).
    Retorna dV/dq_tras con shape (9,).
    """
    rS = q_tras[0:3]; rE = q_tras[3:6]; rL = q_tras[6:9]
    x_SL = rL-rS; r_SL = np.linalg.norm(x_SL)
    x_EL = rL-rE; r_EL = np.linalg.norm(x_EL)
    x_ES = rS-rE; r_ES = np.linalg.norm(x_ES)

    fSL = G*mS*mL/r_SL**3
    fEL = G*mE*mL/r_EL**3
    fES = G*mE*mS/r_ES**3

    dV_S = -fSL*x_SL + fES*x_ES
    dV_E =  fEL*x_EL - fES*x_ES
    dV_L =  fSL*x_SL + fEL*x_EL
    return np.concatenate([dV_S, dV_E, dV_L])


# ── 3b. H_rot libre: rotor asimétrico ───────────────────────────────────────
# H_rot = p_θ²/(2I⊥) + A²/(2I⊥ sin²θ) + p_ψ²/(2Iz)
# donde A = p_φ - p_ψ cosθ
#
# Ecuaciones canónicas:
#   θ̇  = p_θ/I⊥
#   φ̇  = A/(I⊥ sin²θ)
#   ψ̇  = p_ψ/Iz - A cosθ/(I⊥ sin²θ)
#   ṗ_θ = A²cosθ/(I⊥ sin³θ) - A p_ψ/(I⊥ sinθ)
#   ṗ_φ = 0   (φ es cíclica en H_rot)
#   ṗ_ψ = 0   (ψ es cíclica en H_rot)

def sv_rot_step(angles: np.ndarray, p_angles: np.ndarray, h: float):
    """
    SV para H_rot libre. Splitting: T_θ + T_φψ
    T_θ  = p_θ²/(2I⊥)                          → mueve θ, actualiza p_θ
    T_φψ = A²/(2I⊥sin²θ) + p_ψ²/(2Iz)          → mueve φ,ψ  (p_φ,p_ψ ctes)

    Dado que p_φ y p_ψ son constantes de movimiento de H_rot,
    el SV se reduce a un integrador de θ con fuerza conservativa efectiva.
    """
    theta, phi, psi  = angles
    p_theta, p_phi, p_psi = p_angles

    def torque_theta(th, pt, pph, pps):
        """ṗ_θ = -dH_rot/dθ"""
        st = np.sin(th); ct = np.cos(th)
        A  = pph - pps*ct
        return A**2*ct/(I_perp*st**3) - A*pps/(I_perp*st)

    def vel_angles(th, pt, pph, pps):
        """(θ̇, φ̇, ψ̇) = dH_rot/dp"""
        st = np.sin(th); ct = np.cos(th)
        A  = pph - pps*ct
        s2 = st**2
        return (pt/I_perp,
                A/(I_perp*s2),
                pps/I_z - A*ct/(I_perp*s2))

    # Paso SV: kick p_θ(h/2), drift ángulos(h), kick p_θ(h/2)
    # p_φ y p_ψ son constantes → no cambian
    p_theta += (h/2)*torque_theta(theta, p_theta, p_phi, p_psi)

    vt, vp, vs = vel_angles(theta, p_theta, p_phi, p_psi)
    theta += h*vt
    phi   += h*vp
    psi   += h*vs

    p_theta += (h/2)*torque_theta(theta, p_theta, p_phi, p_psi)

    return np.array([theta, phi, psi]), np.array([p_theta, p_phi, p_psi])


_c = 2.0**(1.0/3.0)
W4: list[float] = [1/(2-_c), -_c/(2-_c), 1/(2-_c)]

def yoshida4_rot(angles: np.ndarray, p_ang: np.ndarray, h: float):
    for w in W4:
        angles, p_ang = sv_rot_step(angles, p_ang, w*h)
    return angles, p_ang


# ── 3c. V_multi: kick de acoplamiento ───────────────────────────────────────
# V_multi depende de (q_tras, θ, φ) → actualiza p_tras, p_θ, p_φ

def kick_vmulti(q_tras: np.ndarray, angles: np.ndarray,
                p_tras: np.ndarray, p_ang: np.ndarray, h: float):
    """
    Actualiza p_tras y p_ang con el potencial multipolar.
    V_multi = -GmL dI/(2r_EL³)(3cos²α_L-1) - GmS dI/(2r_ES³)(3cos²α_S-1)
    """
    rS = q_tras[0:3]; rE = q_tras[3:6]; rL = q_tras[6:9]
    theta, phi = angles[0], angles[1]
    st = np.sin(theta); ct = np.cos(theta)
    sp = np.sin(phi);   cp = np.cos(phi)

    u_hat     = np.array([st*sp, -st*cp, ct])
    du_dtheta = np.array([ct*sp, -ct*cp, -st])
    du_dphi   = np.array([st*cp,  st*sp,  0.0])

    x_EL = rL-rE; r_EL = np.linalg.norm(x_EL)
    x_ES = rS-rE; r_ES = np.linalg.norm(x_ES)

    uL = float(np.dot(x_EL, u_hat))
    uS = float(np.dot(x_ES, u_hat))

    C_L = 3*G*mL*dI/(2*r_EL**5)
    C_S = 3*G*mS*dI/(2*r_ES**5)

    # ── Kicks a p_tras ──
    # dV/dr_L = C_L*[(5uL²/r_EL²-1)*x_EL - 2uL*u_hat]
    dV_rL = C_L*((5*uL**2/r_EL**2-1)*x_EL - 2*uL*u_hat)
    dV_rS = C_S*((5*uS**2/r_ES**2-1)*x_ES - 2*uS*u_hat)
    dV_rE = -dV_rL - dV_rS

    p_tras_new = p_tras - h*np.concatenate([dV_rS, dV_rE, dV_rL])

    # ── Kicks a p_ang ──
    dV_dtheta = -(2*C_L*uL*float(np.dot(x_EL,du_dtheta))
                + 2*C_S*uS*float(np.dot(x_ES,du_dtheta)))
    dV_dphi   = -(2*C_L*uL*float(np.dot(x_EL,du_dphi))
                + 2*C_S*uS*float(np.dot(x_ES,du_dphi)))

    p_ang_new = p_ang.copy()
    p_ang_new[0] -= h*dV_dtheta
    p_ang_new[1] -= h*dV_dphi
    # p_ψ no cambia con V_multi

    return p_tras_new, p_ang_new


# ══════════════════════════════════════════════════════════════════════════════
# 4. INTEGRADOR RESPA
# ══════════════════════════════════════════════════════════════════════════════
# Un paso orbital completo:
#
#   kick_vmulti(h/2)
#   for n_sub:  yoshida4_rot(h_rot)      ← rotación rápida
#   sv_tras(h_orb)                        ← traslación orbital
#   for n_sub:  yoshida4_rot(h_rot)
#   kick_vmulti(h/2)
#
# Esto es el esquema RESPA-1 de Tuckerman et al. (1992), adaptado a
# coordenadas canónicas.

H_ORB_S  = 0.5 * DIA         # 12 horas en segundos
H_ROT_S  = 60.0               # 1 minuto para resolver el spin
N_SUB    = int(H_ORB_S / H_ROT_S)   # 720 substeps/paso


def sv_tras_step(q_tras: np.ndarray, p_tras: np.ndarray, h: float):
    """SV estándar para la traslación orbital."""
    p_mid   = p_tras - (h/2)*grad_V_grav(q_tras)
    q_new   = q_tras + h*p_mid*inv_m_tras
    p_new   = p_mid  - (h/2)*grad_V_grav(q_new)
    return q_new, p_new


def yoshida4_tras(q_t: np.ndarray, p_t: np.ndarray, h: float):
    for w in W4:
        q_t, p_t = sv_tras_step(q_t, p_t, w*h)
    return q_t, p_t


def respa_step(q: np.ndarray, p: np.ndarray, h_orb: float):
    """
    Un paso RESPA completo de tamaño h_orb.
    """
    q_t = q[0:9].copy(); p_t = p[0:9].copy()
    ang = q[9:12].copy(); p_a = p[9:12].copy()

    # kick lento (h/2)
    p_t, p_a = kick_vmulti(q_t, ang, p_t, p_a, h_orb/2)

    # substeps rápidos de rotación (h/2)
    for _ in range(N_SUB//2):
        ang, p_a = yoshida4_rot(ang, p_a, H_ROT_S)

    # paso orbital completo
    q_t, p_t = yoshida4_tras(q_t, p_t, h_orb)

    # substeps rápidos de rotación (h/2)
    for _ in range(N_SUB//2):
        ang, p_a = yoshida4_rot(ang, p_a, H_ROT_S)

    # kick lento (h/2)
    p_t, p_a = kick_vmulti(q_t, ang, p_t, p_a, h_orb/2)

    q_new = np.concatenate([q_t, ang])
    p_new = np.concatenate([p_t, p_a])
    return q_new, p_new


# ══════════════════════════════════════════════════════════════════════════════
# 5. ENERGÍA
# ══════════════════════════════════════════════════════════════════════════════

def energia(q: np.ndarray, p: np.ndarray) -> float:
    rS=q[0:3]; rE=q[3:6]; rL=q[6:9]
    theta,phi = q[9],q[10]
    p_theta,p_phi,p_psi = p[9],p[10],p[11]

    T_tras = float(0.5*np.dot(p[0:9], p[0:9]*inv_m_tras))

    st=np.sin(theta); ct=np.cos(theta)
    A = p_phi - p_psi*ct
    T_rot = p_theta**2/(2*I_perp) + A**2/(2*I_perp*st**2) + p_psi**2/(2*I_z)

    r_SL=np.linalg.norm(rL-rS); r_EL=np.linalg.norm(rL-rE); r_ES=np.linalg.norm(rS-rE)
    V0 = -G*mS*mL/r_SL - G*mE*mL/r_EL - G*mE*mS/r_ES

    sp=np.sin(phi); cp=np.cos(phi)
    u_hat=np.array([st*sp,-st*cp,ct])
    uL=float(np.dot(rL-rE, u_hat))
    uS=float(np.dot(rS-rE, u_hat))
    V_m = (-G*mL*dI/(2*r_EL**3)*(3*(uL/r_EL)**2-1)
           -G*mS*dI/(2*r_ES**3)*(3*(uS/r_ES)**2-1))

    return float(T_tras + T_rot + V0 + V_m)

# ══════════════════════════════════════════════════════════════════════════════
# 6. INTEGRACIÓN
# ══════════════════════════════════════════════════════════════════════════════

T_AÑOS  = 50
T_DIAS  = T_AÑOS * 365.25
N_ORB   = int(np.ceil(T_DIAS / 0.5))
GUARD_C = 20
M       = N_ORB // GUARD_C + 2

print("="*65)
print(f"Sistema Sol-Tierra-Luna + Rotación  (RESPA)")
print(f"h_orb={H_ORB_S/DIA:.1f}d  |  h_rot={H_ROT_S:.0f}s  |  {N_SUB} substeps/paso")
print(f"Pasos orbitales: {N_ORB}  |  Total {N_ORB*N_SUB:,} pasos de rotación")
print("="*65)

t_arr = np.zeros(M); q_arr = np.zeros((M,12)); E_arr = np.zeros(M)
q, p  = q0.copy(), p0.copy()
E0    = energia(q, p)
idx   = 0

for n in range(N_ORB):
    if n % GUARD_C == 0:
        t_arr[idx] = n*0.5
        q_arr[idx] = q
        E_arr[idx] = energia(q, p)
        if idx % 200 == 0 and idx > 0:
            dE = (E_arr[idx]-E0)/abs(E0)
            print(f"  t={t_arr[idx]/365.25:.1f}a  ΔE/E₀={dE:.2e}  θ={np.degrees(q[9]):.3f}°")
        idx += 1
    q, p = respa_step(q, p, H_ORB_S)

t_arr[idx]=N_ORB*0.5; q_arr[idx]=q; E_arr[idx]=energia(q,p); idx+=1
t_arr=t_arr[:idx]; q_arr=q_arr[:idx]; E_arr=E_arr[:idx]
print(f"\nCompleto. {idx} puntos.")

# ══════════════════════════════════════════════════════════════════════════════
# 7. OBSERVABLES Y VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

t_a       = t_arr/365.25
rE_AU     = q_arr[:,3:6]/AU
rL_AU     = q_arr[:,6:9]/AU
theta_deg = np.degrees(q_arr[:,9])
phi_deg   = np.degrees(q_arr[:,10])
d_ES      = np.linalg.norm(q_arr[:,3:6]-q_arr[:,0:3], axis=1)/AU
d_EL      = np.linalg.norm(q_arr[:,6:9]-q_arr[:,3:6], axis=1)/1e6
dE_rel    = (E_arr-E0)/abs(E0)

fig = plt.figure(figsize=(20,18))
fig.suptitle(f"Sol-Tierra-Luna + Rotación  ·  RESPA Y4  ·  {T_AÑOS} años",
             fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(4,3, hspace=0.45, wspace=0.32)

ax1=fig.add_subplot(gs[0,0])
ax1.plot(rE_AU[:,0],rE_AU[:,1],'b-',lw=0.5,alpha=0.7,label='Tierra')
ax1.plot(rL_AU[:,0],rL_AU[:,1],'m-',lw=0.3,alpha=0.3,label='Luna')
ax1.plot(0,0,'yo',ms=8,zorder=5,label='Sol')
ax1.set_xlabel('x [AU]'); ax1.set_ylabel('y [AU]')
ax1.set_title('Órbitas'); ax1.legend(fontsize=7)
ax1.grid(True,alpha=0.25); ax1.set_aspect('equal')

ax2=fig.add_subplot(gs[0,1])
rL_r=(q_arr[:,6:9]-q_arr[:,3:6])/1e6
ax2.plot(rL_r[:,0],rL_r[:,1],'m-',lw=0.4,alpha=0.6)
ax2.plot(0,0,'bo',ms=7,label='Tierra',zorder=5)
ax2.set_xlabel('x [10³ km]'); ax2.set_ylabel('y [10³ km]')
ax2.set_title('Órbita lunar'); ax2.legend(fontsize=8)
ax2.grid(True,alpha=0.25); ax2.set_aspect('equal')

ax3=fig.add_subplot(gs[0,2],projection='3d')
s=slice(None,None,10)
ax3.plot(rE_AU[s,0],rE_AU[s,1],rE_AU[s,2],'b-',lw=0.5)
ax3.plot(rL_AU[s,0],rL_AU[s,1],rL_AU[s,2],'m-',lw=0.3,alpha=0.5)
ax3.scatter(0,0,0,color='yellow',s=60,zorder=5)
ax3.set_title('Vista 3D'); ax3.tick_params(labelsize=6)
ax3.set_xlabel('x[AU]',fontsize=7); ax3.set_ylabel('y[AU]',fontsize=7)
ax3.set_zlabel('z[AU]',fontsize=7)

ax4=fig.add_subplot(gs[1,0:2])
ax4.plot(t_a, dE_rel,'g-',lw=0.7)
ax4.axhline(0,color='k',lw=0.5,ls=':')
ax4.set_xlabel('Tiempo [años]'); ax4.set_ylabel('(E−E₀)/|E₀|')
ax4.set_title(f'Error relativo de energía  (RMS={np.sqrt(np.mean(dE_rel**2)):.1e})')
ax4.grid(True,alpha=0.25)
ax4.ticklabel_format(style='sci',axis='y',scilimits=(0,0))

ax5=fig.add_subplot(gs[1,2])
ax5.plot(t_a,d_ES,'b-',lw=0.6)
ax5.set_xlabel('Tiempo [años]'); ax5.set_ylabel('r [AU]')
ax5.set_title('Distancia Tierra-Sol'); ax5.grid(True,alpha=0.25)

ax6=fig.add_subplot(gs[2,0])
ax6.plot(t_a,theta_deg,'r-',lw=0.8)
ax6.axhline(23.4393,color='gray',lw=0.8,ls='--',label='Valor inicial')
ax6.set_xlabel('Tiempo [años]'); ax6.set_ylabel('θ [°]')
ax6.set_title('Oblicuidad θ del eje terrestre')
ax6.legend(fontsize=8); ax6.grid(True,alpha=0.25)

ax7=fig.add_subplot(gs[2,1])
ax7.plot(t_a,phi_deg%360,'darkorange',lw=0.7)
ax7.set_xlabel('Tiempo [años]'); ax7.set_ylabel('φ mod 360° [°]')
ax7.set_title('Ángulo de precesión φ'); ax7.grid(True,alpha=0.25)

ax8=fig.add_subplot(gs[2,2])
ax8.plot(t_a,d_EL,'m-',lw=0.5,alpha=0.7)
ax8.axhline(384.4,color='gray',lw=0.8,ls='--',label='384.4×10³ km')
ax8.set_xlabel('Tiempo [años]'); ax8.set_ylabel('r [10³ km]')
ax8.set_title('Distancia Tierra-Luna')
ax8.legend(fontsize=8); ax8.grid(True,alpha=0.25)

ax9=fig.add_subplot(gs[3,0:2],projection='3d')
th=q_arr[:,9]; ph=q_arr[:,10]
ux=np.sin(th)*np.sin(ph); uy=-np.sin(th)*np.cos(ph); uz=np.cos(th)
s2=slice(None,None,5)
sc=ax9.scatter(ux[s2].tolist(),uy[s2].tolist(),uz[s2].tolist(),  # type: ignore[arg-type]
               c=t_a[s2],cmap='plasma',s=2,alpha=0.7)
plt.colorbar(sc,ax=ax9,label='Tiempo [años]',shrink=0.6)
ax9.set_xlabel('ux',fontsize=7); ax9.set_ylabel('uy',fontsize=7)
ax9.set_zlabel('uz',fontsize=7)
ax9.set_title('Traza del eje polar terrestre (precesión)')
ax9.tick_params(labelsize=6)

ax10=fig.add_subplot(gs[3,2])
ax10.plot(t_a,d_ES,'b-',lw=0.6,label='T-S [AU]')
ax10_r=ax10.twinx()
ax10_r.plot(t_a,d_EL,'m-',lw=0.5,alpha=0.7,label='T-L [10³km]')
ax10.set_xlabel('Tiempo [años]'); ax10.set_ylabel('r T-S [AU]',color='b')
ax10_r.set_ylabel('r T-L [10³km]',color='m')
ax10.set_title('Distancias orbitales'); ax10.grid(True,alpha=0.25)

plt.savefig('sistema_completo.png',dpi=150,bbox_inches='tight')
plt.show()

# ── Diagnóstico ──────────────────────────────────────────────────────────────
print("\n"+"═"*60)
print("DIAGNÓSTICO")
print("═"*60)
print(f"  E₀                   = {E0:.4e} J")
print(f"  max |ΔE/E₀|          = {np.max(np.abs(dE_rel)):.2e}")
print(f"  RMS |ΔE/E₀|          = {np.sqrt(np.mean(dE_rel**2)):.2e}")
print(f"  θ inicial/final [°]  = {theta_deg[0]:.4f} / {theta_deg[-1]:.4f}")
print(f"  Δθ total [°]         = {theta_deg[-1]-theta_deg[0]:.6f}")
print(f"  d_ES ∈ [{d_ES.min():.4f}, {d_ES.max():.4f}] AU")
print(f"  d_EL ∈ [{d_EL.min():.1f}, {d_EL.max():.1f}] ×10³ km")
print("═"*60)