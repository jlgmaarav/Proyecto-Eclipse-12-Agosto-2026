"""
model_s3.py
===========
Modelo S3 — Sombra Besseliana + Delta T + Topografía Lunar (LOLA).

Física:
  - Órbita: propagada con el modelo M5 (N-body + J2 + EIH + Marea CTL).
  - Geometría: Plano de Bessel riguroso con elipsoide WGS84.
  - Tiempo: Delta T dinámico (UT1 corregido).
  - Topografía: Carga y decodifica el modelo de elevación LOLA de la NASA
                (ldem_4.img) para trazar el limbo lunar real de montañas
                y valles y simular las Perlas de Baily (Baily's beads).
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from benchmark import (
    AU_KM,
    DAY_S,
    R_EQ_EARTH,
    R_POL_EARTH,
    R_SUN,
    R_MOON,
    get_delta_t,
    days_to_year_frac
)
from model_m5 import (
    integrate,
    build_Y0,
    f_m5,
    BODIES,
    IDX_EARTH,
    IDX_MOON,
    IDX_SUN
)

ECLIPSE_UTC = datetime(2026, 8, 12, 17, 47, 6)
EPSILON = np.radians(23.4392911)

def ecliptic_to_equatorial(vec_ecl: np.ndarray) -> np.ndarray:
    """Rota un vector del marco Eclíptico J2000 al Ecuatorial J2000."""
    x, y, z = vec_ecl
    cos_e = np.cos(EPSILON)
    sin_e = np.sin(EPSILON)
    return np.array([
        x,
        y * cos_e - z * sin_e,
        y * sin_e + z * cos_e
    ])

def gmst_angle(dt_utc: datetime) -> float:
    """Calcula el Greenwich Mean Sidereal Time (GMST) en radianes."""
    J2000 = datetime(2000, 1, 1, 12, 0, 0)
    d = (dt_utc - J2000).total_seconds() / 86400.0
    gmst_hours = 18.697374558 + 24.06570982441908 * d
    gmst_deg = (gmst_hours * 15.0) % 360.0
    return np.radians(gmst_deg)

def calcular_elementos_besselianos(
    rS_eq: np.ndarray,
    rE_eq: np.ndarray,
    rL_eq: np.ndarray,
    gmst: float
) -> dict:
    """Calcula los elementos besselianos del eclipse en el Plano Fundamental."""
    R_S = rS_eq - rE_eq
    R_L = rL_eq - rE_eq
    
    # Eje de la sombra k_hat (de Luna a Sol - dirección estándar)
    G_vec = R_S - R_L
    dist_G = np.linalg.norm(G_vec)
    k_hat = G_vec / dist_G
    
    # Declinación d y Ascensión Recta a del eje de la sombra
    d = np.arcsin(k_hat[2])
    a = np.arctan2(k_hat[1], k_hat[0])
    
    # Vectores del Plano Fundamental
    i_hat = np.array([-np.sin(a), np.cos(a), 0.0])
    j_hat = np.cross(k_hat, i_hat)
    
    x = R_L @ i_hat
    y = R_L @ j_hat
    mu = (gmst - a) % (2 * np.pi)
    
    sin_f1 = (R_SUN * 1000.0 + R_MOON * 1000.0) / dist_G
    sin_f2 = (R_SUN * 1000.0 - R_MOON * 1000.0) / dist_G
    z_L = R_L @ k_hat
    
    l1 = z_L * np.tan(np.arcsin(sin_f1)) + (R_MOON * 1000.0) / np.cos(np.arcsin(sin_f1))
    l2 = z_L * np.tan(np.arcsin(sin_f2)) - (R_MOON * 1000.0) / np.cos(np.arcsin(sin_f2))
    
    return {
        "x": x, "y": y, "d": d, "mu": mu, "l1": l1, "l2": l2,
        "a": a, "i": i_hat, "j": j_hat, "k": k_hat
    }

def intersect_wgs84(elem: dict) -> tuple:
    """Interseca el eje de la sombra con el elipsoide WGS84. Devuelve (lat, lon, alt, P_eq)."""
    a = R_EQ_EARTH * 1000.0
    b = R_POL_EARTH * 1000.0
    
    x = elem["x"]
    y = elem["y"]
    i_hat = elem["i"]
    j_hat = elem["j"]
    k_hat = elem["k"]
    
    u = x * i_hat + y * j_hat
    
    A = (k_hat[0]**2 + k_hat[1]**2) / a**2 + k_hat[2]**2 / b**2
    B = 2.0 * ((u[0] * k_hat[0] + u[1] * k_hat[1]) / a**2 + u[2] * k_hat[2] / b**2)
    C = (u[0]**2 + u[1]**2) / a**2 + u[2]**2 / b**2 - 1.0
    
    disc = B**2 - 4 * A * C
    if disc < 0:
        return None
        
    z_b = max((-B - np.sqrt(disc)) / (2.0 * A), (-B + np.sqrt(disc)) / (2.0 * A))
    P_eq = u + z_b * k_hat
    
    mu = elem["mu"]
    a_rad = elem["a"]
    gmst = (mu + a_rad) % (2 * np.pi)
    
    cos_g = np.cos(gmst)
    sin_g = np.sin(gmst)
    
    X_ecef = P_eq[0] * cos_g + P_eq[1] * sin_g
    Y_ecef = -P_eq[0] * sin_g + P_eq[1] * cos_g
    Z_ecef = P_eq[2]
    
    p = np.sqrt(X_ecef**2 + Y_ecef**2)
    e2 = (a**2 - b**2) / a**2
    ep2 = (a**2 - b**2) / b**2
    
    if p < 1e-5:
        lat = 90.0 if Z_ecef > 0 else -90.0
        lon = 0.0
        return lat, lon, np.abs(Z_ecef) - b, P_eq
        
    theta = np.arctan2(Z_ecef * a, p * b)
    lat_rad = np.arctan2(Z_ecef + ep2 * b * np.sin(theta)**3, p - e2 * a * np.cos(theta)**3)
    lon_rad = np.arctan2(Y_ecef, X_ecef)
    
    lat = np.degrees(lat_rad)
    lon = np.degrees(lon_rad)
    lon = (lon + 180.0) % 360.0 - 180.0
    
    N = a / np.sqrt(1.0 - e2 * np.sin(lat_rad)**2)
    alt = p / np.cos(lat_rad) - N
    
    return lat, lon, alt, P_eq

def run_s3():
    t0_dt = datetime(2026, 1, 1, 0, 0, 0)
    
    # 1. Cargar LOLA DEM
    print("[S3] Cargando topografía lunar LOLA (ldem_4.img)...")
    try:
        lola_data = np.fromfile("ldem_4.img", dtype="<i2").reshape((720, 1440))
        print("     LOLA DEM cargado correctamente.")
    except Exception as e:
        print(f"  ❌ Error al cargar ldem_4.img: {e}")
        return
        
    # 2. Integrar órbita con M5
    Y0, _ = build_Y0()
    t_eclipse_days = (ECLIPSE_UTC - t0_dt).total_seconds() / DAY_S
    
    print("\n[S3] Integrando órbita usando M5...")
    result = integrate(f_m5, Y0, t_end=t_eclipse_days + 0.01, h0=0.1, tol=1e-10, t_out=[t_eclipse_days])
    
    Y_val = result["Y_out"][0]
    pos_S = Y_val[6*IDX_SUN   : 6*IDX_SUN   + 3]
    pos_E = Y_val[6*IDX_EARTH : 6*IDX_EARTH + 3]
    pos_L = Y_val[6*IDX_MOON  : 6*IDX_MOON  + 3]
    
    rS_eq = ecliptic_to_equatorial(pos_S) * (AU_KM * 1000.0)
    rE_eq = ecliptic_to_equatorial(pos_E) * (AU_KM * 1000.0)
    rL_eq = ecliptic_to_equatorial(pos_L) * (AU_KM * 1000.0)
    
    delta_t_s = get_delta_t(days_to_year_frac(t_eclipse_days))
    t_ut1 = ECLIPSE_UTC - timedelta(seconds=delta_t_s)
    gmst = gmst_angle(t_ut1)
    
    elem = calcular_elementos_besselianos(rS_eq, rE_eq, rL_eq, gmst)
    inter = intersect_wgs84(elem)
    
    if inter is None:
        print("  ❌ La sombra no toca la Tierra en el momento del eclipse.")
        return
        
    lat, lon, alt, r_obs = inter
    rL_geo = rL_eq - rE_eq
    rS_geo = rS_eq - rE_eq
    
    print(f"\n[S3] Centro de totalidad localizado en: {lat:.4f}° N, {lon:.4f}° W")
    print(f"     Distancia Tierra-Luna (geocéntrica): {np.linalg.norm(rL_geo)/1000:,.1f} km")
    
    # 3. Geometría desde el punto del observador
    v_L_obs = rL_geo - r_obs
    d_L = np.linalg.norm(v_L_obs)
    u_L = v_L_obs / d_L  # Vector unitario observador -> Luna
    
    v_S_obs = rS_geo - r_obs
    d_S = np.linalg.norm(v_S_obs)
    u_S = v_S_obs / d_S  # Vector unitario observador -> Sol
    
    # Base ortonormal para el plano del cielo perpendicular a u_L
    up = np.array([0.0, 0.0, 1.0])
    North = up - (u_L @ up) * u_L
    North = North / np.linalg.norm(North)
    East = np.cross(u_L, North)
    
    # Base de orientación de la Luna (Sincronizada)
    # Eje X hacia la Tierra:
    x_M = -rL_geo / np.linalg.norm(rL_geo)
    # Eje Z perpendicular al plano de la eclíptica (ecliptic normal en ecuatorial)
    z_M = np.array([0.0, -np.sin(EPSILON), np.cos(EPSILON)])
    y_M = np.cross(z_M, x_M)
    y_M = y_M / np.linalg.norm(y_M)
    z_M = np.cross(x_M, y_M)
    
    # Matriz de rotación J2000_eq -> Selenográfica (Moon body-fixed)
    R_Moon = np.stack([x_M, y_M, z_M], axis=0)
    
    # Ángulo del horizonte de la Luna (semi-diámetro angular aparente)
    sin_th_horiz = (R_MOON * 1000.0) / d_L
    cos_th_horiz = np.sqrt(1.0 - sin_th_horiz**2)
    
    # 4. Calcular el contorno del limbo para 360 grados
    angulos_deg = np.linspace(0, 360, 720)
    alturas_lola = []
    radios_aparentes_as = []
    
    print("[S3] Calculando perfil de elevación del limbo lunar...")
    
    for ang_deg in angulos_deg:
        ang_rad = np.radians(ang_deg)
        
        # Dirección en el plano perpendicular a la visual
        e = np.cos(ang_rad) * North + np.sin(ang_rad) * East
        
        # Punto en el limbo (horizonte lunar tridimensional)
        P_limb = (R_MOON * 1000.0) * cos_th_horiz * e - (R_MOON * 1000.0) * sin_th_horiz * u_L
        
        # Transformar a selenográfico
        P_M = R_Moon @ P_limb
        nP_M = P_M / np.linalg.norm(P_M)
        
        # Selenográficas lat/lon
        lat_sel = np.degrees(np.arcsin(nP_M[2]))
        lon_sel = np.degrees(np.arctan2(nP_M[1], nP_M[0]))
        
        # Búsqueda en LOLA
        row = int((90.0 - lat_sel) * 4.0) % 720
        col = int((lon_sel % 360.0) * 4.0) % 1440
        height = float(lola_data[row, col]) * 0.5  # metros
        
        alturas_lola.append(height)
        
        # Radio angular aparente en segundos de arco
        theta_moon_rad = (R_MOON * 1000.0 + height) / d_L
        radios_aparentes_as.append(theta_moon_rad * 206264.806)
        
    alturas_lola = np.array(alturas_lola)
    radios_aparentes_as = np.array(radios_aparentes_as)
    
    # 5. Simular Perlas de Baily (Baily's beads)
    # Obtenemos los radios angulares medios en segundos de arco
    rad_moon_mean_as = ((R_MOON * 1000.0) / d_L) * 206264.806
    rad_sun_as = ((R_SUN * 1000.0) / d_S) * 206264.806
    
    # Diferencia de tamaño angular promedio
    diff_mean = rad_moon_mean_as - rad_sun_as  # ~45 segundos de arco (Luna más grande)
    
    # Para simular el segundo contacto (C2) donde el limbo de la Luna y el del Sol se cruzan,
    # desfasamos el centro de la Luna por una distancia angular D_MS comparable a la diferencia de radios.
    # Así, el limbo solar sobresaldrá en los valles más profundos.
    # Desfase: 42.0 segundos de arco en la dirección del movimiento orbital lunar (aprox. a 240 grados).
    D_MS = diff_mean - 1.0  # ligeramente menor que la diferencia, permitiendo que sobresalgan las perlas
    direc_mov_deg = 240.0   # posición angular del desfase
    direc_mov_rad = np.radians(direc_mov_deg)
    
    # El limbo aparente del Sol en relación al centro de la Luna es:
    radios_sol_aparente_as = rad_sun_as + D_MS * np.cos(np.radians(angulos_deg) - direc_mov_rad)
    
    # Las perlas de Baily ocurren donde el limbo del Sol supera el limbo de la Luna
    beads_mask = radios_sol_aparente_as > radios_aparentes_as
    beads_indices = np.where(beads_mask)[0]
    
    print(f"\n[S3] Simulación de Perlas de Baily en el Contacto C2:")
    print(f"     Semi-diámetro angular medio Luna: {rad_moon_mean_as:.2f} arcsec")
    print(f"     Semi-diámetro angular Sol       : {rad_sun_as:.2f} arcsec")
    print(f"     Desfase de simulación de C2     : {D_MS:.2f} arcsec (Ángulo de contacto: {direc_mov_deg}°)")
    print(f"     Perlas de Baily detectadas      : {len(beads_indices)} puntos de limbo")
    
    # 6. Graficar con Matplotlib
    print("\n[S3] Generando gráfico de perfil de limbo y perlas...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Panel 1: Topografía del limbo en metros
    ax1.plot(angulos_deg, alturas_lola, color="#4A90E2", lw=2, label="Perfil LOLA")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("Topografía Física en el Limbo Lunar (LRO LOLA GDR)")
    ax1.set_xlabel("Ángulo de Posición (grados, 0=Norte, 90=Este)")
    ax1.set_ylabel("Elevación (metros)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Limbo solar vs lunar (segundos de arco)
    ax2.plot(angulos_deg, radios_aparentes_as, color="black", lw=2, label="Limbo Lunar Real")
    ax2.plot(angulos_deg, radios_sol_aparente_as, color="#FF9500", lw=1.5, linestyle="--", label="Limbo Solar (C2)")
    
    # Resaltar perlas de Baily
    if len(beads_indices) > 0:
        # Agrupar perlas contiguas para reportarlas
        bead_groups = []
        current_group = [angulos_deg[beads_indices[0]]]
        for idx in range(1, len(beads_indices)):
            if beads_indices[idx] - beads_indices[idx-1] <= 2:
                current_group.append(angulos_deg[beads_indices[idx]])
            else:
                bead_groups.append(current_group)
                current_group = [angulos_deg[beads_indices[idx]]]
        bead_groups.append(current_group)
        
        print(f"     Detalle de perlas (ángulos en grados):")
        for i, grp in enumerate(bead_groups):
            mean_angle = np.mean(grp)
            max_depth = np.max(radios_sol_aparente_as[beads_indices] - radios_aparentes_as[beads_indices])
            print(f"       Perla {i+1}: Centro en {mean_angle:.1f}° | Ancho: {len(grp)*0.5:.1f}°")
            
        ax2.scatter(angulos_deg[beads_mask], radios_sol_aparente_as[beads_mask], 
                    color="red", s=15, zorder=5, label="Perla de Baily (Brillo Solar)")
                    
    ax2.set_title("Intersección de Limbos en Contacto C2 (Simulación)")
    ax2.set_xlabel("Ángulo de Posición (grados)")
    ax2.set_ylabel("Radio Angular (arcsec)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    output_img = "lunar_limb_bailys_beads.png"
    plt.savefig(output_img, dpi=300)
    print(f"\n✅ Gráfico guardado como '{output_img}'")
    print(f"======================================================================\n")

if __name__ == "__main__":
    run_s3()
