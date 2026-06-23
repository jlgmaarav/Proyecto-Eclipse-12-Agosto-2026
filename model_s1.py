"""
model_s1.py
===========
Modelo S1 — Sombra Besseliana Clásica (Delta T = 0).

Física:
  - Órbita: propagada con el modelo M5 (N-body + J2 + EIH + Marea CTL).
  - Geometría: Plano Fundamental de Bessel (Sol y Luna esferas perfectas).
  - Tierra: Elipsoide WGS84.
  - Tiempo: Asume Delta T = 0 (UT1 = TDB/TT), sirviendo de control.
"""

import time
import numpy as np
from datetime import datetime, timedelta
from benchmark import (
    AU_KM,
    DAY_S,
    R_EQ_EARTH,
    R_POL_EARTH,
    R_SUN,
    R_MOON,
    load_reference_data
)
from model_m5 import (
    integrate,
    build_Y0,
    f_m5,
    result_to_model_states,
    BODIES,
    IDX_EARTH,
    IDX_MOON,
    IDX_SUN
)

ECLIPSE_UTC = datetime(2026, 8, 12, 17, 47, 6)

# Oblicuidad de la Eclíptica J2000
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
    """
    Calcula los elementos besselianos del eclipse en el Plano Fundamental.
    Vectores en metros, en el marco geocéntrico ecuatorial.
    """
    # Origen en la Tierra:
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
    
    # Coordenadas rectangulares de la sombra (x, y) en metros
    x = R_L @ i_hat
    y = R_L @ j_hat
    
    # Ángulo horario de la sombra en Greenwich (mu)
    mu = (gmst - a) % (2 * np.pi)
    
    # Conos de sombra (radios l1 y l2) en metros
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
    """
    Interseca el eje de la sombra con el elipsoide WGS84 de forma rigurosa.
    Retorna (latitud, longitud, altura) o None si no interseca.
    """
    a = R_EQ_EARTH * 1000.0
    b = R_POL_EARTH * 1000.0
    
    x = elem["x"]
    y = elem["y"]
    i_hat = elem["i"]
    j_hat = elem["j"]
    k_hat = elem["k"]
    
    # Eje de sombra: P(z_b) = u + z_b * k_hat
    u = x * i_hat + y * j_hat
    
    # Ecuación de intersección con el elipsoide: A z_b^2 + B z_b + C = 0
    A = (k_hat[0]**2 + k_hat[1]**2) / a**2 + k_hat[2]**2 / b**2
    B = 2.0 * ((u[0] * k_hat[0] + u[1] * k_hat[1]) / a**2 + u[2] * k_hat[2] / b**2)
    C = (u[0]**2 + u[1]**2) / a**2 + u[2]**2 / b**2 - 1.0
    
    disc = B**2 - 4 * A * C
    if disc < 0:
        return None  # No toca la Tierra
        
    # Escogemos la raíz del hemisferio que apunta a la Luna (z_b > 0 en este sistema)
    z_b = max((-B - np.sqrt(disc)) / (2.0 * A), (-B + np.sqrt(disc)) / (2.0 * A))
    
    # Punto de intersección geocéntrico ecuatorial
    P_eq = u + z_b * k_hat
    
    # Recuperamos el GMST para rotar a coordenadas terrestres fijas (ECEF)
    mu = elem["mu"]
    a_rad = elem["a"]
    gmst = (mu + a_rad) % (2 * np.pi)
    
    cos_g = np.cos(gmst)
    sin_g = np.sin(gmst)
    
    # ECEF
    X_ecef = P_eq[0] * cos_g + P_eq[1] * sin_g
    Y_ecef = -P_eq[0] * sin_g + P_eq[1] * cos_g
    Z_ecef = P_eq[2]
    
    # Conversión a Latitud, Longitud mediante el método de Bowring
    p = np.sqrt(X_ecef**2 + Y_ecef**2)
    e2 = (a**2 - b**2) / a**2
    ep2 = (a**2 - b**2) / b**2
    
    if p < 1e-5:
        lat = 90.0 if Z_ecef > 0 else -90.0
        lon = 0.0
        alt = np.abs(Z_ecef) - b
        return lat, lon, alt
        
    theta = np.arctan2(Z_ecef * a, p * b)
    lat_rad = np.arctan2(Z_ecef + ep2 * b * np.sin(theta)**3, p - e2 * a * np.cos(theta)**3)
    lon_rad = np.arctan2(Y_ecef, X_ecef)
    
    lat = np.degrees(lat_rad)
    lon = np.degrees(lon_rad)
    lon = (lon + 180.0) % 360.0 - 180.0  # Normalizar a [-180, 180]
    
    N = a / np.sqrt(1.0 - e2 * np.sin(lat_rad)**2)
    alt = p / np.cos(lat_rad) - N
    
    return lat, lon, alt

def run_s1():
    t0_dt = datetime(2026, 1, 1, 0, 0, 0)
    
    # 1. Integración de órbita con M5
    Y0, _ = build_Y0()
    t_eclipse_days = (ECLIPSE_UTC - t0_dt).total_seconds() / DAY_S
    
    # Generar puntos de salida durante el eclipse: de 16:30 a 18:30 UTC cada 30 segundos
    t_start_ecl = ECLIPSE_UTC - timedelta(minutes=60)
    t_end_ecl   = ECLIPSE_UTC + timedelta(minutes=60)
    
    t_out = []
    t_current = t_start_ecl
    while t_current <= t_end_ecl:
        days = (t_current - t0_dt).total_seconds() / DAY_S
        t_out.append(days)
        t_current += timedelta(seconds=30)
        
    # Añadir también el momento exacto del eclipse del benchmark si no está
    t_out.append(t_eclipse_days)
    t_out = sorted(list(set(t_out)))
    
    print("\n[S1] Integrando órbita usando M5...")
    t_wall = time.perf_counter()
    result = integrate(f_m5, Y0, t_end=t_eclipse_days + 0.1, h0=0.1, tol=1e-10, t_out=t_out)
    print(f"     Pasos: {result['steps']} | CPU: {time.perf_counter() - t_wall:.2f} s")
    
    # 2. Calcular sombra besseliana para cada paso
    print("\n[S1] Proyectando sombra Besseliana (Delta T = 0)...")
    path_points = []
    
    for t_val, Y_val in zip(result["t_out"], result["Y_out"]):
        t_utc = t0_dt + timedelta(days=t_val)
        
        # Posiciones en eclíptica (AU)
        pos_S = Y_val[6*IDX_SUN   : 6*IDX_SUN   + 3]
        pos_E = Y_val[6*IDX_EARTH : 6*IDX_EARTH + 3]
        pos_L = Y_val[6*IDX_MOON  : 6*IDX_MOON  + 3]
        
        # Convertir a ecuatorial (metros)
        rS_eq = ecliptic_to_equatorial(pos_S) * (AU_KM * 1000.0)
        rE_eq = ecliptic_to_equatorial(pos_E) * (AU_KM * 1000.0)
        rL_eq = ecliptic_to_equatorial(pos_L) * (AU_KM * 1000.0)
        
        # S1: Delta T = 0 -> usar hora UTC directa
        gmst = gmst_angle(t_utc)
        
        elem = calcular_elementos_besselianos(rS_eq, rE_eq, rL_eq, gmst)
        inter = intersect_wgs84(elem)
        
        if inter is not None:
            lat, lon, alt = inter
            path_points.append({
                "time": t_utc,
                "lat": lat,
                "lon": lon,
                "x": elem["x"],
                "y": elem["y"],
                "l2": elem["l2"]
            })
            
    # 3. Reportar resultados
    print(f"\n======================================================================")
    print(f"  MODELO S1 — Bessel clásico (Delta T = 0) WGS84")
    print(f"======================================================================")
    print(f"  Puntos proyectados en la Tierra: {len(path_points)}")
    
    if len(path_points) > 0:
        # Encontrar momento más cercano al máximo del eclipse
        target_time = ECLIPSE_UTC
        best_pt = min(path_points, key=lambda pt: abs(pt["time"] - target_time))
        
        print(f"\n  Momento central del eclipse (simulado): {best_pt['time'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"     Coordenadas del centro de la sombra:")
        print(f"       Latitud : {best_pt['lat']:.4f}° N")
        print(f"       Longitud: {best_pt['lon']:.4f}° W / E")
        print(f"       Radio de la Umbra (l2): {best_pt['l2']/1000:.3f} km")
        print(f"       (Nota: l2 < 0 indica eclipse total, radio es |l2|)")
        print(f"======================================================================\n")
    else:
        print("  ¡El eje de la sombra no llegó a intersecar con la superficie de la Tierra!")
        print(f"======================================================================\n")

if __name__ == "__main__":
    run_s1()
