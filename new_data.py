import os
import re
import requests
import earthaccess
from astropy.time import Time
from astropy.utils.iers import IERS_Auto

# =====================================================================
# 1. PARÁMETROS DE ORIENTACIÓN TERRESTRE (EOP - IERS)
# =====================================================================
def get_eop_data():
    print("[1/3] Cargando Parámetros de Orientación Terrestre (IERS)...")
    iers_table = IERS_Auto.open()
    t_test = Time("2026-01-01T00:00:00", scale="utc")
    ut1_utc_obj = iers_table.ut1_utc(t_test)
    ut1_utc = float(ut1_utc_obj.value) if hasattr(ut1_utc_obj, 'value') else float(ut1_utc_obj)
    print(f" OK. Caché EOP actualizada. UT1-UTC = {ut1_utc:.5f} s")
    return iers_table

# =====================================================================
# 2. MASAS SEMILLA DE ASTEROIDES (Parseo mejorado)
# =====================================================================
def get_asteroid_seed_masses_pck(asteroid_ids=None):
    print("\n[2/3] Descargando y parseando masas (gm_de440.tpc)...")
    url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc"
    r = requests.get(url, timeout=30)
    
    os.makedirs("kernels", exist_ok=True)
    pck_path = "kernels/gm_de440.tpc"
    with open(pck_path, "wb") as f:
        f.write(r.content)
    
    masses = {}
    content = r.text
    # Buscamos patrones tipo BODY2000001_GM = ( valor )
    # El archivo de la NASA usa paréntesis y espacios variados
    pattern = r"BODY(\d+)_GM\s*=\s*\(\s*([0-9eE+.-]+)"
    matches = re.findall(pattern, content)
    
    for body_id_str, gm_value_str in matches:
        masses[int(body_id_str)] = float(gm_value_str)
    
    if asteroid_ids:
        selected = {aid: masses.get(aid, None) for aid in asteroid_ids}
        return pck_path, selected
    return pck_path, masses

# =====================================================================
# 3. DESCARGA LLR (Argumentos corregidos para earthaccess)
# =====================================================================
def download_all_llr_telemetry(start_year=1969, end_year=2026):
    print(f"\n[3/3] Iniciando descarga masiva LLR ({start_year}-{end_year})...")
    
    earthaccess.login(strategy="interactive")
    
    # Intentamos primero por short_name oficial del CDDIS
    results = earthaccess.search_data(
        short_name="MIRAGE_LLR_NPT", 
        temporal=(f"{start_year}-01-01", f"{end_year}-12-31")
    )
    
    # Si falla, usamos 'keyword' (NO 'query')
    if not results:
        print(" [!] No encontrado por Short Name. Intentando por palabra clave...")
        results = earthaccess.search_data(
            keyword="Lunar Laser Ranging",
            temporal=(f"{start_year}-01-01", f"{end_year}-12-31")
        )

    if not results:
        print(" Error: No se encontraron archivos LLR. Verifica tu conexión o el rango de fechas.")
        return False
    
    print(f" Encontrados {len(results)} archivos. Descargando...")
    os.makedirs("llr_data", exist_ok=True)
    downloaded = earthaccess.download(results, local_path="llr_data/")
    print(f"[OK] {len(downloaded)} archivos en 'llr_data/'")
    return True

# =====================================================================
# EJECUCIÓN
# =====================================================================
if __name__ == "__main__":
    print("=== INICIO DE PROCESO CORREGIDO ===\n")
    get_eop_data()
    
    # Ceres, Pallas, Vesta...
    top_asteroids = [2000001, 2000002, 2000004]
    _, asteroid_masses = get_asteroid_seed_masses_pck(top_asteroids)
    
    print("\nMasas GM obtenidas (km³/s²):")
    for aid, gm in asteroid_masses.items():
        status = f"{gm:.6e}" if gm is not None else "No encontrado"
        print(f"  Asteroide {aid} -> GM = {status}")

    download_all_llr_telemetry(1969, 2026)
    print("\n=== SCRIPT FINALIZADO ===")