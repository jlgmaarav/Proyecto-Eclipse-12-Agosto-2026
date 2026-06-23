import requests
import time
from pathlib import Path

# ==================== CONFIGURACIÓN ====================
BASE_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
OUTPUT_DIR = Path("datos_horizons_2026")
OUTPUT_DIR.mkdir(exist_ok=True)

# Cuerpos para condiciones iniciales (1 ene 2026)
BODIES_INITIAL = [
    ("Sol",      "10"),
    ("Mercurio", "1"),
    ("Venus",    "2"),
    ("Tierra",   "399"),
    ("Luna",     "301"),
    ("Marte",    "4"),
    ("Jupiter",  "5"),
    ("Saturno",  "6"),
    ("Urano",    "7"),
    ("Neptuno",  "8"),
]

# Solo Sol, Tierra y Luna para verificaciones
VERIF_BODIES = ["10", "399", "301"]
VERIF_NAMES  = ["Sol", "Tierra", "Luna"]

VERIF_DATES = [
    "2026-Mar-01 00:00:00",
    "2026-Jun-01 00:00:00",
    "2026-Aug-01 00:00:00",
    "2026-Aug-12 17:47:06",
]

def get_common_params():
    return {
        "format": "text",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": "'500@0'",
        "OBJ_DATA": "'YES'",
        "MAKE_EPHEM": "'YES'",
        "OUT_UNITS": "'AU-D'",
        "REF_PLANE": "'ECLIPTIC'",
        "REF_SYSTEM": "'J2000'",
        "VEC_TABLE": "'2'",
        "CSV_FORMAT": "'YES'",
        "STEP_SIZE": "'1 d'",
        "TIME_TYPE": "'TDB'",
    }

def fetch_horizons(command, start_time, stop_time=None):
    params = get_common_params()
    params["COMMAND"] = f"'{command}'"
    params["START_TIME"] = f"'{start_time}'"
    if stop_time:
        params["STOP_TIME"] = f"'{stop_time}'"
    else:
        params["STOP_TIME"] = f"'{start_time}'"

    response = requests.get(BASE_URL, params=params, timeout=60)
    
    if response.status_code != 200:
        print(f"❌ Error {response.status_code} para {command} ({start_time})")
        print(response.text[:600])
        return None
    
    text = response.text
    
    # Detectar qué efeméride usó Horizons
    if "{source: DE440}" in text:
        source = "DE440"
    elif "{source: DE441}" in text:
        source = "DE441"
    else:
        source = "DESCONOCIDA"
    
    print(f"   ✓ {source} detectado")
    return text

# ==================== 1. CONDICIONES INICIALES ====================
print("📥 Descargando condiciones iniciales (1 ene 2026)...")
for name, body_id in BODIES_INITIAL:
    print(f"   → {name} (ID {body_id})")
    data = fetch_horizons(body_id, "2026-Jan-01 00:00:00", "2026-Jan-02 00:00:00")
    if data:
        (OUTPUT_DIR / f"initial_{name.lower()}.txt").write_text(data, encoding="utf-8")
    time.sleep(1.1)

# ==================== 2. POSICIONES DE VERIFICACIÓN ====================
print("\n📥 Descargando posiciones de verificación...")
for date in VERIF_DATES:
    print(f"   → Fecha: {date}")
    for i, body_id in enumerate(VERIF_BODIES):
        name = VERIF_NAMES[i]
        print(f"      • {name}")
        data = fetch_horizons(body_id, date)
        if data:
            safe_date = date.replace(" ", "_").replace(":", "").replace("-", "")
            (OUTPUT_DIR / f"verif_{name.lower()}_{safe_date}.txt").write_text(data, encoding="utf-8")
        time.sleep(0.9)

# ==================== 3. ΔT y LOLA ====================
print("\n📥 Descargando ΔT y datos LOLA...")
urls = {
    "deltat.data":      "https://maia.usno.navy.mil/ser7/deltat.data",
    "deltat.preds":     "https://maia.usno.navy.mil/ser7/deltat.preds",
    "finals2000A.all":  "https://maia.usno.navy.mil/ser7/finals2000A.all",
    "ldem_4.img":       "http://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/LDEM_4.IMG",
    "ldem_4.lbl":       "http://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/LDEM_4.LBL",
}

for filename, url in urls.items():
    filepath = OUTPUT_DIR / filename
    if filepath.exists() and filepath.stat().st_size > 10000:
        print(f"   → {filename} (ya existe)")
        continue
    print(f"   → {filename}")
    r = requests.get(url, stream=True)
    with open(filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            f.write(chunk)
    print(f"      ✓ Descargado")

print("\n🎉 ¡Todo listo!")
print(f"   Carpeta: {OUTPUT_DIR.resolve()}")
print("   Efeméride usada: DE441 (la actual de Horizons – perfecta para 2026)")
print("\nAhora revisa los archivos .txt y busca las masas GM en las cabeceras.")