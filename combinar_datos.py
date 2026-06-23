import os
import csv
from pathlib import Path
import re

OUTPUT_DIR = Path("datos_horizons_2026")
COMBINED_DIR = OUTPUT_DIR / "consolidados"
COMBINED_DIR.mkdir(exist_ok=True)

def extraer_datos_initial(archivo):
    texto = archivo.read_text(encoding="utf-8")
    # Nombre del cuerpo
    body_match = re.search(r"Target body name:\s*(.+?)\s*\((\d+)\)", texto)
    body_name = body_match.group(1).strip() if body_match else "Desconocido"
    
    # GM
    gm_match = re.search(r"GM \(km\^3/s\^2\):\s*([\d.]+)", texto)
    gm = float(gm_match.group(1)) if gm_match else None
    
    # Datos de posición y velocidad (línea después de $$SOE)
    soe_match = re.search(r"\$\$SOE\n(.+?)\n\$\$EOE", texto, re.DOTALL)
    if soe_match:
        linea = soe_match.group(1).strip()
        campos = [x.strip() for x in linea.split(",")]
        fecha = campos[0]
        x, y, z = float(campos[2]), float(campos[3]), float(campos[4])
        vx, vy, vz = float(campos[5]), float(campos[6]), float(campos[7])
        return {
            "body": body_name,
            "gm_km3_s2": gm,
            "date": fecha,
            "x_au": x, "y_au": y, "z_au": z,
            "vx_au_d": vx, "vy_au_d": vy, "vz_au_d": vz
        }
    return None

# ==================== 1. INITIAL CONDITIONS ====================
print("📦 Procesando condiciones iniciales (10 cuerpos)...")
initial_data = []
for archivo in OUTPUT_DIR.glob("initial_*.txt"):
    datos = extraer_datos_initial(archivo)
    if datos:
        initial_data.append(datos)
        print(f"   ✓ {datos['body']}")

# Guardar CSV limpio
with open(COMBINED_DIR / "initial_conditions.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["body", "gm_km3_s2", "date", "x_au", "y_au", "z_au", "vx_au_d", "vy_au_d", "vz_au_d"])
    writer.writeheader()
    writer.writerows(initial_data)

print(f"   → initial_conditions.csv creado ({len(initial_data)} cuerpos)")

# ==================== 2. VERIFICATION BENCHMARKS ====================
print("\n📦 Procesando posiciones de verificación...")
verif_data = []
for archivo in OUTPUT_DIR.glob("verif_*.txt"):
    datos = extraer_datos_initial(archivo)  # reutilizamos la misma función
    if datos:
        # Extraer fecha del nombre del archivo
        fecha_archivo = archivo.stem.replace("verif_", "").replace("_", " ")
        datos["date"] = fecha_archivo
        verif_data.append(datos)
        print(f"   ✓ {datos['body']} - {fecha_archivo}")

with open(COMBINED_DIR / "verification_benchmarks.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["body", "date", "x_au", "y_au", "z_au", "vx_au_d", "vy_au_d", "vz_au_d"])
    writer.writeheader()
    writer.writerows(verif_data)

print(f"   → verification_benchmarks.csv creado")

# ==================== 3. CONSTANTES FÍSICAS ====================
print("\n📦 Extrayendo constantes físicas...")
with open(COMBINED_DIR / "physical_constants.txt", "w", encoding="utf-8") as f:
    f.write("=== CONSTANTES FÍSICAS EXTRAÍDAS DE DE441 ===\n\n")
    for archivo in sorted(OUTPUT_DIR.glob("initial_*.txt")):
        texto = archivo.read_text(encoding="utf-8")
        body_match = re.search(r"Target body name:\s*(.+?)\s*\((\d+)\)", texto)
        gm_match = re.search(r"GM \(km\^3/s\^2\):\s*([\d.]+)", texto)
        if body_match and gm_match:
            nombre = body_match.group(1).strip()
            gm = gm_match.group(1)
            f.write(f"{nombre:12} → GM = {gm} km³/s²\n")

    f.write("\n\nOtros valores (de cabeceras):\n")
    f.write("Radio ecuatorial Tierra ≈ 6378.137 km\n")
    f.write("J2 Tierra ≈ 0.00108262545\n")
    f.write("Radio Sol ≈ 696000 km\n")
    f.write("Radio Luna ≈ 1737.4 km\n")

print("   → physical_constants.txt creado")

# ==================== 4. RESUMEN ====================
print("\n✅ ¡Todo consolidado!")
print(f"   Archivos limpios en: {COMBINED_DIR.resolve()}")
print("   - initial_conditions.csv")
print("   - verification_benchmarks.csv")
print("   - physical_constants.txt")
print("   - datos_resumen.txt (se creará automáticamente)")

# Copiar también los ΔT y LOLA a la carpeta consolidados por comodidad
for f in ["deltat.data", "deltat.preds", "finals2000A.all", "ldem_4.img", "ldem_4.lbl"]:
    src = OUTPUT_DIR / f
    if src.exists():
        import shutil
        shutil.copy2(src, COMBINED_DIR / f)
        print(f"   ✓ Copiado {f} a consolidados")