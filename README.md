# Predicción del Eclipse Solar Total del 12 de Agosto de 2026
**Simulación Orbital N-Cuerpos (ABM4) y Proyección Geométrica Besseliana (WGS84 + LOLA)**

![Eclipse](https://img.shields.io/badge/Eclipse-12_Agosto_2026-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3F4F75?style=for-the-badge&logo=matplotlib&logoColor=white)

Este repositorio contiene una simulación de física y geometría celeste de alta precisión desarrollada en Python para predecir las circunstancias geográficas exactas del **Eclipse Solar Total del 12 de agosto de 2026**. 

El proyecto resuelve la astrodinámica del sistema solar utilizando un integrador multipaso **Adams-Bashforth-Moulton de 4º orden (ABM4)** acoplado a correcciones relativistas (EIH) y de marea. Posteriormente, la sombra se proyecta sobre el elipsoide terrestre **WGS84** aplicando el formalismo de los **Elementos Besselianos**, corregido dinámicamente por el desfase de rotación terrestre ($\Delta T$) y la topografía real de la Luna (sensor altimétrico **LOLA de la NASA**) para simular las **Perlas de Baily (Baily's beads)**.

---

## Estructura del Proyecto

```text
📂 predicción_eclipse/
 ├── 📄 model_m1.py             # M1: Problema Newtoniano de 3 cuerpos (Sol-Tierra-Luna)
 ├── 📄 model_m2.py             # M2: N-body Newtoniano completo (10 cuerpos celestes principales)
 ├── 📄 model_m3.py             # M3: N-body + perturbación por achatamiento terrestre J2
 ├── 📄 model_m4.py             # M4: N-body + J2 + correcciones relativistas Einstein-Infeld-Hoffmann (EIH 1PN)
 ├── 📄 model_m5.py             # M5: N-body + J2 + EIH + disipación geomareal (CTL, Mignard 1980)
 ├── 📄 model_s1.py             # S1: Proyección Besseliana clásica sobre WGS84 (Delta T = 0)
 ├── 📄 model_s2.py             # S2: Proyección Besseliana con Delta T dinámico (tablas oficiales USNO)
 ├── 📄 model_s3.py             # S3: Proyección Besseliana + Delta T + Topografía LOLA (Perlas de Baily)
 ├── 📄 benchmark.py            # Core evaluador y comparador de errores frente a efemérides JPL DE441
 ├── 📄 new_data.py             # Descargador de parámetros EOP (IERS), constantes PCK y telemetría LLR
 ├── 📄 api.py & combinar.py    # Scripts para obtención y consolidación de datos JPL Horizons
 ├── 📂 Datos/                  # Directorio con archivos consolidados de efemérides y tablas
 │    ├── initial_conditions.csv       # Condiciones iniciales de efemérides DE441 al 1-Ene-2026
 │    ├── jpl_reference_data.json      # Efemérides JPL DE441 de verdad absoluta para 5 fechas de control
 │    ├── deltat.data & deltat.preds   # Tablas del USNO para el cálculo dinámico de Delta T
 │    ├── ldem_4.img & ldem_4.lbl      # Alturas topográficas lunares del sensor LRO/LOLA (resolución 0.25°)
 └── 📂 legacy/                 # Modelos antiguos redundantes y sandboxes de métodos numéricos (Yoshida O(4))
```

---

## Jerarquía de Modelos Físicos (Parte 1: Órbitas)

Los modelos propagan el estado de los cuerpos celestes desde el **1 de enero de 2026** hasta la fecha del eclipse (**12 de agosto de 2026**) utilizando el método Adams-Bashforth-Moulton de paso cuasi-fijo ($h=0.1$ días). El benchmark evalúa el error absoluto de posición respecto a la efeméride de referencia **JPL DE441**:

1.  **M1 (Newton 3-body)**: Considera únicamente al Sol, la Tierra y la Luna. Error lunar acumulado en el eclipse: **~39,280 km**.
2.  **M2 (N-body completo)**: Añade las perturbaciones gravitacionales de los 7 planetas restantes. Gracias al modelado de Júpiter y Venus, el error de separación orbital Tierra-Luna se reduce drásticamente de ~400 km a **~3.7 km**.
3.  **M3 (N-body + $J_2$)**: Añade el armónico de achatamiento de la Tierra ($J_2 = 0.00108262545$). Afecta principalmente a la órbita de la Luna, reduciendo el error Tierra-Luna acumulado a **~1.0 km**.
4.  **M4 (N-body + $J_2$ + EIH)**: Añade las ecuaciones post-newtonianas de primer orden (1PN) Einstein-Infeld-Hoffmann (las que usa la NASA). Corrige la precesión del perihelio terrestre (que causaba una deriva sistemática de ~57,000 km en la posición absoluta de la Tierra respecto al SSB en los modelos puramente newtonianos), logrando errores residuales en el BCRS de **~100–500 km**.
5.  **M5 (N-body + $J_2$ + EIH + Marea CTL)**: Añade las fuerzas disipativas del bulbo de marea terrestre retardado de la Luna usando el modelo CTL de Mignard (1980) y constante Love $k_2 = 0.3077$, lo que representa la transferencia de momento angular espín-órbita de forma rigurosa.

---

## Jerarquía de Modelos de Proyección (Parte 2: Sombra)

Una vez obtenidas las coordenadas orbitales tridimensionales en el momento del eclipse, los modelos de sombra proyectan el cono umbral y penumbral sobre la Tierra (elipsoide WGS84) mediante los elementos besselianos:

*   **S1 (Bessel clásico - Delta T = 0)**: Asume $\Delta T = 0$, lo que desfasa la proyección geográfica del eclipse al ignorar que la Tierra ha rotado más lento debido al frenado mareal. Predice el centro de la totalidad a las 17:47:06 UTC en la posición **64.3647° N, -24.0838° W** (suroeste de Islandia).
*   **S2 (Bessel + $\Delta T$ dinámico)**: Interpola $\Delta T = TT - UT1$ dinámicamente desde las tablas oficiales del USNO ($\Delta T \approx 69.10$ s para la fecha). Esto corrige el ángulo horario de Greenwich ($\mu$) del eje de la sombra, desplazando la trayectoria real hacia el oeste: **64.3647° N, -23.7951° W**, corrigiendo una deriva geográfica de más de **20 km** en longitud. El radio de la umbra en el plano fundamental se calcula rigurosamente en **53.2 km** (trayectoria total de 106.4 km de ancho).
*   **S3 (Bessel + $\Delta T$ + LOLA)**: Carga y procesa la topografía del sensor de altimetría láser **LOLA** de la sonda LRO de la NASA (`ldem_4.img`). Para el observador situado en el centro de la umbra en España/Islandia, calcula el limbo lunar real de montañas y valles. En los contactos de entrada (C2) y salida (C3), modela las **Perlas de Baily (Baily's beads)** indicando con precisión matemática por qué valles selenográficos la fotosfera solar se filtra y brilla segundos antes de la totalidad.

---

## Requisitos e Instalación

Para clonar y ejecutar este simulador, requiere **Python 3.8+** y las librerías listadas en `requirements.txt`:

```bash
git clone https://github.com/tu_usuario/prediccion_eclipse.git
cd prediccion_eclipse
pip install -r requirements.txt
```

---

## Cómo Ejecutar los Modelos

Para correr cualquiera de los modelos de órbita (M) o de sombra (S) y generar sus reportes respectivos en la terminal:

```bash
# Integrar y evaluar el modelo M1
python model_m1.py

# Integrar y evaluar el modelo M5 (con EIH y Marea)
python model_m5.py

# Proyectar el eclipse con Delta T = 0 (Modelo S1)
python model_s1.py

# Proyectar la sombra real con Delta T interpolado (Modelo S2)
python model_s2.py

# Generar el limbo lunar topográfico y simular perlas de Baily (Modelo S3)
python model_s3.py
```

*Nota: Al ejecutar el modelo S3, se generará de manera automática un archivo de imagen en alta resolución `lunar_limb_bailys_beads.png` que representa el perfil del limbo lunar y las perlas de Baily simuladas.*

---

## Documentación Teórica

Para comprender la deducción analítica de las Lagrangianas de rotación de la Tierra, los ángulos de Euler acoplados a la órbita, el tensor de inercia y los potenciales de EIH, se recomienda encarecidamente consultar:
*   [Diario de Eclipse.pdf](Diario%20de%20Eclipse.pdf) (Documentación matemática definitiva del proyecto).
