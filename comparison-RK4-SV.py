import numpy as np
import matplotlib.pyplot as plt

# --- 1. Definición de los Integradores ---

def rk4_step(f, t, u, h):
    """Avanza un paso 'h' usando el método de Runge-Kutta 4 estándar."""
    k1 = f(t, u)
    k2 = f(t + h/2, u + h*k1/2)
    k3 = f(t + h/2, u + h*k2/2)
    k4 = f(t + h, u + h*k3)
    return u + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

def stormer_verlet_step(x_n, y_n, h, omega):
    """Avanza un paso 'h' usando el esquema simpléctico de Störmer-Verlet."""
    # Formulation: https://en.wikipedia.org/wiki/Verlet_integration
    y_half = y_n - (h/2) * (omega**2) * x_n
    x_next = x_n + h * y_half
    y_next = y_half - (h/2) * (omega**2) * x_next
    return x_next, y_next

# --- 2. Parámetros del Sistema (EXTREMOS para máxima claridad) ---

omega = 1.0
def sistema(t, u):
    x, y = u[0], u[1]
    return np.array([y, -omega**2 * x])

t0 = 0.0
t_end = 200.0  # Tiempo muy largo
h = 1.0        # PASO MUY GRANDE (pero estable para SV: |h*omega| = 1.0 <= 2.0)
N = int(np.ceil((t_end - t0)/h))
t = np.linspace(t0, t_end, N+1)

# Vectores para guardar la solución
u_rk4 = np.zeros((N+1, 2))
u_sv = np.zeros((N+1, 2))

# Condiciones iniciales (x=1, v=0) -> Energía inicial I_n = 0.5
u0 = np.array([1.0, 0.0])
u_rk4[0] = u0
u_sv[0] = u0

# --- 3. Bucle de Integración ---

for n in range(N):
    # Paso de RK4
    u_rk4[n+1] = rk4_step(sistema, t[n], u_rk4[n], h)
    
    # Paso de Störmer-Verlet
    x_next, y_next = stormer_verlet_step(u_sv[n, 0], u_sv[n, 1], h, omega)
    u_sv[n+1, 0] = x_next
    u_sv[n+1, 1] = y_next

# --- 4. Análisis de la Cantidad I_n (Energía) ---

I_rk4 = 0.5 * (u_rk4[:, 1]**2 + omega**2 * u_rk4[:, 0]**2)
I_sv = 0.5 * (u_sv[:, 1]**2 + omega**2 * u_sv[:, 0]**2)

# --- 5. Visualización ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Espacio de Fases: Ahora se verá "super claro"
# Usamos alpha para ver la superposición
ax1.plot(u_rk4[:, 0], u_rk4[:, 1], 'r-', label='RK4 (Pérdida de Área Catastrófica)', alpha=0.9, linewidth=1.0)
ax1.plot(u_sv[:, 0], u_sv[:, 1], 'b-', label='Störmer-Verlet (Conservación del Área)', alpha=0.6, linewidth=1.5)
ax1.set_title(r'Espacio de Fases EXAGERADO: $(x, y)$ con $h=1.0$')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
ax1.legend(loc='best')
ax1.grid(True)
ax1.axis('equal') # Importante para ver la geometría real

# Evolución de In: Ahora se verá "super claro"
ax2.plot(t, I_rk4, 'r-', label='RK4 (Disipación Numérica Completa)', linewidth=2.0)
ax2.plot(t, I_sv, 'b-', label='Störmer-Verlet', linewidth=2.0)
ax2.set_title(r'Evolución de Energía: $I_n = (y^2 + \omega^2 x^2)/2$')
ax2.set_xlabel(r'$t$')
ax2.set_ylabel(r'$I_n$')
ax2.set_ylim(-0.05, 0.55) # Forzamos el rango para ver el colapso a 0
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()