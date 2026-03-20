import numpy as np
import matplotlib.pyplot as plt

def rk4_sistema(f, t0, u0, t_end, h):
    """RK4 adaptado para sistemas de EDOs (vectores)."""
    N = int(np.ceil((t_end - t0) / h))
    t = np.linspace(t0, t_end, N + 1)
    
    # u es una matriz donde cada fila es el estado [x, y] en un instante t
    u = np.zeros((N + 1, len(u0)))
    u[0] = u0
    
    for n in range(N):
        k1 = f(t[n], u[n])
        k2 = f(t[n] + h/2, u[n] + h * k1 / 2)
        k3 = f(t[n] + h/2, u[n] + h * k2 / 2)
        k4 = f(t[n] + h, u[n] + h * k3)
        
        u[n+1] = u[n] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
    return t, u

# 1. Definir el sistema de ecuaciones
omega = 1.0

def sistema(t, u):
    x, y = u[0], u[1]
    dxdt = y
    dydt = -omega**2 * x
    return np.array([dxdt, dydt])

# 2. Parámetros de integración
t0 = 0.0
t_end = 50.0  # Tiempo largo para evidenciar el error acumulado
h = 0.5       # Paso relativamente grande para acelerar la disipación
u0 = np.array([1.0, 0.0]) # Condición inicial: x=1, y=0

# 3. Integración
t, u = rk4_sistema(sistema, t0, u0, t_end, h)
x = u[:, 0]
y = u[:, 1]

# 4. Cálculo de la cantidad conservada I_n (Energía)
energia = 0.5 * (y**2 + omega**2 * x**2)

# 5. Visualización
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Gráfica del espacio de fases
ax1.plot(x, y, 'b-', linewidth=1)
ax1.set_title('Espacio de Fases $(x, y)$')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax1.grid(True)
ax1.axis('equal')

# Gráfica de la energía
ax2.plot(t, energia, 'r-', linewidth=1.5)
ax2.set_title(r'Evolución de $I_n = (y^2 + \omega^2 x^2)/2$')
ax2.set_xlabel('$t$')
ax2.set_ylabel('$I_n$')
ax2.grid(True)

plt.tight_layout()
plt.show()
