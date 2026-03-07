import numpy as np
import matplotlib.pyplot as plt

def rk4(f, x0, y0, x_end, h):
    """
    f: función matemática de la EDO
    t0: valor inicial de la variable independiente
    y0: valor inicial de la variable dependiente
    t_end: valor final de la variable independiente
    h: tamaño del paso de integración

    Devuelve:
    t: array de numpy con los valores de la variable independiente
    y: array de numpy con los valores calculados de la solución
    """

    N = int(np.ceil((x_end - x0) / h))

    x = np.linspace(x0, x_end, N + 1)
    y = np.zeros(N + 1)

    y[0] = y0

    for n in range(N):
        k1 = f(x[n], y[n])
        k2 = f(x[n] + h/2, y[n] + h * k1 / 2)
        k3 = f(x[n] + h/2, y[n] + h * k2 / 2)
        k4 = f(x[n] + h, y[n] + h * k3)

        y[n+1] = y[n] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return x, y


# 1. Definir la función de la EDO
def ecuacion_diferencial(x, y):
    return -2 * x * y

# 2. Configurar los parámetros de integración
x_inicial = 0.0
y_inicial = 1.0
x_final = 2.0
paso_h = 0.1

# 3. Llamar a la función RK4
x_aprox, y_aprox = rk4(ecuacion_diferencial, x_inicial, y_inicial, x_final, paso_h)

# 4. Calcular la solución exacta para comparar
x_exacta = np.linspace(x_inicial, x_final, 100)
y_exacta = np.exp(-x_exacta**2)

# 5. Visualizar los resultados
plt.figure(figsize=(8, 5))
plt.plot(x_exacta, y_exacta, 'k-', label='Solución Exacta $e^{-t^2}$')
plt.plot(x_aprox, y_aprox, 'ro', markersize=4, label='Aproximación RK4')
plt.title("Resolución de dy/dt = -2ty mediante RK4")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid(True)
plt.show()