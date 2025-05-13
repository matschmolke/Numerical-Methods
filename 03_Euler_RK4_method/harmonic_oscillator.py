import numpy as np
import matplotlib.pyplot as plt

def harmonic_euler(omega, x0, v0, T, h):
    N = int(T/h)
    t = np.linspace(0, T, N+1)
    x, v = np.zeros(N+1), np.zeros(N+1)
    x[0], v[0] = x0, v0

    for n in range(N):
        x[n+1] = x[n] + h * v[n]
        v[n+1] = v[n] - h * omega**2 * x[n]

    return t, x, v

def harmonic_rk4(omega, x0, v0, T, h):
    N = int(T/h)
    t = np.linspace(0, T, N+1)
    x, v = np.zeros(N+1), np.zeros(N+1)
    x[0], v[0] = x0, v0

    def dx_dt(v): return v
    def dv_dt(x): return -omega**2 * x

    for n in range(N):
        k1_x = h * dx_dt(v[n])
        k1_v = h * dv_dt(x[n])

        k2_x = h * dx_dt(v[n] + 0.5*k1_v)
        k2_v = h * dv_dt(x[n] + 0.5*k1_x)

        k3_x = h * dx_dt(v[n] + 0.5*k2_v)
        k3_v = h * dv_dt(x[n] + 0.5*k2_x)

        k4_x = h * dx_dt(v[n] + k3_v)
        k4_v = h * dv_dt(x[n] + k3_x)

        x[n+1] = x[n] + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        v[n+1] = v[n] + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6

    return t, x, v

def plot_comparison_oscillator(t, xe, xrk):
    plt.figure(figsize=(12,6))
    plt.plot(t, xe, '--', label='Euler', color='orange')
    plt.plot(t, xrk, '-', label='RK4', color='blue')
    plt.title('Porównanie metod Euler vs RK4 (Drgania harmoniczne)')
    plt.xlabel('Czas [s]')
    plt.ylabel('Wychylenie [m]')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_comparison_oscillator():
    #Parametry
    omega = 2.0     #częstość kołowa
    x0 = 1.0        #początkowe wychylenie
    v0 = 0.0        #początkowa prędkość
    T = 10          #czas symulacji
    h = 0.01        #krok czasowy

    #Symulacje
    t, xe, _ = harmonic_euler(omega, x0, v0, T, h)
    _, xrk, _ = harmonic_rk4(omega, x0, v0, T, h)

    #Wykres
    plot_comparison_oscillator(t, xe, xrk)
