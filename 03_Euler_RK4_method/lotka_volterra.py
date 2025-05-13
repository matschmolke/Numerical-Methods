import numpy as np
import matplotlib.pyplot as plt

def lotka_volterra_euler(alpha, beta, delta, gamma, x0, y0, T, h):
    N = int(T/h) #liczba kroków czasowych
    t = np.linspace(0, T, N+1) #wektor czasu od 0 do T z N+1 punktami
    x, y = np.zeros(N+1), np.zeros(N+1)  #wektory populacji: x – ofiary, y – drapieżniki
    x[0], y[0] = x0, y0 #warunki początkowe

    for n in range(N): #pętla symulacji euler - prostsza metoda (jedno przybliżenie pochodnej różnicą skończoną)
        x[n+1] = x[n] + h * (alpha*x[n] - beta*x[n]*y[n]) #równanie ofiar, alfa*x – rozmnażanie ofiar, beta zabojstwa przez drapieżników
        y[n+1] = y[n] + h * (delta*x[n]*y[n] - gamma*y[n]) #równanie drapieżników, delta wzorst drapieznikow, gamma naturalna smierc

    return t, x, y #zwraca wektory czasu oraz populacji

def lotka_volterra_rk4(alpha, beta, delta, gamma, x0, y0, T, h):
    N = int(T/h)
    t = np.linspace(0, T, N+1)
    x, y = np.zeros(N+1), np.zeros(N+1)
    x[0], y[0] = x0, y0

    def dx_dt(x, y): return alpha*x - beta*x*y #pochodna liczby ofiar
    def dy_dt(x, y): return delta*x*y - gamma*y #pochodna liczby drapieżników

    for n in range(N): #RK4 – dokładniejsza metoda niż Euler (4 przybliżenia pośrednie w każdej iteracji)
        k1_x = h * dx_dt(x[n], y[n]) #pochodna na poczatku przedzialu
        k1_y = h * dy_dt(x[n], y[n])
        k2_x = h * dx_dt(x[n] + 0.5*k1_x, y[n] + 0.5*k1_y) #robimy pól kroku do przodu - środek przedzialu
        k2_y = h * dy_dt(x[n] + 0.5*k1_x, y[n] + 0.5*k1_y) #i sprawdzamy, czy prędkość zmian maleje czy rośnie w polowie kroku
        k3_x = h * dx_dt(x[n] + 0.5*k2_x, y[n] + 0.5*k2_y) #schodzimy do środka jeszcze bardziej
        k3_y = h * dy_dt(x[n] + 0.5*k2_x, y[n] + 0.5*k2_y)
        k4_x = h * dx_dt(x[n] + k3_x, y[n] + k3_y) #pochodna na końcu przedzialu
        k4_y = h * dy_dt(x[n] + k3_x, y[n] + k3_y)

        x[n+1] = x[n] + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6 #wyliczamy średnią ze wspólczynników
        y[n+1] = y[n] + (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6

    return t, x, y #zwraca wektory czasu oraz populacji

def plot_comparison_lv(t, xe, ye, xrk, yrk):
    plt.figure(figsize=(12, 6)) #rozmiar wykresu
    plt.plot(t, xe, '--', label='Ofiary Euler', color='orange') #dane
    plt.plot(t, ye, '--', label='Drapieżniki Euler', color='cyan')
    plt.plot(t, xrk, '-', label='Ofiary RK4', color='red')
    plt.plot(t, yrk, '-', label='Drapieżniki RK4', color='blue')
    plt.title('Porównanie Euler vs RK4 (Lotka-Volterra)')
    plt.xlabel('Czas')
    plt.ylabel('Populacja')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_comparison_lv():
    # Parametry początkowe
    alpha, beta, delta, gamma = 0.7, 1.3, 1.0, 1.0 # parametry biologiczne modelu
    x0, y0 = 2, 1 # warunki początkowe
    T = 50 # czas symulacji
    h = 0.01 # krok czasowy

    # Symulacje
    t, xe, ye = lotka_volterra_euler(alpha, beta, delta, gamma, x0, y0, T, h)
    _, xrk, yrk = lotka_volterra_rk4(alpha, beta, delta, gamma, x0, y0, T, h)

    # Wykres
    plot_comparison_lv(t, xe, ye, xrk, yrk)
