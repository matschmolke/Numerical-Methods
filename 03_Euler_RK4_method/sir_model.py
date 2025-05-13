import numpy as np
import matplotlib.pyplot as plt

def sir_euler(beta, gamma, S0, I0, R0, T, h):
    N = int(T/h)
    t = np.linspace(0, T, N+1)
    S, I, R = np.zeros(N+1), np.zeros(N+1), np.zeros(N+1) #wektory liczby osób w każdej z grup
    S[0], I[0], R[0] = S0, I0, R0
    populacja = S0 + I0 + R0 #calkowita populacja

    for n in range(N): #Obliczamy nowe wartości na podstawie poprzednich
        S[n+1] = S[n] - h * beta * S[n] * I[n] / populacja #maleje gdy ktos zachoruje
        I[n+1] = I[n] + h * (beta * S[n] * I[n] / populacja - gamma * I[n]) #rosnie gdy ktos sie zarazi, maleje kiedy wyzdrowieje
        R[n+1] = R[n] + h * gamma * I[n] #rosnie gdy ktos wyzdrowieje

    return t, S, I, R

def sir_rk4(beta, gamma, S0, I0, R0, T, h):
    N = int(T/h)
    t = np.linspace(0, T, N+1)
    S, I, R = np.zeros(N+1), np.zeros(N+1), np.zeros(N+1)
    S[0], I[0], R[0] = S0, I0, R0
    populacja = S0 + I0 + R0

    #funkcje pochodnych
    def dS_dt(S, I): return -beta * S * I / populacja
    def dI_dt(S, I): return beta * S * I / populacja - gamma * I
    def dR_dt(I):    return gamma * I

    for n in range(N): #W każdej iteracji liczymy 4 oszacowania pochodnych w różnych punktach w obrębie kroku
        k1_S = h * dS_dt(S[n], I[n]) #poczatek przedzialu
        k1_I = h * dI_dt(S[n], I[n])
        k1_R = h * dR_dt(I[n])

        k2_S = h * dS_dt(S[n] + 0.5*k1_S, I[n] + 0.5*k1_I)#srodek
        k2_I = h * dI_dt(S[n] + 0.5*k1_S, I[n] + 0.5*k1_I)
        k2_R = h * dR_dt(I[n] + 0.5*k1_I)

        k3_S = h * dS_dt(S[n] + 0.5*k2_S, I[n] + 0.5*k2_I)#srodek
        k3_I = h * dI_dt(S[n] + 0.5*k2_S, I[n] + 0.5*k2_I)
        k3_R = h * dR_dt(I[n] + 0.5*k2_I)

        k4_S = h * dS_dt(S[n] + k3_S, I[n] + k3_I)#koniec przedzialu
        k4_I = h * dI_dt(S[n] + k3_S, I[n] + k3_I)
        k4_R = h * dR_dt(I[n] + k3_I)

        S[n+1] = S[n] + (k1_S + 2*k2_S + 2*k3_S + k4_S) / 6 #srednia wazona
        I[n+1] = I[n] + (k1_I + 2*k2_I + 2*k3_I + k4_I) / 6
        R[n+1] = R[n] + (k1_R + 2*k2_R + 2*k3_R + k4_R) / 6

    return t, S, I, R

def plot_comparison_sir(t, Se, Ie, Re, Srk, Irk, Rrk):
    plt.figure(figsize=(12, 6))
    plt.plot(t, Se, '--', color='lightblue', label='Podatni (Euler)')
    plt.plot(t, Ie, '--', color='orange', label='Zakażeni (Euler)')
    plt.plot(t, Re, '--', color='lightgreen', label='Ozdrowieńcy (Euler)')

    plt.plot(t, Srk, '-', color='blue', label='Podatni (RK4)')
    plt.plot(t, Irk, '-', color='red', label='Zakażeni (RK4)')
    plt.plot(t, Rrk, '-', color='green', label='Ozdrowieńcy (RK4)')

    plt.title('Porównanie metod Euler vs RK4 (Model SIR)')
    plt.xlabel('Czas [dni]')
    plt.ylabel('Liczba osób')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_comparison_sir():
    #Parametry
    beta, gamma = 0.15, 0.05 #wskaznik transmisji wirusa, powrotu do zdrowia
    S0, I0, R0 = 1000, 1, 0 #warunki początkowe
    T, h = 400, 0.01 #czas symulacji, krok

    #Symulacje
    t, Se, Ie, Re = sir_euler(beta, gamma, S0, I0, R0, T, h)
    _, Srk, Irk, Rrk = sir_rk4(beta, gamma, S0, I0, R0, T, h)

    #Wykres
    plot_comparison_sir(t, Se, Ie, Re, Srk, Irk, Rrk)
