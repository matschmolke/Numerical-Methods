import numpy as np

#Metoda Newtona-Raphsona
#Przyjmuje funkcję f, jej pochodną df, punkt startowy x0 oraz dokładność epsilon.
def newton_raphson(f, df, x0, epsilon=1e-7, max_iter=100):
    x = x0

    for i in range(max_iter):
        fx = f(x) #Obliczenie wartości funkcji w punkcie x
        dfx = df(x) #Obliczenie wartości pochodnej funkcji w punkcie x

        #Jeśli wartość funkcji jest dostatecznie bliska 0, przerywamy iteracje.
        if abs(fx) <= epsilon:
            break
        x = x - fx/dfx
    return x


#Zadanie 1: Pierwiastek kwadratowy

def f1(x):
    return x**2 - 9.06

def df1(x):
    return 2*x

approximation = newton_raphson(f1, df1, 2)
exact = np.sqrt(9.06)

print(f"Przybliżone rozwiązanie: {approximation}")
print(f"Rozwiązanie z sqrt: {exact}")

#Zadanie 2: Równanie logistyczne

# Definicja funkcji logistycznej oraz jej pochodnej (model wzrostu populacji)
def f_logistic(t, K, r, p_star):
    return K / (1 + np.exp(-r*t)) - p_star

def df_logistic(t, K, r):
    exp_rt = np.exp(-r*t)
    return (K * r * exp_rt) / ((1 + exp_rt)**2)

# Parametry zadania
K = 1000       # Maksymalna populacja
r = 0.1        # Współczynnik wzrostu populacji
p_star = 700   # Docelowa populacja
x0 = 10        # Punkt startowy iteracji

# Obliczenie rozwiązania metodą Newtona-Raphsona
logistic_solution = newton_raphson(lambda t: f_logistic(t, K, r, p_star), lambda t: df_logistic(t, K, r), x0)

print(f"Czas t, dla którego populacja osiągnie poziom {p_star}: {logistic_solution:.4f}")

#Zadanie 3: Układ równań nieliniowych