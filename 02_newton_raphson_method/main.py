import numpy as np

def newton_raphson(f, df, x0, epsilon=1e-7, max_iter=100):
    x = x0

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)

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

#Zadanie 3: Układ równań nieliniowych