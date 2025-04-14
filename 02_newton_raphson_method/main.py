import numpy as np
import matplotlib.pyplot as plt

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

def nonlinear_newton_raphson(F, J, X0, epsilon=1e-7, max_iter=50):
    X = np.array(X0, dtype=float)

    iterations = [X.copy()]

    for _ in range(max_iter):

        d = np.linalg.solve(J(X), -F(X))
        X += d

        iterations.append(X.copy()) # Zapisywanie przybliżeń do wyświetlenia

        if np.linalg.norm(d) < epsilon:
            break
    return X, np.array(iterations)

#Wektor funkcji
def F(X):
    x, y = X
    return np.array([
        x**2 + y**2 - 1,
        0.5 * (x + 1)**2 - y
    ])

#Macierz Jacobiego
def J(X):
    x, y = X
    return np.array([
        [2*x, 2*y],
        [x + 1, -1]
    ])


# Rozwiązanie 1
X0 = [1.5, 1.5]
solution1, iterations1 = nonlinear_newton_raphson(F, J, X0)

# Rozwiązanie 2
X0 = [1, 0.5]
solution2, iterations2 = nonlinear_newton_raphson(F, J, X0)

# Rozwiązanie 3
X0 = [0, -1]
solution3, iterations3 = nonlinear_newton_raphson(F, J, X0)

print("Rozwiązanie 1 układu:", solution1)
print("Rozwiązanie 2 układu:", solution2)
print("Rozwiązanie 3 układu:", solution3)

# Wykres
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-2, 2, 400)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

Z1 = X_grid**2 + Y_grid**2 - 1
Z2 = 0.5*(X_grid + 1)**2 - Y_grid

plt.contour(X_grid, Y_grid, Z1, levels=[0], colors='blue')
plt.contour(X_grid, Y_grid, Z2, levels=[0], colors='green')
plt.plot(iterations1[:, 0], iterations1[:, 1], 'ro--', label='Rozwiązanie 1 (start [1.5, 1.5])')
plt.plot(iterations2[:, 0], iterations2[:, 1], 'mo--', label='Rozwiązanie 2 (start [1, 0.5])')
plt.plot(iterations3[:, 0], iterations3[:, 1], 'yo--', label='Rozwiązanie 3 (start [0, -1])')
plt.title('Metoda Newtona-Raphsona dla układu równań')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()