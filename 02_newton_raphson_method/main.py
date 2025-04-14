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

def newton_raphson_system(F, J, X0, epsilon=1e-7, max_iter=50):
    X = np.array(X0, dtype=float)
    history = [X.copy()]
    for _ in range(max_iter):
        d = np.linalg.solve(J(X), -F(X))
        X += d
        history.append(X.copy())
        if np.linalg.norm(d) < epsilon:
            break
    return X, np.array(history)

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


# Przykładowy start
X0 = [1.5, 1.5]
solution, history = newton_raphson_system(F, J, X0)

print("Rozwiązanie układu:", solution)

# Rysowanie
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-2, 2, 400)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

Z1 = X_grid**2 + Y_grid**2 - 1
Z2 = 0.5*(X_grid + 1)**2 - Y_grid

plt.contour(X_grid, Y_grid, Z1, levels=[0], colors='blue', label='f1')
plt.contour(X_grid, Y_grid, Z2, levels=[0], colors='green', label='f2')
plt.plot(history[:, 0], history[:, 1], 'ro--', label='iteracje')
plt.legend(['Iteracje'])
plt.title('Metoda Newtona-Raphsona dla układu równań')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axis('equal')
plt.show()