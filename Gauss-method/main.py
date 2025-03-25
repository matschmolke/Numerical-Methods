import numpy as np
import matplotlib.pyplot as plt

# Zadanie 0: Program rozwiązujący układ równań 3x3 + sprawdzenie wyników metodą np.linalg.solve.
def gauss_elimination_pivot(A, b):
    n = len(A)
    A = A.astype(float) # Zapobiega błędom dzielenia
    b = b.astype(float)
    Ab = np.column_stack((A, b)) # Macierz rozszerzona [A|b]

    # Eliminacja Gaussa z pivotowaniem
    for i in range(n):
        # Pivotowanie: znajdujemy największy element w kolumnie poniżej/powyżej przekątnej (pod względem wart. bezwzgl.)
        max_row = np.argmax(abs(Ab[i:, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]  # Zamiana wierszy

        # Eliminacja współczynników pod przekątną
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i] # Współczynnik eliminacji
            Ab[j, i:] -= factor * Ab[i, i:] # Odejmujemy factor razy

    # Podstawianie wsteczne – znajdujemy rozwiazania x od ostatniego do pierwszego
    x = np.zeros(n) # Tworzy macierz wypełnioną zerami o wymiarze n

    for i in range(n - 1, -1, -1): # for(i = n - 1; i > -1; i--)
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i] # np.dot - mnożenie macierzy

    return x

# Dane do układu 3x3
A_data = np.array(
[[2, -1, 1],
[3, 3, 9],
[3, 3, 5]], dtype=float)

b_data = np.array([8, 6, 7], dtype=float)

# Rozwiązujemy układ 3x3 metodą eliminacji Gaussa (pivoting)
print("Rozwiązanie układu 3x3 metodą eliminacji Gaussa z pivotowaniem:")
print(gauss_elimination_pivot(A_data, b_data))
print("Sprawdzanie wyniku metodą linalg.solve:")
print(np.linalg.solve(A_data, b_data))


# Zadanie 2: Odkształcenia belki z jednej strony zamocowanej
def beam_deformation():
    n = 5  # liczba segmentów
    k = 200  # współczynnik sztywności N\m
    L = 10  # długość całej belki
    l = L / n  # długość jednego segmentu belki

    # Konstruujemy macierz sztywności K
    K = np.zeros((n, n))
    np.fill_diagonal(K, 2)
    np.fill_diagonal(K[1:], -1)
    np.fill_diagonal(K[:, 1:], -1)
    K[-1, -1] = 1  # warunek brzegowy (wolny koniec)
    K *= k / (l ** 2)

    # Tworzymy wektor sił zewnętrznych działających na belkę
    F = np.zeros(n)
    F[-1] = -1000  # obciążenie na ostatnim węźle

    # Rozwiązanie układu równań (znajdujemy wektor przemieszczeń)
    W = gauss_elimination_pivot(K, F)
    print("Przemieszczenia W:", W)

    # Wizualizacja wyników
    plt.figure()
    plt.plot(np.linspace(l, L, n), W, marker='o', linestyle='-', color='blue')
    plt.xlabel("Długość belki [m]")
    plt.ylabel("Ugięcie [m]")
    plt.title("Odkształcenia belki zamocowanej jednostronnie")
    plt.grid(True)
    plt.show()

# Odkształcenia belki zamocowanej jednostronnie
print("Odkształcenia belki zamocowanej jednostronnie:")
beam_deformation()