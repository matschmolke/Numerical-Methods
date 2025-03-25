import numpy as np
from gauss import gauss_elimination_pivot

# Zadanie 1: Otwarty model ekonomiczny Leontiefa
def generate_rand_matrix_and_vector(n):
    A = np.random.rand(n, n)  # Macierz o wymiarach nxn z losowymi wartościami

    for i in range(n):
        column_sum = np.sum(A[:, i])  # Suma wartości kolumny macierzy

        # Skaluje kolumnę, aby jej suma była mniejsza niż 1
        if column_sum >= 1:
            A[:, i] = A[:, i] / (column_sum + 0.5)  # Skala dla wartości, aby suma była mniejsza niż 1

    D = np.random.randint(20, 100, size=(n, 1))  # Wektor z wartościami popytu końcowego

    return A, D

def leontief(A, D):
    n = len(A)

    if n != len(D):
        return

    I = np.eye(n)  # macierz jednostkowa kwadratowa
    IA = I - A

    # Rozwiązanie układu (I-A)X = D
    X = gauss_elimination_pivot(IA, D)
    return X