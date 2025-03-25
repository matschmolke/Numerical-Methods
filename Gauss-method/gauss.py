import numpy as np

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