import numpy as np
import matplotlib.pyplot as plt
from gauss import gauss_elimination_pivot

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