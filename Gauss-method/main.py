import numpy as np
from gauss import gauss_elimination_pivot
from beam import beam_deformation
from leontief import leontief, generate_rand_matrix_and_vector

# Zadanie 0: Rozwiązujemy układ 3x3 metodą eliminacji Gaussa (pivoting)
# Dane do układu 3x3
A_data = np.array(
[[2, -1, 1],
[3, 3, 9],
[3, 3, 5]], dtype=float)

b_data = np.array([8, 6, 7], dtype=float)

print("Rozwiązanie układu 3x3 metodą eliminacji Gaussa z pivotowaniem:")
print(gauss_elimination_pivot(A_data, b_data))
print("Sprawdzanie wyniku metodą linalg.solve:")
print(np.linalg.solve(A_data, b_data))

# Zadanie 1: Otwarty model ekonomiczny Leontiefa
num_sectors = 10

A, D = generate_rand_matrix_and_vector(num_sectors)

X = leontief(A, D)

print("Zadanie 1:")
print("Macierz A:")
print(A)
print("\nWektor D:")
print(D)
print("\nWektor X (produkcja sektorów):")
print(X)


# Zadanie 2: Odkształcenia belki zamocowanej jednostronnie
print("Odkształcenia belki zamocowanej jednostronnie:")
beam_deformation()