import numpy as np
from gauss import gauss_elimination_pivot
from beam import beam_deformation
from leontief import leontief

# Zadanie 0: Rozwiązujemy układ 3x3 metodą eliminacji Gaussa (pivoting)
# Dane do układu 3x3
A = np.array(
[[2, -1, 1],
[3, 3, 9],
[3, 3, 5]])

b = np.array([8, 6, 7])

print("Rozwiązanie układu 3x3 metodą eliminacji Gaussa z pivotowaniem:")
print(gauss_elimination_pivot(A, b))
print("Sprawdzanie wyniku metodą linalg.solve:")
print(np.linalg.solve(A, b))

# Zadanie 1: Otwarty model ekonomiczny Leontiefa
num_sectors = 10

X = leontief(num_sectors)

print("Zadanie 1: Otwarty model ekonomiczny Leontiefa")
print("\nWektor X (produkcja sektorów):")
print(X)


# Zadanie 2: Odkształcenia belki zamocowanej jednostronnie
print("Odkształcenia belki zamocowanej jednostronnie:")

beam_deformation()

