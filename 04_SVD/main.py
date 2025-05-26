import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ZADANIE 1: REDUKCJA WYMIAROWOŚCI MACIERZY

# Krok 1: Wygeneruj losową macierz A 10x10
A = np.random.rand(10, 10)

# Krok 2: Oblicz SVD
U, S, VT = np.linalg.svd(A)

# Krok 3 i 4: Rekonstrukcja i obliczenie błędu
errors = []

for r in range(1, 11):
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    VTr = VT[:r, :]
    A_approx = Ur @ Sr @ VTr
    error = np.linalg.norm(A - A_approx)
    errors.append(error)

# Wykres
plt.plot(range(1, 11), errors, marker='o')
plt.xlabel('Liczba wartości osobliwych (r)')
plt.ylabel('Błąd aproksymacji (norma)')
plt.title('Błąd aproksymacji a liczba wartości osobliwych')
plt.grid(True)
plt.show()

#ZADANIE 2: KOMPRESJA OBRAZU W SKALI SZAROŚCI

# Wczytaj obraz w skali szarości

image_path_gray = os.path.join(os.path.dirname(__file__), "OIP.jpg")
img_gray = Image.open(image_path_gray).convert('L')

A = np.array(img_gray)

# SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)

# Funkcja do kompresji
def compress_svd(U, S, VT, r):
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    VTr = VT[:r, :]
    return Ur @ Sr @ VTr

# Kompresja z r = 10, 50, 100
r_values = [10, 50, 100]
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, r in enumerate(r_values):
    A_compressed = compress_svd(U, S, VT, r)
    axs[i].imshow(A_compressed, cmap='gray')
    axs[i].set_title(f'r = {r}')
    axs[i].axis('off')

plt.tight_layout()
plt.show()

#ZADANIE 3: KOMPRESJA OBRAZÓW KOLOROWYCH

image_path_color = os.path.join(os.path.dirname(__file__), "cat.jpg")
img = Image.open(image_path_color)

img_np = np.array(img)  # kształt: (wysokość, szerokość, 3)

# Funkcja do wykonania SVD i rekonstrukcji jednego kanału (R/G/B)
def compress_channel(channel, k):
    U, S, VT = np.linalg.svd(channel, full_matrices=False)
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    VT_k = VT[:k, :]
    compressed = np.dot(U_k, np.dot(S_k, VT_k))
    return np.clip(compressed, 0, 255)  # wartości muszą być w zakresie [0, 255]

# Liczba wartości osobliwych do zachowania
k_values = [10, 30, 100]

fig, axs = plt.subplots(1, len(k_values), figsize=(15, 5))

# Kompresje z różnymi wartościami k
for i, k in enumerate(k_values):
    # Rozdziel kanały
    R = img_np[:, :, 0]
    G = img_np[:, :, 1]
    B = img_np[:, :, 2]

    R_comp = compress_channel(R, k)
    G_comp = compress_channel(G, k)
    B_comp = compress_channel(B, k)

    # Składanie z powrotem
    compressed_img = np.stack([R_comp, G_comp, B_comp], axis=2).astype(np.uint8)

    axs[i].imshow(compressed_img)
    axs[i].set_title(f'k = {k}')
    axs[i].axis('off')


plt.tight_layout()
plt.show()