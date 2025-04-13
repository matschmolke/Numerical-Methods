import numpy as np

def gauss_elimination_pivot(A, b):

    n = len(A)

    A = A.astype(float)
    b = b.astype(float)

    Ab = np.column_stack((A, b)) # Macierz rozszerzona [A|b]

    # Eliminacja Gaussa z pivotowaniem
    for i in range(n):

        # Szukanie wiersza z największym elementem w kolumnie i 
        max_row = np.argmax(abs(Ab[i:, i])) + i # Ab[i:, i ] elementy w kolumnie i od i-tego wiersza w dół 

        # Zamiana aktualnego wiersza z wierszem, który ma największy element w kolumnie i (pivot)
        Ab[[i, max_row]] = Ab[[max_row, i]]

        # Eliminacja współczynników pod przekątną
        for j in range(i + 1, n): # for (int j = i + 1; j < n; j++)

            factor = Ab[j, i] / Ab[i, i] # Współczynnik eliminacji
            
            # Od wiersza j odejmujemy wiersz i (pivotowy), pomnożony przez współczynnik eliminacji
            Ab[j, i:] -= factor * Ab[i, i:]

    # Podstawianie wsteczne – znajdujemy rozwiazania x od ostatniego do pierwszego
    x = np.zeros(n) # macierz wypełnioną zerami o wymiarze n

    for i in range(n - 1, -1, -1): # for(i = n - 1; i > -1; i--)

        # Ab[i, -1] to wartość po prawej stronie (wyraz wolny)
        # np.dot(...) to suma iloczynów współczynników (Ab[i, i + 1:n]) oraz znane x (x[i + 1:n])
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i] 

    return x