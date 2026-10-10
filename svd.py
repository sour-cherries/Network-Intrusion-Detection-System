import numpy as np
import math

def norma(v):
    suma = 0.0
    for i in range(len(v)):
        suma += v[i] ** 2
    return math.sqrt(suma)

def metoda_puterii(M):
    n = np.shape(M)[0]
    v = np.ones(n)
    v = v / norma(v)
    
    for k in range(1000):
        v_nou = np.dot(M, v)
        valoare = np.dot(v, v_nou)
        
        n_v = norma(v_nou)
        if n_v < 1e-12:
            break
        v_nou = v_nou / n_v
        
        diferenta = np.zeros(n)
        suma_v = np.zeros(n)
        for i in range(n):
            diferenta[i] = v_nou[i] - v[i]
            suma_v[i] = v_nou[i] + v[i]
            
        if norma(diferenta) < 1e-8 or norma(suma_v) < 1e-8:
            break
        v = v_nou
        
    return v, valoare

def SVD_trunchiat(A, k):
    n = np.shape(A)[0]
    m = np.shape(A)[1]
    
    U = np.zeros((n, k))
    S = np.zeros(k)
    Vt = np.zeros((k, m))
    
    AT = A.T
    M = np.dot(AT, A)
    
    factor_scalare = np.max(np.abs(M))
    if factor_scalare == 0:
        factor_scalare = 1.0
        
    M_scaled = M / factor_scalare
    Acalc = np.copy(M_scaled)
    
    for p in range(k):
        v, lam_scaled = metoda_puterii(Acalc)
        
        lam_real = lam_scaled * factor_scalare
        
        if lam_real < 1e-10: 
            break
            
        sigma = math.sqrt(lam_real)
        S[p] = sigma
        
        for j in range(m):
            Vt[p, j] = v[j]
            
        for i in range(m):
            for j in range(m):
                Acalc[i, j] = Acalc[i, j] - lam_scaled * v[i] * v[j]
                
        u = np.dot(A, v)
        for i in range(n):
            U[i, p] = u[i] / sigma
            
    return U, S, Vt

if __name__ == "__main__":
    A = np.zeros((4, 3))
    
    A[0, 0] = 1.0; A[0, 1] = 2.0; A[0, 2] = 3.0
    A[1, 0] = 4.0; A[1, 1] = 5.0; A[1, 2] = 6.0
    A[2, 0] = 7.0; A[2, 1] = 8.0; A[2, 2] = 9.0
    A[3, 0] = 10.0; A[3, 1] = 11.0; A[3, 2] = 12.0
    
    print("Matricea initiala A:")
    print(A)
    
    U, S, Vt = SVD_trunchiat(A, 2)
    
    print("\nMatricea U:")
    print(np.round(U, 4))
    
    print("\nValorile singulare S:")
    print(np.round(S, 4))
    
    print("\nMatricea Vt:")
    print(np.round(Vt, 4))
    
    Sigma = np.diag(S)
    A_reconstruit = np.dot(U, np.dot(Sigma, Vt))
    
    print("\nMatricea reconstruita (k=2):")
    print(np.round(A_reconstruit, 2))