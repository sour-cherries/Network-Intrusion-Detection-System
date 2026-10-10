import numpy as np
import math

def norm(v):
    total = 0.0
    for i in range(len(v)):
        total += v[i] ** 2
    return math.sqrt(total)

def power_method(M):
    n = np.shape(M)[0]
    v = np.ones(n)
    v = v / norm(v)
    
    for k in range(1000):
        v_new = np.dot(M, v)
        value = np.dot(v, v_new)
        
        norm_v = norm(v_new)
        if norm_v < 1e-12:
            break
        v_new = v_new / norm_v
        
        difference = np.zeros(n)
        sum_v = np.zeros(n)
        for i in range(n):
            difference[i] = v_new[i] - v[i]
            sum_v[i] = v_new[i] + v[i]
            
        if norm(difference) < 1e-8 or norm(sum_v) < 1e-8:
            break
        v = v_new
        
    return v, value

def truncated_SVD(A, k):
    n = np.shape(A)[0]
    m = np.shape(A)[1]
    
    U = np.zeros((n, k))
    S = np.zeros(k)
    Vt = np.zeros((k, m))
    
    AT = A.T
    M = np.dot(AT, A)
    
    scale_factor = np.max(np.abs(M))
    if scale_factor == 0:
        scale_factor = 1.0
        
    M_scaled = M / scale_factor
    A_work = np.copy(M_scaled)
    
    for p in range(k):
        v, lam_scaled = power_method(A_work)
        
        lam_real = lam_scaled * scale_factor
        
        if lam_real < 1e-10: 
            break
            
        sigma = math.sqrt(lam_real)
        S[p] = sigma
        
        for j in range(m):
            Vt[p, j] = v[j]
            
        for i in range(m):
            for j in range(m):
                A_work[i, j] = A_work[i, j] - lam_scaled * v[i] * v[j]
                
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
    
    print("Initial matrix A:")
    print(A)
    
    U, S, Vt = truncated_SVD(A, 2)
    
    print("\nMatrix U:")
    print(np.round(U, 4))
    
    print("\nSingular values S:")
    print(np.round(S, 4))
    
    print("\nMatrix Vt:")
    print(np.round(Vt, 4))
    
    Sigma = np.diag(S)
    A_reconstructed = np.dot(U, np.dot(Sigma, Vt))
    
    print("\nReconstructed matrix (k=2):")
    print(np.round(A_reconstructed, 2))