import numpy as np
from dfr_CNTR_Prime import *


n = 256
eta = 2


modulus = [0 for _ in range(n+1)]
modulus[0] = 1
modulus[n-1] = -1
modulus[n] = -1

def poly_mul(a,b):
    prod = np.polymul(a,b)
    _, res = np.polydiv(prod, modulus)
    res = list(map(int,res))[::-1] + [0] * (n - len(res))
    return res

def generate_poly_CBD(k):
    # bits = np.random.randint(0,2,2*k*n).tolist()
    bits = np.random.randint(0,2,2*k*n)
    poly = [0 for _ in range(n)]
    for i in range(n):
        for j in range(k):
            poly[i] += (bits[2*k*i+2*j] - bits[2*k*i+2*j+1])
    return poly

def dfr_test_noalg(k, bound):
    sigma_1 = k/2
    sigma_2 = k/2
    s1 =  sigma_1 * sigma_2
    Mat = covariance_matrix(n,1,0)

    pr = 0
    for i in range(0,int(n/8)):
        V_diag = np.diag(Mat)[(8*i):(8*i+8)]
        pr += Saddlepoint_approximation(V_diag,(bound)**2 / s1)
    return log2(pr)

def dfr_test_alg(k, bound):
    sigma_1 = k/2
    sigma_2 = k/2
    s1 =  sigma_1 * sigma_2
    Mat = covariance_matrix(n,1,0)

    pr = 0
    for i in range(0,int(n/8)):
        M = Mat[(8*i):(8*i+8), (8*i):(8*i+8)]
        M_eigen = np.linalg.eigvalsh(M)
        pr += Saddlepoint_approximation(M_eigen,(bound)**2 / s1)
    return log2(pr)

def dfr_test_real(k,bound, iter, PRINT):
    num = 0
    bound2 = bound**2
    for i in range(iter):
        f = generate_poly_CBD(k)
        g = generate_poly_CBD(k)
        prod = poly_mul(f,g)
        for j in range(int(n/8)):
            lst = prod[8*j:8*j+8]
            norm2 = sum(c**2 for c in lst)
            if norm2 >= bound2:
                num = num + 1
                break
        if PRINT and (i%1000 ==0):
            print(i)
    return log2(num/iter)

def test_func_small(bound):
    dfr_noalg = dfr_test_noalg(eta,bound)
    dfr_alg = dfr_test_alg(eta, bound)
    print("bound = %d, dfr_noalg = %.2f, dfr_alg = %.2f" % (bound, dfr_noalg, dfr_alg))

    # iter = ceil((-dfr_noalg-dfr_alg)/2)
    iter = ceil(max(-dfr_noalg, dfr_alg))
    dfr_real = dfr_test_real(eta, bound,2**iter, False)
    print("iter = 2^%d, dfr_real = %.2f" % (iter, dfr_real), end="\n\n")

def test_func_large(bound):
    dfr_noalg = dfr_test_noalg(eta,bound)
    dfr_alg = dfr_test_alg(eta, bound)
    print("bound = %d, dfr_noalg = %.2f, dfr_alg = %.2f" % (bound, dfr_noalg, dfr_alg))

    iter = ceil((-dfr_noalg-dfr_alg)/2)
    # iter = ceil(max(-dfr_noalg, dfr_alg))
    dfr_real = dfr_test_real(eta, bound,2**iter, False)
    print("iter = 2^%d, dfr_real = %.2f" % (iter, dfr_real), end="\n\n")


if __name__ == "__main__":

    for bound in range(110,161,10):
        test_func_small(bound)

    for bound in range(170,201,10):
        test_func_large(bound)

'''
bound = 110, dfr_noalg = -5.16, dfr_alg = -2.84
iter = 2^6, dfr_real = -3.42

bound = 120, dfr_noalg = -7.89, dfr_alg = -4.63
iter = 2^8, dfr_real = -5.19

bound = 130, dfr_noalg = -10.93, dfr_alg = -6.57
iter = 2^11, dfr_real = -7.42

bound = 140, dfr_noalg = -14.29, dfr_alg = -8.67
iter = 2^15, dfr_real = -9.09

bound = 150, dfr_noalg = -17.96, dfr_alg = -10.91
iter = 2^18, dfr_real = -11.15

bound = 160, dfr_noalg = -21.94, dfr_alg = -13.30
iter = 2^22, dfr_real = -13.42

bound = 170, dfr_noalg = -26.23, dfr_alg = -15.83
iter = 2^22, dfr_real = -15.36
'''
