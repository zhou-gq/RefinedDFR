from proba_util import *
from scipy.stats import chi2,ncx2
import numpy as np
from math import factorial, log2
from decimal import *

BND = 300
getcontext().prec = int(0.30103*BND)+10

def law_square_exact(D):
    A={}
    for (a,pa) in D.items():
        A[a**2] = A.get(a**2, Decimal(0)) + pa
    return A

def law_convolution_exact(A,B):
    C = {}
    for (a,pa) in A.items():
        for (b,pb) in B.items():
            c = a+b
            C[c] = C.get(c, Decimal(0)) + (pa * pb) # output in float
    return C

def iter_law_convolution_exact(A, r=0):

    if r <= 0:
        return {0:1}
    D = A
    r_bin = format(r,'b')[1:]  # binary representation of r without the m.s.bit
    for ch in r_bin:
        D = law_convolution_exact(D, D)
        if ch == '1':
            D = law_convolution_exact(D, A)
    return D

def build_centered_binomial_law_exact(k):
    CBD1 = {-1:Decimal(0.25),0:Decimal(0.5),1:Decimal(0.25)}
    return iter_law_convolution_exact(CBD1,k)

def build_rounding_C2_law_exact(q,q2):
    D = {}
    bnd = int(0.5*q/q2)
    D[bnd] = Decimal(q2)/Decimal(2*q)
    D[-bnd] = Decimal(q2)/Decimal(2*q)
    for i in range(-bnd+1,bnd):
        D[i] = Decimal(q2)/Decimal(q)
    return D

def find_num(mu, tail):
    if mu==0:
        return 0
    n=0
    bit = 0
    tmp = log2(mu/2)
    while(True):
        if bit >= (n+1)*tmp+tail:
            break
        else:
            n+=1
            bit+=log2(n)
    return n

ee = Decimal(1)
fac = Decimal(1)
for k in range(1,find_num(2,BND)+1):
    fac *= k
    ee += Decimal(1)/fac

def manual_sf(x,k,lam,N):
    if lam==0:
        return Decimal(chi2.sf(x,k))

    Lam = Decimal(lam)
    res = 0
    for m in range(0,N+1):
        poisson_w = ee**(-Lam/2) * (Lam/2)**m / Decimal(factorial(m))
        chi2_sf = Decimal(chi2.sf(x,df = k+2*m))
        res += poisson_w*chi2_sf
    return res


def dfr_paper(lm,q,q1,q2,m,n,mb,nb,h1,h2,eta1,eta2,mu,tau):
    s2 = h2*eta1 + 2*h1*(eta2/2+(q**2/q1**2-1)/12)
    s2 += (eta2/2+(q**2/q2**2+2)/12)
    bound = 2*q/(2**tau)
    pr = chi2.logsf( bound**2/s2, 32 ) / log(2)
    pr += log2(lm/mu)
    return pr

def dfr_paper_newbound(lm,q,q1,q2,m,n,mb,nb,h1,h2,eta1,eta2,mu,tau):
    s2 = h2*eta1 + 2*h1*(eta2/2+(q**2/q1**2-1)/12)
    bound = 2*q/(2**tau) - sqrt(32)*(eta2+q/(2*q2))
    pr = chi2.logsf( bound**2/s2, 32 ) / log(2)
    pr += log2(lm/mu)
    return pr

def dfr_refined(lm,q,q1,q2,m,n,mb,nb,h1,h2,eta1,eta2,mu,tau):
    s2 = h2*eta1 + 2*h1*(eta2/2+(q**2/q1**2-1)/12)
    DE2 = build_centered_binomial_law_exact(eta2)
    DF2 = build_rounding_C2_law_exact(q, q2)
    D_II = law_convolution_exact(DE2,DF2)
    D_dis = iter_law_convolution_exact(law_square_exact(D_II),32)
    bound = 2*q/(2**tau)
    pr = Decimal(0)
    for (a,pa) in D_dis.items():
        num = find_num(a/s2, BND)
        pr += Decimal(pa) * (manual_sf(bound**2/s2,32,a/s2,num))
    return (log((pr)*lm/mu,2))


############# test function #################

if __name__ == "__main__":
    names = ["Scloud+-128:","Scloud+-192:","Scloud+-256:"]
    lm = [128,192,256]
    q = [4096,4096,4096]
    q1 = [512,4096,1024]
    q2 = [128,1024,128]
    m = [600,928,1136]
    n = [600,896,1120]
    mb = [8,8,12]
    nb = [8,8,11]
    h1 = [150,224,280]
    h2 = [150,232,284]
    eta1 = [7,2,3]
    eta2 = [7,1,2]
    mu = [64,96,64]
    tau = [3,4,3]

    PRINT_DFR_HEAD()
    for i in range(len(names)):
        PRINT_DFR_RESULT(names[i], "paper", dfr_paper(lm[i],q[i],q1[i],q2[i],m[i],n[i],mb[i],nb[i],h1[i],h2[i],eta1[i],eta2[i],mu[i],tau[i]))
        PRINT_DFR_RESULT(names[i], "refined", dfr_refined(lm[i],q[i],q1[i],q2[i],m[i],n[i],mb[i],nb[i],h1[i],h2[i],eta1[i],eta2[i],mu[i],tau[i]))

'''
--------------------------------------------------------------
name                 method          log2(DFR)
--------------------------------------------------------------
Scloud+-128:         paper           -133.21
Scloud+-128:         refined         -133.39
Scloud+-192:         paper           -199.64
Scloud+-192:         refined         -199.65
Scloud+-256:         paper           -263.74
Scloud+-256:         refined         -265.11
'''
