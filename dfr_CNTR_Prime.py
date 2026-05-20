from math import sqrt, log, log2, erfc
from scipy.stats import chi2
from scipy.optimize import bisect
import numpy as np
from proba_util import *
from dfr_CNTR import build_rounding_law_epsilon


def covariance_matrix(n):
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            M[i,j]=0
    for i in range(n):
        if i==0:
            M[i,i]=n
        else:
            M[i,i]=(2*n-i)
    for i in range(n-1):
        M[i,i+1]= (n-i-1)
        M[i+1,i]= (n-i-1)
    return M

def Satterthwaite_approximation(lst,B):
    lst2 = [lst[i]**2 for i in range(len(lst))]
    coef = sum(lst2) / sum(lst) 
    deg = sum(lst)**2 / sum(lst2)
    pr = chi2.sf( B / coef , deg )
    return pr

def Saddlepoint_approximation(lst,B):
    sigma_max = max(lst)
    t_high = 0.9999/(2*sigma_max)
    t_low = 0
    def K_prime(x):
        res = -B
        for sigma in lst:
            res += sigma / (1-2*sigma*x)
        return res
    if K_prime(t_high)<0 and K_prime(t_low)<0:
        t_root = t_high
    elif K_prime(t_high)>0 and K_prime(t_low)>0:
        t_root = t_low
    else:
        t_root = bisect(K_prime,t_low,t_high,xtol=1e-12)
        
    w = t_root * B
    for sigma in lst:
        w += 0.5* log(1-2*sigma*t_root)
    w = sqrt(2*w)
    
    v = 0
    for sigma in lst:
        v+= 2*sigma**2 / (1  - 2*sigma*t_root)
    v = t_root * sqrt(v)
    
    pr = 0.5 * erfc((w + log(v/w)/w)/sqrt(2))
    return pr    
    
# use the maximal diagonal element
def dfr_CNTR_prime_noalg_naive(n,lm,q,eta1,eta2,q2):
    _,sigma_epsilon = dist_meanvar( build_rounding_law_epsilon(q,q2) )
    sigma_1 = eta1/2
    sigma_2 = eta2/2
    s1 =  sigma_1 * sigma_2 +  (2*q/q2)**2*sigma_epsilon*sigma_1
    Mat = covariance_matrix(n)
    bound = 0.5*q - sqrt(2)*q/q2

    pr = 0
    for i in range(0,int(lm/4)):
        V_diag = np.diag(Mat)[(8*i):(8*i+8)]
        pr += chi2.sf( (bound)**2 / max(V_diag) / s1, 8 )
        # pr += chi2.sf( (bound)**2 / min(V_diag) / s1, 8 )
    return log2(pr)

def dfr_CNTR_prime_noalg_Satterthwaite(n,lm,q,eta1,eta2,q2):
    _,sigma_epsilon = dist_meanvar( build_rounding_law_epsilon(q,q2) )
    sigma_1 = eta1/2
    sigma_2 = eta2/2
    s1 =  sigma_1 * sigma_2 +  (2*q/q2)**2*sigma_epsilon*sigma_1
    Mat = covariance_matrix(n)
    bound = 0.5*q - sqrt(2)*q/q2

    pr = 0
    for i in range(0,int(lm/4)):
        V_diag = np.diag(Mat)[(8*i):(8*i+8)]
        pr += Satterthwaite_approximation(V_diag, (bound)**2 / s1)
    return log2(pr)

def dfr_CNTR_prime_noalg_Saddlepoint(n,lm,q,eta1,eta2,q2):
    _,sigma_epsilon = dist_meanvar( build_rounding_law_epsilon(q,q2) )
    sigma_1 = eta1/2
    sigma_2 = eta2/2
    s1 =  sigma_1 * sigma_2 +  (2*q/q2)**2*sigma_epsilon*sigma_1
    Mat = covariance_matrix(n)
    bound = 0.5*q - sqrt(2)*q/q2

    pr = 0
    for i in range(0,int(lm/4)):
        V_diag = np.diag(Mat)[(8*i):(8*i+8)]
        pr += Saddlepoint_approximation(V_diag,(bound)**2 / s1)
    return log2(pr)

def dfr_CNTR_prime_alg_Satterthwaite(n,lm,q,eta1,eta2,q2):
    _,sigma_epsilon = dist_meanvar( build_rounding_law_epsilon(q,q2) )
    sigma_1 = eta1/2
    sigma_2 = eta2/2
    s1 =  sigma_1 * sigma_2 +  (2*q/q2)**2*sigma_epsilon*sigma_1
    Mat = covariance_matrix(n)
    bound = 0.5*q - sqrt(2)*q/q2

    pr = 0
    for i in range(0,int(lm/4)):
        M = Mat[(8*i):(8*i+8), (8*i):(8*i+8)]
        M_eigen = np.linalg.eigvalsh(M)
        pr += Satterthwaite_approximation(M_eigen, (bound)**2 / s1)
    return log2(pr)

def dfr_CNTR_prime_alg_Saddlepoint(n,lm,q,eta1,eta2,q2):
    _,sigma_epsilon = dist_meanvar( build_rounding_law_epsilon(q,q2) )
    sigma_1 = eta1/2
    sigma_2 = eta2/2
    s1 =  sigma_1 * sigma_2 +  (2*q/q2)**2*sigma_epsilon*sigma_1
    Mat = covariance_matrix(n)
    bound = 0.5*q - sqrt(2)*q/q2

    pr = 0
    for i in range(0,int(lm/4)):
        M = Mat[(8*i):(8*i+8), (8*i):(8*i+8)]
        M_eigen = np.linalg.eigvalsh(M)
        pr += Saddlepoint_approximation(M_eigen,(bound)**2 / s1)
    return log2(pr)


if __name__ == "__main__":
    names = {0:'CNTR-Prime-653', 1:'CNTR-Prime-761',2:'CNTR-Prime-1277'}
    q = [4621,4591,7879]
    n = [653,761,1277]
    lm = [320,376,512]
    q2 = [2**11,2**10,2**10]
    eta1=[3,2,2]
    eta2=[3,2,2]

    PRINT_DFR_HEAD()
    for i in range(len(names)):
        # PRINT_DFR_RESULT(names[i], "noalg_naive", dfr_CNTR_prime_noalg_naive(n[i],lm[i],q[i],eta1[i],eta2[i],q2[i]))
        # PRINT_DFR_RESULT(names[i], "noalg_Sat", dfr_CNTR_prime_noalg_Satterthwaite(n[i],lm[i],q[i],eta1[i],eta2[i],q2[i]))
        PRINT_DFR_RESULT(names[i], "noalg_Sad", dfr_CNTR_prime_noalg_Saddlepoint(n[i],lm[i],q[i],eta1[i],eta2[i],q2[i]))
        # PRINT_DFR_RESULT(names[i], "alg_Sat", dfr_CNTR_prime_alg_Satterthwaite(n[i],lm[i],q[i],eta1[i],eta2[i],q2[i]))
        PRINT_DFR_RESULT(names[i], "alg_Sad", dfr_CNTR_prime_alg_Saddlepoint(n[i],lm[i],q[i],eta1[i],eta2[i],q2[i]))

'''
--------------------------------------------------------------
name                 method          log2(DFR)
--------------------------------------------------------------
CNTR-Prime-653       noalg_Sad       -590.07
CNTR-Prime-653       alg_Sad         -314.48
CNTR-Prime-761       noalg_Sad       -300.16
CNTR-Prime-761       alg_Sad         -162.62
CNTR-Prime-1277      noalg_Sad       -187.91
CNTR-Prime-1277      alg_Sad         -103.03
'''