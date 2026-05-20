from scipy.stats import chi2
from math import erfc, log, log2
from proba_util import *


def build_rounding_law_epsilon(q,p):
    D = {}
    for u in range(0, q):    
        temp = p/q*u
        epsilon = round( temp ) - temp
        D[epsilon] = D.get(epsilon,0)+1./q 
    return D 

def dfr_CNTR_paper(n,q,eta1,eta2,q2):
    _, var_eps = dist_meanvar( build_rounding_law_epsilon(q,q2) )
    var1 = eta1/2
    var2 = eta2/2
    s1 =  1.5*n*(var1*var2 + (2*q/q2)**2 * var_eps * var1)
    s1 += (q/q2)**2* var_eps
    pr = chi2.logsf( (0.5*q)**2 / s1, 8 ) / log(2) 
    return pr+log2(n/8)

def dfr_CNTR_paper_newbound(n,q,eta1,eta2,q2):
    _, var_eps = dist_meanvar( build_rounding_law_epsilon(q,q2) )
    var1 = eta1/2
    var2 = eta2/2
    s1 =  1.5*n*(var1*var2 + (2*q/q2)**2 * var_eps * var1)
    pr = chi2.logsf( (0.5*q-sqrt(2)*q/q2)**2 / s1, 8 ) / log(2) 
    return pr+log2(n/8)

def dfr_CNTR_voronoi(n,q,eta1,eta2,q2):
    _, var_eps = dist_meanvar( build_rounding_law_epsilon(q,q2) )
    var1 = eta1/2
    var2 = eta2/2
    s1 =  1.5*n*(var1*var2 + (2*q/q2)**2 * var_eps * var1)
    s2 = (q/q2)**2* var_eps
    s = sqrt(4*(s1 + s2))
    tau = q/(sqrt(2)*s)
    return log2(224*0.5*erfc(tau))+log2(n/8)


############# test function #################

if __name__ == "__main__":
    names = {0:'CNTR-512',1:'CNTR-768', 2:"CNTR-1024"}
    q = [3457,3457,3457]
    n = [512,768,1024]
    eta1=[5,3,2]
    eta2=[5,3,2]
    q2=[2**10, 2**10, 2**10]

    PRINT_DFR_HEAD()
    for i in range(len(names)):
        PRINT_DFR_RESULT(names[i], "paper", dfr_CNTR_paper(n[i],q[i],eta1[i],eta2[i],q2[i]))
        PRINT_DFR_RESULT(names[i], "voronoi", dfr_CNTR_voronoi(n[i],q[i],eta1[i],eta2[i],q2[i]))

'''
--------------------------------------------------------------
name                 method          log2(DFR)
--------------------------------------------------------------
CNTR-512             paper           -153.89
CNTR-512             voronoi         -169.68
CNTR-768             paper           -209.26
CNTR-768             voronoi         -226.45
CNTR-1024            paper           -264.91
CNTR-1024            voronoi         -283.19
'''
