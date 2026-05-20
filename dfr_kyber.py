from proba_util import *
from math import log2, erfc

# tatal error = ( vec(e) + vec(rt) ) * vec(r) - vec(s) * ( vec(e1) + vec(ru) ) + e2 + rv,
# where partI = ( vec(e) + vec(rt) ) * vec(r) - vec(s) * ( vec(e1) + vec(ru) ), partII = e2 + rv
# D(vec(a)*vec(b)) = iter_law_convolution( law_product(Da,Db) , N ), where N is length of vector.

# no simulation
def dfr_discrete(q,deg,k,eta1,eta2,dt,du,dv):
    Ds = build_centered_binomial_law(eta1)
    De = build_centered_binomial_law(eta1)
    Dr = build_centered_binomial_law(eta1)
    De1 = build_centered_binomial_law(eta2)
    De2 = build_centered_binomial_law(eta2)
    
    Drt = build_mod_switching_error_law(q, 2**dt)
    Dru = build_mod_switching_error_law(q, 2**du)
    Drv = build_mod_switching_error_law(q, 2**dv)

    D1 = law_product(law_convolution(De,Drt), Dr)
    D2 = law_product(law_convolution(De1,Dru), Ds)
    D_partI = iter_law_convolution(law_substract(D1,D2), deg*k)
    D_partII = law_convolution(De2,Drv)
    D_final = law_convolution(D_partI,D_partII)

    pr = tail_probability(D_final, q/4.)
    return log2(deg*pr)

# simulate total error as a Gaussion distribution
def dfr_continuous(q,deg,k,eta1,eta2,dt,du,dv):
    _,dev_t = dist_meanvar(build_mod_switching_error_law(q, 2**dt))
    _,dev_u = dist_meanvar(build_mod_switching_error_law(q, 2**du))
    _,dev_v = dist_meanvar(build_mod_switching_error_law(q, 2**dv))
    _,dev_1 = dist_meanvar(build_centered_binomial_law(eta1))
    _,dev_2 = dist_meanvar(build_centered_binomial_law(eta2))

    dev = k*deg*dev_1* (dev_1 + dev_t + dev_2 + dev_u) + dev_2 + dev_v
    tau = q/(4*sqrt(2*dev))
    return log2(deg*erfc(tau))

# simulate partI as a Gaussion distribution
def dfr_cont_dis(q,deg,k,eta1,eta2,dt,du,dv):
    _,dev_t = dist_meanvar(build_mod_switching_error_law(q, 2**dt))
    _,dev_u = dist_meanvar(build_mod_switching_error_law(q, 2**du))
    _,dev_1 = dist_meanvar(build_centered_binomial_law(eta1))
    _,dev_2 = dist_meanvar(build_centered_binomial_law(eta2))
    dev = k*deg*dev_1* (dev_1 + dev_t + dev_2 + dev_u)

    De2 = build_centered_binomial_law(eta2)
    Drv = build_mod_switching_error_law(q, 2**dv)
    D_part2 = law_convolution(De2,Drv)
    pr = 0
    for (a,pa) in D_part2.items():
        tau1 = (q/4-a)/sqrt(2*dev)
        tau2 = (q/4+a)/sqrt(2*dev)
        pr += 0.5*pa*(erfc(tau1)+erfc(tau2))
    return log2(deg*pr)


############# test function #################
if __name__ == "__main__":
    names = {0:'Kyber-512',1:'Kyber-768', 2:"Kyber-1024"}
    deg = [256,256,256]
    q = [3329,3329,3329]
    eta1=[3,2,2]
    eta2=[2,2,2]
    r1=[0,0,0]
    r2=[10,10,11]
    r3=[4,4,5]
    ks=[2,3,4]
    
    PRINT_DFR_HEAD()
    for i in range(len(names)):
        PRINT_DFR_RESULT(names[i], "continuous", dfr_continuous(q[i],deg[i],ks[i],eta1[i],eta2[i],r1[i],r2[i],r3[i]))
        PRINT_DFR_RESULT(names[i], "cont+dis", dfr_cont_dis(q[i],deg[i],ks[i],eta1[i],eta2[i],r1[i],r2[i],r3[i]))
        PRINT_DFR_RESULT(names[i], "discrete", dfr_discrete(q[i],deg[i],ks[i],eta1[i],eta2[i],r1[i],r2[i],r3[i]))

'''
--------------------------------------------------------------
name                 method          log2(DFR)
--------------------------------------------------------------
Kyber-512            continuous      -75.83
Kyber-512            cont+dis        -147.43
Kyber-512            discrete        -139.14
Kyber-768            continuous      -81.13
Kyber-768            cont+dis        -172.62
Kyber-768            discrete        -165.24
Kyber-1024           continuous      -145.54
Kyber-1024           cont+dis        -181.11
Kyber-1024           discrete        -175.20
'''