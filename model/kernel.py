import math
import scipy.special
import numpy as np


def cal_b_spline(m,x):
    s = math.factorial(m-1)
    sum = 0
    for k in range(m):
        sum = sum + (-1) ** k * (x - k) ** (m - 1)
    return s**(-1)*sum

def cal_derive(m,n,x):
    sum = 0
    for k in range(m):
        sum = sum + (-1) ** k * scipy.special.comb(n, k) * cal_b_spline(m-n,x-k)
    return sum

def cal_spline_wavelet(m,x):
    if m==1:
        if (x>0)&(x<0.5):
            g=1
        elif (x>0.5)&(x<1):
            g=-1
        else:
            g=0

    else:
        s = 2**(-m+1)
        sum = 0
        for k in range(2*m-2):
            sum = sum+(-1) ** k * cal_b_spline(2*m,k+1) * cal_derive(2*m,m,2*x-k)
        g = s * sum
    return g