# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:21:29 2018

@author: yw_10
"""

import numpy as np
from scipy.stats import norm
import scipy as sp
import sys
import os
sys.path.append(os.getcwd())
from neo_common_code import *
import matplotlib.pyplot as plt

def CDF_fx(x,y0,A,x0,w): #4w = FW1/e2
    return A*norm.cdf(x,loc=x0,scale=w) + y0

def CDF_fitter(xs,ys,plot=True):
    y0 = np.min(ys)
    A = np.max(ys) - y0
    x0 = xs[get_nearest_idx_from_list(A/2.+y0,ys)]
    w = np.abs(x0-xs[get_nearest_idx_from_list(A*0.89+y0,ys)])
    p0 = (y0,A,x0,w)
    
    popt, pcov = sp.optimize.curve_fit(CDF_fx, xs, ys, p0)
    perr = np.sqrt(np.diag(pcov))
    if plot:
        plt.figure()
        plt.plot(xs,ys,'o')
        plt.plot(xs,CDF_fx(xs,*popt))
    return popt, perr
    
