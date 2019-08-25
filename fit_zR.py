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
import scipy.interpolate

def zR_fx(z,z0,w0,zR): #w0 = HW1/e2
    return w0*np.sqrt(1+np.square(1.0*(z-z0)/zR))

def zR_fitter(xs,ys,plot=True,title=False):
    x_fx = scipy.interpolate.interp1d(ys,xs,fill_value='extrapolate')
    w0 = min(ys)
    z_sqrt2w0 = x_fx(np.sqrt(2)*w0)
    z0 = x_fx(w0)
    zR = abs(z_sqrt2w0 - z0)
    p0 = (z0,w0,zR)
    
    popt, pcov = sp.optimize.curve_fit(zR_fx, xs, ys, p0)
    perr = np.sqrt(np.diag(pcov))
    if plot:
        if title:
            plt.figure(title)
            plt.title(title)
        else:
            plt.figure()
        plt.plot(xs,ys,'o')
        Xs = np.linspace(min(xs),max(xs),100)
        plt.plot(Xs,zR_fx(Xs,*popt))
    return popt, perr
    
