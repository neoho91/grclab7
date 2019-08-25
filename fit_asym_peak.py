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
import cv2
from matplotlib.colors import LogNorm

def asym_peak_fx(x,x0,a,A,gamma0,bg):
    gamma = 2.*gamma0/(1+np.exp(a*(x-x0)))
    G = A*np.exp(-np.square((x-x0)/gamma))
    return bg + G

def asym_peak_fitter(xs,ys,plot=True,can_be_null=True,p0=None):
    if p0 == None:
        x0 = xs[get_nearest_idx_from_list(max(ys),ys)]
        bg = min(ys)
        A = max(ys) - bg
        a = 0.0
        gamma0 = 2*np.abs(x0-xs[get_nearest_idx_from_list(A/2+bg,ys)])
        p0 = (x0,a,A,gamma0,bg)
    
    try:
        popt, pcov = sp.optimize.curve_fit(asym_peak_fx, xs, ys, p0)
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError as e:
        if can_be_null:
            popt = (x0,a,asym,0,bg)
            perr = (np.NaN,np.NaN,np.NaN,np.NaN,np.NaN)
        else:
            raise Exception(e)
    if can_be_null:
        if np.isnan([popt,perr]).any() or (popt<0).any() or (perr/popt>1).any():
            if is_no_peak(ys):
                popt = (x0,a,asym,0,bg)
                perr = (np.NaN,np.NaN,np.NaN,np.NaN,np.NaN)
    if plot:
        plt.figure()
        plt.plot(xs,ys,'o')
        Xs = np.linspace(min(xs),max(xs),100)
        plt.plot(Xs,asym_peak_fx(Xs,*popt))
    return popt, perr
    
def is_no_peak(spec):
    amp = max(spec)-min(spec)
    baseline = np.median(spec)
    SNR = np.abs(amp-baseline)
    return SNR/baseline < 1