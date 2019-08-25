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

def mallus_fx(theta,A,theta0,R):
    return A*(np.square(np.cos((theta-theta0)/180.*np.pi)) + R*np.square(np.sin((theta-theta0)/180.*np.pi)))

def mallus_fitter(thetas,ys,plot=True,title=False):
    A = max(ys)
    theta0 = thetas[get_nearest_idx_from_list(A,ys)]
    R = A/min(ys)
    p0 = (A,theta0,R)
    
    popt, pcov = sp.optimize.curve_fit(mallus_fx, thetas, ys, p0)
    perr = np.sqrt(np.diag(pcov))
    if plot:
        if title:
            plt.figure(title)
            plt.title(title)
        else:
            plt.figure()
        plt.polar(thetas/180.*np.pi,ys,'o',)
        Xs = np.linspace(min(thetas),max(thetas),100)
        plt.polar(Xs/180.*np.pi,mallus_fx(Xs,*popt))
    return popt, perr
    
