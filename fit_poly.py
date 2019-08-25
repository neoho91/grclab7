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
import warnings
warnings.simplefilter('ignore', np.RankWarning)

def poly_fitter(xs,ys,order=10,plot=True,p0=None):
    popt = np.polyfit(xs,ys,order)
    
    if plot:
        plt.figure()
        plt.plot(xs,ys,'o')
        Xs = np.linspace(min(xs),max(xs),max(100,len(xs)))
        plt.plot(Xs,np.poly1d(popt)(Xs))
    return popt