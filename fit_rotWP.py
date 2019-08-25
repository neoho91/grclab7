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

def rotWPpara_fx(x,A,tilt,retard):
    return A/4*(3+np.cos(2*np.pi*retard)
                +2*np.cos(2*(-2*tilt+2*x)/180.*np.pi)*np.square(np.sin(np.pi*retard)))

def rotWPperp_fx(x,A,tilt,retard):
    return A*np.square(np.sin(np.pi*retard)*np.sin(2*(tilt-x)/180.*np.pi))

def rotWPpara_fitter(xs,ys,plot=True,title=False):
    A = max(ys)
    tilt = 0.
    retard = 0.3
    p0 = (A,tilt,retard)
    
    popt, pcov = sp.optimize.curve_fit(rotWPpara_fx, xs, ys, p0)
    perr = np.sqrt(np.diag(pcov))
    if plot:
        if title:
            plt.figure(title)
            plt.title(title)
        else:
            plt.figure()
        plt.polar(xs/180.*np.pi,ys,'o')
        Xs = np.linspace(min(xs),max(xs),100)
        plt.polar(Xs/180.*np.pi,rotWPpara_fx(Xs,*popt))
    return popt, perr
    
def rotWPperp_fitter(xs,ys,plot=True,title=False):
    A = max(ys)
    tilt = 0.
    retard = 0.3
    p0 = (A,tilt,retard)
    
    popt, pcov = sp.optimize.curve_fit(rotWPperp_fx, xs, ys, p0)
    perr = np.sqrt(np.diag(pcov))
    if plot:
        if title:
            plt.figure(title)
            plt.title(title)
        else:
            plt.figure()
        plt.polar(xs/180.*np.pi,ys,'o')
        Xs = np.linspace(min(xs),max(xs),100)
        plt.polar(Xs/180.*np.pi,rotWPperp_fx(Xs,*popt))
    return popt, perr

def is_bad_fitting(fitting_results):
    popt,perr=fitting_results
    return np.isnan([popt,perr]).any() or (perr/popt>1).any()

def rotWP_fitter(xs,ys,plot=True,title=False):
    fitting_results = rotWPpara_fitter(xs,ys,plot,title)
    if is_bad_fitting(fitting_results):
        fitting_results = rotWPperp_fitter(xs,ys,plot,title)
    return fitting_results

def error_fx(p,xs,ypara,yperp):
    A,tilt,retard=p
    para_err = np.abs(rotWPpara_fx(xs,A,tilt,retard) - ypara)
    perp_err = np.abs(rotWPperp_fx(xs,A,tilt,retard) - yperp)
    return (para_err + perp_err)

def rotWP2_fitter(xs,ypara,yperp,plot=True,title=False):
    A = max(ypara)
    tilt = 0.
    retard = 0.3
    p0 = (A,tilt,retard)
    pfit, pcov, infodict, errmsg, success = sp.optimize.leastsq(error_fx,p0,args=(xs,ypara,yperp),full_output=True)
    if (len(ypara) > len(p0)) and pcov is not None:
        s_sq = (error_fx(pfit, xs, ypara, yperp)**2).sum()/(len(ypara)-len(p0))
        pcov = pcov * s_sq
    else:
        pcov = np.inf

    error = [] 
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_leastsq = pfit
    perr_leastsq = np.array(error) 
    
    if plot:
        if title:
            plt.figure(title)
            plt.title(title)
        else:
            plt.figure()
        Xs = np.linspace(min(xs),max(xs),100)
        plt.polar(xs/180.*np.pi,ypara,'o')
        plt.polar(Xs/180.*np.pi,rotWPpara_fx(Xs,*pfit_leastsq))
        plt.polar(xs/180.*np.pi,yperp,'o')
        plt.polar(Xs/180.*np.pi,rotWPperp_fx(Xs,*pfit_leastsq))
    return pfit_leastsq, perr_leastsq