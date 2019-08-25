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

def _rotPthenA_fx(pol):
    return lambda x,A,tilt,retard: A/8*(4
                + 2*np.cos(2*(pol - x)/180.*np.pi)
                + np.cos(2*(pol - x)/180.*np.pi + 2*np.pi*retard)
                + np.cos(2*(-pol + x)/180.*np.pi + 2*np.pi*retard)
                + 4*np.cos(2*(pol - 2*tilt + x)/180.*np.pi)*np.square(np.sin(np.pi*retard)))

def rotPthenA90_fx(x,A1,A2,A3,tilt,retard):
    return _rotPthenA_fx(90)(x,A1,tilt,retard)

def rotPthenApara_fx(x,A1,A2,A3,tilt,retard):
    return _rotPthenA_fx(45.05-0.84)(x,A2,tilt,retard)

def rotPthenAantipara_fx(x,A1,A2,A3,tilt,retard):
    return _rotPthenA_fx(132.79-0.84)(x,A3,tilt,retard)

def is_bad_fitting(fitting_results):
    popt,perr=fitting_results
    return np.isnan([popt,perr]).any() or (perr/popt>1).any()

def error_fx_4mirrors(p,xs,ynopol,ypara,yapara):
    A1,A2,A3,tilt,retard=p
    err_1 = np.square(rotPthenA90_fx(xs,A1,A2,A3,tilt,retard) - ynopol)
    err_2 = np.square(rotPthenApara_fx(xs,A1,A2,A3,tilt,retard) - ypara)
    err_3 = np.square(rotPthenAantipara_fx(xs,A1,A2,A3,tilt,retard) - yapara)
    return (err_1 + err_2 + err_3)

def rotPthenA_fitter(xs,ys,pol,plot=True,title=False):
    A = max(ys)
    tilt = -2.
    retard = 0.3
    p0 = (A,tilt,retard)
    fx = _rotPthenA_fx(pol)
    
    popt, pcov = sp.optimize.curve_fit(fx, xs, ys, p0)
    perr = np.sqrt(np.diag(pcov))
    if plot:
        if title:
            plt.figure(title)
            plt.title(title)
        else:
            plt.figure()
        plt.polar(xs/180.*np.pi,ys,'o')
        Xs = np.linspace(min(xs),max(xs),100)
        plt.polar(Xs/180.*np.pi,fx(Xs,*popt))
    return popt, perr

def calib_4mirror_fitter(xs,ynopol,ypara,yapara,plot=True,title=False):
    A1 = max(ynopol)
    A2 = max(ypara)
    A3 = max(yapara)
    tilt = -2.
    retard = 0.25
    p0 = (A1,A2,A3,tilt,retard)
    pfit, pcov, infodict, errmsg, success = sp.optimize.leastsq(error_fx_4mirrors,p0,args=(xs,ynopol,ypara,yapara),full_output=True)
    if (len(ypara) > len(p0)) and pcov is not None:
        s_sq = (error_fx_4mirrors(pfit, xs,ynopol,ypara,yapara)**2).sum()/(len(ypara)-len(p0))
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
            _fig=plt.figure(title)
            fig = _fig.add_subplot(111,projection='polar')
            fig.set_title('%s\ntilt = %.3f $\pm$ %.3f deg, retard = %.4f $\pm$ %.4f wave'%(title,pfit_leastsq[3],perr_leastsq[3],pfit_leastsq[4],perr_leastsq[4]))
        else:
            _fig=plt.figure()
            fig = _fig.add_subplot(111,projection='polar')
        Xs = np.linspace(min(xs),max(xs),100)
        fig.plot(xs/180.*np.pi,ynopol/max(ynopol),'o',color='C0')
        fig.plot(Xs/180.*np.pi,rotPthenA90_fx(Xs,*pfit_leastsq)/max(ynopol),color='C0')
        fig.plot(xs/180.*np.pi,ypara/max(ypara),'o',color='C1')
        fig.plot(Xs/180.*np.pi,rotPthenApara_fx(Xs,*pfit_leastsq)/max(ypara),color='C1')
        fig.plot(xs/180.*np.pi,yapara/max(yapara),'o',color='C2')
        fig.plot(Xs/180.*np.pi,rotPthenAantipara_fx(Xs,*pfit_leastsq)/max(yapara),color='C2')
        plt.tight_layout()
        plt.pause(1e-3)
    return pfit_leastsq, perr_leastsq