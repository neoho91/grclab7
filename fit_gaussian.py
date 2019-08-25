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

def gaussian_fx(x,x0,fwhm,A,bg):
    return bg + A*np.exp(-4*np.log(2)*np.square((x-x0)/fwhm))

def gaussian_fx_old(x,x0,fwhm,A,bg):
    return bg + A*np.exp(-np.log(2)/4.*np.square((x-x0)/fwhm))

def gaussian_fitter(xs,ys,plot=True,can_be_null=True):
    x0 = xs[get_nearest_idx_from_list(max(ys),ys)]
    bg = min(ys)
    A = max(ys) - bg
    fwhm = 2*np.abs(x0-xs[get_nearest_idx_from_list(A/2+bg,ys)])
    p0 = (x0,fwhm,A,bg)
    
    try:
        popt, pcov = sp.optimize.curve_fit(gaussian_fx, xs, ys, p0)
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError as e:
        if can_be_null:
            popt = (x0,0,A,bg)
            perr = (np.NaN,np.NaN,np.NaN,np.NaN)
        else:
            raise Exception(e)
    if can_be_null:
        if np.isnan([popt,perr]).any() or (popt<0).any() or (perr/popt>1).any():
            if is_no_peak(ys):
                popt = (x0,0,A,bg)
                perr = (np.NaN,np.NaN,np.NaN,np.NaN)
    if plot:
        plt.figure()
        plt.plot(xs,ys,'o')
        Xs = np.linspace(min(xs),max(xs),100)
        plt.plot(Xs,gaussian_fx(Xs,*popt))
    return popt, perr
    
def gaussian2_fx(x,x01,fwhm1,A1,x02,fwhm2,A2,bg):
    return bg + A1*np.exp(-4*np.log(2)*np.square((x-x01)/fwhm1)) + A2*np.exp(-4*np.log(2)*np.square((x-x02)/fwhm2))

def gaussian2_fitter(xs,ys, plot=True):
    bg = min(ys)
    xs1 = xs[0:len(xs)/2]
    ys1 = ys[0:len(ys)/2]
    x01 = xs1[get_nearest_idx_from_list(max(ys1),ys1)]
    A1 = max(ys1) - bg
    fwhm1 = 2*np.abs(x01-xs1[get_nearest_idx_from_list(A1/2+bg,ys1)])
    xs2 = xs[len(xs)/2:]
    ys2 = ys[len(ys)/2:]
    x02 = xs2[get_nearest_idx_from_list(max(ys2),ys2)]
    A2 = max(ys2) - bg
    fwhm2 = 2*np.abs(x02-xs2[get_nearest_idx_from_list(A2/2+bg,ys2)])
    p0 = (x01,fwhm1,A1,x02,fwhm2,A2,bg)
    
    popt, pcov = sp.optimize.curve_fit(gaussian2_fx, xs, ys, p0)
    perr = np.sqrt(np.diag(pcov))
    if plot:
        plt.figure()
        plt.plot(xs,ys,'o')
        Xs = np.linspace(min(xs),max(xs),100)
        plt.plot(Xs,gaussian2_fx(Xs,*popt))
        plt.plot(Xs,gaussian_fx(Xs,popt[0],popt[1],popt[2],popt[-1]))
        plt.plot(Xs,gaussian_fx(Xs,popt[3],popt[4],popt[5],popt[-1]))
    return popt, perr

def gaussian_amp_to_area(fitter_output):
    popt,perr=fitter_output
    fwhm=popt[1]
    fwhm_e=perr[1]
    amp=popt[2]
    amp_e=perr[2]
    if fwhm > 0:
#        A=np.sqrt(np.pi/32)/np.log(2)*amp*np.square(fwhm)
#        A_e=A*(amp_e/amp+2.*fwhm_e/fwhm)
        A=amp*(fwhm*np.sqrt(np.pi/(np.log(16))))
        A_e=A*(amp_e/amp+fwhm_e/fwhm)
    else:
        A=0
        A_e=0
    new_popt = [popt[0],fwhm,A,popt[3]]
    new_perr = [perr[0],fwhm_e,A_e,perr[3]]
    return np.array(new_popt), np.array(new_perr)

def is_no_peak(spec):
    amp = max(spec)-min(spec)
    baseline = np.median(spec)
    SNR = np.abs(amp-baseline)
    return SNR/baseline < 1

def gaussian2D_fx((x,y),x0,y0,fwhmx,fwhmy,vol,bg):
    A = vol*np.log(16)/(np.pi*fwhmx*fwhmy)
    return bg + A*np.exp( -4*np.log(2)*(np.square((x-x0)/fwhmx) + np.square((y-y0)/fwhmy)) )

def gaussian2D_fitter(img, plot=True, sat = 255):
    img = img.astype(np.uint8)
    xs = np.arange(len(img[0]))
    ys = np.arange(len(img))
    x,y = np.meshgrid(xs, ys)
    bg = np.min(img)
    vol = np.sum(img)*2
    amp = int(np.max(img)-bg)
    br = cv2.boundingRect(img.clip(amp/2)-amp/2)
    x0 = int(br[0]+1.*br[2]/2)-1
    y0 = int(br[1]+1.*br[3]/2)-1
    fwhmx = br[2] #length of x to be bounded
    fwhmy = br[3]
    p0 = (x0,y0,fwhmx,fwhmy,vol,bg)
    
    def _gaussian2D_fx((x,y),x0,y0,fwhmx,fwhmy,vol,bg):
        return  gaussian2D_fx((x,y),x0,y0,fwhmx,fwhmy,vol,bg).ravel()
    def _prepare_img(img):
        img = img.ravel()
        weight = (-1*(img.clip(sat-1)-(sat-1)))*0.999999+1e-6
        return img,weight
    prepared_img,weight=_prepare_img(img)
    popt, pcov = sp.optimize.curve_fit(_gaussian2D_fx, (x,y), prepared_img, p0, absolute_sigma=False,sigma=weight)
    if popt[2] < 0 or popt[3] < 0:
        p0 = popt
        p0[2] = np.abs(p0[2])
        p0[3] = np.abs(p0[3])
        popt, pcov = sp.optimize.curve_fit(_gaussian2D_fx, (x,y), prepared_img, p0, absolute_sigma=False,sigma=weight)
    perr = np.sqrt(np.diag(pcov))
    if plot:
        x0 = int(popt[0])
        y0 = int(popt[1])
        x_data = img[y0]
        y_data = img[:,x0]
        xs_fit = np.linspace(np.min(xs),np.max(xs),100)
        x_data_fit = gaussian2D_fx(np.meshgrid(xs_fit,y0),*popt)[0]
        ys_fit = np.linspace(np.min(ys),np.max(ys),100)
        y_data_fit = gaussian2D_fx(np.meshgrid(x0,ys_fit),*popt)[:,0]
        _fig = plt.figure()
        fig1 = _fig.add_subplot(221)
        fig1.imshow(img,norm=LogNorm(vmin=0.01, vmax=sat))
        fig1.set_title('input')
        fig1.axhline(x0,color='C0')
        fig1.axvline(y0,color='C1')
        fig2 = _fig.add_subplot(222)
        fig2.imshow(gaussian2D_fx((x,y),*popt),norm=LogNorm(vmin=0.01, vmax=sat))
        fig2.set_title('fitted')
        fig2.axhline(x0,color='C0')
        fig2.axvline(y0,color='C1')
        fig3 = _fig.add_subplot(223)
        fig3.semilogy(xs,x_data,'o',color='C0')
        fig3.semilogy(xs_fit,x_data_fit,color='C0')
        fig3.set_title('horizontal')
        fig4 = _fig.add_subplot(224)
        fig4.semilogy(ys,y_data,'o',color='C1')
        fig4.semilogy(ys_fit,y_data_fit,color='C1')
        fig4.set_title('vertical')
        plt.pause(1e-3)
        plt.tight_layout()
    return popt, perr