# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 19:06:35 2017

@author: yw_10
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.interpolate
import scipy as sp
c=299792.458 #divide by wavelength in nm to gives frequency in THz
calib_path=r'D:\Nonlinear_setup\Python_codes\FilterCalib'
#_filter1a=np.load(os.path.join(calib_path,'1','AOIs.npy'))
#_filter1c=np.load(os.path.join(calib_path,'1','fitted_centrals.npy'))
#_filter1f=np.load(os.path.join(calib_path,'1','fitted_fwhms.npy'))
#_filter1s=np.load(os.path.join(calib_path,'1','fitted_slant.npy'))
#_filter1p=np.load(os.path.join(calib_path,'1','fitted_T_peaks.npy'))
#_filter2a=np.load(os.path.join(calib_path,'2','AOIs.npy'))
#_filter2c=np.load(os.path.join(calib_path,'2','fitted_centrals.npy'))
#_filter2f=np.load(os.path.join(calib_path,'2','fitted_fwhms.npy'))
#_filter2s=np.load(os.path.join(calib_path,'2','fitted_slant.npy'))
#_filter2p=np.load(os.path.join(calib_path,'2','fitted_T_peaks.npy'))

_filter1a=np.load(os.path.join(calib_path,'1','AOIs.npy'))
_filter1c=np.load(os.path.join(calib_path,'1','fitted_centrals.npy'))
_filter1f=np.load(os.path.join(calib_path,'1','fitted_fwhms.npy'))
_filter1s=np.load(os.path.join(calib_path,'1','fitted_slant.npy'))
_filter1p=np.load(os.path.join(calib_path,'1','fitted_T_peaks.npy'))
_filter2a=np.load(os.path.join(calib_path,'2','AOIs.npy'))
_filter2c=np.load(os.path.join(calib_path,'2','fitted_centrals.npy'))
_filter2f=np.load(os.path.join(calib_path,'2','fitted_fwhms.npy'))
_filter2s=np.load(os.path.join(calib_path,'2','fitted_slant.npy'))
_filter2p=np.load(os.path.join(calib_path,'2','fitted_T_peaks.npy'))

angle_of_short_edge = scipy.interpolate.interp1d(_filter1c - _filter1f/2,_filter1a,fill_value='cubic')
angle_of_long_edge = scipy.interpolate.interp1d(_filter2c + _filter2f/2,_filter2a,fill_value='cubic')

angle_from_wlc1 = scipy.interpolate.interp1d(_filter1c,_filter1a,fill_value='cubic')
fwhm_from_wlc1 = scipy.interpolate.interp1d(_filter1c,_filter1f,fill_value='cubic')
slant_from_wlc1 = scipy.interpolate.interp1d(_filter1c,_filter1s,fill_value='cubic')
T_peak_from_wlc1 = scipy.interpolate.interp1d(_filter1c,_filter1p,fill_value='cubic')
wlc_from_angle1 = scipy.interpolate.interp1d(_filter1a,_filter1c,fill_value='cubic')
fwhm_from_angle1 = scipy.interpolate.interp1d(_filter1a,_filter1f,fill_value='cubic')
slant_from_angle1 = scipy.interpolate.interp1d(_filter1a,_filter1s,fill_value='cubic')
T_peak_from_angle1 = scipy.interpolate.interp1d(_filter1a,_filter1p,fill_value='cubic')

angle_from_wlc2 = scipy.interpolate.interp1d(_filter2c,_filter2a,fill_value='cubic')
fwhm_from_wlc2 = scipy.interpolate.interp1d(_filter2c,_filter2f,fill_value='cubic')
slant_from_wlc2 = scipy.interpolate.interp1d(_filter2c,_filter2s,fill_value='cubic')
T_peak_from_wlc2 = scipy.interpolate.interp1d(_filter2c,_filter2p,fill_value='cubic')
wlc_from_angle2 = scipy.interpolate.interp1d(_filter2a,_filter2c,fill_value='cubic')
fwhm_from_angle2 = scipy.interpolate.interp1d(_filter2a,_filter2f,fill_value='cubic')
slant_from_angle2 = scipy.interpolate.interp1d(_filter2a,_filter2s,fill_value='cubic')
T_peak_from_angle2 = scipy.interpolate.interp1d(_filter2a,_filter2p,fill_value='cubic')

def _lambda_c(lambda_o,theta,n):
    return lambda_o*np.sqrt(1-np.square(np.sin(theta*np.pi/180.)/n))
    
def lambda_c(theta):
    return _lambda_c(790,theta,1.85)

def _angle(wl_c,lambda_o,n):
    ans = np.arcsin(n*np.sqrt(1-np.square(wl_c/lambda_o)))
    return ans/np.pi*180.

def angle(wl_c):
    return _angle(wl_c,790.,1.85)

def _another_angle(theta1,df):
    """
    df = 3e5*(1/lambda1 - 1/lambda2) in THz
    theta1 in deg
    return theta2 for separation of df in central wavelength
    """
    n=1.85
    f = c/790.
    theta1 = theta1/180.*np.pi
    ans = -np.arcsin(np.sqrt(np.square(n) - np.square(n*f)/np.square(df - (f*n)/np.sqrt((n - np.sin(theta1))*(n + np.sin(theta1))))))
    return ans/np.pi*180.
    
def _HWHM(wl_c):
    """
    returns half width half max for the filter at central wavelength in nm in THz
    """
    FWHM1=12.5*2.5
    FWHM2=9.8*2.5
    wl1=680.1
    wl2=793.9
#    return ((FWHM1-FWHM2)/(wl1-wl2)*(wl_c-wl1)+FWHM1)/2.
    return ((FWHM1-FWHM2)/(1/wl1-1/wl2)*(1./wl_c-1/wl1)+FWHM1)/2.
    
def get_angles(wl,dwl=1):
    """
    wavelength in nm,
    dwl in nm
    return theta1 and theta2 for filter overlap of dwl (actual bandwidth of light output)
    """
    shorter_wl = wl-dwl/2.
    longer_wl = wl+dwl/2.
    a1 = angle_of_short_edge(shorter_wl)
    a2 = angle_of_long_edge(longer_wl)
    return (a1,a2)

def _FWHM_nm(wl_c,df):
    return wl_c/np.sqrt(1+c/df)

def _FWHM_THz(wl_c,dl):
    return c*(1/(wl_c-dl/2.)-1/1/(wl_c+dl/2.))

def trapz_curve(x,h,c,d,fwhm): #T peak, central, slant, fwhm
    sq = np.where(abs(x-c)<=(fwhm-d)/2., h, 0)
    rt = np.where(abs(x-c-fwhm/2.) <= d/2., h-(x-(c+(fwhm-d)/2.))*float(h)/d, 0)
    lt = np.where(abs(x-c+fwhm/2.) < d/2., (x-(c-(fwhm+d)/2.))*float(h)/d, 0)
    return sq + lt + rt

def fit_trapz_curve(Xs,Ys):
    h = np.max(Ys)
    c = Xs[list(Ys).index(h)]
    fwhm = 2*np.abs(c-Xs[get_nearest_idx_from_list(h/2.,Ys)])
    d = 1
    p0 = (h, c, d, fwhm)
    popt, pcov = sp.optimize.curve_fit(trapz_curve, Xs, Ys, p0)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def get_nearest_idx_from_list(val,lis):
    abs_diff_values = [abs(x-val) for x in lis]
    return abs_diff_values.index(min(abs_diff_values))

if __name__ == "__main__":
    dwl = 1
    wls = np.linspace(580,620,100)
    angs = get_angles(wls,dwl=dwl)
    
    _fig = plt.figure('Angles vs central wavelength, FWHM = %.2f nm'%dwl)
    fig = _fig.add_subplot(111)
    xmarker = fig.axvline(wls[0],color='k',linestyle='--')
    ymarker1 = fig.axhline(angs[0][0],color='b',linestyle='--')
    ymarker2 = fig.axhline(angs[1][0],color='r',linestyle='--')
    text1 = fig.text(wls[0],angs[0][0],'',color='b')
    text2 = fig.text(wls[0],angs[0][1],'',color='r')
    fig.plot(wls,angs[0],color='b')
    fig.plot(wls,angs[1],color='r')
    fig.grid()
    fig.set_title('Incident angles vs output central wavelength, FWHM = %.2f nm'%dwl)
    fig.set_xlabel('Output central wavelength, nm')
    fig.set_ylabel('Incident angles, deg')
    
    def onclick(event):
        x = event.xdata
        if x > wls[-1]:
            x = wls[-1]
        if x < wls[0]:
            x = wls[0]
        xmarker.set_xdata(x)
        ang1,ang2 = get_angles(x,dwl=dwl)
        ymarker1.set_ydata(ang1)
        ymarker2.set_ydata(ang2)
        text1.set(text='%.2f\n%.2f'%(x,ang1),x=x,y=ang1)
        text2.set(text='%.2f'%ang2,x=x,y=ang2)
                
        plt.show()
        plt.pause(1e-3)
        
    _fig.canvas.mpl_connect('motion_notify_event', onclick)