# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:18:09 2019

@author: Neo
"""

import numpy as np
import scipy as sp
import sys
import os
import scipy.interpolate
wls_TR_h = np.load(os.path.join(os.getcwd(),'NPBS_TR','wls_TR_h.npy'))
NPBS_TR_h_fx = sp.interpolate.interp1d(wls_TR_h[0],wls_TR_h[1],kind='cubic')

wls_TR_c = np.load(os.path.join(os.getcwd(),'NPBS_TR','wls_TR_c.npy'))
NPBS_TR_c_fx = sp.interpolate.interp1d(wls_TR_c[0],wls_TR_c[1],kind='linear')
#not enough data to use cubic interpolation

wls_TR_gamma = np.load(os.path.join(os.getcwd(),'NPBS_TR','wls_TR_gamma.npy'))
#input wl and gamma, gives TR ratio
NPBS_TR_gamma_fx = scipy.interpolate.interp2d(wls_TR_gamma[0],wls_TR_gamma[1],wls_TR_gamma[2],fill_value=None,kind='linear')
#not enough data to use cubic interpolation
#%%
#Append this column for further wavelength calibration. Do it on polarization_control.py too.
LRHV_TR_742 = [ 2.20521896,  2.29786088,  2.83636363,  1.87723264]
_LRHV_TR_wl_dic = {742:LRHV_TR_742}

#%%
def NPBS_TR_from_pol_wl(pol,wl=742):
    try:
        pol_idx = _LRHV_TR_pol_idx_dic[pol]
    except KeyError:
        print('%s pol not found. Please use one of %s.'%(pol,str(_LRHV_TR_pol_idx_dic.keys())))
    
    try:
        LRHV = _LRHV_TR_wl_dic[wl]
    except KeyError:
        print('wavelength %s not found. Please use one of %s.'%(str(wl),str(_LRHV_TR_wl_dic.keys())))
    
    TR = LRHV[pol_idx]
    return TR
#%%
_LRHV_TR_pol_idx_dic = {'L':0,'R':1,'H':2,'V':3}