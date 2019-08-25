# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:07:06 2019

@author: Neo
"""

import numpy as np
import os
import sys
sys.path.append(r'D:\Nonlinear_setup\Python_codes')
from rot_stages_noGUI import *
try:
    from powermeter_analog import *
except:
    pass
#try:
#    from powermeter_digital import *
#except:
#    pass
try:
    from powermeter_1550_monitor import *
except:
    pass
try:
    from polarimeter import *
except:
    pass
from NPBS_TR_ratio import *

def move_H_sc(ang):
    rot3.mAbs(ang)
def get_H_sc():
    return rot3.getPos()

def get_sc_power_nW(wl,pol=None,c_pol=True,pm='polarimeter',return_raw=False):
    """
    pol = 'L', 'R', 'H', 'V'
    """
    if pm == 'pma':
        pma_wl(wl)
    elif pm == 'polarimeter':
        popt,perr,max_ang = polarimeter_measure_slow(wl=wl,plot=False,live_plot=False,verbose=False,return_max_angle=True)
        A,alpha,gamma = popt
        if return_raw:
            return (popt, perr,max_ang)
        else:
            if pol == None:
                return A*NPBS_TR_gamma_fx(wl,gamma)*1e9
            else:
                return A*NPBS_TR_from_pol_wl(pol,wl)*1e9
    else:
        pm1550m_wl(wl)
#        pmd_wl(wl)
    if c_pol:
        if pm == 'pma':
            return pma_power()*NPBS_TR_c_fx(wl)*1e9
        else:
            return pm1550m_power()*NPBS_TR_c_fx(wl)*1e9
#            return pmd_power()*NPBS_TR_c_fx(wl)*1e9
    else:
        if pm == 'pma':
            return pma_power()*NPBS_TR_h_fx(wl)*1e9
        else:
            return pm1550m_power()*NPBS_TR_h_fx(wl)*1e9
#            return pmd_power()*NPBS_TR_h_fx(wl)*1e9

def _hwp_ang_inten(ang):
    A = 0.99747
    phi = 42.35502/180.*np.pi
    R = 5.51358e-4
    ang = ang/180.*np.pi
    return A*(np.square(np.cos(2*(ang-phi))) + R*np.square(np.sin(2*(ang-phi))))

def _hwp_inten_ang(inten):
    A = 0.99747
    phi = 42.35502
    R = 5.51358e-4
    return phi + 45 - 0.5*np.arctan(np.sqrt((A*R - inten)/(-A + inten)))/np.pi*180

def _get_hwp_ang_needed(nW,wl,pm='polarimeter',c_pol=True,pol=None):
    curr_power = get_sc_power_nW(wl,pol=pol,c_pol=c_pol,pm=pm)
    curr_ang = get_H_sc()
    curr_normed_inten = _hwp_ang_inten(curr_ang)
    max_inten = curr_power/curr_normed_inten
    need_normed_inten = nW/max_inten
    return _hwp_inten_ang(need_normed_inten)

def set_sc_power_nW(nW,wl,pol=None,iteration=3,verbose=True,pm='polarimeter',c_pol=True, return_raw=False):
    """
    pol = 'L', 'R', 'H', 'V'
    """
    for i in range(iteration):
        curr_ang_need = _get_hwp_ang_needed(nW,wl,pol=pol,pm=pm,c_pol=c_pol)
        if np.isnan(curr_ang_need):
            curr_power = get_sc_power_nW(wl,pol=pol,c_pol=c_pol,pm=pm)
            if nW > curr_power:
                if verbose:
                    print('Set power (%d nW) is too high.'%nW)
                move_H_sc(42.35502)
            else:    
                if verbose:
                    print('Set power (%d nW) is too low.'%nW)
                move_H_sc(42.35502+45)
            break
        else:
            move_H_sc(curr_ang_need)
    return get_sc_power_nW(wl,pol=pol,c_pol=c_pol,pm=pm,return_raw=return_raw)