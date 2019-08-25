# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:56:19 2019

@author: Neo

Input calibration for waveplates manually.
The order is LCP (1,i), RCP (1,-i), H (1,0) and V (0,1)
Each element consists of ((HWP loc, QWP rel loc),(ref alpha, ref gamma))
QWP rel loc is used for: QWP abs loc = 0.5*HWP loc + QWP rel loc
"""

import copy
import sys
import os
import time
import numpy as np
import scipy as sp
import csv
import matplotlib.pyplot as plt
import winsound
import threading
from neo_common_code import *
sys.path.append(os.path.abspath(r"D:\WMP_setup\Python_codes"))
sys.path.append(os.path.abspath(r"D:\Nonlinear_setup\Python_codes"))
from pygame import mixer
mixer.init()

from filter_rot_stages import *
from polarimeter import *
import rmh
import rmq

#%%
#Append this cell for further wavelength calibration. Do it on NPBS_TR_ratio.py too.
LRHV_742 = [((15.1,47.1), (122.70,-25.79)), #each element consists of ((HWP loc, QWP rel loc),(ref alpha, ref gamma))
            ((54.5,13.8), (53.92,26.52)),
            ((0.3,0.5), (175.61,0.39)),
            ((43.7,-24.9), (85.75,0.11))] 

_LRHV_wl_dic = {742:LRHV_742}

#%%
def go_to_pol(pol,wl=742,check_pol=False,verbose=True):
    try:
        pol_idx = _LRHV_pol_idx_dic[pol]
    except KeyError:
        print('%s pol not found. Please use one of %s.'%(pol,str(_LRHV_pol_idx_dic.keys())))
    
    try:
        LRHV = _LRHV_wl_dic[wl]
    except KeyError:
        print('wavelength %s not found. Please use one of %s.'%(str(wl),str(_LRHV_wl_dic.keys())))
    
    ((h,q),(ref_a,ref_g)) = LRHV[pol_idx]
    H,Q = HQ_rel_to_abs(h,q)
    set_h_q(H,Q)
    if check_pol:
        popt,perr = polarimeter_measure_slow(wl,angs = np.arange(0,180,15), timesleep = 0.1, ave_num = 3, take_bg=0, live_plot=0, plot=0, verbose=0)
        init_a,init_g = put_polarimeter_angles_to_smallest_range(popt[1],popt[2])
        if verbose:
            print('Target ref a = %.2fdeg, measured ref a = %.2fdeg (%.2fdeg)\nTarget ref g = %.2fdeg, measured ref g = %.2fdeg (%.2fdeg)'%(
                    ref_a,init_a,init_a-ref_a,
                    ref_g,init_g,init_g-ref_g))
    


#%%
_LRHV_pol_idx_dic = {'L':0,'R':1,'H':2,'V':3}

def HQ_rel_to_abs(H,Q):
    return (H,0.5*H+Q)

def home_h_q():
    def h_loop():
        rmh.home_rot()
    def q_loop():
        rmq.home_rot()
    h_th = threading.Thread(target=h_loop)
    q_th = threading.Thread(target=q_loop)
    h_th.start()
    q_th.start()
    h_th.join()
    q_th.join()

def set_h_q(h,q):
    def h_loop():
        rmh.set_ang(h)
    def q_loop():
        rmq.set_ang(q)
    h_th = threading.Thread(target=h_loop)
    q_th = threading.Thread(target=q_loop)
    h_th.start()
    q_th.start()
    h_th.join()
    q_th.join()

def get_h_q():
    return (rmh.get_ang(),rmq.get_ang())