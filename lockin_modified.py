# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 18:41:43 2016

@author: Neo
"""

import visa
import time
import numpy as np
rm=visa.ResourceManager()
import sys
sys.path.append(r'D:\Nonlinear_setup\Python_codes')
from neo_common_code import *

try:
    srs830 = rm.open_resource(u"GPIB0::8::INSTR")
except:
    print 'Lock-in not connected'
else:
    print 'Lock-in online.'

srs830_sensitivity=[2e-9,5e-9,10e-9,20e-9,50e-9,100e-9,200e-9,500e-9,1e-6,2e-6,5e-6,10e-6,20e-6,50e-6,100e-6,200e-6,500e-6,1e-3,2e-3,5e-3,10e-3,20e-3,50e-3,100e-3,200e-3,500e-3,1]
sr830_timeconstant = [10e-6,30e-6,100e-6,300e-6,1e-3,3e-3,10e-3,30e-3,100e-3,300e-3,1.,3.,10.,30.,100.,300.,1e3,3e3,10e3,30e3]

max_sens = 0.85
min_sens = 0.3

srs830.write('SRAT 13') #Set sampling rate to 512Hz


# basic definitions for common things to ask the lockin
def get_lockin_reading1():
    return float(srs830.ask("OUTP? 3"))
    
def get_lockin_reading2():
    return float(srs830.ask("OUTP? 4"))

def get_lockin_X():
    return float(srs830.ask("OUTP? 1"))

def get_lockin_Y():
    return float(srs830.ask("OUTP? 2"))

def get_lockin_R():
    return float(srs830.ask("OUTP? 3"))

def get_lockin_theta():
    return float(srs830.ask("OUTP? 4"))

def get_lockin_freq():
    return float(srs830.ask("FREQ?"))

def get_lockin_time_const():
    i = int(srs830.ask("OFLT?"))
    return sr830_timeconstant[i]

def get_lockin_slope_index():
    i = int(srs830.ask("OFSL?"))
    return i

def clear_lockin_buffer():
    srs830.write("REST")
    print("Lockin buffer cleared.")
    return

def start_lockin_buffer():
    srs830.write("STRT")
    print("Lockin buffer started.")
    return

def lockin_disp_Xerr():
    srs830.write("DDEF 1,2,0")
    
def lockin_disp_X():
    srs830.write("DDEF 1,0,0")
    
def lockin_disp_Yerr():
    srs830.write("DDEF 2,2,0")

def lockin_disp_Y():
    srs830.write("DDEF 2,0,0")
    

def get_OPT_lockin_reading1():
    optimise_srs830_sensitivity()
    return get_lockin_reading1()

def get_OPT_lockin_reading2():
    optimise_srs830_sensitivity()
    return get_lockin_reading2()

def get_lockin_sens_idx():
    return int(srs830.ask('sens?'))

def is_lockin_sens_opted():
    try:
        srs830.write('*CLS')
        curr_sens_index = int(srs830.ask('sens?'))
        curr_sens = srs830_sensitivity[curr_sens_index]
        curr_val = get_lockin_X()
        return curr_val < max_sens*curr_sens and curr_val > min_sens*curr_sens
    except:
        srs830.write('*CLS')

def is_lockin_locked():
    lockin_disp_Xerr()
    global _auto_opt_Xs
    ds = []
    for i in range(3):
        ds.append(np.array(srs830.ask("SNAP? 1,10,4").split(','),dtype=float))
    ds = np.array(ds)
    thetas = ds[:,2]
    _auto_opt_Xs = np.abs(ds[:,0])
    X_errs = ds[:,1]
    return np.std(thetas)<3 and np.mean(_auto_opt_Xs/X_errs) > 1

def lockin_auto_optimize():
    if is_lockin_locked():
        if is_lockin_sens_opted():
            return
        else:
            lockin_auto_gain()
            return
    else:
        optimise_srs830_sensitivity()
            

def optimise_srs830_sensitivity():
    try:
        srs830.write('*CLS')
        curr_sens_index = int(srs830.ask('sens?'))
        curr_sens = srs830_sensitivity[curr_sens_index]
        
        while get_lockin_reading1() < curr_sens*min_sens:
            curr_sens_index -= 1
            srs830.write('sens %s'%curr_sens_index)
            curr_sens = srs830_sensitivity[curr_sens_index]
            time.sleep(0.1)
            
        while get_lockin_reading1() > curr_sens*max_sens:
            curr_sens_index += 1
            srs830.write('sens %s'%curr_sens_index)
            curr_sens = srs830_sensitivity[curr_sens_index]
            time.sleep(0.1)
        srs830.write('*CLS')
    except:
        srs830.write('*CLS')

def lockin_auto_gain():
    srs830.write('AGAN')
    try:
        srs830.ask('*stb?')
    except visa.VisaIOError:
        srs830.read()

def lockin_auto_phase():
    srs830.write('APHS')
    try:
        srs830.ask('*stb?')
    except visa.VisaIOError:
        srs830.read()
    
def lockin_aux3out(o=None):
    if o == None:
        return float(srs830.ask('AUXV ? 3'))
    else:
        srs830.write('AUXV 3,%f'%o)
        return lockin_aux3out()
    
def lockin_aux4in(): #get pmt gain voltage
    return float(srs830.ask("OAUX? 4"))

def lockin_timeconst(t=None): #get time constant of lock in in s, response time is ~10*this output in s
    if t == None:
        return sr830_timeconstant[int(srs830.ask("OFLT?"))]
    else:
        idx = get_nearest_idx_from_list(t,sr830_timeconstant)
        srs830.write("OFLT %i"%idx)
        return lockin_timeconst()
    

def lockin_get_X_Xerr_theta_aux4in_freq(theta_th=10,max_n=10):
    ds = []
    ds.append(np.array(srs830.ask("SNAP? 1,10,4,8,9").split(','),dtype=float))
    n = 0
    while np.abs(ds[-1][2]) > theta_th:
        ds.append(np.array(srs830.ask("SNAP? 1,10,4,8,9").split(','),dtype=float))
        if n > max_n:
            break
        n += 1
    return ds[-1]


def lockin_get_X_Y_Xnoise_Y_noise():
    #Warning: need to set the lockin displays to show xnoise and ynoise first
    #Note: The noises are in V/sqrt(Hz) - need to multiply by sqrt(ENBW) after
    return np.array(srs830.ask("SNAP? 1,10,2,11").split(','),dtype=float)