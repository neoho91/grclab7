# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 15:31:30 2018

@author: Neo
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
sys.path.append(os.path.abspath(r"D:\WMP_setup\Python_codes"))
sys.path.append(os.path.abspath(r"D:\Nonlinear_setup\Python_codes"))
from neo_common_code import *
import scipy.interpolate

def anal_calib_PD(sample,fig=None):
    main_path = os.path.join('D:\Nonlinear_setup\Experimental_data\isat_measurement',sample)
    power_data = np.load(os.path.join(main_path,'power_data.npy'))
    power_data_dev = np.load(os.path.join(main_path,'power_data_dev.npy'))
    ref_data = np.load(os.path.join(main_path,'ref_data.npy'))
    ref_data_dev = np.load(os.path.join(main_path,'ref_data_dev.npy'))
    HWP_angs = np.load(os.path.join(main_path,'HWP_angs.npy'))
    
    if not fig:
        _fig = plt.figure(sample)
        fig = _fig.add_subplot(111)
    fig.cla()
    PR = power_data/ref_data
    PR_e = PR*np.sqrt(np.square(power_data_dev/power_data)+np.square(ref_data_dev/ref_data))
    fig.errorbar(HWP_angs[:len(power_data)],PR,yerr=PR_e,marker='o',capsize=2)
    fig.set_title('%s\nref max/min = %fm/%fu\npow max/min = %fu/%fn'%(sample,max(ref_data)*1e3,min(ref_data)*1e6,max(power_data)*1e6,min(power_data)*1e9))
    fig.set_xlabel('HWP angle, deg')
    fig.set_ylabel('pow/ref, mW/mV')
    plt.pause(1e-6)
    
    min_idx = get_nearest_idx_from_list(min(power_data),power_data)
    max_idx = get_nearest_idx_from_list(max(power_data),power_data)
    crop_power_data = power_data[min_idx:max_idx]
    crop_power_data_dev = power_data_dev[min_idx:max_idx]
    crop_ref_data = ref_data[min_idx:max_idx]
    crop_ref_data_dev = ref_data_dev[min_idx:max_idx]
    lower_th_idx = get_nearest_idx_from_list(50e-3,crop_ref_data)
    high_pow_ref_ratio = np.mean(crop_power_data[lower_th_idx:]/crop_ref_data[lower_th_idx:])
    
    _pow_ref_ratio_in = scipy.interpolate.interp1d(crop_ref_data,crop_power_data,kind='cubic')
    def _pow_ref_ratio(ref):
        try:
            return _pow_ref_ratio_in(ref)
        except:
            return ref*high_pow_ref_ratio
    def pow_ref_ratio(ref):
        return np.array(map(_pow_ref_ratio,ref))
    pow_ref_ratio_dev = scipy.interpolate.interp1d(crop_ref_data,crop_power_data_dev,fill_value='extrapolate')
    
    return [fig,pow_ref_ratio,pow_ref_ratio_dev]
    
def real_time_anal_calib_PD(sample,timesleep=1):
    outputs = anal_calib_PD(sample)
    while True:
        outputs = anal_calib_PD(sample,outputs[0])
        plt.pause(timesleep+0.01)

def anal_measure_isat(sample,with_chopper=True,fig=None,fig2=None):
    main_path = os.path.join('D:\Nonlinear_setup\Experimental_data\isat_measurement',sample)
    sig_data = np.load(os.path.join(main_path,'sig_data.npy'))
    sig_data_dev = np.load(os.path.join(main_path,'sig_data_dev.npy'))
    ref_data = np.load(os.path.join(main_path,'ref_data.npy'))
    ref_data_dev = np.load(os.path.join(main_path,'ref_data_dev.npy'))
    diff_data = np.load(os.path.join(main_path,'diff_data.npy'))
    diff_data_dev = np.load(os.path.join(main_path,'diff_data_dev.npy'))
    HWP_angs = np.load(os.path.join(main_path,'HWP_angs.npy'))
    if with_chopper:
        chopper_coeff = 2
    else:
        chopper_coeff = 1
    def get_calib_PD_name():
        _log=open(os.path.join(main_path,'log.txt'),'r')
        log=_log.read()
        calib_PD_file = log.split('calib_PD')[1].split(';')[0].split('=')[1]
        if ' ' in calib_PD_file:
            calib_PD_file = calib_PD_file[1:]
        return calib_PD_file
    def get_z_from_log():
        _log=open(os.path.join(main_path,'log.txt'),'r')
        log=_log.read()
        s = log.split('z_offset')[1].split(';')[0].split('=')[1]
        if ' ' in s:
            s = s[1:]
        return float(s)
    try:
        calib_PD_file = get_calib_PD_name()
        zfig,pow_ref_ratio,pow_ref_ratio_dev = anal_calib_PD(calib_PD_file)
        z_offset = get_z_from_log()
        spot_D = spot_diameter_um(z_offset)
    except:
        if not fig:
            _fig = plt.figure(sample)
            fig = _fig.add_subplot(111)
            print('\ncalib_PD file not found/z_offset not specified in log file.')
        plt.cla()
        plt.title(sample)
        plt.plot(HWP_angs[:len(ref_data)],ref_data,label='ref' )
        plt.plot(HWP_angs[:len(sig_data)],sig_data,label='sig' )
        plt.plot(HWP_angs[:len(diff_data)],diff_data,label='diff' )
        plt.plot(HWP_angs[:len(sig_data)],np.abs(diff_data)/ref_data,label='|diff|/ref' )
        plt.plot(HWP_angs[:len(sig_data)],sig_data/ref_data,label='sig/ref' )
        plt.plot(HWP_angs[:len(sig_data)],sig_data-ref_data,label='sig - ref' )
        plt.plot(HWP_angs[:len(sig_data)],(sig_data-ref_data)/ref_data,label='(sig - ref)/ref' )
        plt.xlabel('HWP angle, deg')
        plt.legend()
        return [fig]
        
    power_data = pow_ref_ratio(ref_data)*chopper_coeff
    power_data_dev = power_data*np.sqrt(np.square(ref_data_dev/ref_data)+np.square(pow_ref_ratio_dev(ref_data)/power_data))
    inten_data = Ipeak_from_Pave(power_data*1e6,z_offset)
    inten_data_dev = inten_data*np.sqrt(np.square(power_data_dev/power_data))
    dT_data = diff_data
    dT_data_dev = diff_data_dev
    
    T_data = sig_data/ref_data
    T_data_dev = T_data*np.sqrt(np.square(sig_data_dev/sig_data)+np.square(ref_data_dev/ref_data))
    
    if not fig:
        _fig = plt.figure(sample)
        fig = _fig.add_subplot(111)
        fig2 = fig.twinx()
    fig.clear()
    fig.errorbar(inten_data,dT_data,xerr=inten_data_dev,yerr=dT_data_dev,marker='o',color='C0',capsize=2)
    fig.set_title('%s, spot diameter = %.2f um'%(sample,spot_D))
    fig.set_xlabel('I$_{peak}$, MW/cm$^2$')
    fig.set_ylabel('difference, V',color='C0')
    fig.set_xscale('log')
    plt.pause(1e-6)
    fig2.clear()
    fig2.errorbar(inten_data,T_data,yerr=T_data_dev,xerr=inten_data_dev,marker='o',color='C1',capsize=2)
    fig2.set_ylabel('signal/reference',color='C1')
    fig2.set_xscale('log')
    plt.pause(1e-6)
    return [fig,fig2,inten_data,dT_data]
    
    
def real_time_anal_measure_isat(sample,timesleep=11.5):
    outputs = anal_measure_isat(sample)
    while True:
        if len(outputs) > 1:
            outputs = anal_measure_isat(sample,fig=outputs[0],fig2=outputs[1])
        else:
            outputs = anal_measure_isat(sample,fig=outputs[0])
        plt.pause(timesleep+0.01)

def anal_long_term_measurement(sample,fig=None,fig2=None):
    main_path = os.path.join('D:\Nonlinear_setup\Experimental_data\isat_measurement',sample)
    sig_data = np.load(os.path.join(main_path,'sig_data.npy'))*1e3
    sig_data_dev = np.load(os.path.join(main_path,'sig_data_dev.npy'))*1e3
    ref_data = np.load(os.path.join(main_path,'ref_data.npy'))*1e3
    ref_data_dev = np.load(os.path.join(main_path,'ref_data_dev.npy'))*1e3
    diff_data = np.load(os.path.join(main_path,'diff_data.npy'))*1e3
    diff_data_dev = np.load(os.path.join(main_path,'diff_data_dev.npy'))*1e3
    elapsed_time_s = np.load(os.path.join(main_path,'elapsed_time_s.npy'))/3600.
    
    if not fig:
        _fig = plt.figure(sample)
        fig = _fig.add_subplot(111)
        fig2 = fig.twinx()
    fig.clear()
    fig2.clear()
    fig.errorbar(elapsed_time_s,sig_data,yerr=sig_data_dev,marker='o',color='C1',capsize=2,label='sig')
    fig.errorbar(elapsed_time_s,ref_data,yerr=ref_data_dev,marker='o',color='C2',capsize=2,label='ref')
    fig.legend(loc='upper right')
    plt.pause(1e-6)
    fig2.errorbar(elapsed_time_s,diff_data,yerr=diff_data_dev,marker='o',color='C0',capsize=2,label='diff')
    fig2.legend(loc='upper left')
    fig.set_title(sample)
    fig.set_xlabel('time, hr')
    fig2.set_ylabel('diff, mV')
    fig.set_ylabel('sig and ref, mV')
    return [fig,fig2]

def real_time_anal_long_term_measurement(sample,timesleep=14):
    outputs = anal_long_term_measurement(sample)
    while True:
        outputs = anal_long_term_measurement(sample,outputs[0],outputs[1])
        plt.pause(timesleep)

def Ipeak_from_Pave(Pave,z,r0=0.745,zR=6.58):
    """
    Ipeak in MW/cm-2
    Pave = average power from power meter, if using chopper then need to x2, uW.
    z = position away from focus plane, um.
    """
    RR = 80e6
    tau = 200e-15
    T_20x = 0.9
    uW = 1e-6
    MW = 1e-6
    coeff = T_20x/RR/tau * uW * MW
    A = spot_area_cm2(z,r0=r0,zR=zR)
    return coeff* Pave/A

def spot_area_cm2(z,r0=0.745,zR=6.58):
    """
    returns area in cm^2, for gaussian beam with r0 = HW1/e2 in um and zR in um. z is the offset distance from waist position, in um.
    """
    um2 = 1e-8
    r2 = np.square(r0)*(1 + np.square(z/zR))*um2
    return np.pi*r2

def spot_diameter_um(z,r0=0.745,zR=6.58):
    return np.sqrt(1+np.square(1.*z/zR))*r0*2