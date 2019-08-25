# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 21:57:28 2018

@author: Neo
"""

import copy
import sys
import os
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import winsound
import threading
from scipy.optimize import *
#from simple_pid import PID
from neo_common_code import *
sys.path.append(os.path.abspath(r"D:\WMP_setup\Python_codes"))
sys.path.append(os.path.abspath(r"D:\Nonlinear_setup\Python_codes"))

from sc_piezo import *
from powermeter_sc_monitor import *
#from TS_CCS200 import *

def init_background_optimize_sc_piezo():
    pmscm_wl(780)
#    global CCS200, CCS200_wls
#    CCS200=CCS(0)
#    CCS200.set_int_time(1e-5)
#    CCS200_wls = CCS200.get_wavelengths()
#    CCS200.get_optimised_spec()

def stop_background_optimize_sc_piezo():
    pass
#    CCS200.shutdown()

def get_pid_signal():
    return pmscm_power()*1e3
#    spec = CCS200.get_spec()
#    return np.sum(spec)
#    spec = np.convolve(spec, np.array((1,2,4,8,10,8,4,2,1))/40., mode='same')
#    if spec[2000] < np.max(spec[0:500])*2:
#        return 1e-6
#    tenth_max = (np.max(spec)-np.min(spec))/10.
#    max_idx = spec.argmax()
#    nearest_above = (np.abs(spec[max_idx:-1] - tenth_max)).argmin()
#    nearest_below = (np.abs(spec[0:max_idx] - tenth_max)).argmin()
#    return (np.mean(CCS200_wls[nearest_above + max_idx]) - np.mean(CCS200_wls[nearest_below]))

def opt_scpz_y(target=200,plot=True):
    init_background_optimize_sc_piezo()
    try:
        init_x = scpz_get_x_pos()
        init_y = scpz_get_y_pos()
        init_z = scpz_get_z_pos()
        print 'Initial positions = (%.2f, %.2f, %.2f)'%(init_x,init_y,init_z)
        
        if get_pid_signal() < 0.01:
            print 'Signal too small, aborted opt_scpz_y.'
            raise KeyboardInterrupt
        
#        find_peak(scpz_move_to_y,scpz_get_y_pos,get_pid_signal,0.1,2,0.1,average=10,fluctuation='auto',timesleep=0.1,auto_max_range=False,plot=plot,threshold_val=target,stablize_time=0.1)
        find_peak_2(scpz_move_to_y,scpz_get_y_pos,get_pid_signal,0.1,2,average=1,fluctuation='auto',timesleep=0.1,auto_max_range=False,plot=plot,threshold_val=target,stablize_time=0.1)
        
        fin_x = scpz_get_x_pos()
        fin_y = scpz_get_y_pos()
        fin_z = scpz_get_z_pos()
        print 'Final positions = (%.2f, %.2f, %.2f)'%(fin_x,fin_y,fin_z)
    except KeyboardInterrupt:
        pass
    finally:
        stop_background_optimize_sc_piezo()
        
def opt_scpz_z(target=200,plot=True):
    init_background_optimize_sc_piezo()
    try:
        init_x = scpz_get_x_pos()
        init_y = scpz_get_y_pos()
        init_z = scpz_get_z_pos()
        print 'Initial positions = (%.2f, %.2f, %.2f)'%(init_x,init_y,init_z)
        
        if get_pid_signal() < 0.01:
            print 'Signal too small, aborted opt_scpz_z.'
            raise KeyboardInterrupt
        
        find_peak_2(scpz_move_to_z,scpz_get_z_pos,get_pid_signal,0.1,2,average=1,fluctuation='auto',timesleep=0.1,auto_max_range=False,plot=plot,threshold_val=target,stablize_time=0.1)
        
        fin_x = scpz_get_x_pos()
        fin_y = scpz_get_y_pos()
        fin_z = scpz_get_z_pos()
        print 'Final positions = (%.2f, %.2f, %.2f)'%(fin_x,fin_y,fin_z)
    except KeyboardInterrupt:
        pass
    finally:
        stop_background_optimize_sc_piezo()
        
def opt_scpz_x(target=200,plot=True):
    init_background_optimize_sc_piezo()
    try:
        init_x = scpz_get_x_pos()
        init_y = scpz_get_y_pos()
        init_z = scpz_get_z_pos()
        print 'Initial positions = (%.2f, %.2f, %.2f)'%(init_x,init_y,init_z)
        
        if get_pid_signal() < 0.01:
            print 'Signal too small, aborted opt_scpz_x.'
            raise KeyboardInterrupt
        
        find_peak_2(scpz_move_to_x,scpz_get_x_pos,get_pid_signal,1,20,average=1,fluctuation='auto',timesleep=0.1,auto_max_range=False,plot=plot,threshold_val=target,stablize_time=0.3)
        
        fin_x = scpz_get_x_pos()
        fin_y = scpz_get_y_pos()
        fin_z = scpz_get_z_pos()
        print 'Final positions = (%.2f, %.2f, %.2f)'%(fin_x,fin_y,fin_z)
    except KeyboardInterrupt:
        pass
    finally:
        stop_background_optimize_sc_piezo()

def opt_scpz(target=200,num=1,verbose=True):
    init_background_optimize_sc_piezo()
    try:
        init_x = scpz_get_x_pos()
        init_y = scpz_get_y_pos()
        init_z = scpz_get_z_pos()
        if verbose:
            print 'Initial positions = (%.2f, %.2f, %.2f)'%(init_x,init_y,init_z)
        
        if get_pid_signal() < 0.01:
            if verbose:
                print 'Signal too small, aborted opt_scpz.'
            raise KeyboardInterrupt
        
        _fig = plt.figure('Optimize Piezo')
        figx = _fig.add_subplot(131)
        figx_plot, = figx.plot([],[],marker='x',ls='',color='C0')
        figx_data = [_fig,figx,figx_plot]
        figy = _fig.add_subplot(132)
        figy_plot, = figy.plot([],[],marker='x',ls='',color='C1')
        figy_data = [_fig,figy,figy_plot]
        figz = _fig.add_subplot(133)
        figz_plot, = figz.plot([],[],marker='x',ls='',color='C2')
        figz_data = [_fig,figz,figz_plot]
        plt.get_current_fig_manager().window.showMaximized()
        plt.pause(0.1)
        _fig.tight_layout(rect=(0,0,1,0.95))
        
        _n = 1
        while _n != num+1:
            _fig.suptitle('run #%i'%_n)
            figy_data = find_peak_2(scpz_move_to_y,scpz_get_y_pos,get_pid_signal,0.1,1,average=1,fluctuation='auto',timesleep=1e-6,auto_max_range=False,plot=True,threshold_val=target,stablize_time=0.2,fig_data=figy_data,poss=[],vals=[])
            figz_data = find_peak_2(scpz_move_to_z,scpz_get_z_pos,get_pid_signal,0.1,1,average=1,fluctuation='auto',timesleep=1e-6,auto_max_range=False,plot=True,threshold_val=target,stablize_time=0.2,fig_data=figz_data,poss=[],vals=[])
            figx_data = find_peak_2(scpz_move_to_x,scpz_get_x_pos,get_pid_signal,1,10,average=1,fluctuation='auto',timesleep=1e-6,auto_max_range=False,plot=True,threshold_val=target,stablize_time=0.2,fig_data=figx_data,poss=[],vals=[])
            _n += 1
        
        fin_x = scpz_get_x_pos()
        fin_y = scpz_get_y_pos()
        fin_z = scpz_get_z_pos()
        if verbose:
            print 'Final positions = (%.2f, %.2f, %.2f)'%(fin_x,fin_y,fin_z)
    except KeyboardInterrupt:
        ans = raw_input('Return to initial positions? [Y/n]')
        if ans.lower() == 'n':
            pass
        else:
            scpz_move_to_x(init_x)
            scpz_move_to_y(init_y)
            scpz_move_to_z(init_z)
    finally:
        stop_background_optimize_sc_piezo()