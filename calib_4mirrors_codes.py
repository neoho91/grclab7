# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 15:39:57 2018

@author: Millie
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
import winsound
import os
import sys
sys.path.append(os.path.abspath('D:\WMP_setup\Python_codes'))
sys.path.append(os.path.abspath('D:\Nonlinear_setup\Python_codes'))
import threading
import scipy.interpolate

from calibrate_circular_pol3_1560 import *
try:
    import TL_slider_3
except:
    print "Laser shutter not connected"
try:
    from powermeter_analog import *
except:
    print "powermeter analog not connected"
try:
    from powermeter_digital import *
except:
    print "powermeter digital not connected"
try:
    from powermeter_usb_interface import *
except:
    print "powermeter usb interface not connected"
#from Laser_Control import *
from rot_stages_noGUI import *
from anal_calib_4mirrors_codes import *
from neo_common_code import *
from read_TH_temperature import *
from lockin import *


def calib_4mirrors(sample=None,log=''):
    if sample == None:
        curr_time_str = create_date_str()
        sample='calib_4mirrors_%s'%curr_time_str
    else:
        pass
    main_path = os.path.join('D:\Nonlinear_setup\Experimental_data\calibrate_4mirrors',sample)
    os.mkdir(main_path)

    init_line = 'calib_4mirrors (%s)'%(sample)
    print init_line
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    ana_angs=np.arange(0,360.01,5)
    np.save(os.path.join(main_path,'ana_angs'),ana_angs)
    
    def prepare_take_data():
        global powers, powers_dev
        powers, powers_dev = [],[]
        pma_wl(780)
        move_rot2(0.1)
        home_A()
    
    def take_data():
        _powers=[]
        for i in range(10):
            _powers.append(pma_power())
            time.sleep(0.00)
        
        curr_power = np.mean(_powers)
        curr_power_dev = np.std(_powers)
        
        powers.append(curr_power)
        powers_dev.append(curr_power_dev)
        
    def save_data(suffix):
        try:
            np.save(os.path.join(main_path,'powers_%s'%suffix),np.array(powers))
            np.save(os.path.join(main_path,'powers_dev_%s'%suffix),np.array(powers_dev))
        except:
            time.sleep(1)
            save_data()
            
    total_len = len(ana_angs)
    
    prepare_take_data()
    raw_input('\nInstall dielectric prism before 4mirrors and powermeter analog after analyzer. Press enter to start.')
    prints('\nCompleted ')
    prev_line=''
    _n = 0
    start_time_nopol = time.time()
    for ana_ang in ana_angs:
        move_A(ana_ang)
        time.sleep(0.1)
        take_data()
        save_data('nopol')
        _n += 1
        curr_line = '%.2f percent'%(float(_n)/total_len*100.)
        prints(curr_line,prev_line)
        prev_line = curr_line
    end_time_nopol = time.time()
    play_sound(complete)
    np.save(os.path.join(main_path,'timestamps'),np.array([start_time_nopol,end_time_nopol]))
    
    prepare_take_data()
    raw_input('\nInstall fixed polarizer with arrow parallel to beam propagation (i.e. facing toward 4 mirrors). Press enter to start.')
    prints('\nCompleted ')
    prev_line=''
    _n = 0
    start_time_para = time.time()
    for ana_ang in ana_angs:
        move_A(ana_ang)
        time.sleep(0.1)
        take_data()
        save_data('para')
        _n += 1
        curr_line = '%.2f percent'%(float(_n)/total_len*100.)
        prints(curr_line,prev_line)
        prev_line = curr_line
    end_time_para = time.time()
    play_sound(complete)
    np.save(os.path.join(main_path,'timestamps'),np.array([start_time_nopol,end_time_nopol,
            start_time_para,end_time_para]))
    
    prepare_take_data()
    raw_input('\nFlip the fixed polarizer so its arrow anti-parallel to beam propagation (i.e. facing away 4 mirrors). Press enter to start.')
    prints('\nCompleted ')
    prev_line=''
    _n = 0
    start_time_apara = time.time()
    for ana_ang in ana_angs:
        move_A(ana_ang)
        time.sleep(0.1)
        take_data()
        save_data('apara')
        _n += 1
        curr_line = '%.2f percent'%(float(_n)/total_len*100.)
        prints(curr_line,prev_line)
        prev_line = curr_line
    end_time_apara = time.time()
    play_sound(complete)    
    np.save(os.path.join(main_path,'timestamps'),np.array([start_time_nopol,end_time_nopol,
            start_time_para,end_time_para,
            start_time_apara,end_time_apara]))
    prints('\n')
    
    try:
        return anal_calib_4mirrors(sample)
    except:
        pass

def background_check_sc_circular_polarization(sample=None,main_path=None,timeconst=0.01,ave_num=3,run_num=0):
    if sample == None:
        curr_time_str = create_date_str()
    else:
        curr_time_str = sample
    if main_path == None:
        main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\background_check_sc_circular_polarization',curr_time_str)
        os.makedirs(main_path)
    timeconst=lockin_timeconst(timeconst)
    timesleep_full = timeconst*10
    timesleep = timeconst*5
    
    def prepare_take_data_outside_loop():
        global elapsed_times
        elapsed_times = []
        lockin_disp_Xerr()
        plt.pause(timesleep_full)
    
    def prepare_take_data(suffix):
        global Xs, Xerrs, thetas, theta_errs, gains, freqs, data_path
        Xs, Xerrs, thetas, theta_errs, gains, freqs = [],[],[],[],[],[]
        data_path = os.path.join(main_path,'data_%s.npz'%suffix)

    def take_data():
        curr_Xs, curr_Xerrs, curr_thetas, curr_gains, curr_freqs = [],[],[],[],[]
        lockin_auto_optimize()
        for i in range(ave_num):
            X, Xerr, theta, gain, freq = lockin_get_X_Xerr_theta_aux4in_freq(theta_th=360)
            curr_Xs.append(X)
            curr_Xerrs.append(Xerr)
            curr_thetas.append(theta)
            curr_gains.append(gain)
            curr_freqs.append(freq)
            plt.pause(timesleep)
        Xs.append(np.mean(curr_Xs))
        Xerrs.append(np.sqrt( np.square(np.std(curr_Xs)/np.sqrt(ave_num)) + np.square(np.mean(curr_Xerrs)) )) #taking account of both signal fluctuation and its error
        thetas.append(np.mean(curr_thetas))
        theta_errs.append(np.std(curr_thetas))
        gains.append(np.mean(curr_gains))
        freqs.append(np.mean(curr_freqs))
        
    def save_data():
        try:
            np.savez(data_path,
             start_time=start_time,
             end_time=end_time,
             Xs=Xs,
             Xerrs=Xerrs,
             thetas=thetas,
             theta_errs=theta_errs,
             gains=gains,
             freqs=freqs)
            elapsed_times.append(np.mean([end_time,start_time])-global_start_time)
            np.save(os.path.join(main_path,'elapsed_times.npy'),np.array(elapsed_times))
        except:
            time.sleep(1)
            save_data()
    
    def finishing_outside_loop():
        lockin_disp_X()
        
    global global_start_time,start_time,end_time
    global_start_time = time.time()
    prev_completed = ''
    _n = 1
    prints('\n')
    
    prepare_take_data_outside_loop()
    while _n != run_num+1:
        prepare_take_data('%i'%(_n))
        start_time = time.time()
        for ana_ang in np.arange(0,360.01,10):
            move_A(ana_ang)
            plt.pause(timesleep_full)
            take_data()
            
            if _n == 1:
                completed = 'Started on %s, %s elapsed. Now taking #%i, analyzer at %.1f deg.'%(
                        time.strftime("%d%b%Y %H:%M", time.localtime(global_start_time)),
                        '0s',
                        _n,
                        ana_ang)
            else:
                completed = 'Started on %s, %s elapsed. Now taking #%i, analyzer at %.1f deg.'%(
                        time.strftime("%d%b%Y %H:%M", time.localtime(global_start_time)),
                        sec_to_hhmmss(elapsed_times[-1]),
                        _n,
                        ana_ang)
            prints(completed,prev_completed)
            prev_completed = completed
        _n += 1
        end_time = time.time()
        save_data()
    finishing_outside_loop()
    