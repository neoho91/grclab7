# -*- coding: utf-8 -*-
"""
Created on Thu Aug 08 16:57:38 2019

@author: Neo
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

from background_optimize_sc_piezo import *
from filter_rot_stages import *
from polarimeter import *
from polarimeter_2 import *
import rmh
import rmq

def calibrate_sc_waveplates(sample=None,wl=742,h_angs=np.arange(0,180,1),q_angs=np.arange(0,360,1),log=None):
    if sample == None:
        curr_time_str = create_date_str()
    else:
        curr_time_str = sample
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\sc_waveplates_calib',curr_time_str)
    os.makedirs(main_path)
    
    # write log file with parameters used for experiment
    input_line = np.array([get_last_input_line()])
    log_txt = [unicode(input_line),u'\nwavelength = %.2f nm'%wl,u'\n'+unicode(log)]
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    np.save(os.path.join(main_path,'h_angs'),np.array(h_angs))
    np.save(os.path.join(main_path,'q_angs'),np.array(q_angs))
    
    rmh.home_rot()
    rmq.home_rot()
    opt_scpz()
    
    global data
    data = []
    #data structure is going to be:
    #      q_angs[0]   q_angs[1]      q_angs[-1]
    # [ [   datum   ,   datum, ... ,   datum   ], #HWP = h_angs[0]
    #   [   datum   ,   datum, ... ,   datum   ], #HWP = h_angs[1]
    #   ...,
    #   [   datum   ,   datum, ... ,   datum   ]  #HWP = h_angs[-1]]
    #
    # while each datum is:
    # datum = ( (a_aft100x, g_aft100x),(a_ref, g_ref) )
    
    total_len = len(h_angs)*len(q_angs)
    start_time = time.time()
    _i = 0
    prev = ''
    for i,h_ang in enumerate(h_angs):
        rmh.set_ang(h_ang)
        datum_same_H = []
        start_time_for_this_run = time.time()
        for j,q_ang in enumerate(q_angs):
            rmq.set_ang(q_ang)
            curr_datum = [[0,0],[0,0]]
            
            def pola_aft_100x():
                popt,perr = polarimeter2_measure_slow(wl,angs = np.arange(0,360,15), timesleep = 0.1, ave_num = 3, take_bg=0, live_plot=0, verbose=0, plot = 0)
                curr_datum[0] = popt[1:]
            pola_aft_100x_th = threading.Thread(target=pola_aft_100x)
            
            def pola_ref():
                popt,perr = polarimeter_measure_slow(wl,angs = np.arange(0,360,15), timesleep = 0.1, ave_num = 3, take_bg=0, live_plot=0, verbose=0, plot = 0)
                curr_datum[1] = popt[1:]
            pola_ref_th = threading.Thread(target=pola_ref)
            
            pola_aft_100x_th.start()
            pola_ref_th.start()
            
            pola_aft_100x_th.join()
            pola_ref_th.join()
            
            datum_same_H.append(copy.copy(curr_datum))
            
            elapsed_time = time.time() - start_time
            time_left = elapsed_time*(1.*(total_len)/(_i+1)-1)
            elapsed_time_for_this_run = time.time() - start_time_for_this_run
            time_left_for_this_run = elapsed_time_for_this_run*(1.*len(q_angs)/(j+1)-1)
            completed = u'HWP at %.2f deg (%.2f percent) %s left | QWP at %.2f (%.2f percent) %s left for this run.'%(
                    h_ang,100.0*(float(i+1)/len(h_angs)),sec_to_hhmmss(time_left),
                    q_ang,100.0*(float(j+1)/len(q_angs)),sec_to_hhmmss(time_left_for_this_run))
            prints(completed,prev)
            prev = completed
            _i += 1
        
        data.append(copy.copy(datum_same_H))
    
    data = np.array(data)
    np.save(os.path.join(main_path,'data'),data)

def calibrate_sc_waveplates2(sample=None,wl=742,h_angs=np.arange(0,90,1),q_angs_off=np.arange(-45,45,1),log=None,opt_scpz_every_h_loop=True):
    """
    h_angs = hwp angles (absolute values)
    q_angs_off = qwp offset angles from 0.5*hwp angles (relative values), i.e. q_ang = 0.5*h_ang + q_ang_off
    
    default:
        h_angs = np.arange(0,90,1), q_angs_off = np.arange(-45,45,1)
    if fails to find good angle sets:
        h_angs = np.arange(90,180,1), q_angs_off = np.arange(45,135,1)
    """
    if sample == None:
        curr_time_str = create_date_str()
    else:
        curr_time_str = sample
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\sc_waveplates_calib2',curr_time_str)
    os.makedirs(main_path)
    
    # write log file with parameters used for experiment
    input_line = np.array([get_last_input_line()])
    log_txt = [unicode(input_line),u'\nwavelength = %.2f nm'%wl,u'\n'+unicode(log)]
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    np.save(os.path.join(main_path,'h_angs'),np.array(h_angs))
    np.save(os.path.join(main_path,'q_angs_off'),np.array(q_angs_off))
    
    home_h_q()
    
    global data
    data = []
    #data structure is going to be:
    #     q_angs_off[0]  q_angs_off[1]  q_angs_off[-1]
    # [ [   datum   ,   datum, ... ,   datum   ], #HWP = h_angs[0]
    #   [   datum   ,   datum, ... ,   datum   ], #HWP = h_angs[1]
    #   ...,
    #   [   datum   ,   datum, ... ,   datum   ]  #HWP = h_angs[-1]]
    #
    # while each datum is:
    # datum = ( (a_aft100x, g_aft100x),(a_ref, g_ref) )
    
    total_len = len(h_angs)*len(q_angs_off)
    start_time = time.time()
    _i = 0
    prev = ''
    for i,h_ang in enumerate(h_angs):
        if opt_scpz_every_h_loop:
            start_time_for_opt_scpz = time.time()
            opt_scpz(verbose=False)
            elapsed_time_for_opt_scpz = time.time() - start_time_for_opt_scpz
        rmh.set_ang(h_ang)
        datum_same_H = []
        start_time_for_this_run = time.time()
        for j,q_ang_off in enumerate(q_angs_off):
            q_ang = 0.5*h_ang + q_ang_off
            rmq.set_ang(q_ang)
            curr_datum = [[0,0],[0,0]]
            
            def pola_aft_100x():
                popt,perr = polarimeter2_measure_slow(wl,angs = np.arange(0,180,15), timesleep = 0.1, ave_num = 3, take_bg=0, live_plot=0, verbose=0, plot = 0)
                curr_datum[0] = popt[1:]
            pola_aft_100x_th = threading.Thread(target=pola_aft_100x)
            
            def pola_ref():
                popt,perr = polarimeter_measure_slow(wl,angs = np.arange(0,180,15), timesleep = 0.1, ave_num = 3, take_bg=0, live_plot=0, verbose=0, plot = 0)
                curr_datum[1] = popt[1:]
            pola_ref_th = threading.Thread(target=pola_ref)
            
            pola_aft_100x_th.start()
            pola_ref_th.start()
            
            pola_aft_100x_th.join()
            pola_ref_th.join()
            
            datum_same_H.append(copy.copy(curr_datum))
            
            if opt_scpz_every_h_loop:
                elapsed_time = time.time() - start_time - elapsed_time_for_opt_scpz
                time_left = elapsed_time*(1.*(total_len)/(_i+1)-1) + elapsed_time_for_opt_scpz*(len(h_angs)-1-i)
            else:
                elapsed_time = time.time() - start_time
                time_left = elapsed_time*(1.*(total_len)/(_i+1)-1)
            elapsed_time_for_this_run = time.time() - start_time_for_this_run
            time_left_for_this_run = elapsed_time_for_this_run*(1.*len(q_angs_off)/(j+1)-1)
            completed = u'HWP at %.2f deg (%.2f percent) %s left | QWP offset at %.2f (%.2f percent) %s left for this run.'%(
                    h_ang,100.0*(float(i+1)/len(h_angs)),sec_to_hhmmss(time_left),
                    q_ang_off,100.0*(float(j+1)/len(q_angs_off)),sec_to_hhmmss(time_left_for_this_run))
            prints(completed,prev)
            prev = completed
            _i += 1
        
        data.append(copy.copy(datum_same_H))
    
    data = np.array(data)
    np.save(os.path.join(main_path,'data'),data)
    prints('\n')

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
