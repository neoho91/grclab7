# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 09:59:18 2017

@author: Neo
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import winsound
import threading
import copy
import time
sys.path.append(os.path.abspath(r"D:\WMP_setup\Python_codes"))
sys.path.append(os.path.abspath(r"D:\Nonlinear_setup\Python_codes"))
from neo_common_code import *
from anal_isat_measurement import *
from pygame import mixer
mixer.init()

#from powermeter_analog import *
from powermeter_digital import *
from piezo import *
#from lockin import *
from rot_stages_noGUI_isat import *
try:
    from yokogawa import *
except:
    from yokogawa import *
from TL_slider_2 import *

def calib_PD(sample,HWP_angs=np.arange(-15,45,0.5)):
    main_path=r'D:\Nonlinear_setup\Experimental_data\isat_measurement\%s'%sample 
    os.makedirs(main_path)
    try:
        global power_data,power_data_dev,det_data,det_data_dev
        power_data = []
        power_data_dev = []
        ref_data = []
        ref_data_dev = []
        np.save(os.path.join(main_path,'HWP_angs'),np.array(HWP_angs))
        prints('Completed ')
        prev_line=''
        for n,HWP_ang in enumerate(HWP_angs):
            block_laser_2()
            move_HWP(HWP_ang)
            plt.pause(1)
            pmd_zero()
            plt.pause(1.5)
            unblock_laser_2()
            plt.pause(1)
            def power_loop():
                #modify this according to device acquiring power before objective
                curr_power_data = []
                powermeter2.input.pdiode.filter.lpass.state=0
                for i in range(10):
                    curr_power_data.append(pmd_power())
                power_datum = np.average(curr_power_data)
                power_dev_datum = np.std(curr_power_data)/np.sqrt(9)
                #end
                power_data.append(power_datum)
                power_data_dev.append(power_dev_datum)
            def ref_loop():
                #modify this according to device acquiring reference signal
    #            curr_ref_data = []
    #            for i in range(5):
    #                curr_ref_data.append(pma_power())
    #                time.sleep(0.1)
    #            ref_datum = np.average(curr_ref_data)
    #            ref_dev_datum = np.std(curr_ref_data)
                
                yoko_ch2_autoscale_amp()
                time.sleep(1)
                ref_datum,ref_dev_datum = get_ch2_reading()
                #end
                ref_data.append(ref_datum)
                ref_data_dev.append(ref_dev_datum)
            
            power_th = threading.Thread(target=power_loop)
            ref_th = threading.Thread(target=ref_loop)
            power_th.start()
            ref_th.start()
            power_th.join()
            ref_th.join()
            np.save(os.path.join(main_path,'power_data'),np.array(power_data))
            np.save(os.path.join(main_path,'power_data_dev'),np.array(power_data_dev))
            np.save(os.path.join(main_path,'ref_data'),np.array(ref_data))
            np.save(os.path.join(main_path,'ref_data_dev'),np.array(ref_data_dev))
            
            curr_line = '%.2f percent'%(float(n+1)/len(HWP_angs)*100.)
            prints(curr_line,prev_line)
            prev_line = curr_line
        play_sound(complete)
    except Exception as e:
        play_sound(error_sound)
        raise Exception(e)
    try:
        anal_calib_PD(sample)
    except:
        pass

def measure_isat(sample,piezo_pos=(get_pos()[0],get_pos()[2]),num_of_power=100,integrate_time=10,log='calib_PD = None; z_offset = 0;'):
    """
    In log field, write:
        calib_PD=*the calibration sample name*; <-- remember the semicolon.
        z_offset=*insert offset from focal plane in um*; <-- remember the semicolon.
    piezo_pos = (x,z)
    """
    main_path=r'D:\Nonlinear_setup\Experimental_data\isat_measurement\%s'%sample 
    os.makedirs(main_path)
    try:
        HWP_angs = log_power_separation(num_of_power)
        move(*piezo_pos)
        plt.pause(1)
        init_line = 'measure_isat (%s) with integrate_time = %.2f s\nHWP angles = %s\npiezo stage at %s\n'%(sample,integrate_time,str(HWP_angs),str(piezo_pos))
        print init_line
        log_txt = [init_line,
                   unicode(log)+u'\n\n']
        np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
        global sig_data,sig_data_dev,ref_data,ref_data_dev,diff_data,diff_data_dev
        sig_data = []
        sig_data_dev = []
        ref_data = []
        ref_data_dev = []
        diff_data = []
        diff_data_dev = []
        np.save(os.path.join(main_path,'HWP_angs'),np.array(HWP_angs))
        prints('Completed ')
        prev_line=''
        for n,HWP_ang in enumerate(HWP_angs):
            move_HWP(HWP_ang)
            plt.pause(0.1)
    #        def sig_loop():
    #            #modify this according to device acquiring signal (T-Tref)
    #            yoko_ch1_autoscale()
    #            yoko_restart_stat()
    #            time.sleep(10)
    #            sig_datum,sig_dev_datum = get_ch1_reading()
    #            #end
    #            sig_data.append(sig_datum)
    #            sig_data_dev.append(sig_dev_datum)
    #        def ref_loop():
    #            #modify this according to device acquiring reference signal
    #            plt.pause(1.5)
    #            yoko_ch2_autoscale()
    #            yoko_restart_stat()
    #            time.sleep(10)
    #            ref_datum,ref_dev_datum = get_ch2_reading()
    #            #end
    #            ref_data.append(ref_datum)
    #            ref_data_dev.append(ref_dev_datum)
    #        
    #        sig_th = threading.Thread(target=sig_loop)
    #        ref_th = threading.Thread(target=ref_loop)
    #        sig_th.start()
    #        ref_th.start()
    #        sig_th.join()
    #        ref_th.join()
    
            #special code for yoko, as it wont work properly in thread
    #        yoko_ch1_autoscale()
            yoko_autoscale_all()
            yoko_restart_meas()
            plt.pause(integrate_time)
            diff_datum,diff_dev_datum = copy.copy(get_BPD_reading())
            ref_datum,ref_dev_datum = get_ch2_reading()
            sig_datum,sig_dev_datum = get_ch3_reading()
            diff_data.append(diff_datum)
            diff_data_dev.append(diff_dev_datum)
            sig_data.append(sig_datum)
            sig_data_dev.append(sig_dev_datum)
            ref_data.append(ref_datum)
            ref_data_dev.append(ref_dev_datum)
            #end
            
            np.save(os.path.join(main_path,'diff_data'),np.array(diff_data))
            np.save(os.path.join(main_path,'diff_data_dev'),np.array(diff_data_dev))
            np.save(os.path.join(main_path,'sig_data'),np.array(sig_data))
            np.save(os.path.join(main_path,'sig_data_dev'),np.array(sig_data_dev))
            np.save(os.path.join(main_path,'ref_data'),np.array(ref_data))
            np.save(os.path.join(main_path,'ref_data_dev'),np.array(ref_data_dev))
            
            curr_line = '%.2f percent'%(float(n+1)/len(HWP_angs)*100.)
            prints(curr_line,prev_line)
            prev_line = curr_line
        play_sound(complete)
    except Exception as e:
        play_sound(error_sound)
        raise Exception(e)
    try:
        anal_measure_isat(sample)
    except:
        pass

def long_term_measurement(sample,integrate_time=2,HWP_ang=30,log=''):
    main_path=r'D:\Nonlinear_setup\Experimental_data\isat_measurement\%s'%sample 
    os.makedirs(main_path)
    try:
        init_line = 'long_term_measurement (%s) with integrate_time = %.2f s at HWP_angle = %f\n'%(sample,integrate_time,HWP_ang)
        print init_line
        log_txt = [init_line,
                   unicode(log)+u'\n\n']
        np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
        move_HWP(HWP_ang)
        global sig_data,sig_data_dev,ref_data,ref_data_dev,diff_data,diff_data_dev, elapsed_time_s
        sig_data = []
        sig_data_dev = []
        ref_data = []
        ref_data_dev = []
        diff_data = []
        diff_data_dev = []
        elapsed_time_s = []
        prints('Time elapsed ')
        prev_line=''
        start_time = time.time()
        while True:
            yoko_autoscale_all()
            yoko_restart_stat()
            yoko_restart_meas()
            plt.pause(integrate_time)
            diff_datum,diff_dev_datum = copy.copy(get_BPD_reading())
            ref_datum,ref_dev_datum = get_ch2_reading()
            sig_datum,sig_dev_datum = get_ch3_reading()
            elapsed_time_s.append(time.time()-start_time)
            diff_data.append(diff_datum)
            diff_data_dev.append(diff_dev_datum)
            sig_data.append(sig_datum)
            sig_data_dev.append(sig_dev_datum)
            ref_data.append(ref_datum)
            ref_data_dev.append(ref_dev_datum)
            #end
            
            np.save(os.path.join(main_path,'diff_data'),np.array(diff_data))
            np.save(os.path.join(main_path,'diff_data_dev'),np.array(diff_data_dev))
            np.save(os.path.join(main_path,'sig_data'),np.array(sig_data))
            np.save(os.path.join(main_path,'sig_data_dev'),np.array(sig_data_dev))
            np.save(os.path.join(main_path,'ref_data'),np.array(ref_data))
            np.save(os.path.join(main_path,'ref_data_dev'),np.array(ref_data_dev))
            np.save(os.path.join(main_path,'elapsed_time_s'),np.array(elapsed_time_s))
            
            curr_line = '%s'%(sec_to_hhmmss(elapsed_time_s[-1]))
            prints(curr_line,prev_line)
            prev_line = curr_line
    except Exception as e:
        play_sound(error_sound)
        raise Exception(e)

#----------------------------------------------------------------------------#
def get_ch1_reading():
    try:
        return [yoko_ch1_amp(),yoko_ch1_std()]
    except:
        return get_ch1_reading()
    
def get_ch2_reading():
    try:
        return [yoko_ch2_amp(),yoko_ch2_std()]
    except:
        return get_ch2_reading()
    
def get_ch3_reading():
    try:
        return [yoko_ch3_amp(),yoko_ch3_std()]
    except:
        return get_ch3_reading()

def yoko_autoscale_all():
    yoko_ch1_autoscale_hist()
    time.sleep(0.1)
    yoko_ch2_autoscale_amp()
    time.sleep(0.1)
    yoko_ch3_autoscale_amp()

def get_BPD_reading(total_time=10,sleep=0.1):
    return yoko_histogram_peaks_diff()

def prints(s,prev_s=''):
    if prev_s == '':
        sys.stdout.write(s)
        sys.stdout.flush()
    else:
        last_len = len(prev_s)
        sys.stdout.write('\b' * last_len)
        sys.stdout.write(' ' * last_len)
        sys.stdout.write('\b' * last_len)
        sys.stdout.write(s)
        sys.stdout.flush()

sound_dir = 'D:\Nonlinear_setup\Python_codes\soundfx\\'
complete = sound_dir + 'scancompleted.wav'
error_sound = sound_dir + 'error.wav'

def play_sound(sound):
    typ = sound[-3:]
    if typ == 'wav' or typ == 'WAV':
        winsound.PlaySound(sound, winsound.SND_ASYNC)
    elif typ == 'mp3' or typ == 'MP3':
        mixer.music.load(sound)
        mixer.music.play()

def move(a,b):
    move_to_x(a)
    move_to_z(b)

def const_power_separation(num_of_power,hwp_max_power_angle=34.83):
    I = np.linspace(0,1,num_of_power)
    HWP_angs = -np.degrees(np.arccos(np.sqrt(I)))/2. + hwp_max_power_angle
    return HWP_angs

def log_power_separation(num_of_power,hwp_max_power_angle=34.83):
    I = np.logspace(0,7,num_of_power)/10**7
    HWP_angs = -np.degrees(np.arccos(np.sqrt(I)))/2. + hwp_max_power_angle
    return HWP_angs