# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:48:48 2017

@author: Neo
"""
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import threading
import scipy.io as io
import visa
rm=visa.ResourceManager()
sys.path.append(r'D:\Nonlinear_setup\Python_codes')
from piezo import *
try:
    from powermeter_analog import *
except:
    print("powermeter analog not connected")
try:
    from yokogawa_waveform import *
except:
    print("yokogawa not connected")
from anal_sharp_blade_experiment_neo import *
from neo_common_code import *
from fit_CDF import *

def sharp_blade_scan(sample,wavelength,cut_in_z=True,z_poss=np.arange(0,70.01,1),log=''):    
    main_path= os.path.join(r'D:\Nonlinear_setup\Experimental_data\sharp_blade_scan',sample)
    os.makedirs(main_path)
    init_line = '\nStarted sharp_blade_scan (%s) on %s, cut_in_z = %s\n z_poss = %s\n'%(sample,time.strftime("%d%b%Y %H:%M", time.localtime()),str(cut_in_z), str(z_poss))
    print(init_line)
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    blade_pos=np.array([], float)
    powermeter_read=np.array([], float)
    pma_wl(wavelength)
    
    time.sleep(1)
    pwr_d= os.path.join(main_path, sample+".mat")
    prev_completed = ''
    prints('')
    total_len=len(z_poss)
    for i,z_pos in enumerate(z_poss):
        if cut_in_z:
            move_to_z(z_pos)
        else:
            move_to_x(z_pos)
        blade_pos=np.append(blade_pos, z_pos)
        time.sleep(1)
        powermeter_read=np.append(powermeter_read, pma_power())
        
        mat_dic={'blade_pos':blade_pos,'powermeter_read':powermeter_read}
        io.savemat(pwr_d,mat_dic)  
        np.save(os.path.join(main_path,sample+'_power'),powermeter_read)
        np.save(os.path.join(main_path,sample+'_blade_pos'),blade_pos)
        
        if cut_in_z:
            completed = 'z at %.1f um (%.2f percent)'%(z_pos,i*100./total_len)
        else:
            completed = 'x at %.1f um (%.2f percent)'%(z_pos,i*100./total_len)
        prints(completed,prev_completed)
        prev_completed = completed

def sharp_blade_scan_fast(sample,wavelength,cut_in_z=True,z_min=0,z_max=70,log=''):    
    main_path= os.path.join(r'D:\Nonlinear_setup\Experimental_data\sharp_blade_scan',sample)
    os.makedirs(main_path)
    speed=1.0#um/s
    ts = 0.05
    inc = speed*ts
    z_poss=np.arange(z_min,z_max+0.01,inc)
    init_line = '\nStarted sharp_blade_scan (%s) on %s, cut_in_z = %s\n z_poss = %s\n'%(sample,time.strftime("%d%b%Y %H:%M", time.localtime()),str(cut_in_z), str(z_poss))
    print(init_line)
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    global powermeter_read,blade_pos
    powermeter_read=[]
    pma_wl(wavelength)
    powermeter1.sense.average.count = _pause_time_to_ave_num(ts)
    if cut_in_z:
        move_to_z(z_poss[0])
    else:
        move_to_x(z_poss[0])
    
    def move_blade():
        time.sleep(1)
        for z_pos in z_poss:
            if cut_in_z:
                move_to_z(z_pos)
            else:
                move_to_x(z_pos)
            time.sleep(ts)
    
    def take_pm_data():
        time.sleep(1)
        while move_blade_th.isAlive():
            powermeter_read.append(pma_power())
            
    
    move_blade_th = threading.Thread(target=move_blade)
    take_pm_data_th = threading.Thread(target=take_pm_data)
    move_blade_th.start()
    take_pm_data_th.start()
    move_blade_th.join()
    take_pm_data_th.join()
 
    np.save(os.path.join(main_path,sample+'_power'),np.array(powermeter_read))
    blade_pos = np.linspace(min(z_poss),max(z_poss),len(powermeter_read))
    np.save(os.path.join(main_path,sample+'_blade_pos'),blade_pos) 
#    powermeter1.sense.average.count = 300
    
    print 'Done!'

def sharp_blade_scan_loop(sample,wavelength,cut_in_z=True,z_poss=np.arange(0,70.01,1),y_poss=np.arange(0,70.01,5),log=''):
    main_path= os.path.join(r'D:\Nonlinear_setup\Experimental_data\sharp_blade_scan',sample)
    os.makedirs(main_path)
    init_line = '\nStarted sharp_blade_scan_loop (%s) on %s, cut_in_z = %s\n z_poss or x_poss = %s, y_poss = %s\n'%(sample,time.strftime("%d%b%Y %H:%M", time.localtime()),str(cut_in_z), str(z_poss),str(y_poss))
    print(init_line)
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    pma_wl(wavelength)
    
    total_len=len(y_poss)*len(z_poss)
    prev_completed = ''
    print('')
    _n = 1
    for i,y_pos in enumerate(y_poss):
        curr_sample = sample+'_y%i'%y_pos
        move_to_y(y_pos)
        if cut_in_z:
            move_to_z(z_poss[0])
        else:
            move_to_x(z_poss[0])
        time.sleep(1)
        pwr_d= os.path.join(main_path, curr_sample+".mat")
        blade_pos=np.array([], float)
        powermeter_read=np.array([], float)
        for j,z_pos in enumerate(z_poss):
            if cut_in_z:
                move_to_z(z_pos)
            else:
                move_to_x(z_pos)
            blade_pos=np.append(blade_pos, z_pos)
            time.sleep(1)
            powermeter_read=np.append(powermeter_read, pma_power())
            
            mat_dic={'blade_pos':blade_pos,'powermeter_read':powermeter_read}
            io.savemat(pwr_d,mat_dic)  
            np.save(os.path.join(main_path,curr_sample+'_power'),powermeter_read)
            np.save(os.path.join(main_path,curr_sample+'_blade_pos'),blade_pos)
            
            if cut_in_z:
                completed = 'y at %.1f um, z at %.1f um (%.2f percent)'%(y_pos,z_pos,_n*100./total_len)
            else:
                completed = 'y at %.1f um, x at %.1f um (%.2f percent)'%(y_pos,z_pos,_n*100./total_len)
            prints(completed,prev_completed+' ')
            prev_completed = completed
            _n += 1
    print('')
    try:
        anal_sharp_blade_scan_loop(sample)
    except:
        pass

def sharp_blade_scan_loop_fast(sample,wavelength,cut_in_z=True,z_min=0,z_max=70,y_poss=np.arange(0,70.01,5),log=''):
    main_path= os.path.join(r'D:\Nonlinear_setup\Experimental_data\sharp_blade_scan',sample)
    os.makedirs(main_path)
    speed=1.0#um/s
    ts = 0.05
    inc = speed*ts
    z_poss=np.arange(z_min,z_max+0.01,inc)
    init_line = '\nStarted sharp_blade_scan_loop_fast (%s) on %s, cut_in_z = %s\n z_poss or x_poss = %s, y_poss = %s\n'%(sample,time.strftime("%d%b%Y %H:%M", time.localtime()),str(cut_in_z), str(z_poss),str(y_poss))
    print(init_line)
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    global powrmeter_read
    pma_wl(wavelength)
    powermeter1.sense.average.count = _pause_time_to_ave_num(ts)
    
    total_len=len(y_poss)
    prev_completed = ''
    print('')
    for i,y_pos in enumerate(y_poss):
        completed = 'y at %.1f um (%.2f percent)'%(y_pos,i*100./total_len)
        prints(completed,prev_completed+' ')
        prev_completed = completed
        
        curr_sample = sample+'_y%i'%y_pos
        move_to_y(y_pos)
        powermeter_read=[]
        pma_wl(wavelength)
        if cut_in_z:
            move_to_z(z_poss[0])
        else:
            move_to_x(z_poss[0])
        
        def move_blade():
            time.sleep(1)
            for z_pos in z_poss:
                if cut_in_z:
                    move_to_z(z_pos)
                else:
                    move_to_x(z_pos)
                time.sleep(ts)
        
        def take_pm_data():
            time.sleep(1)
            while move_blade_th.isAlive():
                powermeter_read.append(pma_power())
                time.sleep(0.05)
        
        move_blade_th = threading.Thread(target=move_blade)
        take_pm_data_th = threading.Thread(target=take_pm_data)
        move_blade_th.start()
        take_pm_data_th.start()
        move_blade_th.join()
        take_pm_data_th.join()
     
        np.save(os.path.join(main_path,curr_sample+'_power'),np.array(powermeter_read))
        blade_pos = np.linspace(min(z_poss),max(z_poss),len(powermeter_read))
        np.save(os.path.join(main_path,curr_sample+'_blade_pos'),blade_pos) 

    print('')
    powermeter1.sense.average.count = 300
    try:
        anal_sharp_blade_scan_loop(sample)
    except:
        pass
    
def sharp_blade_scan_loop_ultrafast(sample,cut_in_z=True,y_poss=np.arange(0,70.01,5),ave_num=3,log=''):
    main_path= os.path.join(r'D:\Nonlinear_setup\Experimental_data\sharp_blade_scan',sample)
    os.makedirs(main_path)
    
    init_line = '\nStarted sharp_blade_scan_loop_ultrafast (%s) on %s, cut_in_z = %s\ny_poss = %s\n'%(sample,time.strftime("%d%b%Y %H:%M", time.localtime()),str(cut_in_z),str(y_poss))
    print(init_line)
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    if cut_in_z:
        def move_pz(pos):
            return move_to_z(pos)
    else:
        def move_pz(pos):
            return move_to_x(pos)
    
    yoko_ave()
    total_len=len(y_poss)
    prev_completed = ''
    print('')
    for i,y_pos in enumerate(y_poss):
        completed = 'y at %.1f um (%.2f percent)'%(y_pos,i*100./total_len)
        prints(completed,prev_completed+' ')
        prev_completed = completed
        
        curr_sample = sample+'_y%i'%y_pos
        move_to_y(y_pos)
        time.sleep(0.1)
        
        yoko_start()
        time.sleep(0.5)
        for i in range(ave_num):
            move_pz(0)
            time.sleep(0.3)
            move_pz(80)
            time.sleep(0.3)
        yoko_stop()
        
        blade_pos = yoko_get_pos_wf()*7.99744081893794#7.582938388625592#7.99632169202167#8.000320012800513#7.8125#7.979808056824758 #um/V conversion factor
        pd_read = yoko_get_pd_wf()
     
        np.save(os.path.join(main_path,curr_sample+'_power'),pd_read)
        np.save(os.path.join(main_path,curr_sample+'_blade_pos'),blade_pos) 

    print('')
    yoko_norm()
    try:
        anal_sharp_blade_scan_loop(sample)
    except:
        pass

def single_ultrafast_chop(ave_num=1,cut_in_z=True,plot=False):
    yoko_ave()
    if cut_in_z:
        def move_pz(pos):
            return move_to_z(pos)
    else:
        def move_pz(pos):
            return move_to_x(pos)
    try:
        yoko_start()
        time.sleep(0.5)
        for i in range(ave_num):
            move_pz(0)
            time.sleep(0.3)
            move_pz(80)
            time.sleep(0.3)
        yoko_stop()
        
        blade_pos = yoko_get_pos_wf()*7.99744081893794#7.582938388625592#7.99632169202167#7.8125#7.979808056824758 #um/V conversion factor
        pd_read = yoko_get_pd_wf()
        yoko_norm()
        
        popt,perr = CDF_fitter(blade_pos,pd_read,plot)
        return 'x0 = %.3f +- %.3f um        FW1/e2 = %.3f +- %.3f um           FWHM = %.3f +- %.3f um'%(popt[2],perr[2],4*popt[3],4*perr[3],4*popt[3]/1.699,4*perr[3]/1.699)
    except KeyboardInterrupt:
        yoko_norm()
        raise KeyboardInterrupt

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

def _pause_time_to_ave_num(s):
    return int(round(3023*s-15.796))


