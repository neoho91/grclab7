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
import time
sys.path.append(os.path.abspath(r"D:\WMP_setup\Python_codes"))
sys.path.append(os.path.abspath(r"D:\Nonlinear_setup\Python_codes"))
from pygame import mixer
mixer.init()

from powermeter_analog import *
from powermeter_digital import *
from piezo import *
from lockin import *
from rot_stages_noGUI_isat import *
try:
    from yokogawa import *
except:
    from yokogawa import *


def calib_PD(sample):
    main_path=r'D:\Nonlinear_setup\Experimental_data\calib_isat\%s'%sample 
    os.makedirs(main_path)
    global pma_data,pma_data_dev,pd_data,pd_data_dev
    pma_data = []
    pma_data_dev = []
    pd_data = []
    pd_data_dev = []
    n = 0
    while True:
        def pma_loop():
            curr_pma_data = []
            for i in range(10):
                curr_pma_data.append(pma_power())
                time.sleep(0.1)
            pma_data.append(np.average(curr_pma_data))
            pma_data_dev.append(np.std(curr_pma_data))
        def pd_loop():
            curr_pd_data = []
            optimise_srs830_sensitivity()
            for i in range(10):
                curr_pd_data.append(get_pd_reading())
                time.sleep(0.1)
            pd_data.append(np.average(curr_pd_data))
            pd_data_dev.append(np.std(curr_pd_data))
        
        pma_th = threading.Thread(target=pma_loop)
        pd_th = threading.Thread(target=pd_loop)
        pma_th.start()
        pd_th.start()
        pma_th.join()
        pd_th.join()
        np.save(os.path.join(main_path,'pma_data'),np.array(pma_data))
        np.save(os.path.join(main_path,'pma_data_dev'),np.array(pma_data_dev))
        np.save(os.path.join(main_path,'pd_data'),np.array(pd_data))
        np.save(os.path.join(main_path,'pd_data_dev'),np.array(pd_data_dev))
        
        n+=1
        ans=raw_input('Took %i data. Last power = %.2f mW. Continue?y/n '%(n,pma_data[-1]*1e3))
        if ans =='N' or ans =='n':
            break
        else:
            continue
    return pma_data,pma_data_dev,pd_data,pd_data_dev

def calib_PD_auto(sample,HWP_angs=np.arange(93,93+45,0.5)):
    main_path=r'D:\Nonlinear_setup\Experimental_data\calib_isat\%s'%sample 
    os.makedirs(main_path)
    global pma_data,pma_data_dev,pd_data,pd_data_dev
    pma_data = []
    pma_data_dev = []
    pd_data = []
    pd_data_dev = []
    prints('Completed ')
    prev_line=''
    for n,HWP_ang in enumerate(HWP_angs):
        move_HWP(HWP_ang)
        def pma_loop():
            curr_pma_data = []
            for i in range(60):
                curr_pma_data.append(pma_power())
                time.sleep(0.1)
            pma_data.append(np.average(curr_pma_data))
            pma_data_dev.append(np.std(curr_pma_data))
        def pd_loop():
            yoko_ch1_autoscale()
            yoko_restart_stat()
            time.sleep(5)
            pd_amp,pd_std = get_pd_reading()
            pd_data.append(pd_amp)
            pd_data_dev.append(pd_std)
        
        pma_th = threading.Thread(target=pma_loop)
        pd_th = threading.Thread(target=pd_loop)
        pma_th.start()
        pd_th.start()
        pma_th.join()
        pd_th.join()
        np.save(os.path.join(main_path,'pma_data'),np.array(pma_data))
        np.save(os.path.join(main_path,'pma_data_dev'),np.array(pma_data_dev))
        np.save(os.path.join(main_path,'pd_data'),np.array(pd_data))
        np.save(os.path.join(main_path,'pd_data_dev'),np.array(pd_data_dev))
        
        curr_line = '%.2f percent'%(float(n+1)/len(HWP_angs)*100.)
        prints(curr_line,prev_line)
        prev_line = curr_line
    play_sound(complete)
    
def calib_PD_auto_2(sample,HWP_angs=np.arange(93,93+45,0.5)):
    main_path=r'D:\Nonlinear_setup\Experimental_data\calib_isat\%s'%sample 
    os.makedirs(main_path)
    pma_data = []
    pma_data_dev = []
    pmd_data = []
    pmd_data_dev = []
    prints('Completed ')
    prev_line=''
    for n,HWP_ang in enumerate(HWP_angs):
        move_HWP(HWP_ang)
        def pma_loop():
            curr_pma_data = []
            for i in range(60):
                curr_pma_data.append(pma_power())
                time.sleep(0.1)
            pma_data.append(np.average(curr_pma_data))
            pma_data_dev.append(np.std(curr_pma_data))
        def pmd_loop():
            curr_pmd_data = []
            for i in range(60):
                curr_pmd_data.append(pmd_power())
                time.sleep(0.1)
            pmd_data.append(np.average(curr_pmd_data))
            pmd_data_dev.append(np.std(curr_pmd_data))
        
        pma_th = threading.Thread(target=pma_loop)
        pmd_th = threading.Thread(target=pmd_loop)
        pma_th.start()
        pmd_th.start()
        pma_th.join()
        pmd_th.join()
        np.save(os.path.join(main_path,'pma_data'),np.array(pma_data))
        np.save(os.path.join(main_path,'pma_data_dev'),np.array(pma_data_dev))
        np.save(os.path.join(main_path,'pmd_data'),np.array(pmd_data))
        np.save(os.path.join(main_path,'pmd_data_dev'),np.array(pmd_data_dev))
        
        curr_line = '%.2f percent'%(float(n+1)/len(HWP_angs)*100.)
        prints(curr_line,prev_line)
        prev_line = curr_line
    play_sound(complete)
        

def measure_isat(sample,pos_s,pos_b):
    main_path=r'D:\Nonlinear_setup\Experimental_data\calib_isat\%s'%sample 
    os.makedirs(main_path)
    global pds_data,pds_data_dev,apds_data,apds_data_dev
    global pdb_data,pdb_data_dev,apdb_data,apdb_data_dev
    pds_data = []
    pds_data_dev = []
    apds_data = []
    apds_data_dev = []
    pdb_data = []
    pdb_data_dev = []
    apdb_data = []
    apdb_data_dev = []
    n = 0
    while True:
        #measure sample
        move_to_x(pos_s[0])
        move_to_z(pos_s[1])
        time.sleep(1)
        yoko_ch1_autoscale()
        yoko_ch2_autoscale()
        
        yoko_restart_stat()
        time.sleep(5)
        apds_amp,apds_std = get_apd_reading()
        pds_amp,pds_std = get_pd_reading()
        apds_data.append(apds_amp)
        apds_data_dev.append(apds_std)
        pds_data.append(pds_amp)
        pds_data_dev.append(pds_std)
        
        #measure background
        move_to_x(pos_b[0])
        move_to_z(pos_b[1])
        time.sleep(1)
        yoko_ch1_autoscale()
        yoko_ch2_autoscale()
        
        yoko_restart_stat()
        time.sleep(5)
        apdb_amp,apdb_std = get_apd_reading()
        pdb_amp,pdb_std = get_pd_reading()
        apdb_data.append(apdb_amp)
        apdb_data_dev.append(apdb_std)
        pdb_data.append(pdb_amp)
        pdb_data_dev.append(pdb_std)
        
        #save data
        np.save(os.path.join(main_path,'apds_data'),np.array(apds_data))
        np.save(os.path.join(main_path,'apds_data_dev'),np.array(apds_data_dev))
        np.save(os.path.join(main_path,'pds_data'),np.array(pds_data))
        np.save(os.path.join(main_path,'pds_data_dev'),np.array(pds_data_dev))
        
        np.save(os.path.join(main_path,'apdb_data'),np.array(apdb_data))
        np.save(os.path.join(main_path,'apdb_data_dev'),np.array(apdb_data_dev))
        np.save(os.path.join(main_path,'pdb_data'),np.array(pdb_data))
        np.save(os.path.join(main_path,'pdb_data_dev'),np.array(pdb_data_dev))
        
        play_sound(complete)
        n+=1
        ans=raw_input('Took %i data. Continue?y/n '%(n))
        if ans =='N' or ans =='n':
            break
        else:
            continue
    return pds_data,pds_data_dev,apds_data,apds_data_dev,pdb_data,pdb_data_dev,apdb_data,apdb_data_dev

def measure_isat_auto(sample,pos_s,pos_b,pos_g,HWP_angs=np.arange(0,44.5,0.2)):
    main_path=r'D:\Nonlinear_setup\Experimental_data\calib_isat\%s'%sample 
    os.makedirs(main_path)
    pds_data = []
    pds_data_dev = []
    apds_data = []
    apds_data_dev = []
    pdb_data = []
    pdb_data_dev = []
    apdb_data = []
    apdb_data_dev = []
    pdg_data = []
    pdg_data_dev = []
    apdg_data = []
    apdg_data_dev = []
    
    prints('Completed ')
    prev_line=''
    for n,HWP_ang in enumerate(HWP_angs):
        move_HWP(HWP_ang)
        #measure sample
        move_to_x(pos_s[0])
        move_to_z(pos_s[1])
        time.sleep(1)
        yoko_ch1_autoscale()
        yoko_ch2_autoscale()
        
        yoko_restart_stat()
        time.sleep(5)
        apds_amp,apds_std = get_apd_reading()
        pds_amp,pds_std = get_pd_reading()
        apds_data.append(apds_amp)
        apds_data_dev.append(apds_std)
        pds_data.append(pds_amp)
        pds_data_dev.append(pds_std)
        
        #measure background
        move_to_x(pos_b[0])
        move_to_z(pos_b[1])
        time.sleep(1)
        yoko_ch1_autoscale()
        yoko_ch2_autoscale()
        
        yoko_restart_stat()
        time.sleep(5)
        apdb_amp,apdb_std = get_apd_reading()
        pdb_amp,pdb_std = get_pd_reading()
        apdb_data.append(apdb_amp)
        apdb_data_dev.append(apdb_std)
        pdb_data.append(pdb_amp)
        pdb_data_dev.append(pdb_std)
        
        #measure glass
        move_to_x(pos_g[0])
        move_to_z(pos_g[1])
        time.sleep(1)
        yoko_ch1_autoscale()
        yoko_ch2_autoscale()
        
        yoko_restart_stat()
        time.sleep(5)
        apdg_amp,apdg_std = get_apd_reading()
        pdg_amp,pdg_std = get_pd_reading()
        apdg_data.append(apdg_amp)
        apdg_data_dev.append(apdg_std)
        pdg_data.append(pdg_amp)
        pdg_data_dev.append(pdg_std)
        
        #save data
        np.save(os.path.join(main_path,'apds_data'),np.array(apds_data))
        np.save(os.path.join(main_path,'apds_data_dev'),np.array(apds_data_dev))
        np.save(os.path.join(main_path,'pds_data'),np.array(pds_data))
        np.save(os.path.join(main_path,'pds_data_dev'),np.array(pds_data_dev))
        
        np.save(os.path.join(main_path,'apdb_data'),np.array(apdb_data))
        np.save(os.path.join(main_path,'apdb_data_dev'),np.array(apdb_data_dev))
        np.save(os.path.join(main_path,'pdb_data'),np.array(pdb_data))
        np.save(os.path.join(main_path,'pdb_data_dev'),np.array(pdb_data_dev))
        
        np.save(os.path.join(main_path,'apdb_data'),np.array(apdb_data))
        np.save(os.path.join(main_path,'apdb_data_dev'),np.array(apdb_data_dev))
        np.save(os.path.join(main_path,'pdb_data'),np.array(pdb_data))
        np.save(os.path.join(main_path,'pdb_data_dev'),np.array(pdb_data_dev))
        
        np.save(os.path.join(main_path,'apdg_data'),np.array(apdg_data))
        np.save(os.path.join(main_path,'apdg_data_dev'),np.array(apdg_data_dev))
        np.save(os.path.join(main_path,'pdg_data'),np.array(pdg_data))
        np.save(os.path.join(main_path,'pdg_data_dev'),np.array(pdg_data_dev))
        
        curr_line = '%.2f percent'%(float(n+1)/len(HWP_angs)*100.)
        prints(curr_line,prev_line)
        prev_line = curr_line
    play_sound(complete)

def measure_isat_auto_2(sample,pos_s,pos_b,pos_g,HWP_angs=np.arange(0,44.5,0.2)):
    main_path=r'D:\Nonlinear_setup\Experimental_data\calib_isat\%s'%sample 
    os.makedirs(main_path)
    pmas_data = []
    pmas_data_dev = []
    apds_data = []
    apds_data_dev = []
    pmab_data = []
    pmab_data_dev = []
    apdb_data = []
    apdb_data_dev = []
    pmag_data = []
    pmag_data_dev = []
    apdg_data = []
    apdg_data_dev = []
    
    prints('Completed ')
    prev_line=''
    for n,HWP_ang in enumerate(HWP_angs):
        move_HWP(HWP_ang)
        #measure sample
        move_to_x(pos_s[0])
        move_to_z(pos_s[1])
        time.sleep(1)
        yoko_ch2_autoscale()
        
        def yoko_loop():
            yoko_restart_stat()
            time.sleep(5)
            apds_amp,apds_std = get_apd_reading()
            apds_data.append(apds_amp)
            apds_data_dev.append(apds_std)
        def pma_loop():
            curr_pma_data=[]
            for i in range(50):
                curr_pma_data.append(pma_power())
                time.sleep(0.1)
            pmas_data.append(np.average(curr_pma_data))
            pmas_data_dev.append(np.std(curr_pma_data))
        pma_th = threading.Thread(target=pma_loop)
        yoko_th = threading.Thread(target=yoko_loop)
        pma_th.start()
        yoko_th.start()
        pma_th.join()
        yoko_th.join()
        
        #measure background
        move_to_x(pos_b[0])
        move_to_z(pos_b[1])
        time.sleep(1)
        yoko_ch2_autoscale()
        
        def yoko_loop():
            yoko_restart_stat()
            time.sleep(5)
            apdb_amp,apdb_std = get_apd_reading()
            apdb_data.append(apdb_amp)
            apdb_data_dev.append(apdb_std)
        def pma_loop():
            curr_pma_data=[]
            for i in range(50):
                curr_pma_data.append(pma_power())
                time.sleep(0.1)
            pmab_data.append(np.average(curr_pma_data))
            pmab_data_dev.append(np.std(curr_pma_data))
        pma_th = threading.Thread(target=pma_loop)
        yoko_th = threading.Thread(target=yoko_loop)
        pma_th.start()
        yoko_th.start()
        pma_th.join()
        yoko_th.join()
        
        #measure glass
        move_to_x(pos_g[0])
        move_to_z(pos_g[1])
        time.sleep(1)
        yoko_ch2_autoscale()
        
        def yoko_loop():
            yoko_restart_stat()
            time.sleep(5)
            apdg_amp,apdg_std = get_apd_reading()
            apdg_data.append(apdg_amp)
            apdg_data_dev.append(apdg_std)
        def pma_loop():
            curr_pma_data=[]
            for i in range(50):
                curr_pma_data.append(pma_power())
                time.sleep(0.1)
            pmag_data.append(np.average(curr_pma_data))
            pmag_data_dev.append(np.std(curr_pma_data))
        pma_th = threading.Thread(target=pma_loop)
        yoko_th = threading.Thread(target=yoko_loop)
        pma_th.start()
        yoko_th.start()
        pma_th.join()
        yoko_th.join()
        
        #save data
        np.save(os.path.join(main_path,'apds_data'),np.array(apds_data))
        np.save(os.path.join(main_path,'apds_data_dev'),np.array(apds_data_dev))
        np.save(os.path.join(main_path,'pmas_data'),np.array(pmas_data))
        np.save(os.path.join(main_path,'pmas_data_dev'),np.array(pmas_data_dev))
        
        np.save(os.path.join(main_path,'apdb_data'),np.array(apdb_data))
        np.save(os.path.join(main_path,'apdb_data_dev'),np.array(apdb_data_dev))
        np.save(os.path.join(main_path,'pmab_data'),np.array(pmab_data))
        np.save(os.path.join(main_path,'pmab_data_dev'),np.array(pmab_data_dev))
        
        np.save(os.path.join(main_path,'apdb_data'),np.array(apdb_data))
        np.save(os.path.join(main_path,'apdb_data_dev'),np.array(apdb_data_dev))
        np.save(os.path.join(main_path,'pmab_data'),np.array(pmab_data))
        np.save(os.path.join(main_path,'pmab_data_dev'),np.array(pmab_data_dev))
        
        np.save(os.path.join(main_path,'apdg_data'),np.array(apdg_data))
        np.save(os.path.join(main_path,'apdg_data_dev'),np.array(apdg_data_dev))
        np.save(os.path.join(main_path,'pmag_data'),np.array(pmag_data))
        np.save(os.path.join(main_path,'pmag_data_dev'),np.array(pmag_data_dev))
        
        curr_line = '%.2f percent'%(float(n+1)/len(HWP_angs)*100.)
        prints(curr_line,prev_line)
        prev_line = curr_line
    play_sound(complete)


#----------------------------------------------------------------------------#
def get_pd_reading():
    try:
        return [yoko_ch1_amp(),yoko_ch1_std()]
    except:
        return get_pd_reading()

def get_apd_reading():
    try:
        return [yoko_ch2_amp(),yoko_ch2_std()]
    except:
        return get_apd_reading()

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