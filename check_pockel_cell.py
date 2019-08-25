# -*- coding: utf-8 -*-
"""
Created on Mon Dec 03 12:25:39 2018

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
from neo_common_code import *
sys.path.append(os.path.abspath(r"D:\WMP_setup\Python_codes"))
sys.path.append(os.path.abspath(r"D:\Nonlinear_setup\Python_codes"))
try:
    from powermeter_analog import *
except:
    print("powermeter analog not connected.")

from rot_stages_noGUI_4 import *
from fit_mallus import *
from lockin import *

from pygame import mixer
mixer.init()
sound_dir = 'D:\Nonlinear_setup\Python_codes\soundfx\\'
complete = sound_dir + 'scancompleted.wav'
def play_sound(sound):
    typ = sound[-3:]
    if typ == 'wav' or typ == 'WAV':
        winsound.PlaySound(sound, winsound.SND_ASYNC)
    elif typ == 'mp3' or typ == 'MP3':
        mixer.music.load(sound)
        mixer.music.play()

def check_circular_polarization_pck(reset_each_round=True, pm=True, rot4_align_h_angle=26.291508):
    global A_data, I_data
    def press(event):
        global A_data, I_data
        if event.key == 'm':
            print(mallus_fitter(np.array(A_data)*180/np.pi, np.array(I_data)))
        elif event.key == 'c':
            I_data, A_data = [],[]
    if pm:
        _fig = plt.figure('powermeter')
    else:
        _fig = plt.figure('lockin')
        srs830.clear()
    _fig.canvas.mpl_connect('key_press_event',press)
    plt.ion()
    _fig.clf()
    fig = _fig.add_subplot(111,projection='polar')
    fig.axhline(color='k')
    I_data=[]
    A_data=[]
    line, = fig.plot(A_data,I_data,'o',ms=1)
    if pm:
        fig.set_ylabel('powermeter, mW')
    else:
        fig.set_ylabel('lockin, mW')
    plt.pause(0.05)
    
    try:
        while True:
            if pm:
                I_data.append(pma_power()*1e3)
            else:
#                I_data.append(get_lockin_reading1())
                I_data.append(get_lockin_X())
            A_data.append((move_rot4()-rot4_align_h_angle)/180.*np.pi)
            line.set_xdata(A_data)
            line.set_ydata(I_data)
            fig.relim()
            fig.autoscale_view(True,True,True)
            plt.pause(0.05)
            if reset_each_round and len(A_data)>5 and (np.abs(np.diff(A_data))>6.2).any():
                print mallus_fitter(np.array(A_data)*180./np.pi,I_data,False)
                I_data=[]
                A_data=[]
    except KeyboardInterrupt:
        A_data = np.array(A_data)/np.pi*180
        I_data = np.array(I_data)
        return np.array(A_data), np.array(I_data)

def check_circular_polarization_pck_accurate(ana_angs=np.linspace(0,360,num=25),ana_offset_angle=26.291508):
    """
    rotate rot4, get pma data
    """
#    unblock_laser()
    curr_time_str = create_date_str()
    sample='rot_ana_%s'%curr_time_str
    global A_data,I_data,aux1,aux1_e
    def prepare_take_data():
        global powers, powers_dev,A_data,I_data,aux1,aux1_e
        powers, powers_dev,A_data,I_data,aux1,aux1_e = [],[],[],[],[],[]
        pma_wl(780)
        move_rot4(0.1)
        home_rot4()
    
    def take_data(a):
        aux1.extend(get_aux1_val_raw())
        _powers=[]
        for i in range(10):
            _powers.append(pma_power()*1e3)
            time.sleep(0.00)
        
        A_data.append(a)
        
        curr_power = np.mean(_powers)
        curr_power_dev = np.std(_powers)
        
        powers.append(curr_power)
        powers_dev.append(curr_power_dev)
    
    total_len = len(ana_angs)
    prints('Completed ')
    prev_line=''
    _n = 0
    prepare_take_data()
    _fig = plt.figure('powermeter')
    _fig.clf()
    fig = _fig.add_subplot(111,projection='polar')
    fig.axhline(color='k')
    line, = fig.plot(A_data,I_data,'o',ms=1)
    for ana_ang in ana_angs:
        move_rot4(ana_ang+ana_offset_angle)
        take_data(ana_ang)
        
        line.set_xdata(np.array(A_data)/180.*np.pi)
        line.set_ydata(powers)
        fig.relim()
        fig.autoscale_view(True,True,True)
        plt.pause(0.001)
        
        _n += 1
        curr_line = '%.2f percent'%(float(_n+1)/total_len*100.)
        prints(curr_line,prev_line)
        prev_line = curr_line
    I_data = np.array(powers)
    A_data = np.array(A_data)
    aux1_e = np.std(aux1)
    aux1 = np.mean(aux1)
    aux1,aux1_e = round_to_error(aux1,aux1_e)
    play_sound(complete)
    return mallus_fitter(A_data,I_data,0)

def get_aux1_val():
    ans = []
    for i in range(20):
        ans.append(float(srs830.ask('OAUX? 1')))
    return (np.mean(ans)*1e4,np.std(ans)*1e4)

def get_aux1_val_raw():
    ans = []
    for i in range(20):
        ans.append(float(srs830.ask('OAUX? 1'))*1e4)
    return ans