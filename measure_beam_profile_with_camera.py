# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 15:47:06 2018

@author: Neo
"""
import copy
import sys
import os
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import threading
import winsound
from pygame import mixer
mixer.init()

sys.path.append(os.path.abspath(r"D:\WMP_setup\Python_codes"))
sys.path.append(os.path.abspath(r"D:\Nonlinear_setup\Python_codes"))
from neo_common_code import *
from instrumental.drivers.cameras import uc480
from anal_measure_beam_profile_with_camera import *
global cam
try:
    if not cam.model == 'C1285R12M':
        raise Exception
    else:
        print('Camera C1285R12M already online.')    
except:
    cam=uc480.UC480_Camera('C1285R12M')
    print('Camera C1285R12M online.')
cam.auto_exposure=False
cam.auto_framerate=False
cam.auto_gain=False
try:
    import TL_slider_2 #sc
    def block_sc():
        TL_slider_2.block_laser_2()
    def unblock_sc():
        TL_slider_2.unblock_laser_2()
except:
    print "SC shutter not connected"
    def block_sc():
        pass
    def unblock_sc():
        pass
try:
    import TL_slider_3 #toptica
    def block_laser():
        TL_slider_3.block_laser_3()
    def unblock_laser():
        TL_slider_3.unblock_laser_3()
except:
    print "1560 shutter not connected"
    def block_laser():
        pass
    def unblock_laser():
        pass
try:
    from piezo import *
except:
    def get_pos():
        return []
    print "piezo not connected"

#%%
def scan_beam_profile(y_poss,sample=None,log=''):
    input_line = np.array([get_last_input_line()])
    if sample == None:
        curr_time_str = create_date_str2()
    else:
        curr_time_str = sample
    main_path=r'D:\Nonlinear_setup\Experimental_data\scan_beam_profile\%s'%curr_time_str 
    os.makedirs(main_path)
    move_to_y(y_poss[0])
    total_len = len(y_poss)
    total_time = 2*(total_len*( 0.12 + 0.5 ) + 2)
    global start_time,end_time
    start_time = time.time()
    init_line = '\nStarted scan_beam_profile (%s) on %s, expected to complete on %s.\n'%(curr_time_str,time.strftime("%d%b%Y %H:%M:%S", time.localtime()),time.strftime("%d%b%Y %H:%M:%S", time.localtime(time.time()+total_time)))
    init_line2 = 'y_poss = %s.'%(str(y_poss))
    print(init_line)
    print(init_line2)
    log_txt = [unicode(input_line),unicode(init_line),unicode(init_line2),
               u'\n\n'+unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    def prepare_take_data(is_sc):
        global images, data_path, curr_y_poss
        images = []
        curr_y_poss = []
        move_to_y(y_poss[0])
        if is_sc:
            block_laser()
            unblock_sc()
            data_path = os.path.join(main_path,'sc_data.npz')
        else:
            unblock_laser()
            block_sc()
            data_path = os.path.join(main_path,'toptica_data.npz')
        plt.pause(2)
        while abs(get_pos()[1] - y_poss[0]) > 1:
            plt.pause(0.1)
    
    def take_data():
        curr_y_poss.append(get_pos()[1])
        try:
            curr_img = cam.grab_image(exposure_time='0.00001s')
        except:
            cam.start_live_video()
            curr_img = cam.grab_image(exposure_time='0.00001s')
        images.append(copy.copy(curr_img))
    
    def save_data():
        try:
            np.savez(data_path,
             y_poss=curr_y_poss,
             images=images)
        except:
            time.sleep(1)
            save_data()
    
    def finishing():
        cam.stop_live_video()
        block_laser()
        block_sc()
    
    prev_completed = ''
    _n = 1
    prints('\n')
    prepare_take_data(is_sc = True)
    start_time = time.time()
    for y_pos in y_poss:
        move_to_y(y_pos)
        plt.pause(0.5)
        take_data()
        
        completed = 'Scanning SC beam profile at %.3f um (%.2f percent)'%(y_pos,_n*100./total_len)
        prints(completed,prev_completed)
        prev_completed = completed
        _n += 1
    save_data()
    
    prev_completed = ''
    _n = 1
    prints('\n')
    prepare_take_data(is_sc = False)
    for y_pos in y_poss:
        move_to_y(y_pos)
        plt.pause(0.5)
        take_data()
        
        completed = 'Scanning 1550 beam profile at %.3f um (%.2f percent)'%(y_pos,_n*100./total_len)
        prints(completed,prev_completed)
        prev_completed = completed
        _n += 1
    save_data()
        
    end_time = time.time()
    finishing()
    
    print '\nDone! Time spent = %.1fs'%(time.time()-start_time)
    play_sound(complete)

    try:
        anal_scan_beam_profile(curr_time_str)
        plt.pause(1e-6)
    except:
        print("%s not analyzed"%curr_time_str)

#%%
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


def keyboard_control_piezo(step=0.1):
    global prev
    prev = ''
    def _up():
        global prev
        try:
            move_to_z(get_pos()[2]-step)
        except:
            pass
        curr_pos = str(get_pos())
        prints(curr_pos,prev)
        prev = curr_pos
    def _down():
        global prev
        try:
            move_to_z(get_pos()[2]+step)
        except:
            pass
        curr_pos = str(get_pos())
        prints(curr_pos,prev)
        prev = curr_pos
    def _left():
        global prev
        try:
            move_to_x(get_pos()[0]-step)
        except:
            pass
        curr_pos = str(get_pos())
        prints(curr_pos,prev)
        prev = curr_pos
    def _right():
        global prev
        try:
            move_to_x(get_pos()[0]+step)
        except:
            pass
        curr_pos = str(get_pos())
        prints(curr_pos,prev)
        prev = curr_pos
    def _closer():
        global prev
        try:
            move_to_y(get_pos()[1]+step)
        except:
            pass
        curr_pos = str(get_pos())
        prints(curr_pos,prev)
        prev = curr_pos
    def _further():
        global prev
        try:
            move_to_y(get_pos()[1]-step)
        except:
            pass
        curr_pos = str(get_pos())
        prints(curr_pos,prev)
        prev = curr_pos
    start_keyboard_listening(['up','down','left','right','page_up','page_down'],[_up,_down,_left,_right,_closer,_further])