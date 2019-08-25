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
sys.path.append(os.path.abspath(r"D:\WMP_setup\Python_codes"))
sys.path.append(os.path.abspath(r"D:\Nonlinear_setup\Python_codes"))
from pygame import mixer
mixer.init()

from TL_slider import *
import avaspec as H
from powermeter_analog import *
#from rot_stages import *
from rot_stages_noGUI import *
hero_pixel_wavelengths = np.load('D:/WMP_setup/Python_codes/hero_pixel_wavelengths.npy')
from THG_1560_anal import *
from piezo import *

def THG_1560(sample,hero_int_time=2000,hero_avg_num=1,num_of_spec=5,alphas=np.arange(0,360,5),timesleep=0.5,log=''):
    """
    Obtain Hero spectrum for each alpha and gamma

    for horizontal reference frame (0deg at horizontal, and counter-clockwise rotation in the view of the microscope)
    POL = -4 (actually a HWP ZO for 1560 is bing used here)
    ANA = -47.8
    """
    main_path=r'D:\Nonlinear_setup\Experimental_data\THG_1560\%s'%sample 
    os.makedirs(main_path)
    block_laser()
    total_len = len(alphas)
    total_time = total_len*hero_avg_num*num_of_spec*hero_int_time/1000. + total_len*(0.5+timesleep+1)
    start_time = time.time()

    init_line = 'Started THG_1560 (%s) on %s, expected to complete on %s\nhero_int_time = %.2fms, hero_avg_num = %i, num_of_spec = %i, timesleep = %s\nalphas = %s'%(sample,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+total_time)),hero_int_time,hero_avg_num,num_of_spec,timesleep,str(alphas))
    print(init_line)
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    np.save(os.path.join(main_path,'alphas'),alphas)
    _alphas_csv=open(os.path.join(main_path, "alphas.csv"), "wb") 
    alphas_csv=csv.writer(_alphas_csv)
    alphas_csv.writerow(alphas)
    _alphas_csv.close()

#    np.save(os.path.join(main_path,'gammas'),gammas)
#    _gammas_csv=open(os.path.join(main_path, "gammas.csv"), "wb") 
#    gammas_csv=csv.writer(_gammas_csv)
#    gammas_csv.writerow(gammas)
#    _gammas_csv.close()

    H.initialise_hero()
    H.hero_int_time(hero_int_time)
    H.hero_avg_num(hero_avg_num)

    def HOME_POL_LOOP():
        home_POL()
#    def HOME_QWP_LOOP():
#        home_QWP()
#    def HOME_ANA_LOOP():
#        home_ANA()
    print('Homing POL')
    POL_th = threading.Thread(target=HOME_POL_LOOP)
#    QWP_th = threading.Thread(target=HOME_QWP_LOOP)
#    ANA_th = threading.Thread(target=HOME_ANA_LOOP)
    POL_th.start()
#    QWP_th.start()
#    ANA_th.start()
#
    BCKGND_SPEC = H.hero_spec()
    
    np.save(os.path.join(main_path,'BCKGND_SPEC'),BCKGND_SPEC)
    _BCKGND_SPEC_csv=open(os.path.join(main_path, "BCKGND_SPEC.csv"), "wb")
    BCKGND_SPEC_csv=csv.writer(_BCKGND_SPEC_csv)
    BCKGND_SPEC_csv.writerow(BCKGND_SPEC)
    _BCKGND_SPEC_csv.close()
     
    POL_th.join()
#    QWP_th.join()
#    ANA_th.join()

    def prepare_take_data():
        global specs
        specs = []

    def prepare_take_data_pm():
            global pm_ref
            pm_ref = []
            pma_wl(1560)

    def take_data():
        curr_spec = copy.copy(H.hero_spec())
        specs.append(curr_spec)

    def take_data_2():
        curr_power = pma_power()
        pm_ref.append(curr_power)

    def save_data(a):
        try:
            np.save(os.path.join(main_path,'SPECS_a%s'%(a)),np.array(specs))
            np.save(os.path.join(main_path,'PM_ref'),np.array(pm_ref))
        except:
            time.sleep(1)
            save_data(a)
    
    prepare_take_data_pm()
    
    prev_completed = ''
    _n = 0
    prints('\n')
    for alpha in alphas:
        completed = 'alpha at %.1fdeg (%.2f percent)'%(alpha,_n*100./total_len)
        prints(completed,prev_completed)
        prev_completed = completed

        move_PQA(alpha)
        prepare_take_data()
        
        unblock_laser()
        plt.pause(0.5)
        for i in range(num_of_spec):
            take_data()
        take_data_2()
        block_laser()
        
        save_data(a='%i'%(alpha*100))

        plt.pause(timesleep)
        _n += 1

    block_laser()
    print 'Done! Time spent = %is'%(time.time()-start_time)
    play_sound(complete)

    H.hero_shutdown()
    try:
        anal_THG_1560(sample)
        plt.pause(1e-6)
    except:
        pass



def polarized_SHG_singlebeam2(sample,hero_int_time=2000,hero_avg_num=1,num_of_spec=5,alpha=-5,gammas=np.arange(-45,45.1,5),betas=np.arange(0,360,5),timesleep=0.5,log=''):
    """
    Obtain Hero spectrum for fix alpha, change gamma and scan beta (analyzer angle + offset)

    for horizontal reference frame (0deg at horizontal, and counter-clockwise rotation in the view of the microscope)
    POL = -37.9
    QWP = 25.5 (fast/slow), use +-45 around this
    ANA = -51.5
    """
    main_path=r'D:\Nonlinear_setup\Experimental_data\reVP1\%s'%sample 
    os.makedirs(main_path)
    block_laser()
    total_len = len(betas)*len(gammas)
    total_time = total_len*hero_avg_num*num_of_spec*hero_int_time/1000. + total_len*(0.5+timesleep+1)
    start_time = time.time()

    init_line = 'Started polarized_SHG_singlebeam2 (%s) on %s, expected to complete on %s\nhero_int_time = %.2fms, hero_avg_num = %i, num_of_spec = %i, timesleep = %.2fs, alpha = %.2fdeg\ngammas = %s\nbetas = %s'%(sample,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+total_time)),hero_int_time,hero_avg_num,num_of_spec,timesleep,alpha,str(gammas),str(betas))
    print(init_line)
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    np.save(os.path.join(main_path,'betas'),betas)

    np.save(os.path.join(main_path,'gammas'),gammas)

    H.initialise_hero()
    H.hero_int_time(hero_int_time)
    H.hero_avg_num(hero_avg_num)
  
    align_pol_ang = -37.9
    align_QWP_ang = 25.5
    align_ana_ang = -51.5
    def HOME_MOVE_POL_LOOP():
        home_POL()
        move_POL(alpha+align_pol_ang)
    def HOME_QWP_LOOP():
        home_QWP()
    def HOME_ANA_LOOP():
        home_ANA()
    print('Homing POL, QWP, ANA')
    POL_th = threading.Thread(target=HOME_MOVE_POL_LOOP)
    QWP_th = threading.Thread(target=HOME_QWP_LOOP)
    ANA_th = threading.Thread(target=HOME_ANA_LOOP)
    POL_th.start()
    QWP_th.start()
    ANA_th.start()

    BCKGND_SPEC = H.hero_spec()
    
    np.save(os.path.join(main_path,'BCKGND_SPEC'),BCKGND_SPEC)
    _BCKGND_SPEC_csv=open(os.path.join(main_path, "BCKGND_SPEC.csv"), "wb")
    BCKGND_SPEC_csv=csv.writer(_BCKGND_SPEC_csv)
    BCKGND_SPEC_csv.writerow(BCKGND_SPEC)
    _BCKGND_SPEC_csv.close()
     
    POL_th.join()
    QWP_th.join()
    ANA_th.join()

    def prepare_take_data():
        global specs
        specs = []
        
    def prepare_take_data_pm():
        global pm_ref
        pm_ref = []
        pma_wl(1560)

    def take_data():
        curr_spec = copy.copy(H.hero_spec())
        specs.append(curr_spec)
        
    def take_data_2():
        curr_power = pma_power()
        pm_ref.append(curr_power)

    def save_data(g,b):
        try:
            np.save(os.path.join(main_path,'SPECS_g%s_b%s'%(g,b)),np.array(specs))
            np.save(os.path.join(main_path,'PM_g%s'%(g)),np.array(pm_ref))
        except:
            time.sleep(1)
            save_data(g,b)
    
    prev_completed = ''
    _n = 0
    prints('\n')
    for gamma in gammas:
        def MOVE_QWP_LOOP():
            move_QWP(alpha+align_QWP_ang+gamma)
        def HOME_ANA_LOOP():
            home_ANA()
        QWP_th = threading.Thread(target=MOVE_QWP_LOOP)
        ANA_th = threading.Thread(target=HOME_ANA_LOOP)
        QWP_th.start()
        ANA_th.start()
        QWP_th.join()
        ANA_th.join()
        
        prepare_take_data_pm()
        
        for beta in betas:
            completed = 'gamma at %.1fdeg, beta at %.1fdeg (%.2f percent)'%(gamma,beta,_n*100./total_len)
            prints(completed,prev_completed)
            prev_completed = completed
            
            move_ANA(beta+align_ana_ang)
           
            prepare_take_data()
            
            unblock_laser()
            time.sleep(1.0)
            plt.pause(0.5)
            for i in range(num_of_spec):
                take_data()
            take_data_2()
            block_laser()
            
            save_data(g='%i'%(gamma*100),b='%i'%(beta*100))

            plt.pause(timesleep)
            _n += 1

    block_laser()
    print 'Done! Time spent = %is'%(time.time()-start_time)
    play_sound(complete)

    H.hero_shutdown()
    try:
        anal_polarized_SHG_singlebeam2(sample)
        plt.pause(1e-6)
    except:
        pass

#-----------------------------------------------------------------------#
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

def move_PQA(alpha):
    align_pol_ang = -4
#    align_QWP_ang = 25.5
#    align_ana_ang = -47.8
    def pol_loop():
        move_POL(alpha+align_pol_ang)
#    def QWP_loop():
#        move_QWP(alpha+align_QWP_ang)
#    def ANA_loop():
#        move_ANA(alpha+align_ana_ang)
    pol_th=threading.Thread(target=pol_loop)
#    QWP_th=threading.Thread(target=QWP_loop)
#    ANA_th=threading.Thread(target=ANA_loop)
    pol_th.start()
#    QWP_th.start()
#    ANA_th.start()
    pol_th.join()
#    QWP_th.join()
#    ANA_th.join()

def home_PQA():
    def HOME_POL_LOOP():
        home_POL()
#    def HOME_QWP_LOOP():
#        home_QWP()
#    def HOME_ANA_LOOP():
#        home_ANA()
    POL_th = threading.Thread(target=HOME_POL_LOOP)
#    QWP_th = threading.Thread(target=HOME_QWP_LOOP)
#    ANA_th = threading.Thread(target=HOME_ANA_LOOP)
    POL_th.start()
#    QWP_th.start()
#    ANA_th.start()
    POL_th.join()
#    QWP_th.join()
#    ANA_th.join()


def create_date_str():
    return time.strftime("%d%b%y_%H%M", time.localtime())

def uniArray(array_unicode):
    items = [x.encode('utf-8') for x in array_unicode]
    array_unicode = np.array([items]) # remove the brackets for line breaks
    return array_unicode

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