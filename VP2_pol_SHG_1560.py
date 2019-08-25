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
from shutter_rotational_stage import *
import avaspec as H
from powermeter_analog import *
from rot_stages_noGUI import * #NEED TO CHANGE OFFSET ANGLES FOR ROT STAGES IN VP2_calib_1560_rot
from VP2_calib_1560_rot import *
hero_pixel_wavelengths = np.load('D:/WMP_setup/Python_codes/hero_pixel_wavelengths.npy')
from VP2_pol_SHG_1560_anal import *
from piezo import *
from WMP_fine_dl import *


sys.path.append(r'D:/WMP_setup/Python_codes')
hero_pixel_wavelengths = np.load(r'D:/WMP_setup/Python_codes/hero_pixel_wavelengths.npy')
try:
    reVP1test780coeff = np.load(r'D:\Nonlinear_setup\Python_codes\reVP1test780coeff.npy')
    reVP1test390coeff = np.load(r'D:\Nonlinear_setup\Python_codes\reVP1test390coeff.npy')
except:
    print('reVP1test calibration file(s) not found.')
    reVP1test780coeff = np.array([1])
    reVP1test390coeff = np.array([1])

#---------------------------------------------------#
#Definition of codes to be used in actual experiments

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

#---------------------------------------------------#
#codes for the actual experiments
#code #1: for polarized SHG experiment (6 petals)


def VP2_pol_SHG(sample,hero_int_time=2000,hero_avg_num=1,num_of_spec=5,alphas=np.arange(0,360,2),ana_offset=-50.99,pump_wl=1560,timesleep=0.5,log=''):
    """
    Obtain Hero spectrum for each alpha.
    Alpha is defined as the angle for the ANALYZER.
    The HWP controls the input polarization, and therefore its angle is Alpha/2.
    QWP is not needed in this experiment code, since circular poalrization parameters are fixed in the supercontinuum laser beam.
    SHG pump beam wavelength is 1560nm.
    The laser beam we are using comes from the 780nm output from Toptica, this is the reminiscent 1560nm beam.

    As carachterized on 21-nov-2017, 1560nm average power before 100x lens is ~3.2mW, and 100x lens transmittance is ~40%.

    for horizontal reference frame (0deg at horizontal, and counter-clockwise rotation in the view of the microscope)
    HWP = -9.67
    ANA = -50.99
    """
    main_path=r'D:\Nonlinear_setup\Experimental_data\VP2_pol_SHG\%s'%sample 
    os.makedirs(main_path)
    block_laser()
    total_len = len(alphas)
    total_time = total_len*hero_avg_num*num_of_spec*hero_int_time/1000. + total_len*(0.5+timesleep+1)
    start_time = time.time()
    init_pos = get_pos()

    init_line = '\nStarted VP2_pol_SHG (%s) on %s, expected to complete on %s\nhero_int_time = %.2f ms, hero_avg_num = %i, num_of_spec = %i, timesleep = %.2f s, alphas = %s, piezo stage position at %s\n'%(sample,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+total_time)),hero_int_time,hero_avg_num,num_of_spec,timesleep,str(alphas),str(init_pos))
    print(init_line)
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    np.save(os.path.join(main_path,'alphas'),alphas)
    _alphas_csv=open(os.path.join(main_path, "alphas.csv"), "wb") 
    alphas_csv=csv.writer(_alphas_csv)
    alphas_csv.writerow(alphas)
    _alphas_csv.close()

    H.initialise_hero()
    H.hero_int_time(hero_int_time)
    H.hero_avg_num(hero_avg_num)

    def HOME_HWP_LOOP():
        home_HWP()
    def HOME_ANA_LOOP():
        home_ANA()
    def HOME_QWP_LOOP():
        home_QWP()
    print('Homing HWP, QWP, ANA')
    HWP_th = threading.Thread(target=HOME_HWP_LOOP)
    ANA_th = threading.Thread(target=HOME_ANA_LOOP)
    QWP_th = threading.Thread(target=HOME_QWP_LOOP)
    HWP_th.start()
    ANA_th.start()
    QWP_th.start()

    BCKGND_SPEC = H.hero_spec()
    
    np.save(os.path.join(main_path,'BCKGND_SPEC'),BCKGND_SPEC)
    _BCKGND_SPEC_csv=open(os.path.join(main_path, "BCKGND_SPEC.csv"), "wb")
    BCKGND_SPEC_csv=csv.writer(_BCKGND_SPEC_csv)
    BCKGND_SPEC_csv.writerow(BCKGND_SPEC)
    _BCKGND_SPEC_csv.close()
     
    HWP_th.join()
    ANA_th.join()
    QWP_th.join()

    pma_wl(pump_wl)

    def prepare_take_data_inside_loop():
        global specs
        specs = []
        
    def prepare_take_data_outside_loop():
        global powers, powers_dev
        powers = []
        powers_dev = []

    def take_data_specs():
        curr_spec = copy.copy(H.hero_spec())
        specs.append(curr_spec)

    def take_data_powers():
        _powers=[]
        for i in range(10):
            _powers.append(pma_power())
            time.sleep(0.05)
        
        curr_power = np.mean(_powers)
        curr_power_dev = np.std(_powers)
        
        powers.append(curr_power)
        powers_dev.append(curr_power_dev)

    def save_data_specs(a):
        try:
            np.save(os.path.join(main_path,'SPECS_a%s'%a),np.array(specs))
        except:
            time.sleep(1)
            save_data_specs(a)

    def save_data_powers():
        try:
            np.save(os.path.join(main_path,'powers'),np.array(powers))
            np.save(os.path.join(main_path,'powers_dev'),np.array(powers_dev))
        except:
            time.sleep(1)
            save_data_powers()
    
    prev_completed = ''
    _n = 0
    prints('\n')
    
    prepare_take_data_outside_loop()
    for alpha in alphas:
        completed = 'alpha at %.1fdeg (%.2f percent)'%(alpha,_n*100./total_len)
        prints(completed,prev_completed)
        prev_completed = completed

        move_alpha(alpha,ANA_off=ana_offset)
        prepare_take_data_inside_loop()
        
        unblock_laser()
        plt.pause(0.5)
        for i in range(num_of_spec):
            take_data_specs()
        take_data_powers()

        block_laser()
        save_data_specs(a='%i'%(alpha*100))
        save_data_powers()

        plt.pause(timesleep)
        _n += 1

    block_laser()
    print 'Done! Time spent = %is'%(time.time()-start_time)
    play_sound(complete)

    H.hero_shutdown()
    try:
        VP2_pol_SHG_1560_anal(sample)
        plt.pause(1e-6)
    except:
        pass


#---------------------------------------------------#
#code #2: for SHG mapping experiment, with defined alphas

def VP2_mapping_SHG(sample,ini_pos,map_size,resol=1,hero_int_time=2000,hero_avg_num=1,num_of_spec=5,alphas=[233,180],ana_offset=-50.99,pump_wl=1560,timesleep=0.5,log=''):

    Xi,Zi=ini_pos
    Xl,Zl=map_size
    Xs=np.arange(Xi,Xi+Xl+resol,resol)
    Zs=np.arange(Zi,Zi+Zl+resol,resol)

    block_laser()
    block_sc()
    total_len = len(Xs)*len(Zs)*len(alphas)
    total_time = total_len*(hero_avg_num*num_of_spec*hero_int_time/1000. +2)
    start_time = time.time()

    init_line = '\nStarted VP2_mapping_SHG (%s) on %s, expected to complete on %s\nhero_int_time = %.2f ms, hero_avg_num = %i, num_of_spec = %i, timesleep = %.2f s, alphas = %s, initial position = %s, mapping size = %s, and resolution = %.1f um\n'%(
        sample,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+total_time)),hero_int_time,hero_avg_num,num_of_spec,timesleep,str(alphas),str(ini_pos),str(map_size),resol)
    print(init_line)
    ans=raw_input('Continue? Input n to cancel. ')
    if ans=='n' or ans=='N':
        return
    main_path=r'D:\Nonlinear_setup\Experimental_data\VP2_mapping_SHG\%s'%sample 
    os.makedirs(main_path)
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    np.save(os.path.join(main_path,'alphas'),alphas)
    np.save(os.path.join(main_path,'Xs'),Xs)
    np.save(os.path.join(main_path,'Zs'),Zs)
    _alphas_csv=open(os.path.join(main_path, "alphas.csv"), "wb") 
    alphas_csv=csv.writer(_alphas_csv)
    alphas_csv.writerow(alphas)
    _alphas_csv.close()

    H.initialise_hero()
    H.hero_int_time(hero_int_time)
    H.hero_avg_num(hero_avg_num)

    def HOME_HWP_LOOP():
        home_HWP()
    def HOME_QWP_LOOP():
        home_QWP()
    def HOME_ANA_LOOP():
        home_ANA()
    print('Homing HWP, QWP, ANA')
    HWP_th = threading.Thread(target=HOME_HWP_LOOP)
    QWP_th = threading.Thread(target=HOME_QWP_LOOP)
    ANA_th = threading.Thread(target=HOME_ANA_LOOP)
    HWP_th.start()
    QWP_th.start()
    ANA_th.start()

    BCKGND_SPEC = H.hero_spec()
    
    np.save(os.path.join(main_path,'BCKGND_SPEC'),BCKGND_SPEC)
    _BCKGND_SPEC_csv=open(os.path.join(main_path, "BCKGND_SPEC.csv"), "wb")
    BCKGND_SPEC_csv=csv.writer(_BCKGND_SPEC_csv)
    BCKGND_SPEC_csv.writerow(BCKGND_SPEC)
    _BCKGND_SPEC_csv.close()
     
    HWP_th.join()
    QWP_th.join()
    ANA_th.join()

    pma_wl(pump_wl)

    def prepare_take_data_spec():
        global specs
        specs = []
        
    def prepare_take_data_pm_Z():
        global powers, powers_dev
        powers = []
        powers_dev = []

    def prepare_take_data_pm_X():
        powers.append([])
        powers_dev.append([])

    def take_data_specs():
        curr_spec = copy.copy(H.hero_spec())
        specs.append(curr_spec)

    def take_data_powers():
        _powers=[]
        for i in range(10):
            _powers.append(pma_power())
            time.sleep(0.05)
        
        curr_power = np.mean(_powers)
        curr_power_dev = np.std(_powers)
        
        powers[-1].append(curr_power)
        powers_dev[-1].append(curr_power_dev)

    def save_data_specs(a,x,z):
        try:
            np.save(os.path.join(main_path,'SPECS_a%s_x%s_z%s'%(a,x,z)),np.array(specs))
        except:
            time.sleep(1)
            save_data_specs(a,x,z)

    def save_data_powers(a):
        try:
            np.save(os.path.join(main_path,'powers_a%s'%a),np.array(powers))
            np.save(os.path.join(main_path,'powers_dev_a%s'%a),np.array(powers_dev))
        except:
            time.sleep(1)
            save_data_powers(a)
    
    prev_completed = ''
    _n = 0
    prints('\n')
    unblock_laser()
    unblock_sc()
    
    for alpha in alphas:
        move_alpha(alpha,ANA_off=ana_offset)
        prepare_take_data_pm_Z()
        for Z in Zs:
            move_to_z(Z)
            prepare_take_data_pm_X()
            for X in Xs:
                move_to_x(X)
                completed = 'alpha at %.1fdeg, Z at %.1f um, X at %.1f um (%.2f percent)'%(alpha,Z,X,_n*100./total_len)
                prints(completed,prev_completed)
                prev_completed = completed
                
                prepare_take_data_spec()
                
#                unblock_laser()
                plt.pause(0.5)
                for i in range(num_of_spec):
                    take_data_specs()
                take_data_powers()

#                block_laser()
                save_data_specs(a='%i'%(alpha*100),x='%i'%(X*100),z='%i'%(Z*100))
                save_data_powers(a='%i'%(alpha*100))

                plt.pause(timesleep)
                _n += 1

    block_laser()
    block_sc()
    print 'Done! Time spent = %is'%(time.time()-start_time)
    play_sound(complete)

    H.hero_shutdown()
    try:
        VP2_mapping_SHG_anal(sample)
        plt.pause(1e-6)
    except:
        pass

#---------------------------------------------------#
#code #3: for SHG power dependence, with defined alphas

def VP2_power_dep_SHG(sample,hero_int_time=2000,hero_avg_num=1,num_of_spec=5,alphas=np.arange(0,360,2),hwp_offset=-9.67,ana_offset=-50.99,pump_wl=1560,timesleep=0.5,log=''):
    """

    In this power dependence measurement, analyser holds its position.

    After aligning HWO and ANA to alpha for maximum SHG, a polazizer must be inserted after HWP,
    so by rotating HWP we change the power pumping SHG, and do not change the alpha.
    
    The objective is to observe how SHG intensity varies with varying pump power.

    Obtain Hero spectrum for each HWP position (alpha/2).
    Alpha is defined as the angle for the polarization after HWP.

    The HWP controls the input polarization, and therefore its angle is Alpha/2.

    SHG pump beam wavelength is 1560nm.
    The laser beam we are using comes from the 780nm output from Toptica, this is the reminiscent 1560nm beam.

    As carachterized on 21-nov-2017, 1560nm average power before 100x lens is ~3.2mW, and 100x lens transmittance is ~40%.

    for horizontal reference frame (0deg at horizontal, and counter-clockwise rotation in the view of the microscope)
    HWP = -9.67
    ANA = -50.99
    """
    main_path=r'D:\Nonlinear_setup\Experimental_data\VP2_power_dep_SHG\%s'%sample 
    os.makedirs(main_path)
    block_laser()
    total_len = len(alphas)
    total_time = total_len*hero_avg_num*num_of_spec*hero_int_time/1000. + total_len*(0.5+timesleep+1)
    start_time = time.time()
    init_pos = get_pos()

    init_line = '\nStarted VP2_power_dep_SHG (%s) on %s, expected to complete on %s\nhero_int_time = %.2f ms, hero_avg_num = %i, num_of_spec = %i, timesleep = %.2f s, alphas = %s, piezo stage position at %s\n'%(sample,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+total_time)),hero_int_time,hero_avg_num,num_of_spec,timesleep,str(alphas),str(init_pos))
    print(init_line)
    ans=raw_input('To run this experiment, a polarizer must be introduced after HWP. Have you done this modification? Press n to cancel, press any other key to proceed.')
    if ans=='n' or ans=='N':
        return
    
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')

    np.save(os.path.join(main_path,'alphas'),alphas)
    _alphas_csv=open(os.path.join(main_path, "alphas.csv"), "wb") 
    alphas_csv=csv.writer(_alphas_csv)
    alphas_csv.writerow(alphas)
    _alphas_csv.close()

    H.initialise_hero()
    H.hero_int_time(hero_int_time)
    H.hero_avg_num(hero_avg_num)

    def HOME_HWP_LOOP():
        home_HWP()
    print('Homing HWP')
    HWP_th = threading.Thread(target=HOME_HWP_LOOP)
    HWP_th.start()

    BCKGND_SPEC = H.hero_spec()

    np.save(os.path.join(main_path,'BCKGND_SPEC'),BCKGND_SPEC)
    _BCKGND_SPEC_csv=open(os.path.join(main_path, "BCKGND_SPEC.csv"), "wb")
    BCKGND_SPEC_csv=csv.writer(_BCKGND_SPEC_csv)
    BCKGND_SPEC_csv.writerow(BCKGND_SPEC)
    _BCKGND_SPEC_csv.close()

    HWP_th.join()


    pma_wl(pump_wl)

    def prepare_take_data_inside_loop():
        global specs
        specs = []

    def prepare_take_data_outside_loop():
        global powers, powers_dev
        powers = []
        powers_dev = []

    def take_data_specs():
        curr_spec = copy.copy(H.hero_spec())
        specs.append(curr_spec)

    def take_data_powers():
        _powers=[]
        for i in range(10):
            _powers.append(pma_power())
            time.sleep(0.05)

        curr_power = np.mean(_powers)
        curr_power_dev = np.std(_powers)

        powers.append(curr_power)
        powers_dev.append(curr_power_dev)

    def save_data_specs(a):
        try:
            np.save(os.path.join(main_path,'SPECS_a%s'%a),np.array(specs))
        except:
            time.sleep(1)
            save_data_specs(a)

    def save_data_powers():
        try:
            np.save(os.path.join(main_path,'powers'),np.array(powers))
            np.save(os.path.join(main_path,'powers_dev'),np.array(powers_dev))
        except:
            time.sleep(1)
            save_data_powers()

    prev_completed = ''
    _n = 0
    prints('\n')

    prepare_take_data_outside_loop()
    for alpha in alphas:
        completed = 'alpha at %.1fdeg (%.2f percent)'%(alpha,_n*100./total_len)
        prints(completed,prev_completed)
        prev_completed = completed

        move_HWP(alpha/2 + hwp_offset)
        prepare_take_data_inside_loop()

        unblock_laser()
        plt.pause(0.5)
        for i in range(num_of_spec):
            take_data_specs()
        take_data_powers()

        block_laser()
        save_data_specs(a='%i'%(alpha*100))
        save_data_powers()

        plt.pause(timesleep)
        _n += 1

    block_laser()
    print 'Done! Time spent = %is'%(time.time()-start_time)
    play_sound(complete)

    H.hero_shutdown()



#_______________________________________________________________
def move_HWP(ang=None):
    return move_rot1(ang)
def home_HWP():
    return home_rot1()
def move_ANA(ang=None):
    return move_rot2(ang)
def home_ANA():
    return home_rot2()
def move_QWP(ang=None):
    return move_rot3(ang)
def home_QWP():
    return home_rot3()
    
def move_alpha(alpha,ANA_off=-50.99):
    def HWP_LOOP():
        move_1560_to_alpha(alpha)
    def ANA_LOOP():
        move_ANA(alpha + ANA_off)
    HWP_th = threading.Thread(target=HWP_LOOP)
    ANA_th = threading.Thread(target=ANA_LOOP)
    HWP_th.start()
    ANA_th.start()
    HWP_th.join()
    ANA_th.join()

def home_all_rot():
    def HWP_loop():
        home_rot1()
    def QWP_loop():
        home_rot3()
    def ANA_loop():
        home_rot2()
    pol_th=threading.Thread(target=HWP_loop)
    QWP_th=threading.Thread(target=QWP_loop)
    ANA_th=threading.Thread(target=ANA_loop)
    pol_th.start()
    QWP_th.start()
    ANA_th.start()
    pol_th.join()
    QWP_th.join()
    ANA_th.join()

#def home_HA():
#    def HWP_LOOP():
#        home_HWP()
#    def ANA_LOOP():
#        home_ANA()
#    HWP_th = threading.Thread(target=HWP_LOOP)
#    ANA_th = threading.Thread(target=ANA_LOOP)
#    HWP_th.start()
#    ANA_th.start()
#    HWP_th.join()
#    ANA_th.join()
    