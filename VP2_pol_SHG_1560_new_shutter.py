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

try:
    import TL_slider #filter
except:
    print "Filter shutter not connected"
try:
    import TL_slider_2 #sc
except:
    print "SC shutter not connected"
try:
    import TL_slider_3 #toptica
except:
    print "1560 shutter not connected"
try:
    import avaspec as H
except:
    print "Hero not connected"
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
try:
    from lockin import *
except:
    print "lock-in not connected"
from rot_stages_noGUI import * #NEED TO CHANGE OFFSET ANGLES FOR ROT STAGES IN VP2_calib_1560_rot
#from rot_stages_noGUI_4 import *
from VP2_calib_1560_rot import *
hero_pixel_wavelengths = np.load('D:/WMP_setup/Python_codes/hero_pixel_wavelengths.npy')
from VP2_pol_SHG_1560_anal import *
from piezo import *
from WMP_fine_dl import *

#from sharp_blade_experiment_neo import *


sys.path.append(r'D:/WMP_setup/Python_codes')
hero_pixel_wavelengths = np.load(r'D:/WMP_setup/Python_codes/hero_pixel_wavelengths.npy')
try:
    reVP1test780coeff = np.load(r'D:\Nonlinear_setup\Python_codes\reVP1test780coeff.npy')
    reVP1test390coeff = np.load(r'D:\Nonlinear_setup\Python_codes\reVP1test390coeff.npy')
except:
    print('reVP1test calibration file(s) not found.')
    reVP1test780coeff = np.array([1])
    reVP1test390coeff = np.array([1])
#%%
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
#%%
#---------------------------------------------------#
#codes for the actual experiments
#code #1: for polarized SHG experiment (6 petals)


def VP2_pol_SHG(sample,hero_int_time=2000,hero_avg_num=1,num_of_spec=5,alphas=np.arange(0,360,2),ana_offset=-50.99,pump_wl=1560,sc_wl=756,sc_on=False,timesleep=0.5,log=''):
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
    total_len = len(alphas)
    total_time = (2*total_len)*hero_avg_num*num_of_spec*hero_int_time/1000. + total_len*(0.5+timesleep+1) + total_len*1.5
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

    if sc_on:
        unblock_sc()
    else:
        block_sc()

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
    plt.pause(0.01)
    ANA_th.start()
    plt.pause(0.01)
    QWP_th.start()

    block_laser()
    time.sleep(1)
    BCKGND_SPEC = H.hero_spec()
    unblock_laser()
    time.sleep(1)
    
    np.save(os.path.join(main_path,'BCKGND_SPEC'),BCKGND_SPEC)
     
    HWP_th.join()
    ANA_th.join()
    QWP_th.join()
    
    move_alpha(alphas[0])
    lockin_auto_gain()
    lockin_auto_phase()
    optimise_srs830_sensitivity()
    get_lockin_reading1()
    time.sleep(1)
    block_laser()

#    pma_wl(pump_wl)
#    pmd_wl(sc_wl)

    def prepare_take_data_inside_loop():
        global specs, specs_bg
        specs = []
        specs_bg = []
        
    def prepare_take_data_outside_loop():
        global powers, powers_dev, powers_sc, powers_sc_dev
        powers = []
#        powers_sc = []
        powers_dev = []
#        powers_sc_dev = []

    def take_data_specs():
        curr_spec = copy.copy(H.hero_spec())
        specs.append(curr_spec)
        block_laser()
        time.sleep(1)
        specs_bg.append(copy.copy(H.hero_spec()))
        unblock_laser()
        time.sleep(0.5)

    def take_data_powers():
        _powers=[]
#        _powers_sc=[]
        for i in range(10):
#            _powers.append(pma_power())
            _powers.append(get_lockin_reading1())
#        for i in range(10):
#            _powers_sc.append(pmd_power())
#            time.sleep(0.05)
        
        curr_power = np.mean(_powers)
        curr_power_dev = np.std(_powers)
#        curr_power_sc = np.mean(_powers_sc)
#        curr_power_sc_dev = np.std(_powers_sc)
        
        powers.append(curr_power)
        powers_dev.append(curr_power_dev)
#        powers_sc_dev.append(curr_power_sc_dev)

    def save_data_specs(a):
        try:
            np.save(os.path.join(main_path,'SPECS_a%s'%a),np.array(specs))
            np.save(os.path.join(main_path,'SPECS_BG_a%s'%a),np.array(specs_bg))
        except:
            time.sleep(1)
            save_data_specs(a)

    def save_data_powers():
        try:
            np.save(os.path.join(main_path,'powers'),np.array(powers))
            np.save(os.path.join(main_path,'powers_dev'),np.array(powers_dev))
#            np.save(os.path.join(main_path,'powers_sc'),np.array(powers_sc))
#            np.save(os.path.join(main_path,'powers_sc_dev'),np.array(powers_sc_dev))
        except:
            time.sleep(1)
            save_data_powers()

    prev_completed = ''
    _n = 0
    prints('\n')
    
    SP_filter_out()
    unblock_laser()
    time.sleep(1)
    
    prepare_take_data_outside_loop()
    for alpha in alphas:
        completed = 'alpha at %.1fdeg (%.2f percent)'%(alpha,_n*100./total_len)
        prints(completed,prev_completed)
        prev_completed = completed

        move_alpha(alpha,ANA_off=ana_offset)
        plt.pause(0.5)
        prepare_take_data_inside_loop()
        
        for i in range(num_of_spec):
            take_data_specs()
        take_data_powers()

        save_data_specs(a='%i'%(alpha*100))
        save_data_powers()

        plt.pause(timesleep)
        _n += 1

    block_laser()
#    SP_filter_in()
    block_sc()
    
    print 'Done! Time spent = %is'%(time.time()-start_time)
    play_sound(complete)

    H.hero_shutdown()
    try:
        VP2_pol_SHG_1560_anal(sample)
        plt.pause(1e-6)
    except:
        pass

#%%
#---------------------------------------------------#
#code #2: for SHG mapping experiment, with defined alphas

def VP2_mapping_SHG(sample,ini_pos,map_size,resol=1,hero_int_time=2000,hero_avg_num=1,num_of_spec=5,alphas=[233,180],analyzer_off_from_alpha=0,ana_offset=-50.99,pump_wl=1560,timesleep=0.5,pump_with_1560=True,pm_as_ref=False,log=''):

    Xi,Zi=ini_pos
    Xl,Zl=map_size
    Xs=np.arange(Xi,Xi+Xl+resol,resol)
    Zs=np.arange(Zi,Zi+Zl+resol,resol)

    block_laser()
    block_sc()
    SP_filter_out()
    total_len = len(Xs)*len(Zs)*len(alphas)
    total_time = total_len*(hero_avg_num*num_of_spec*hero_int_time/1000. +2)
    start_time = time.time()

    init_line = '\nStarted VP2_mapping_SHG (%s) on %s, expected to complete on %s\nhero_int_time = %.2f ms, hero_avg_num = %i, num_of_spec = %i, timesleep = %.2f s, alphas = %s, initial position = %s, mapping size = %s, and resolution = %.1f um\n'%(
        sample,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+total_time)),hero_int_time,hero_avg_num,num_of_spec,timesleep,str(alphas),str(ini_pos),str(map_size),resol)
    print(init_line)
#    ans=raw_input('Continue? Input n to cancel. ')
#    if ans=='n' or ans=='N':
#        return
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
    plt.pause(0.01)
    QWP_th.start()
    plt.pause(0.01)
    ANA_th.start()

    BCKGND_SPEC = copy.copy(H.hero_spec())
    
    np.save(os.path.join(main_path,'BCKGND_SPEC'),BCKGND_SPEC)
    _BCKGND_SPEC_csv=open(os.path.join(main_path, "BCKGND_SPEC.csv"), "wb")
    BCKGND_SPEC_csv=csv.writer(_BCKGND_SPEC_csv)
    BCKGND_SPEC_csv.writerow(BCKGND_SPEC)
    _BCKGND_SPEC_csv.close()
     
    HWP_th.join()
    QWP_th.join()
    ANA_th.join()
    
    unblock_laser()
    move_alpha(alphas[0])
    time.sleep(1)
    lockin_auto_gain()
    lockin_auto_phase()
    lockin_auto_gain()
    optimise_srs830_sensitivity()
    time.sleep(0.5)
    for i in range(10):
        get_lockin_reading1()
        time.sleep(0.1)
    block_laser()

#    pma_wl(pump_wl)

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
        for i in range(1):
#            _powers.append(pma_power())
            _powers.append(get_lockin_reading1())
#            time.sleep(0.05)
        
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
    if pump_with_1560:
        unblock_laser()
        block_sc()
#        SP_filter_out()
    else:
        unblock_sc()
        block_laser()
#        SP_filter_in()
    time.sleep(1)
    for alpha in alphas:
        move_alpha(alpha,ANA_off=ana_offset)
        move_A(move_A()+analyzer_off_from_alpha)
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
#%%
def VP2_zscan_SHG(sample,center_z,distance_from_center,resol=1,hero_int_time=2000,hero_avg_num=1,num_of_spec=5,alphas=[233,180],ana_offset=-50.99,pump_wl=1560,timesleep=0.5,pump_with_1560=True,log=''):

    y0=center_z
    yl=distance_from_center
    Ys=np.arange(y0-yl,y0+yl+resol,resol)

    block_laser()
#    block_sc()
    total_len = len(Ys)*len(alphas)
    total_time = total_len*(hero_avg_num*num_of_spec*hero_int_time/1000. +2)
    start_time = time.time()

    init_line = '\nStarted VP2_zscan_SHG (%s) on %s, expected to complete on %s\nhero_int_time = %.2f ms, hero_avg_num = %i, num_of_spec = %i, timesleep = %.2f s, alphas = %s, central Z = %s, distance from central Z = %s, and resolution = %.1f um\n'%(
        sample,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+total_time)),hero_int_time,hero_avg_num,num_of_spec,timesleep,str(alphas),str(y0),str(yl),resol)
    print(init_line)
#    ans=raw_input('Continue? Input n to cancel. ')
#    if ans=='n' or ans=='N':
#        return
    main_path=r'D:\Nonlinear_setup\Experimental_data\VP2_zscan_SHG\%s'%sample 
    os.makedirs(main_path)
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    np.save(os.path.join(main_path,'alphas'),alphas)
    np.save(os.path.join(main_path,'Ys'),Ys)
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
        
    def prepare_take_data_pm():
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

    def save_data_specs(a,y):
        try:
            np.save(os.path.join(main_path,'SPECS_a%s_y%s'%(a,y)),np.array(specs))
        except:
            time.sleep(1)
            save_data_specs(a,y)

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
    if pump_with_1560:
        unblock_laser()
        SP_filter_out()
    else:
#        unblock_sc()
        SP_filter_in()
    
    for alpha in alphas:
        move_alpha(alpha,ANA_off=ana_offset)
        prepare_take_data_pm()
        for Y in Ys:
            move_to_y(Y)
            completed = 'alpha at %.1fdeg, Y at %.1f um (%.2f percent)'%(alpha,Y,_n*100./total_len)
            prints(completed,prev_completed)
            prev_completed = completed
            
            prepare_take_data_spec()
            
#            unblock_laser()
            plt.pause(0.5)
            for i in range(num_of_spec):
                take_data_specs()
            take_data_powers()

#            block_laser()
            save_data_specs(a='%i'%(alpha*100),y='%i'%(Y*100))
            save_data_powers(a='%i'%(alpha*100))

            plt.pause(timesleep)
            _n += 1

    block_laser()
#    block_sc()
    print 'Done! Time spent = %is'%(time.time()-start_time)
    play_sound(complete)

    H.hero_shutdown()
    try:
        VP2_zscan_SHG_anal(sample)
        plt.pause(1e-6)
    except:
        pass
#%%
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
#%%
#code #4: for polarized SHG experiment (2 petals)


def VP2_pol_SHG_rotA(sample,hero_int_time=2000,hero_avg_num=1,num_of_spec=5,alphas=np.arange(0,90.001,15),betas=np.arange(0,360,2),ana_offset=-50.99,pump_wl=1560,sc_wl=756,sc_on=False,timesleep=0.5,log=''):
    """
    Obtain Hero spectrum for each alpha.
    Beta is defined as the angle for the ANALYZER.
    Alpha is defined as the polarization angle for the 1560 beam.
    The laser beam we are using comes from the 780nm output from Toptica, this is the reminiscent 1560nm beam.

    As carachterized on 21-nov-2017, 1560nm average power before 100x lens is ~3.2mW, and 100x lens transmittance is ~40%.

    for horizontal reference frame (0deg at horizontal, and counter-clockwise rotation in the view of the microscope)
    HWP = -9.67
    ANA = -50.99
    """
    main_path=r'D:\Nonlinear_setup\Experimental_data\VP2_pol_SHG_rotA\%s'%sample 
    os.makedirs(main_path)
    block_laser()
    total_len = len(alphas)*len(betas)
    total_time = total_len*hero_avg_num*num_of_spec*hero_int_time/1000. + total_len*(0.5+timesleep+1)
    start_time = time.time()
    init_pos = get_pos()

    init_line = '\nStarted VP2_pol_SHG_rotA (%s) on %s, expected to complete on %s\nhero_int_time = %.2f ms, hero_avg_num = %i, num_of_spec = %i, timesleep = %.2f s, alphas = %s, betas = %s, piezo stage position at %s\n'%(sample,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+total_time)),hero_int_time,hero_avg_num,num_of_spec,timesleep,str(alphas),str(betas),str(init_pos))
    print(init_line)
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    np.save(os.path.join(main_path,'alphas'),alphas)
    np.save(os.path.join(main_path,'betas'),betas)

    H.initialise_hero()
    H.hero_int_time(hero_int_time)
    H.hero_avg_num(hero_avg_num)

    if sc_on:
        SP_filter_in()
        unblock_sc()
    else:
        SP_filter_out()
        block_sc()

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
    plt.pause(0.01)
    ANA_th.start()
    plt.pause(0.01)
    QWP_th.start()

    block_laser()
    BCKGND_SPEC = H.hero_spec()
    unblock_laser()
    
    np.save(os.path.join(main_path,'BCKGND_SPEC'),BCKGND_SPEC)
     
    HWP_th.join()
    ANA_th.join()
    QWP_th.join()

    pma_wl(pump_wl)
#    pmd_wl(sc_wl)

    def prepare_take_data_inside_loop():
        global specs
        specs = []
        
    def prepare_take_data_outside_loop():
        global powers, powers_dev, powers_sc, powers_sc_dev
        powers = []
#        powers_sc = []
        powers_dev = []
#        powers_sc_dev = []

    def take_data_specs():
        curr_spec = copy.copy(H.hero_spec())
        specs.append(curr_spec)

    def take_data_powers():
        _powers=[]
#        _powers_sc=[]
        for i in range(10):
            _powers.append(pma_power())
#        for i in range(10):
#            _powers_sc.append(pmd_power())
#            time.sleep(0.05)
        
        curr_power = np.mean(_powers)
        curr_power_dev = np.std(_powers)
#        curr_power_sc = np.mean(_powers_sc)
#        curr_power_sc_dev = np.std(_powers_sc)
        
        powers.append(curr_power)
        powers_dev.append(curr_power_dev)
#        powers_sc_dev.append(curr_power_sc_dev)

    def save_data_specs(a,b):
        try:
            np.save(os.path.join(main_path,'SPECS_a%s_b%s'%(a,b)),np.array(specs))
        except:
            time.sleep(1)
            save_data_specs(a,b)

    def save_data_powers(a):
        try:
            np.save(os.path.join(main_path,'powers_a%s'%a),np.array(powers))
            np.save(os.path.join(main_path,'powers_dev_a%s'%a),np.array(powers_dev))
#            np.save(os.path.join(main_path,'powers_sc'),np.array(powers_sc))
#            np.save(os.path.join(main_path,'powers_sc_dev'),np.array(powers_sc_dev))
        except:
            time.sleep(1)
            save_data_powers(a)
    
    prev_completed = ''
    _n = 0
    prints('\n')
    
    for alpha in alphas:
        move_1560_to_alpha(alpha)
        prepare_take_data_outside_loop()
        for beta in betas:
            completed = 'alpha at %.1fdeg, beta at %.1fdeg (%.2f percent)'%(alpha,beta,_n*100./total_len)
            prints(completed,prev_completed)
            prev_completed = completed
    
            move_A(beta)
            plt.pause(0.5)
            prepare_take_data_inside_loop()
            
            for i in range(num_of_spec):
                take_data_specs()
            take_data_powers()
    
    
            save_data_specs(a='%i'%(alpha*100),b='%i'%(beta*100))
            save_data_powers(a='%i'%(alpha*100))
    
            plt.pause(timesleep)
            _n += 1

    block_laser()
#    SP_filter_in()
    block_sc()
    
    print 'Done! Time spent = %is'%(time.time()-start_time)
    play_sound(complete)

    H.hero_shutdown()
    try:
        VP2_pol_SHG_1560_anal(sample)
        plt.pause(1e-6)
    except:
        pass
#%%
def scan_delayline(sample=None,fdl_min=0,fdl_max=12,fdl_incre=0.01,hero_int_ms=1000,hero_ave_num=1):
    H.initialise_hero()
    if sample == None:
        curr_time_str = create_date_str()
    else:
        curr_time_str = sample
    lockin_auto_gain()
    lockin_auto_phase()
    optimise_srs830_sensitivity()
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline')
    H.hero_int_time(hero_int_ms)
    H.hero_avg_num(hero_ave_num)
    
    fdl_poss = np.arange(fdl_min,fdl_max+fdl_incre,fdl_incre)
    prev=''
    total_poss = len(fdl_poss)
    data = []
    for i,fdl_pos in enumerate(fdl_poss):
        move_fdl_abs(fdl_pos)
        curr_fdl_pos = get_fdl_pos()
        global curr_spec, curr_lockin
        curr_spec = []
        curr_lockin = []
        def hero_loop():
            curr_spec.append(H.hero_spec())
        hero_th = threading.Thread(target=hero_loop)
        def lockin_loop():
            while hero_th.isAlive():
                curr_lockin.append(get_lockin_reading1())
                time.sleep(0.1)
        lockin_th = threading.Thread(target=lockin_loop)
        hero_th.start()
        lockin_th.start()
        hero_th.join()
        lockin_th.join()
        data.append([(curr_fdl_pos,np.average(curr_lockin)),curr_spec[0]])
        np.save(os.path.join(main_path,curr_time_str+'.npy'),np.array(data))
            
        completed = u'%f (%.2f percent)'%(curr_fdl_pos,100.0*(float(i+1)/total_poss))
        prints(completed,prev)
        prev = completed
    
    print 'Done!'
    H.hero_shutdown()

#%%

def scan_delayline_with_bg(sample=None,fdl_min=0,fdl_max=12,fdl_incre=0.01,hero_int_ms=1000,hero_ave_num=1,chop_laser_as_bg=True,lockin_as_pump_ref=True):
    H.initialise_hero()
    if sample == None:
        curr_time_str = create_date_str()
    else:
        curr_time_str = sample
    global curr_spec, curr_lockin, curr_bg_spec, curr_lockin_bg
    unblock_laser()
    unblock_sc()
    time.sleep(1)
    lockin_auto_gain()
    lockin_auto_phase()
    optimise_srs830_sensitivity()
    time.sleep(1)
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_with_bg')
    H.hero_int_time(hero_int_ms)
    H.hero_avg_num(hero_ave_num)
    
    fdl_poss = np.arange(fdl_min,fdl_max+fdl_incre,fdl_incre)
    prev=''
    total_poss = len(fdl_poss)
    data = [(curr_time_str,'fdl_min = %f'%fdl_min,'fdl_max = %f'%fdl_max,'fdl_incre = %f'%fdl_incre,'hero_int_ms = %f'%hero_int_ms,'hero_ave_num = %i'%hero_ave_num,'chop_laser_as_bg = %s'%chop_laser_as_bg,'lockin_as_pump_ref = %s'%lockin_as_pump_ref)]
    np.save(os.path.join(main_path,curr_time_str+'.npy'),np.array(data))
    i = 0
    for fdl_pos in fdl_poss:
        move_fdl_abs(fdl_pos)
        curr_fdl_pos = get_fdl_pos()
        
        curr_spec = []
        curr_spec_bg = []
        curr_lockin = []
        curr_lockin_bg = []
        def hero_loop():
            curr_spec.append(H.hero_spec())
        hero_th = threading.Thread(target=hero_loop)
        def lockin_loop():
            while hero_th.isAlive():
                curr_lockin.append(get_lockin_reading1())
                time.sleep(0.1)
        
        def hero_bg_loop():
            curr_spec_bg.append(H.hero_spec())
        hero_bg_th = threading.Thread(target=hero_bg_loop)
        def lockin_bg_loop():
            while hero_bg_th.isAlive():
                curr_lockin_bg.append(get_lockin_reading1())
                time.sleep(0.1)
        
        if chop_laser_as_bg:
            block_laser()
            unblock_sc()
        else:
            unblock_laser()
            block_sc()
        time.sleep(1)
        lockin_bg_th = threading.Thread(target=lockin_loop)
        hero_bg_th.start()
        lockin_bg_th.start()
        hero_bg_th.join()
        lockin_bg_th.join()
                
        
        completed = u'%f BG (%.2f percent)'%(curr_fdl_pos,50.0*(float(i+1)/total_poss))
        prints(completed,prev+' '*4)
        prev = completed
        i+=1
        
        if chop_laser_as_bg:
            unblock_laser()
        else:
            unblock_sc()
        time.sleep(1)
        lockin_th = threading.Thread(target=lockin_loop)
        hero_th.start()
        lockin_th.start()
        hero_th.join()
        lockin_th.join()
        
        data.append([(curr_fdl_pos,np.average(curr_lockin),np.average(curr_lockin_bg)),curr_spec[0],curr_spec_bg[0]])
        np.save(os.path.join(main_path,curr_time_str+'.npy'),np.array(data))
            
        completed = u'%f DATA (%.2f percent)'%(curr_fdl_pos,50.0*(float(i+1)/total_poss))
        prints(completed,prev+' '*4)
        prev = completed
        i+=1
    
    print '\nDone!'
    H.hero_shutdown()

#%%
def scan_delayline_multiple_alpha_with_bg(sample=None,fdl_min=0,fdl_max=12,fdl_incre=0.01,hero_int_ms=1000,hero_ave_num=1,chop_laser_as_bg=True,lockin_as_pump_ref=True,alphas=np.arange(0,60,5),log=''):
    if sample == None:
        curr_time_str = create_date_str()
    else:
        curr_time_str = sample
    fdl_poss = np.arange(fdl_min,fdl_max+fdl_incre,fdl_incre)
    total_poss = len(fdl_poss)
    total_time = len(alphas)*total_poss*((hero_ave_num*hero_int_ms/1000.)*2 + 3)
    ini_line = [('\nStarted scan_delayline_multiple_alpha_with_bg (%s) on %s, expected to complete on %s'%(
        curr_time_str,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+total_time))),
             'fdl_min = %f'%fdl_min,
             'fdl_max = %f'%fdl_max,
             'fdl_incre = %f'%fdl_incre,
             'hero_int_ms = %f'%hero_int_ms,
             'hero_ave_num = %i'%hero_ave_num,
             'chop_laser_as_bg = %s'%chop_laser_as_bg,
             'lockin_as_pump_ref = %s'%lockin_as_pump_ref,
             'alphas = %s'%str(alphas)
             )]
    for l in ini_line:
        print(l)
    log_txt = [str(ini_line),
               unicode(log)+u'\n\n']
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_multiple_alpha_with_bg',curr_time_str)
    os.makedirs(main_path)
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    H.initialise_hero()
    home_all_rot()
    move_alpha(alphas[0])
    
    unblock_laser()
    unblock_sc()
    time.sleep(1)
    lockin_auto_gain()
    lockin_auto_phase()
    optimise_srs830_sensitivity()
    time.sleep(1)
    
    H.hero_int_time(hero_int_ms)
    H.hero_avg_num(hero_ave_num)
    
    global curr_spec, curr_lockin, curr_bg_spec, curr_lockin_bg
    prev=''
    for j,alpha in enumerate(alphas):
        def loop1():
            move_alpha(alpha)
        def loop2():
            move_fdl_abs(fdl_poss[0])
        th1=threading.Thread(target=loop1)
        th2=threading.Thread(target=loop2)
        th1.start()
        th2.start()
        th1.join()
        th2.join()
        curr_data = [(curr_time_str,
                 'fdl_min = %f'%fdl_min,
                 'fdl_max = %f'%fdl_max,
                 'fdl_incre = %f'%fdl_incre,
                 'hero_int_ms = %f'%hero_int_ms,
                 'hero_ave_num = %i'%hero_ave_num,
                 'chop_laser_as_bg = %s'%chop_laser_as_bg,
                 'lockin_as_pump_ref = %s'%lockin_as_pump_ref,
                 'alpha = %f'%alpha
                 )]
        i = 0
        for fdl_pos in fdl_poss:
            move_fdl_abs(fdl_pos)
            curr_fdl_pos = get_fdl_pos()
            
            curr_spec = []
            curr_spec_bg = []
            curr_lockin = []
            curr_lockin_bg = []
            def hero_loop():
                curr_spec.append(H.hero_spec())
            hero_th = threading.Thread(target=hero_loop)
            def lockin_loop():
                while hero_th.isAlive():
                    curr_lockin.append(get_lockin_reading1())
                    time.sleep(0.1)
            
            def hero_bg_loop():
                curr_spec_bg.append(H.hero_spec())
            hero_bg_th = threading.Thread(target=hero_bg_loop)
            def lockin_bg_loop():
                while hero_bg_th.isAlive():
                    curr_lockin_bg.append(get_lockin_reading1())
                    time.sleep(0.1)
            
            if chop_laser_as_bg:
                block_laser()
                unblock_sc()
            else:
                unblock_laser()
                block_sc()
            time.sleep(1)
            lockin_bg_th = threading.Thread(target=lockin_loop)
            hero_bg_th.start()
            lockin_bg_th.start()
            hero_bg_th.join()
            lockin_bg_th.join()
                    
            completed = u'Alpha at %04.1f deg (%05.2f percent). Delayline at %06.3f mm BG (%05.2f percent)'%(alpha,100.*(j)/len(alphas),curr_fdl_pos,50.0*(float(i+1)/total_poss))
            prints(completed,prev+' '*6)
            prev = completed
            i+=1
            
            if chop_laser_as_bg:
                unblock_laser()
            else:
                unblock_sc()
            time.sleep(1)
            lockin_th = threading.Thread(target=lockin_loop)
            hero_th.start()
            lockin_th.start()
            hero_th.join()
            lockin_th.join()
            
            curr_data.append([(curr_fdl_pos,np.average(curr_lockin),np.average(curr_lockin_bg)),curr_spec[0],curr_spec_bg[0]])
            np.save(os.path.join(main_path,'%s_a%i.npy'%(curr_time_str,alpha*100)),np.array(curr_data))
                
            completed = u'Alpha at %04.1f deg (%05.2f percent). Delayline at %06.3f mm DATA (%05.2f percent)'%(alpha,100.*(j)/len(alphas),curr_fdl_pos,50.0*(float(i+1)/total_poss))
            prints(completed,prev+' '*6)
            prev = completed
            i+=1
    
    print '\nDone!'
    H.hero_shutdown()


#%%
# ------------------------------------------------------------------
# ------------------------Auxillary functions-----------------------
# ------------------------------------------------------------------
def create_date_str():
    return time.strftime("%d%b%y_%H%M", time.localtime())

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
    plt.pause(0.01)
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
    plt.pause(0.01)
    QWP_th.start()
    plt.pause(0.01)
    ANA_th.start()
    pol_th.join()
    QWP_th.join()
    ANA_th.join()

#def SP_filter_in():
#    TL_slider.block_laser()
#def SP_filter_out():
#    TL_slider.unblock_laser()
def block_laser():
    TL_slider_3.block_laser_3()
def unblock_laser():
    TL_slider_3.unblock_laser_3()
def block_sc():
    TL_slider_2.block_laser_2()
def unblock_sc():
    TL_slider_2.unblock_laser_2()
#SP_filter_out()