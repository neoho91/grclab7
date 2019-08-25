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
try:
    illumination = np.load('D:/Nonlinear_setup/Python_codes/illumination.npy')
except:
    illumination = 1
    
try:
    from filter_rot_stages import *
except:
    print "filter rotational stages not connected"
    
try:
    from rot_stage_DDR25M import *
    ANA_off=20.8359
    def move_A(a=None):
        if a==None:
            return move_rot5() - ANA_off
        return move_rot5(a+ANA_off) - ANA_off
    
    def home_A():
        move_rot5(0)
        home_rot5()
except:
    print('Analyzer not connected')
    
try:
    from WMP_fine_dl import *
except:
    print "fine_delayline not connected"
    
try:
    from VP_stage_x import *
except:
    print('VP stage x-axis not connected')
    
try:
    from VP_stage_y import *
except:
    print('VP stage y-axis not connected')
    
try:
    import TL_slider #"dichroic mirror"
    def insert_DM():
        TL_slider.insert_DM()
    def remove_DM():
        TL_slider.remove_DM()
except:
    def insert_DM():
        pass
    def remove_DM():
        pass
    print "Filter shutter not connected"
    
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
    import avaspec as H
except:
    print "Hero not connected"
    
try:
    from lockin import *
except:
    print "lock-in not connected"
    def lockin_auto_gain():
        pass
    def lockin_auto_phase():
        pass
    def optimise_srs830_sensitivity():
        pass
    def get_lockin_reading1():
        return 0
    
try:
    from piezo import *
except:
    def get_pos():
        return []
    print "piezo not connected"
    
try:
    from rot_stages_noGUI import *
except:
    print "rotational stages not connected"

try:
    from andor_main import *
    print "Andor spectrometer online."
except:
    print "Andor not connected."
    
try:
    from sc_power_control import *
except:
    print("SC power control failed to initialize.")

from background_optimize_sc_piezo import *
#from send_whatsapp import *
try:
    from LED_control import *
except:
    print('LED power control not connected.')
    
from pco import *
try:
    from polarimeter import *
except:
    print('polarimeter not connected.')
    
try:
    from polarization_control import *
except:
    print('polarization control not connected.')

#from calib_4mirrors_codes import *
hero_pixel_wavelengths = np.load('D:/WMP_setup/Python_codes/hero_pixel_wavelengths.npy')
from anal_VP2_with_cryostat import *
from NPBS_TR_ratio import *
#from sharp_blade_experiment_neo import *


sys.path.append(r'D:/WMP_setup/Python_codes')
hero_pixel_wavelengths = np.load(r'D:/WMP_setup/Python_codes/hero_pixel_wavelengths.npy')

#---------------------------------------------------#
#Definition of codes to be used in actual experiments


#%%

def pSHG_with_andor(sample,ana_angs=np.arange(0,360,5),sc_on=False,andor_ave_num=10,andor_int_ms=100,log='',
                    readjust_sample_pos_every_n_points=1,led_p=0.5,ref_img=None):
    """
    Obtain andor spectrum for each analyzer angle.
    SHG pump beam wavelength is 1560nm.
    """
    input_line = np.array([get_last_input_line()])
    main_path=r'D:\Nonlinear_setup\Experimental_data\pSHG_with_andor\%s'%sample 
    os.makedirs(main_path)
    
    set_exposure_time(andor_int_ms)
    wls = get_wl_vector()
    
    total_len = len(ana_angs)
    total_time = total_len*( andor_int_ms/1000.*(andor_ave_num+1) + 1 )
    global start_time,end_time
    start_time = time.time()

    init_line = '\nStarted pSHG_with_andor (%s) on %s, expected to complete on %s.\n'%(sample,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+total_time)))
    init_line2 = 'ana_angs = %s, sc_on = %s, andor_ave_num = %i, andor_int_ms = %f ms.'%(str(ana_angs),str(sc_on),andor_ave_num,andor_int_ms)
    print(init_line)
    print(init_line2)
    log_txt = [unicode(input_line),unicode(init_line),unicode(init_line2),
               u'\n\n'+unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    unblock_laser()
    if sc_on:
        unblock_sc()
    else:
        block_sc()
    
    home_A()
        
    def prepare_take_data():
        global specs, bg_specs, wls, data_path, ref_img
        specs = []
        bg_specs = []
        wls = get_wl_vector()
        data_path = os.path.join(main_path,'data.npz')
        try:
            initialise_pco()
        except WindowsError:
            pass
        set_pco_exposure_time(50)
        LED_power(led_p)
        if ref_img==None:
            ref_img = get_pco_image(10)

    def take_data():
        block_laser()
        time.sleep(0.1)
        bg_specs.append(copy.copy(get_spectrum()))
        unblock_laser()
        time.sleep(0.1)
        
        curr_specs = []
        for i in range(andor_ave_num):
            curr_specs.append(copy.copy(get_spectrum()))
        specs.append(np.array(curr_specs))

    def save_data():
        try:
            np.savez(data_path,
             wls=wls,
             ana_angs=ana_angs,
             specs=specs,
             bg_specs=bg_specs
             )
        except:
            time.sleep(1)
            save_data()
    
    def finishing():
        block_laser()
        block_sc()
        close_pco()
        LED_power(0)

    prev_completed = ''
    _n = 0
    prints('\n')
    
    prepare_take_data()
    start_time = time.time()
    for j,ana_ang in enumerate(ana_angs):
        if j%readjust_sample_pos_every_n_points == 0 and j > 0:
            insert_DM()
            plt.pause(0.1)
            x_comp, y_comp = get_offset_um(ref_img,get_pco_image(10))
            if abs(x_comp) > 10 or abs(y_comp) > 10:
                print('Unable to find sample.')
                play_sound(error_sound)
            else:
                move_vpx_abs(get_vpx_pos() + x_comp/1000.)
                move_vpy_abs(get_vpy_pos() - y_comp/1000.)
                x_comp, y_comp = get_offset_um(ref_img,get_pco_image(10))
                move_vpx_abs(get_vpx_pos() + x_comp/2./1000.)
                move_vpy_abs(get_vpy_pos() - y_comp/2./1000.)
            remove_DM()
        completed = 'Analyzer at %.1fdeg (%.2f percent)'%(ana_ang,_n*100./total_len)
        prints(completed,prev_completed)
        prev_completed = completed

        move_A(ana_ang)
        plt.pause(1)
        take_data()
        save_data()

        _n += 1
    end_time = time.time()
    finishing()
    
    print 'Done! Time spent = %is'%(time.time()-start_time)
    play_sound(complete)

    try:
        anal_pSHG_with_andor(sample)
        plt.pause(1e-6)
    except:
        print("%s not analyzed"%sample)



#%%
#---------------------------------------------------#
#code #2: for SHG mapping experiment

def VP2_mapping_SHG(sample,init_pos,final_pos,resol=1,andor_int_time=100,andor_spec_num=1,timesleep=0.2,slow_save=True,log='',pump_with_1550=True):

    Xi,Zi=init_pos
    Xf,Zf=final_pos
    Xs=np.arange(Xi,Xf+resol,resol)
    Zs=np.arange(Zi,Zf+resol,resol)

    block_laser()
    block_sc()
    total_len = len(Xs)*len(Zs)
    total_time = total_len*((andor_int_time+0.032)*andor_spec_num/1000. + timesleep)
    start_time = time.time()

    init_line = '\nStarted VP2_mapping_SHG (%s) on %s, expected to complete on %s\nandor_int_time = %.2f ms, andor_spec_num = %i, timesleep = %.2f s, init position = %s, final position = %s, and resolution = %.1f um\n'%(
        sample,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+total_time)),andor_int_time,andor_spec_num,timesleep,str(init_pos),str(final_pos),resol)
    print(init_line)
    if not query_yes_no('Continue?'):
        return 
    main_path=r'D:\Nonlinear_setup\Experimental_data\VP2_mapping_SHG\%s'%sample 
    os.makedirs(main_path)
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')

    np.save(os.path.join(main_path,'Xs'),Xs)
    np.save(os.path.join(main_path,'Zs'),Zs)
    
    def _get_spec():
        _spec = []
        for i in range(andor_spec_num):
            _spec.append(copy.copy(get_spectrum()))
        return np.median(_spec,0)

    def prepare_take_data():
        global data
        #datum = ((x,z), pm_data, median-ed spec)
        #data = [(spec_wl,bg_spec), datum 1, datum2, ...] 
        data = []
        set_exposure_time(andor_int_time)
        move_to_x(Xs[0])
        move_to_z(Zs[0])
    
    def take_background():
        block_laser()
        block_sc()
        plt.pause(2)
        pma_zero()
        bg_spec = _get_spec()
        spec_wl = get_wl_vector()
        data.append((spec_wl,bg_spec))
        
    def take_datum(x,z):
        move_to_x(x)
        move_to_z(z)
        plt.pause(timesleep)
        _curr_spec = [0]
        _curr_pm_data = [0]
        
        def andor_loop():
            _curr_spec[0] = _get_spec()
        andor_th = threading.Thread(target=andor_loop)
        
        def pm_loop():
            _pm_data = []
            while andor_th.isAlive():
                _pm_data.append(pma_power())
                plt.pause(0.01)
            _curr_pm_data[0] = np.mean(_pm_data)
        pm_th = threading.Thread(target=pm_loop)
        
        andor_th.start()
        pm_th.start()
        andor_th.join()
        pm_th.join()
        
        datum = ((x,z),_curr_pm_data[0],_curr_spec[0])
        data.append(datum)
        return datum

    def save_data():
        try:
            np.save(os.path.join(main_path,'data'),np.array(data))
        except:
            time.sleep(1)
            save_data()
    
    def finishing():
        block_laser()
        block_sc()
        save_data()
        play_sound(complete)
        send_whatsapp_message('Mapping %s done.'%sample)
    
    prepare_take_data()
    print('Taking background...')
    take_background()
    
    prev_completed = ''
    _n = 0
    prints('\n')
    if pump_with_1550:
        unblock_laser()
    else:
        unblock_sc()
    plt.pause(2)
    for Z in Zs:
        for X in Xs:
            completed = 'Z at %.1f um, X at %.1f um (%.2f percent)'%(Z,X,_n*100./total_len)
            prints(completed,prev_completed)
            prev_completed = completed
            
            take_datum(X,Z)
            if not slow_save:
                save_data()
            _n += 1
        if slow_save:
            save_data()

    finishing()
    print 'Done! Time spent = %is'%(time.time()-start_time)
    
    try:
        VP2_mapping_SHG_anal(sample)
        plt.pause(1e-6)
    except:
        pass



#%%

def scan_delayline_with_andor(sample=None,fdl_min=0,fdl_max=12,fdl_incre=0.01,andor_int_ms=100,andor_median_num=5):
    if sample == None:
        curr_time_str = create_date_str()
    else:
        curr_time_str = sample
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline')
    set_exposure_time(andor_int_ms)
    wls = get_wl_vector()
    
    fdl_poss = np.arange(fdl_min,fdl_max+fdl_incre,fdl_incre)
    prev=''
    total_poss = len(fdl_poss)
    data = [None,wls]
    for i,fdl_pos in enumerate(fdl_poss):
        move_fdl_abs(fdl_pos)
        curr_fdl_pos = get_fdl_pos()
        
        curr_spec = []
        for j in range(andor_median_num):
            curr_spec.append(copy.copy(get_spectrum()))
        spec = np.median(curr_spec,0)
        
        data.append([curr_fdl_pos,spec])
        np.save(os.path.join(main_path,curr_time_str+'.npy'),np.array(data))
            
        completed = u'%f (%.2f percent)'%(curr_fdl_pos,100.0*(float(i+1)/total_poss))
        prints(completed,prev)
        prev = completed
    
    print 'Done!'


    
#%%
def scan_delayline_with_andor6(sample=None,
                               fdl_min=0,fdl_max=12,fdl_incre=0.01,
                               fdl_poss = None,
                               andor_int_ms=100,andor_num=5,andor_warm_up_sec=60,
                               repeat=1,
                               opt_sc=False,slow_save=True,
                               sc_wl=742,sc_power_nW=2000,set_sc_power_every_n_point=10,sc_pol='L',
                               pco_exp_time_ms=50,led_power=0.5,num_imgs=10,
                               take_img_every_n_point=1,
                               compensate_stage_xy=False,
                               manual_compensate_stage_xy=False):
    '''Scan with options to optimise SC for each run, slow save.
    Also scans with laser only, SC only, and both beams.
    Polarimeter attached to get reference SC power and polarisation.
    
    pm1550m = laser powermeter (Need to zero by hand before starting run!)
    pmp = polarimeter powermeter = sc powermeter'''
    global curr_time_str
    if sample == None:
        curr_time_str = create_date_str()
    else:
        curr_time_str = sample
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_with_andor6',curr_time_str)
    os.makedirs(main_path)
    img_path = os.path.join(main_path,'images')
    os.makedirs(img_path)
    if fdl_poss == None:
        fdl_poss = np.arange(fdl_min,fdl_max+fdl_incre,fdl_incre)
    else:
        fdl_poss = np.array(fdl_poss)
    # write log file with parameters used for experiment
    input_line = np.array([get_last_input_line()])
    delay_params_line = 'Scanning from delay line position %s mm to %s mm with step size %s mm.'%(str(fdl_min), str(fdl_max), str(fdl_incre))
    andor_params_line = 'Andor integration time %s ms, take median of %i points, warming up for %s seconds, repeating %i times.'%(str(andor_int_ms), andor_num, str(andor_warm_up_sec), repeat)
    image_params_line = 'PCO camera integration time %s ms, averaging over %i images.'%(str(pco_exp_time_ms), num_imgs)
    misc_params_line = 'SC wavelength %s nm, power at sample %s nW, sc polairzation = %s'%(str(sc_wl), str(sc_power_nW), sc_pol)
    log_txt = [unicode(input_line),u'\n'+unicode(delay_params_line),u'\n'+unicode(andor_params_line),
               u'\n'+unicode(image_params_line), u'\n'+unicode(misc_params_line)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    def prepare_take_data():
        global data,wls,img_info
        set_exposure_time(andor_int_ms)
        remove_DM()
        initialise_pco()
        set_pco_exposure_time(pco_exp_time_ms)
        
        wls = get_wl_vector()
        # data will be [[sc wavelength], wavelengths, [both on], [laser only], [sc only]]
        # each data point is appended in those last 3 lists, [fdl_pos, spectrum, sc_power, laser_power, sc_gamma (if applicable)]
        # sc_gamma is a measure of the reference polarisation of the SC (after 1 reflection by NPBS) parametrised by alpha, gamma
        data = [[sc_wl],wls,[],[],[]]
        # img info will be [[img npy name, rep, fdl pos, time since start]]
        img_info = []
        
        block_laser()
        block_sc()
        go_to_pol(sc_pol,sc_wl)
        pm1550m_wl(1550)
        pmp_wl(sc_wl)
        time.sleep(1)
        # Need to zero pmd by hand as laser slider is after the pmd.
#        pmd_zero()
        pmp_zero()
        time.sleep(5)
    
    def take_data(curr_fdl_pos,laser_on,sc_on):
        _specs = [0]
        _sc_power = [0]
        _sc_gamma = [0]
        _laser_power = [0]
        
        def spec_loop():
            curr_spec = []
            for k in range(andor_num):
                curr_spec.append(copy.copy(get_spectrum()))
            specs = np.array(curr_spec)
            _specs[0] = specs
        
        def measure_power_and_pol_loop():
            curr_sc_power = []
            curr_laser_power = []
            curr_sc_gamma = gamma #from the periodic setting of SC power in actual measurement
            while spec_th.isAlive():
                curr_laser_power.append(pm1550m_power())
                sc_power = pmp_power()
                curr_sc_power.append(sc_power)
                time.sleep(0.01)
            if laser_on:
                # Laser power is recorded in nW
                _laser_power[0] = np.mean(curr_laser_power)*1e9
            if sc_on:
                # SC power is recorded in nW
                _sc_power[0] = np.mean(curr_sc_power)*NPBS_TR_from_pol_wl(sc_pol,sc_wl)*1e9
#                _sc_power[0] = np.mean(curr_sc_power)*NPBS_TR_gamma_fx(sc_wl,gamma)*1e9
                _sc_gamma[0] = curr_sc_gamma   
        
        spec_th = threading.Thread(target=spec_loop)        
        measure_power_and_pol_th = threading.Thread(target=measure_power_and_pol_loop)
        
        spec_th.start()
        measure_power_and_pol_th.start()
        spec_th.join()
        measure_power_and_pol_th.join()
        
        if laser_on:
            if sc_on: #both on
                data[2].append([curr_fdl_pos,_specs[0],_sc_power[0],_laser_power[0],_sc_gamma[0]])
            else: #only laser on
                data[3].append([curr_fdl_pos,_specs[0],_sc_power[0],_laser_power[0]])
        else: #only sc on
            data[4].append([curr_fdl_pos,_specs[0],_sc_power[0],_laser_power[0],_sc_gamma[0]])
    
    def save_data():
        np.save(os.path.join(main_path,curr_time_str+'.npy'),np.array(data))
    
    def take_and_save_img(name,rep,fdl_pos,laser_only_img):
        block_sc()
        unblock_laser()
        img = DMin_LEDon_TAKEimg(LED_p=led_power, num_imgs=num_imgs, DM_out_after=True)
        img_info_entry = (name,rep,fdl_pos,time.time()-experiment_start_time)
        img_info.append(img_info_entry)
        img_comp = equalize_histogram_and_8bit(img)
        np.save(os.path.join(main_path, 'img_info.npy'), img_info)
        np.save(os.path.join(img_path, name+'.npy'), img_comp)
        np.save(os.path.join(img_path,name+'_laser.npy'), laser_only_img)
#        if compensate_stage_xy:
#            return _process_img(os.path.join(img_path,name+'.npy'))
        return img_comp
    
    def finishing():
        block_laser()
        block_sc()
        close_pco()
        LED_power(0)
        remove_DM()
        
    def take_background_imgs():
        unblock_laser()
        block_sc()
        img_laser_only = DMin_LEDon_TAKEimg(LED_p=0, num_imgs=num_imgs, DM_out_after=False)
        _img_laser_only = copy.copy(img_laser_only)
        img_laser_only = equalize_histogram_and_8bit(img_laser_only)
        
        block_laser()
        img_LED_only = DMin_LEDon_TAKEimg(LED_p=led_power, num_imgs=num_imgs, DM_out_after=True)
        _img_LED_only = copy.copy(img_LED_only)
        img_LED_only = equalize_histogram_and_8bit(img_LED_only)
        
        np.save(os.path.join(main_path, 'img_LED_only.npy'), img_LED_only)
        np.save(os.path.join(main_path, 'img_laser_only.npy'), img_laser_only)
        
#        if compensate_stage_xy:
#            _process_img(os.path.join(main_path,'img_LED_only.npy')) #to remove background
        img_both = equalize_histogram_and_8bit(_img_laser_only + _img_LED_only)
        origin_coords = extract_coords_of_laser_on_sample(img_both,img_LED_only,img_laser_only,um_per_px=1,plot=False)
        print("Original laser coordinates (px): %s"%str(origin_coords))
        return img_LED_only,img_laser_only,origin_coords,img_both
    
    def equalize_histogram_and_8bit(img):
        return ((img-np.min(img))/(np.max(img)-np.min(img)) * 255).astype(np.uint8)
        
    
    # Andor somehow is more sensitive when starting to take data after a long pause. This decays back to normal.
    # "timewasting" to get Andor to warm up and remove (?) this artifact.
    def warm_up_andor(warm_up_time):
        s = time.time()
        i = 0
        while (time.time() - s) < warm_up_time:
            get_spectrum()
            time.sleep(0.05)
            i += 1
        #print("Warmed up for "+ str(time.time()-s) + " seconds. ")
        #print("Took " + str(i) + " spectra. ")
        
        
    warm_up_th = threading.Thread(target=warm_up_andor, args=(andor_warm_up_sec,))
    warm_up_th.start()
    
    prepare_take_data()
    prev=''
    total_poss = len(fdl_poss)
    total_poss_repeat = total_poss*repeat
    sample_ref_img,laser_ref_img,origin_coords,img_both = take_background_imgs()
    
    #unblock lasers for more effective warmup
    unblock_laser()
    unblock_sc()
    warm_up_th.join()
    experiment_start_time = time.time()
    
    #brought outside the take_img_every_n_point compensation to avoid excessive plt.imshow
    insert_DM()
    plt.pause(0.1)
    LED_power(0)
    unblock_laser()
    curr_laser_img = equalize_histogram_and_8bit(sp.ndimage.median_filter(get_pco_image(num_imgs),3))
    LED_power(led_power)
    
    real_time_img_fig = plt.figure('Sample and laser tracking')
    ref_img_fig = real_time_img_fig.add_subplot(121)
    ref_img_fig.set_title('Reference image')
    ref_img_fig.imshow((sample_ref_img.astype(float)+laser_ref_img.astype(float))/illumination)
    live_img_fig = real_time_img_fig.add_subplot(122)
    curr_img_raw = get_pco_image(num_imgs)
    curr_img = (equalize_histogram_and_8bit(curr_img_raw).astype(float) + curr_laser_img.astype(float))/illumination
    curr_img_axes = live_img_fig.imshow(curr_img)
    remove_DM()
    plt.pause(0.1)
    LED_power(0)
    # /end image showing.
    
    _i = 0
    for j in range(repeat):
        if opt_sc:
            opt_scpz(verbose=False)
            block_laser()
            unblock_sc()
            plt.pause(5)
            set_sc_power_nW(sc_power_nW,sc_wl,pm='polarimeter')
        start_time = time.time()
        _j = 0
        for i,fdl_pos in enumerate(fdl_poss):
            move_fdl_abs(fdl_pos)
            if i%set_sc_power_every_n_point == 0:
                popt,perr,max_ang = set_sc_power_nW(sc_power_nW,sc_wl,pm='polarimeter',return_raw=True)
                gamma = popt[2] #needed for measure_power_and_pol loop
                set_pola_QWP_ang(max_ang)
                
            if i%take_img_every_n_point == 0:
                if compensate_stage_xy:
                    insert_DM()
                    plt.pause(0.1)
                    LED_power(0)
                    unblock_laser()
                    curr_laser_img = equalize_histogram_and_8bit(sp.ndimage.median_filter(get_pco_image(num_imgs),3))
                    LED_power(led_power)
                    
                    curr_img_raw = get_pco_image(num_imgs)
                    curr_img = (equalize_histogram_and_8bit(curr_img_raw).astype(float) + curr_laser_img.astype(float))/illumination
                    curr_img_axes.set_data(curr_img)
                    curr_img_axes.set_clim(np.min(curr_img),np.max(curr_img))
                    
                    force_manual_compensate_stage_xy = False
                    
                    if not manual_compensate_stage_xy:
                        is_good_pos = False
                        comp_run_num = 0
                        stage_resolution = 0.02#0.15 #um
                        
                        while (not is_good_pos) and (comp_run_num < 3):
                            curr_coords = []
                            for k in range(5):
                                curr_coords.append(extract_coords_of_laser_on_sample_2(equalize_histogram_and_8bit(get_pco_image(num_imgs/2)),curr_laser_img,sample_ref_img,laser_ref_img,plot=False,um_per_px=1))
                            curr_coords = np.mean(curr_coords,axis=0)
                            print("Before correction %i_%i: laser coords = %s"%(j, _j, str(curr_coords)))
                            x_comp,y_comp = curr_coords - origin_coords
                            x_comp = x_comp*2./17#5.7/155
                            y_comp = y_comp*2./17#5.7/155*3 #don't know why the px-um conversion factor is different for x and y
                            if abs(x_comp) > 1.5 or abs(y_comp) > 1.5: #if sample drifted too far away, means something is wrong.
                                print('Unable to find sample.')
                                play_sound(error_sound)
                                force_manual_compensate_stage_xy = True
                                break
                            elif x_comp < stage_resolution and y_comp < stage_resolution: #sample reached initial position.
                                is_good_pos = True
                                break
                            else:
                                if x_comp > stage_resolution: #um. if sample drifted just a little bit, better not move it, as limited by the current actuator resolution.
                                    move_x(-x_comp)
        #                                move_vpx_abs(get_vpx_pos() - x_comp/1000.)
                                if y_comp > stage_resolution:
                                    move_z(-y_comp)
        #                                move_vpy_abs(get_vpy_pos() + y_comp/1000.)
                                comp_run_num += 1
                                curr_img_raw = get_pco_image(num_imgs)
                                curr_img = equalize_histogram_and_8bit(curr_img_raw)
                                curr_img = (curr_img.astype(float) + curr_laser_img.astype(float))/illumination
                                curr_img_axes.set_data(curr_img)
                                curr_img_axes.set_clim(np.min(curr_img),np.max(curr_img))
                    
                    if manual_compensate_stage_xy or force_manual_compensate_stage_xy:
                        curr_img_raw = get_pco_image(num_imgs)
                        curr_img = equalize_histogram_and_8bit(curr_img_raw)
                        curr_img = (curr_img.astype(float) + curr_laser_img.astype(float))/illumination
                        curr_img_axes.set_data(curr_img)
                        curr_img_axes.set_clim(np.min(curr_img),np.max(curr_img))
                        curr_coords = extract_coords_of_laser_on_sample_2(equalize_histogram_and_8bit(curr_img_raw),curr_laser_img,sample_ref_img,laser_ref_img,plot=False,um_per_px=1)
                        offset_from_init = (np.array(curr_coords) - np.array(origin_coords))*2./17
                        live_img_fig.set_title('Laser coord offset = %s um.'%offset_from_init)
                        plt.pause(0.01)
                        plt.get_current_fig_manager().window.showMaximized()
                        not_statisfy = [True]
                        step = [1.]
                        real_time_img_fig.suptitle('Arrow keys to move, PageUp/PageDown to focus, +/- keys to change increment (%.3f um), Esc to stop.'%step[0])
                        def press(event):
                            if event.key == 'left':
                                move_x(step[0])
                            elif event.key == 'right':
                                move_x(-step[0])
                            elif event.key == 'down':
                                move_z(-step[0])
                            elif event.key == 'up':
                                move_z(step[0])
                            elif event.key == '+':
                                step[0] = step[0]*2.
                            elif event.key == '-':
                                step[0] = step[0]/2.
                            elif event.key == 'pageup':
                                move_y(-step[0])
                            elif event.key == 'pagedown':
                                move_y(step[0])
                            elif event.key == 'escape':
                                not_statisfy[0] = False
                            else:
                                pass
                        real_time_img_fig.canvas.mpl_connect('key_press_event', press)
                        play_sound(adjust_piezo_now)
                        print('Adjust piezo now.')
                        while not_statisfy[0]:
                            curr_img_raw = get_pco_image(num_imgs)
                            curr_img = equalize_histogram_and_8bit(curr_img_raw)
                            curr_img = (curr_img.astype(float) + curr_laser_img.astype(float))/illumination
                            curr_img_axes.set_data(curr_img)
                            curr_img_axes.set_clim(np.min(curr_img),np.max(curr_img))
                            curr_coords = extract_coords_of_laser_on_sample_2(equalize_histogram_and_8bit(curr_img_raw),curr_laser_img,sample_ref_img,laser_ref_img,plot=False,um_per_px=1)
                            offset_from_init = (np.array(curr_coords) - np.array(origin_coords))*2./17
                            live_img_fig.set_title('Laser coord offset = %s um.'%offset_from_init)
                            real_time_img_fig.suptitle('Arrow keys to move, PageUp/PageDown to focus, +/- keys to change increment (%.3f um), Esc to stop.'%step[0])
                            plt.pause(0.01)
                        play_sound(scan_continued)
                        print('Scan continued.')
                    
                curr_data_img = take_and_save_img('%i_%i'%(j,_j),j,fdl_pos,curr_laser_img)
                print("After correction %i_%i: laser coords = %s"%(j, _j, extract_coords_of_laser_on_sample_2(curr_data_img,curr_laser_img,sample_ref_img,laser_ref_img,plot=False,um_per_px=1)))
                curr_img = (curr_data_img.astype(float) + curr_laser_img.astype(float))/illumination
                curr_img_axes.set_data(curr_img)
                curr_img_axes.set_clim(np.min(curr_img),np.max(curr_img))
                _j += 1
            
            block_laser()
            unblock_sc()
            plt.pause(1)
            take_data(curr_fdl_pos=fdl_pos,laser_on=False,sc_on=True)
                
            unblock_laser()
            block_sc()
            plt.pause(1)
            take_data(curr_fdl_pos=fdl_pos,laser_on=True,sc_on=False)
            
            unblock_laser()
            unblock_sc()
            plt.pause(1)
            take_data(curr_fdl_pos=fdl_pos,laser_on=True,sc_on=True)
            
            if not slow_save:
                save_data()
            
            elapsed_time = time.time() - start_time
            time_left = elapsed_time*(1.*len(fdl_poss)/(i+1)-1)
            completed = u'run %i (%.2f percent): %f (%.2f percent) %s left for this run.'%(j,100.0*(float(_i+1)/total_poss_repeat),fdl_pos,100.0*(float(i+1)/total_poss),sec_to_hhmmss(time_left))
            prints(completed,prev)
            prev = completed
            _i += 1
            
        if slow_save:
            save_data()
    
    finishing()
    time_taken = time.time() - experiment_start_time
    print(u'\n' + 'Time taken for experiment = %s'%sec_to_hhmmss(time_taken))
    print(u'\n' + 'Done!')
    play_sound(complete)
    

#%%
# SCAN DELAYLINE WITH ANDOR 7
# Auxiliary methods for scan_delayline_with_andor7()

def take_spectrum_with_power(specs=[], laser_powers=[], sc_powers=[], num_spectrum=1):
    #threading like this means I need to put a while loop in the individual functions
    def measure_spectrum(specs=[], num_spectrum=1):
        for i in range(num_spectrum):
            specs.append(copy.copy(get_spectrum()))
        return specs
    
    spec_th = threading.Thread(target=measure_spectrum, args=(specs, num_spectrum))

    def measure_laser_power(powers=[]):
        while spec_th.isAlive():
            powers.append(pm1550m_power())
            time.sleep(0.01)
        return powers
    def measure_sc_power(powers=[]):
        while spec_th.isAlive():
            powers.append(pmp_power())
            time.sleep(0.01)
        return powers
    laser_th = threading.Thread(target=measure_laser_power, args=(laser_powers,))
    sc_th = threading.Thread(target=measure_sc_power, args=(sc_powers,))
    threading.Thread()
    spec_th.start()
    laser_th.start()
    sc_th.start()
    spec_th.join()
    laser_th.join()
    sc_th.join()
    return specs, np.mean(laser_powers), np.mean(sc_powers) # need to later *NPBS_TR_from_pol_wl(sc_pol,sc_wl)

def take_images(led_power=0.1, num_imgs=3):
    """Takes images of laser only and LED only and returns 8-bit versions of the images.
    *Alters state of lasers to both blocked.
    *Alters state of DM to be inserted.
    *Alters state of LED to be at led_power (0.1 by default)."""
    # Take images
    block_sc()
    unblock_laser()
    img_laser_only = copy.copy(DMin_LEDon_TAKEimg(LED_p=0, num_imgs=num_imgs, DM_out_after=False))
    
    block_laser()
    img_LED_only = copy.copy(DMin_LEDon_TAKEimg(LED_p=led_power, num_imgs=num_imgs, DM_out_after=False))
    
    # Treat images and create combined image
    img_both = img_laser_only + img_LED_only
    img_laser_only = equalize_histogram_and_8bit(img_laser_only)
    img_LED_only = equalize_histogram_and_8bit(img_LED_only)
    img_both = equalize_histogram_and_8bit(img_both)
    
    return img_laser_only, img_LED_only, img_both

def take_deilluminated_image(num_imgs=1):
    img = get_pco_image(num_imgs)
    img = equalize_histogram_and_8bit(img)
    img = img.astype(float) / illumination
    img = equalize_histogram_and_8bit(img)
    return img
    

def compensate(manual=False, num_imgs=3, led_power=0.1, sample_ref_img=None, laser_ref_img=None, ref_coords=None):
    """Compensates the sample drift with respect to some initial reference (set of images+calculated coords).
    *Alters state of lasers to sc blocked, laser unblocked.
    *Alters state of DM to be inserted.
    *Alters state of LED to be at led_power (0.1 by default)."""
    # Never touch these again while compensating.
    insert_DM()
    block_sc()
    time.sleep(0.1)
    #global sample ref image to be deilluminated
    deillum_sample_ref_img = sample_ref_img.astype(float) / illumination
    deillum_sample_ref_img = equalize_histogram_and_8bit(deillum_sample_ref_img)
    
    # Take laser reference image
    LED_power(0)
    unblock_laser()
    curr_laser_img = equalize_histogram_and_8bit(sp.ndimage.median_filter(get_pco_image(num_imgs), 3))
    # Take current sample image (with laser spot)
    LED_power(led_power)
    curr_img_8bit = take_deilluminated_image(num_imgs=num_imgs)
    
    # Compensate.
    if manual:
        _manual_compensate(num_imgs=num_imgs, led_power=led_power,
                           sample_img=curr_img_8bit, laser_img=curr_laser_img,
                           sample_ref_img=deillum_sample_ref_img, laser_ref_img=laser_ref_img, ref_coords=ref_coords)
    else:
        good = _auto_compensate(num_imgs=num_imgs, led_power=led_power,
                                sample_img=curr_img_8bit, laser_img=curr_laser_img,
                                sample_ref_img=deillum_sample_ref_img, laser_ref_img=laser_ref_img, ref_coords=ref_coords)
        if not good:
            _manual_compensate(num_imgs=num_imgs, led_power=led_power,
                               sample_img=curr_img_8bit, laser_img=curr_laser_img,
                               sample_ref_img=deillum_sample_ref_img, laser_ref_img=laser_ref_img, ref_coords=ref_coords)
    return True


def _manual_compensate(num_imgs=3, led_power=0.1, sample_img=None, laser_img=None, sample_ref_img=None, laser_ref_img=None, ref_coords=None):
    """Internal method for compensate(). Manual compensation (user inputs commands via keypresses).
    pyplot window opened for user to see live images, left open at termination.
    Returns the main Figure of the pyplot window."""
    # Calculate current offset of laser wrt original (reference) position
    curr_coords = extract_coords_of_laser_on_sample_2(sample_img, laser_img, sample_ref_img, laser_ref_img, plot=False, um_per_px=1)
    offset_from_init = (np.array(curr_coords) - np.array(ref_coords)) * 2. / 17
    print("\nBefore correction: laser coords = %s; " % (str(curr_coords),))
    # Create current combined laser and sample image to display to user
    curr_img = sample_img.astype(float) + laser_img.astype(float)
    
    # set step size for piezo (in um)
    step = [1.]
    satisfied = [False]
    
    #Create window for user to see ref and live sample images
    compensation_img_fig = plt.figure("Manual sample position compensation")
    ref_img_subplot = compensation_img_fig.add_subplot(121)
    ref_img_subplot.set_title("Reference image")
    ref_img = (sample_ref_img.astype(float) + laser_ref_img.astype(float)) / illumination
    ref_img_subplot.imshow(ref_img)
    live_img_subplot = compensation_img_fig.add_subplot(122)
    curr_img_axim = live_img_subplot.imshow(curr_img)

    live_img_subplot.set_title("Laser coord offset = %s um" % offset_from_init)
    plt.pause(0.01)
    plt.get_current_fig_manager().window.showMaximized()
    compensation_img_fig.suptitle("Arrow keys to move, PageUp/PageDown to focus, +/- keys to change increment (now %.3f um), Esc to stop." % step[0])
    
    #Set up user controls for compensation - see corresponding method
    def compensation_press(event):
        if event.key == 'left':
            move_x(step[0])
        elif event.key == 'right':
            move_x(-step[0])
        elif event.key == 'down':
            move_z(-step[0])
        elif event.key == 'up':
            move_z(step[0])
        elif event.key == '+':
            step[0] = step[0]*2.
        elif event.key == '-':
            step[0] = step[0]/2.
        elif event.key == 'pageup':
            move_y(-step[0])
        elif event.key == 'pagedown':
            move_y(step[0])
        elif event.key == 'escape':
            satisfied[0] = True
        else:
            pass
    compensation_img_fig.canvas.mpl_connect("key_press_event", compensation_press)
    
    #Begin compensation.
    play_sound(adjust_piezo_now)
    print("Adjust piezo now.")
    LED_power(led_power)
    
    while not satisfied[0]:
        # Take current sample image (with laser spot) for user
        curr_img_8bit = take_deilluminated_image(num_imgs)
        curr_img = curr_img_8bit.astype(float) + laser_img.astype(float)
        
        # Calculate current offset of laser wrt original (reference) position
        curr_coords = extract_coords_of_laser_on_sample_2(curr_img_8bit, laser_img, sample_ref_img, laser_ref_img, plot=False, um_per_px=1)
        offset_from_init = (np.array(curr_coords) - np.array(ref_coords)) * 2. / 17
        
        # Display current image and coords
        curr_img_axim.set_data(curr_img)
        curr_img_axim.set_clim(np.min(curr_img),np.max(curr_img))
        live_img_subplot.set_title('Laser coord offset = %s um.'%offset_from_init)
        compensation_img_fig.suptitle('Arrow keys to move, PageUp/PageDown to focus, +/- keys to change increment (%.3f um), Esc to stop.'%step[0])
        plt.pause(0.1)
        
    #while loop broken by user pressing Escape, compensation completed
    curr_img_8bit = take_deilluminated_image(num_imgs)
    print("After correction: laser coords = %s" % (extract_coords_of_laser_on_sample_2(curr_img_8bit, laser_img, sample_ref_img, laser_ref_img, plot=False, um_per_px=1)))
    play_sound(scan_continued)
    print('Scan continued.\n')
    return compensation_img_fig


def _auto_compensate(num_imgs=3, led_power=0.1, sample_img=None, laser_img=None, sample_ref_img=None, laser_ref_img=None, ref_coords=None):
    """Internal method for compensate(). Automatic compensation (using jena piezo cube that cryostat is mounted on).
    Returns True if compensation was successful, False otherwise."""
    satisfied = False
    autocomp_run_num = 0
    stage_resolution = 0.02 #um #0.015
    
    while (not satisfied) and (autocomp_run_num < 3):
        # get current coordinates of laser on sample and work out needed correction
        curr_coords = []
        for k in range(5):
            curr_img_8bit = take_deilluminated_image(num_imgs)
            curr_coords.append(extract_coords_of_laser_on_sample_2(curr_img_8bit, laser_img, sample_ref_img, laser_ref_img, plot=False, um_per_px=1))
        curr_coords = np.mean(curr_coords, axis=0)
        print("Before correction: laser coords = %s; " % (str(curr_coords),))
        # correction needed in um
        x_comp, y_comp = curr_coords - ref_coords
        x_comp = x_comp * 2. / 17 #5.7/155
        y_comp = y_comp * 2. / 17 #5.7/155*3 #don't know why the px-um conversion factor is different for x and y
        
        # if sample drifted too far away, means something is wrong. Break and leave while loop.
        if abs(x_comp) > 1.5 or abs(y_comp) > 1.5:
            print('Unable to find sample.')
            play_sound(error_sound)
            break
        # sample reached a good position within stage resolution, break and leave while loop.
        elif x_comp < stage_resolution and y_comp < stage_resolution:
            satisfied = True
            break
        # perform a correction with the piezo
        else:
            if x_comp > stage_resolution: #um
                move_x(-x_comp)
#                move_vpx_abs(get_vpx_pos() - x_comp/1000.) #for actuator
            if y_comp > stage_resolution:
                move_z(-y_comp)
#                move_vpy_abs(get_vpy_pos() + y_comp/1000.) #for actuator
            autocomp_run_num += 1
    #end while        
    if satisfied:
        curr_img_8bit = take_deilluminated_image(num_imgs)
        print("After correction: laser coords = %s" % (extract_coords_of_laser_on_sample_2(curr_img_8bit, laser_img, sample_ref_img, laser_ref_img, plot=False, um_per_px=1)))
        return True
    else:
        return False

#TODO: Analysis code for this, real-time sample viewing, real-time data viewing
def scan_delayline_with_andor7(dir_name=None,
                               fdl_min=0,fdl_max=12,fdl_incre=0.01,
                               fdl_poss = None,
                               andor_int_ms=100,andor_num=5,andor_warm_up_sec=60,
                               repeat=1,
                               opt_sc=False,slow_save=True,
                               sc_wl=742,sc_power_nW=2000,#set_sc_power_every_n_point=10,
                               pco_exp_time_ms=50,led_power=0.5,num_imgs=10,
                               take_img_every_n_point=1,
                               compensate_stage_xy=False,
                               manual_compensate_stage_xy=False,
                               ana_angs=(90,),
                               sc_pols = ("L","R","H","V") ):
    # Create data directory, write log file
    main_dir_name = dir_name
    if main_dir_name is None:
        main_dir_name = create_date_str()
        
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_with_andor7',main_dir_name)
    os.makedirs(main_path)
    img_path = os.path.join(main_path,'images')
    os.makedirs(img_path)
    # write log file with parameters used for experiment
    input_line = np.array([get_last_input_line()])
    delay_params_line = 'Scanning from delay line position %s mm to %s mm with step size %s mm.'%(str(fdl_min), str(fdl_max), str(fdl_incre))
    andor_params_line = 'Andor integration time %s ms, take median of %i points, warming up for %s seconds, repeating %i times.'%(str(andor_int_ms), andor_num, str(andor_warm_up_sec), repeat)
    image_params_line = 'PCO camera integration time %s ms, averaging over %i images.'%(str(pco_exp_time_ms), num_imgs)
    misc_params_line = 'SC wavelength %s nm, power at sample %s nW, sc polarisation = %s'%(str(sc_wl), str(sc_power_nW), str(sc_pols))
    log_txt = [unicode(input_line),
               u'\n' + unicode(delay_params_line),
               u'\n' + unicode(andor_params_line),
               u'\n' + unicode(image_params_line),
               u'\n' + unicode(misc_params_line) + u'\n\n']
    np.savetxt(os.path.join(main_path, 'log.txt'), uniArray(log_txt), fmt='%s')
    
    #Set up equipment
    #PCO
    kill_process("CamWare.exe")
    try:
        close_pco()
    except (NameError, WindowsError):
        pass
    initialise_pco()
    set_pco_exposure_time(pco_exp_time_ms)
    
    #Andor + start warmup
    def warm_up_andor(warm_up_time):
        s = time.time()
        i = 0
        while (time.time() - s) < warm_up_time:
            get_spectrum()
            time.sleep(0.05)
            i += 1
        #print("Warmed up for "+ str(time.time()-s) + " seconds. " + u"\n" + "Took " + str(i) + " spectra. ")
    set_exposure_time(andor_int_ms)
    andor_wls = get_wl_vector()
    warm_up_th = threading.Thread(target=warm_up_andor, args=(andor_warm_up_sec,))
    warm_up_th.start()
    
    #Powermeters
    block_sc()
    block_laser()
    remove_DM()
    pm1550m_wl(1550)
    pmp_wl(sc_wl)
    time.sleep(1)
    pmp_zero()
    time.sleep(5)
    
    # Bookkeeping    
    prev = "" #just for printing experiment progress percentage%
    if fdl_poss is not None:
        fdl_poss = np.array(fdl_poss)
    else:
        fdl_poss = np.arange(fdl_min, fdl_max + fdl_incre, fdl_incre)
    laser_combis = {"SC": 0, "laser": 1, "both": 2}
    pol_dic = {"L":0, "R":1, "H":2, "V":3}
#    laser_combis = [0, 1, 2] # Only SC, only laser, both
    repeat_num_points = len(fdl_poss) * len(sc_pols) * len(ana_angs) * len(laser_combis)
    total_num_points = repeat_num_points * repeat
    data = [ [andor_wls, sc_wl], [] ] # data will be saved here. Structure is [[global params],[datums]]
    #datum structure is [[params],[meas]]
    #so datum = [[repeat_num,fdlpos,laserflags,anaang,scpol], [specs,laserpower,scpower,scalpha,scgamma]]
    img_info = [] # image info will be saved here (name_img, repeat_num, fdl_pos, time.time()-experiment_start_time)
    
    # Start experiment
    warm_up_th.join()
    experiment_start_time = time.time()
    
    # Take and save reference images at start of experiment, print original coordinates of laser wrt sample
    ref_laser_only, ref_LED_only, ref_both = take_images(led_power=led_power, num_imgs=num_imgs)
    np.save(os.path.join(main_path, 'img_LED_only.npy'), ref_LED_only)
    np.save(os.path.join(main_path, 'img_laser_only.npy'), ref_laser_only)
    origin_coords = extract_coords_of_laser_on_sample(ref_both, ref_LED_only, ref_laser_only, um_per_px=1, plot=False)
    print("Original laser coordinates (px): %s"%str(origin_coords))
    
    for repeat_num in range(repeat):
        # if opt_sc, then optimise at start of each repeat
        if opt_sc:
            opt_scpz(verbose=False)
            
        repeat_start_time = time.time()
        num_imgs_taken_this_repeat = 0
        
        for fdl_pos_num, fdl_pos in enumerate(fdl_poss):
            move_fdl_abs(fdl_pos)
            
            # if it has been sufficiently long, take images and compensate position
            corr_num = position_in_list_of_lists([fdl_pos_num],
                                                 [fdl_poss])
            if corr_num % take_img_every_n_point == 0: # this is what we use to determine when to compensate
                if compensate_stage_xy:
                    compensate(manual=manual_compensate_stage_xy, num_imgs=num_imgs, led_power=led_power,
                               sample_ref_img=ref_LED_only, laser_ref_img=ref_laser_only, ref_coords=origin_coords)
                #take images and image metadata
                img_laser_only, img_LED_only, img_both = take_images(led_power=led_power, num_imgs=num_imgs)
                name_img = str(repeat_num) + str(num_imgs_taken_this_repeat)
                img_info_entry = (name_img, repeat_num, fdl_pos, time.time()-experiment_start_time)
                img_info.append(img_info_entry)
                #save images and image metadata
                np.save(os.path.join(main_path, 'img_info.npy'), img_info)
                np.save(os.path.join(img_path, name_img+'.npy'), img_LED_only)
                np.save(os.path.join(img_path, name_img+'_laser.npy'), img_laser_only)
                num_imgs_taken_this_repeat += 1
            #end compensate and take images
            
            for sc_pol_num, sc_pol in enumerate(sc_pols):
                block_laser()
                unblock_sc()
                go_to_pol(sc_pol, sc_wl)
                #adjust sc power
                popt,perr,max_ang = set_sc_power_nW(sc_power_nW,sc_wl,pm='polarimeter',return_raw=True)
                sc_alpha, sc_gamma = popt[1], popt[2] #polarisation params
                set_pola_QWP_ang(max_ang) #make sure polarimeter powermeter is maximally sensitive

                for ana_ang_num, ana_ang in enumerate(ana_angs):
                    move_A(ana_ang)
                                   
                    for laser_combi in laser_combis.values():
                        # Prepare to take data. Open/close sc and laser as needed, remove DM.
                        if laser_combi == 0:
                            block_laser()
                            unblock_sc()
                        elif laser_combi == 1:
                            unblock_laser()
                            block_sc()
                        elif laser_combi == 2:
                            unblock_laser()
                            unblock_sc()
                        else:
                            close_pco()
                            raise Exception("Something went wrong, laser_combi was %s" % str(laser_combi))
                        remove_DM()
                        # take data
                        datum_params = [repeat_num, fdl_pos, laser_combi, ana_ang, pol_dic[sc_pol], sc_alpha, sc_gamma]
                        meas = [[], [], []]
                        specs, ave_laser_power, ave_sc_power = take_spectrum_with_power(specs=meas[0], laser_powers=meas[1], sc_powers=meas[2], num_spectrum = andor_num)
                        # convert data to nW and account for T/R ratio for SC passing through NPBS
                        ave_laser_power *= 1e9
                        ave_sc_power *= NPBS_TR_from_pol_wl(sc_pol,sc_wl) * 1e9
                        # save data
                        datum = [datum_params, [specs, ave_laser_power, ave_sc_power]]
                        data[1].append(datum)
                        if not slow_save:
                            np.save(os.path.join(main_path, 'data.npy'), np.array(data))
                        
                        #print experiment progress %
                        point_num = position_in_list_of_lists([laser_combi, ana_ang_num, sc_pol_num, fdl_pos_num],
                                                              [laser_combis.values(), ana_angs, sc_pols, fdl_poss])
                        elapsed_time = time.time() - repeat_start_time
                        time_left = elapsed_time*(1.*len(fdl_poss)/(fdl_pos_num+1)-1)
                        overall_point_num = position_in_list_of_lists([laser_combi, ana_ang_num, sc_pol_num, fdl_pos_num, repeat_num],
                                                                      [laser_combis.values(), ana_angs, sc_pols, fdl_poss, range(repeat)])
                        completed = u'Repeat number %i (%.2f percent): %f (%.2f percent) %s left for this repeat.'%(repeat_num,100.0*(float(overall_point_num+1)/total_num_points),
                                                        fdl_pos,100.0*(float(point_num+1)/repeat_num_points),sec_to_hhmmss(time_left))
                        prints(completed,prev)
                        prev = completed
                    #end for laser_combi
                #end for ana_ang
            #end for sc_pol
            if slow_save:
                np.save(os.path.join(main_path, main_dir_name+'.npy'), np.array(data))
        #end for fdl_pos
    #end for repeat
    #Experiment done, clean up.
    block_laser()
    block_sc()
    close_pco()
    LED_power(0)
    remove_DM()
    time_taken = time.time() - experiment_start_time
    print(u'\n' + 'Time taken for experiment = %s' % sec_to_hhmmss(time_taken))
    print(u'\n' + 'Done!')
    play_sound(complete)
    return


#def scan_delayline_with_andor7(sample=None,
#                               fdl_min=0,fdl_max=12,fdl_incre=0.01,
#                               fdl_poss = None,
#                               andor_int_ms=100,andor_num=5,andor_warm_up_sec=60,
#                               repeat=1,
#                               opt_sc=False,slow_save=True,
#                               sc_wl=742,sc_power_nW=2000,set_sc_power_every_n_point=10,sc_pol='L',
#                               pco_exp_time_ms=50,led_power=0.5,num_imgs=10,
#                               take_img_every_n_point=1,
#                               compensate_stage_xy=False,
#                               manual_compensate_stage_xy=False):
#    
#    '''Scan with options to optimise SC for each run, slow save.
#    Also scans with laser only, SC only, and both beams.
#    Polarimeter attached to get reference SC power and polarisation.
#    
#    pm1550m = laser powermeter (Need to zero by hand before starting run!)
#    pmp = polarimeter powermeter = sc powermeter'''
#    curr_time_str = ""
#    if sample == None:
#        curr_time_str = create_date_str()
#    else:
#        curr_time_str = sample
#    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_with_andor7',curr_time_str)
#    os.makedirs(main_path)
#    img_path = os.path.join(main_path,'images')
#    os.makedirs(img_path)
#    if fdl_poss == None:
#        fdl_poss = np.arange(fdl_min,fdl_max+fdl_incre,fdl_incre)
#    else:
#        fdl_poss = np.array(fdl_poss)
#    # write log file with parameters used for experiment
#    input_line = np.array([get_last_input_line()])
#    delay_params_line = 'Scanning from delay line position %s mm to %s mm with step size %s mm.'%(str(fdl_min), str(fdl_max), str(fdl_incre))
#    andor_params_line = 'Andor integration time %s ms, take median of %i points, warming up for %s seconds, repeating %i times.'%(str(andor_int_ms), andor_num, str(andor_warm_up_sec), repeat)
#    image_params_line = 'PCO camera integration time %s ms, averaging over %i images.'%(str(pco_exp_time_ms), num_imgs)
#    misc_params_line = 'SC wavelength %s nm, power at sample %s nW, sc polairzation = %s'%(str(sc_wl), str(sc_power_nW), sc_pol)
#    log_txt = [unicode(input_line),u'\n'+unicode(delay_params_line),u'\n'+unicode(andor_params_line),
#               u'\n'+unicode(image_params_line), u'\n'+unicode(misc_params_line)+u'\n\n']
#    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
#        
#        
#    warm_up_th = threading.Thread(target=warm_up_andor, args=(andor_warm_up_sec,))
#    warm_up_th.start()
#    
#    ###prepare_take_data
#    data = []
#    wls = []
#    img_info = []
#    
#    set_exposure_time(andor_int_ms)
#    remove_DM()
#    initialise_pco()
#    set_pco_exposure_time(pco_exp_time_ms)
#    
#    wls = get_wl_vector()
#    # data will be [[sc wavelength], wavelengths, [both on], [laser only], [sc only]]
#    # each data point is appended in those last 3 lists, [fdl_pos, spectrum, sc_power, laser_power, sc_gamma (if applicable)]
#    # sc_gamma is a measure of the reference polarisation of the SC (after 1 reflection by NPBS) parametrised by alpha, gamma
#    data = [[sc_wl],wls,[],[],[]]
#    # img info will be [[img npy name, rep, fdl pos, time since start]]
#    img_info = []
#    
#    block_laser()
#    block_sc()
#    go_to_pol(sc_pol,sc_wl)
#    pm1550m_wl(1550)
#    pmp_wl(sc_wl)
#    time.sleep(1)
#    # Need to zero pmd by hand as laser slider is after the pmd.
##        pmd_zero()
#    pmp_zero()
#    time.sleep(5)
#    ###end prepare_take_data
#    
#    prev=''
#    total_poss = len(fdl_poss)
#    total_poss_repeat = total_poss*repeat
#    sample_ref_img,laser_ref_img,origin_coords,img_both = take_background_imgs()
#    
#    #unblock lasers for more effective warmup
#    unblock_laser()
#    unblock_sc()
#    warm_up_th.join()
#    experiment_start_time = time.time()
#    
#    #brought outside the take_img_every_n_point compensation to avoid excessive plt.imshow - creates window for real-time tracking
#    insert_DM()
#    plt.pause(0.1)
#    LED_power(0)
#    unblock_laser()
#    curr_laser_img = equalize_histogram_and_8bit(sp.ndimage.median_filter(get_pco_image(num_imgs),3))
#    LED_power(led_power)
#    
#    real_time_img_fig = plt.figure('Sample and laser tracking')
#    ref_img_fig = real_time_img_fig.add_subplot(121)
#    ref_img_fig.set_title('Reference image')
#    ref_img_fig.imshow((sample_ref_img.astype(float)+laser_ref_img.astype(float))/illumination)
#    live_img_fig = real_time_img_fig.add_subplot(122)
#    curr_img_raw = get_pco_image(num_imgs)
#    curr_img = (equalize_histogram_and_8bit(curr_img_raw).astype(float) + curr_laser_img.astype(float))/illumination
#    curr_img_axes = live_img_fig.imshow(curr_img)
#    remove_DM()
#    plt.pause(0.1)
#    LED_power(0)
#    # /end image showing.
#    
#    _i = 0
#    for j in range(repeat): #for each repeat
#        if opt_sc:
#            opt_scpz(verbose=False)
#            block_laser()
#            unblock_sc()
#            plt.pause(5)
#            set_sc_power_nW(sc_power_nW,sc_wl,pm='polarimeter')
#        start_time = time.time()
#        _j = 0
#        for i,fdl_pos in enumerate(fdl_poss):
#            move_fdl_abs(fdl_pos)
#            if i%set_sc_power_every_n_point == 0:
#                popt,perr,max_ang = set_sc_power_nW(sc_power_nW,sc_wl,pm='polarimeter',return_raw=True)
#                gamma = popt[2] #needed for measure_power_and_pol loop
#                set_pola_QWP_ang(max_ang)
#                
#            if i%take_img_every_n_point == 0:
#                if compensate_stage_xy:
#                    insert_DM()
#                    plt.pause(0.1)
#                    LED_power(0)
#                    unblock_laser()
#                    curr_laser_img = equalize_histogram_and_8bit(sp.ndimage.median_filter(get_pco_image(num_imgs),3))
#                    LED_power(led_power)
#                    
#                    curr_img_raw = get_pco_image(num_imgs)
#                    curr_img = (equalize_histogram_and_8bit(curr_img_raw).astype(float) + curr_laser_img.astype(float))/illumination
#                    curr_img_axes.set_data(curr_img)
#                    curr_img_axes.set_clim(np.min(curr_img),np.max(curr_img))
#                    
#                    force_manual_compensate_stage_xy = False
#                    
#                    if not manual_compensate_stage_xy:
#                        is_good_pos = False
#                        comp_run_num = 0
#                        stage_resolution = 0.02#0.15 #um
#                        
#                        while (not is_good_pos) and (comp_run_num < 3):
#                            curr_coords = []
#                            for k in range(5):
#                                curr_coords.append(extract_coords_of_laser_on_sample_2(equalize_histogram_and_8bit(get_pco_image(num_imgs/2)),curr_laser_img,sample_ref_img,laser_ref_img,plot=False,um_per_px=1))
#                            curr_coords = np.mean(curr_coords,axis=0)
#                            print("Before correction %i_%i: laser coords = %s"%(j, _j, str(curr_coords)))
#                            x_comp,y_comp = curr_coords - origin_coords
#                            x_comp = x_comp*2./17#5.7/155
#                            y_comp = y_comp*2./17#5.7/155*3 #don't know why the px-um conversion factor is different for x and y
#                            if abs(x_comp) > 1.5 or abs(y_comp) > 1.5: #if sample drifted too far away, means something is wrong.
#                                print('Unable to find sample.')
#                                play_sound(error_sound)
#                                force_manual_compensate_stage_xy = True
#                                break
#                            elif x_comp < stage_resolution and y_comp < stage_resolution: #sample reached initial position.
#                                is_good_pos = True
#                                break
#                            else:
#                                if x_comp > stage_resolution: #um. if sample drifted just a little bit, better not move it, as limited by the current actuator resolution.
#                                    move_x(-x_comp)
#        #                                move_vpx_abs(get_vpx_pos() - x_comp/1000.)
#                                if y_comp > stage_resolution:
#                                    move_z(-y_comp)
#        #                                move_vpy_abs(get_vpy_pos() + y_comp/1000.)
#                                comp_run_num += 1
#                                curr_img_raw = get_pco_image(num_imgs)
#                                curr_img = equalize_histogram_and_8bit(curr_img_raw)
#                                curr_img = (curr_img.astype(float) + curr_laser_img.astype(float))/illumination
#                                curr_img_axes.set_data(curr_img)
#                                curr_img_axes.set_clim(np.min(curr_img),np.max(curr_img))
#                    
#                    if manual_compensate_stage_xy or force_manual_compensate_stage_xy:
#                        curr_img_raw = get_pco_image(num_imgs)
#                        curr_img = equalize_histogram_and_8bit(curr_img_raw)
#                        curr_img = (curr_img.astype(float) + curr_laser_img.astype(float))/illumination
#                        curr_img_axes.set_data(curr_img)
#                        curr_img_axes.set_clim(np.min(curr_img),np.max(curr_img))
#                        curr_coords = extract_coords_of_laser_on_sample_2(equalize_histogram_and_8bit(curr_img_raw),curr_laser_img,sample_ref_img,laser_ref_img,plot=False,um_per_px=1)
#                        offset_from_init = (np.array(curr_coords) - np.array(origin_coords))*2./17
#                        live_img_fig.set_title('Laser coord offset = %s um.'%offset_from_init)
#                        plt.pause(0.01)
#                        plt.get_current_fig_manager().window.showMaximized()
#                        not_statisfy = [True]
#                        step = [1.]
#                        real_time_img_fig.suptitle('Arrow keys to move, PageUp/PageDown to focus, +/- keys to change increment (%.3f um), Esc to stop.'%step[0])
#                        def press(event):
#                            if event.key == 'left':
#                                move_x(step[0])
#                            elif event.key == 'right':
#                                move_x(-step[0])
#                            elif event.key == 'down':
#                                move_z(-step[0])
#                            elif event.key == 'up':
#                                move_z(step[0])
#                            elif event.key == '+':
#                                step[0] = step[0]*2.
#                            elif event.key == '-':
#                                step[0] = step[0]/2.
#                            elif event.key == 'pageup':
#                                move_y(-step[0])
#                            elif event.key == 'pagedown':
#                                move_y(step[0])
#                            elif event.key == 'escape':
#                                not_statisfy[0] = False
#                            else:
#                                pass
#                        real_time_img_fig.canvas.mpl_connect('key_press_event', press)
#                        play_sound(adjust_piezo_now)
#                        print('Adjust piezo now.')
#                        while not_statisfy[0]:
#                            curr_img_raw = get_pco_image(num_imgs)
#                            curr_img = equalize_histogram_and_8bit(curr_img_raw)
#                            curr_img = (curr_img.astype(float) + curr_laser_img.astype(float))/illumination
#                            curr_img_axes.set_data(curr_img)
#                            curr_img_axes.set_clim(np.min(curr_img),np.max(curr_img))
#                            curr_coords = extract_coords_of_laser_on_sample_2(equalize_histogram_and_8bit(curr_img_raw),curr_laser_img,sample_ref_img,laser_ref_img,plot=False,um_per_px=1)
#                            offset_from_init = (np.array(curr_coords) - np.array(origin_coords))*2./17
#                            live_img_fig.set_title('Laser coord offset = %s um.'%offset_from_init)
#                            real_time_img_fig.suptitle('Arrow keys to move, PageUp/PageDown to focus, +/- keys to change increment (%.3f um), Esc to stop.'%step[0])
#                            plt.pause(0.01)
#                        play_sound(scan_continued)
#                        print('Scan continued.')
#                    
#                curr_data_img = take_and_save_img('%i_%i'%(j,_j),j,fdl_pos,curr_laser_img)
#                print("After correction %i_%i: laser coords = %s"%(j, _j, extract_coords_of_laser_on_sample_2(curr_data_img,curr_laser_img,sample_ref_img,laser_ref_img,plot=False,um_per_px=1)))
#                curr_img = (curr_data_img.astype(float) + curr_laser_img.astype(float))/illumination
#                curr_img_axes.set_data(curr_img)
#                curr_img_axes.set_clim(np.min(curr_img),np.max(curr_img))
#                _j += 1
#            
#            block_laser()
#            unblock_sc()
#            plt.pause(1)
#            take_data(curr_fdl_pos=fdl_pos,laser_on=False,sc_on=True)
#                
#            unblock_laser()
#            block_sc()
#            plt.pause(1)
#            take_data(curr_fdl_pos=fdl_pos,laser_on=True,sc_on=False)
#            
#            unblock_laser()
#            unblock_sc()
#            plt.pause(1)
#            take_data(curr_fdl_pos=fdl_pos,laser_on=True,sc_on=True)
#            
#            if not slow_save:
#                save_data()
#            
#            elapsed_time = time.time() - start_time
#            time_left = elapsed_time*(1.*len(fdl_poss)/(i+1)-1)
#            completed = u'run %i (%.2f percent): %f (%.2f percent) %s left for this run.'%(j,100.0*(float(_i+1)/total_poss_repeat),fdl_pos,100.0*(float(i+1)/total_poss),sec_to_hhmmss(time_left))
#            prints(completed,prev)
#            prev = completed
#            _i += 1
#            
#        if slow_save:
#            save_data()
#    
#    #finishing
#    block_laser()
#    block_sc()
#    close_pco()
#    LED_power(0)
#    remove_DM()
#    time_taken = time.time() - experiment_start_time
#    print(u'\n' + 'Time taken for experiment = %s'%sec_to_hhmmss(time_taken))
#    print(u'\n' + 'Done!')
#    play_sound(complete)
#    
#    
#    ### FN DEFINITIONS ###
#
#    def take_data(curr_fdl_pos,laser_on,sc_on):
#        _specs = [0]
#        _sc_power = [0]
#        _sc_gamma = [0]
#        _laser_power = [0]
#        
#        def spec_loop():
#            curr_spec = []
#            for k in range(andor_num):
#                curr_spec.append(copy.copy(get_spectrum()))
#            specs = np.array(curr_spec)
#            _specs[0] = specs
#        
#        def measure_power_and_pol_loop():
#            curr_sc_power = []
#            curr_laser_power = []
#            curr_sc_gamma = gamma #from the periodic setting of SC power in actual measurement
#            while spec_th.isAlive():
#                curr_laser_power.append(pm1550m_power())
#                sc_power = pmp_power()
#                curr_sc_power.append(sc_power)
#                time.sleep(0.01)
#            if laser_on:
#                # Laser power is recorded in nW
#                _laser_power[0] = np.mean(curr_laser_power)*1e9
#            if sc_on:
#                # SC power is recorded in nW
#                _sc_power[0] = np.mean(curr_sc_power)*NPBS_TR_from_pol_wl(sc_pol,sc_wl)*1e9
##                _sc_power[0] = np.mean(curr_sc_power)*NPBS_TR_gamma_fx(sc_wl,gamma)*1e9
#                _sc_gamma[0] = curr_sc_gamma   
#        
#        spec_th = threading.Thread(target=spec_loop)        
#        measure_power_and_pol_th = threading.Thread(target=measure_power_and_pol_loop)
#        
#        spec_th.start()
#        measure_power_and_pol_th.start()
#        spec_th.join()
#        measure_power_and_pol_th.join()
#        
#        if laser_on:
#            if sc_on: #both on
#                data[2].append([curr_fdl_pos,_specs[0],_sc_power[0],_laser_power[0],_sc_gamma[0]])
#            else: #only laser on
#                data[3].append([curr_fdl_pos,_specs[0],_sc_power[0],_laser_power[0]])
#        else: #only sc on
#            data[4].append([curr_fdl_pos,_specs[0],_sc_power[0],_laser_power[0],_sc_gamma[0]])
#    
#    def save_data():
#        np.save(os.path.join(main_path,curr_time_str+'.npy'),np.array(data))
#    
#    def take_and_save_img(name,rep,fdl_pos,laser_only_img):
#        block_sc()
#        unblock_laser()
#        img = DMin_LEDon_TAKEimg(LED_p=led_power, num_imgs=num_imgs, DM_out_after=True)
#        img_info_entry = (name,rep,fdl_pos,time.time()-experiment_start_time)
#        img_info.append(img_info_entry)
#        img_comp = equalize_histogram_and_8bit(img)
#        np.save(os.path.join(main_path, 'img_info.npy'), img_info)
#        np.save(os.path.join(img_path, name+'.npy'), img_comp)
#        np.save(os.path.join(img_path,name+'_laser.npy'), laser_only_img)
##        if compensate_stage_xy:
##            return _process_img(os.path.join(img_path,name+'.npy'))
#        return img_comp
#        
#    def take_background_imgs():
#        unblock_laser()
#        block_sc()
#        img_laser_only = DMin_LEDon_TAKEimg(LED_p=0, num_imgs=num_imgs, DM_out_after=False)
#        _img_laser_only = copy.copy(img_laser_only)
#        img_laser_only = equalize_histogram_and_8bit(img_laser_only)
#        
#        block_laser()
#        img_LED_only = DMin_LEDon_TAKEimg(LED_p=led_power, num_imgs=num_imgs, DM_out_after=True)
#        _img_LED_only = copy.copy(img_LED_only)
#        img_LED_only = equalize_histogram_and_8bit(img_LED_only)
#        
#        np.save(os.path.join(main_path, 'img_LED_only.npy'), img_LED_only)
#        np.save(os.path.join(main_path, 'img_laser_only.npy'), img_laser_only)
#        
##        if compensate_stage_xy:
##            _process_img(os.path.join(main_path,'img_LED_only.npy')) #to remove background
#        img_both = equalize_histogram_and_8bit(_img_laser_only + _img_LED_only)
#        origin_coords = extract_coords_of_laser_on_sample(img_both,img_LED_only,img_laser_only,um_per_px=1,plot=False)
#        print("Original laser coordinates (px): %s"%str(origin_coords))
#        return img_LED_only,img_laser_only,origin_coords,img_both        
#    
#    # Andor somehow is more sensitive when starting to take data after a long pause. This decays back to normal.
#    # "timewasting" to get Andor to warm up and remove (?) this artifact.
#    def warm_up_andor(warm_up_time):
#        s = time.time()
#        i = 0
#        while (time.time() - s) < warm_up_time:
#            get_spectrum()
#            time.sleep(0.05)
#            i += 1
#        #print("Warmed up for "+ str(time.time()-s) + " seconds. ")
#        #print("Took " + str(i) + " spectra. ")

#%%
# ------------------------------------------------------------------
# ------------------------Auxiliary functions-----------------------
# ------------------------------------------------------------------


sound_dir = 'D:\Nonlinear_setup\Python_codes\soundfx\\'
complete = sound_dir + 'scancompleted.wav'
error_sound = sound_dir + 'error.wav'
scan_continued = os.path.join(sound_dir,'scan_continued.mp3')
adjust_piezo_now = os.path.join(sound_dir,'adjust_piezo_now.mp3')

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

def move_analyzer_and_acquire_data(acq_fx,speed_deg_per_s=25,acq_fx_walltime=0.012706,timesleep=0.1,ana_range=360,measure_only_const_spd=True):
    """
    Rotate analyzer (rot2) contineously while acquiring data with acq_fx. Returns [interpolated analyzer angles,data,raw analyzer angle,timestamp for angle, timestamp for data]
    WARNING: there is an offset angle compared to data acquired with usual rot -> stop -> measure -> repeat scheme. Do NOT compare angle rotation between this scheme and old one.
    acq_fx = function to be called to get data, should return a value or list of values.
    speed_deg_per_s = rotational speed of analyzer, default 25 deg/s (maximum available)
    acq_fx_walltime = time in second needed to acquire data, default to 0.012706s for lock-in without optimised gain.
    timesleep = time in second paused between data point, default 0.1 s.
    ana_range = analyzer range to scan, default 360 deg.
    measure_only_const_spd = True to account variation in speed during acceleration and deceleration, discarding the data when the speed is not constant.
    """
    input_line = np.array([get_last_input_line()])
    global _ana_angs,_data,_ang_times,_data_times,_ana_angs_interp
    _ana_angs = []
    _data = []
    _ang_times = []
    _data_times = []
    
    if measure_only_const_spd:
        acc = 24.5 #deg/s/s
        time_to_reach_const_spd = speed_deg_per_s/acc #s
        dist_to_reach_const_spd = 0.5*speed_deg_per_s**2/acc #deg
#        data_len_to_discard = int(time_to_reach_const_spd/timesleep)
        
        rot2.mcRel(-dist_to_reach_const_spd,25)
        
        def move_loop():
            rot2.mcRel(ana_range+2*dist_to_reach_const_spd,speed_deg_per_s)
    else:
        def move_loop():
            rot2.mcRel(ana_range,speed_deg_per_s)
    move_th=threading.Thread(target=move_loop)
    
    def acquire_data_loop():
        while move_th.isAlive():
            _data.append(acq_fx())
            _data_times.append(time.time()-start_time)
            time.sleep(timesleep - acq_fx_walltime)
    acquire_data_th=threading.Thread(target=acquire_data_loop)
    
    def read_ang_loop():
        while move_th.isAlive():
            _ana_angs.append(move_A())
            _ang_times.append(time.time()-start_time)
    read_ang_th=threading.Thread(target=read_ang_loop)
    
    move_th.start()
    start_time=time.time()
    acquire_data_th.start()
    read_ang_th.start()
    
    move_th.join()
    acquire_data_th.join()
    read_ang_th.join()
    
    if measure_only_const_spd:
        data_len_to_discard = get_nearest_idx_from_list(time_to_reach_const_spd,_data_times)
        _data = _data[data_len_to_discard:-data_len_to_discard]
        _data_times = _data_times[data_len_to_discard:-data_len_to_discard]
    
    _ana_angs = put_angles_to_same_range(_ana_angs,360)
    ana_angs_fx = sp.interpolate.interp1d(_ang_times,_ana_angs,kind='cubic')
    _ana_angs_interp = ana_angs_fx(np.array(_data_times)-acq_fx_walltime)
    
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\temp',create_date_str2()+'.npz')
    np.savez(main_path,
             input_line=input_line,
             _ana_angs_interp=_ana_angs_interp,
             _ana_ang=_ana_angs,
             _data=_data,
             _ang_times=_ang_times,
             _data_times=_data_times)
    
    plt.polar(np.array(_ana_angs_interp)/180.*np.pi,_data)
    return [_ana_angs_interp,_data,_ana_angs,_ang_times,_data_times]
    
def keyboard_control_delayline(step=0.01):
    def _f():
        print(move_fdl_abs(get_fdl_pos()+step))
    def _b():
        print(move_fdl_abs(get_fdl_pos()-step))
    start_keyboard_listening([',','.'],[_b,_f])

def keyboard_control_piezo():
    step = [1]
    def _higher_step():
        step[0] = step[0]*2.
    def _lower_step():
        step[0] = step[0]/2.
    def _up():
        move_z(step[0])
    def _down():
        move_z(-step[0])
    def _left():
        move_x(step[0])
    def _right():
        move_x(-step[0])
    start_keyboard_listening(['up','down','left','right','+','-'],[_up,_down,_left,_right,_higher_step,_lower_step])

def check_circular_polarization():
    global A_data, li_data
    def press(event):
        global li_data,A_data
        if event.key == 'm':
            print mallus_fitter(np.array(A_data)*180./np.pi,np.array(li_data))
        elif event.key == 'c':
            li_data,A_data = [],[]
    _fig = plt.figure('lockin')
    _fig.suptitle('c to clear, m to fit.')
    _fig.canvas.mpl_connect('key_press_event',press)
    plt.ion()
    _fig.clf()
    fig = _fig.add_subplot(111,projection='polar')
    fig.axhline(color='k')
    li_data=[]
    A_data=[]
    line, = fig.plot(A_data,li_data,'o')
    fig.set_ylabel('lockin R, V')
    plt.pause(0.01)
    srs830.clear()
    try:
        while True:
            li_data.append(get_lockin_reading1())
#            A_data.append(move_A()/180.*np.pi)
            A_data.append(move_rot4()/180.*np.pi)
            line.set_xdata(A_data)
            line.set_ydata(li_data)
            fig.relim()
            fig.autoscale_view(True,True,True)
            plt.pause(0.01)
    except KeyboardInterrupt:
        return np.array(A_data), np.array(li_data)

def DMin_LEDon_TAKEimg(LED_p=0.017,LED_p_after=0,DM_out_after=True,num_imgs=10):
    insert_DM()
    LED_power(LED_p)
    plt.pause(1)
    
    img = get_pco_image(num_imgs=num_imgs)
    
    LED_power(LED_p_after)
    if DM_out_after:
        remove_DM()
#        plt.pause(0.1)
    return img

def equalize_histogram_and_8bit(img):
    return ((img-np.min(img))/(np.max(img)-np.min(img)) * 255).astype(np.uint8)

def get_offset_um(ref_img,data_img,um_per_px=5.7/155):
    data_img_crop = data_img[100:-100,100:-100]
    try:
        conv_data_sample = cv2.matchTemplate(data_img_crop,ref_img,5)
    except cv2.error:
        data_img_crop = equalize_histogram_and_8bit(data_img_crop)
        ref_img = equalize_histogram_and_8bit(ref_img)
        conv_data_sample = cv2.matchTemplate(data_img_crop,ref_img,5)
    finally:
        min_val,max_val_sample,min_loc,max_loc_sample = cv2.minMaxLoc(conv_data_sample)
        return (np.array(max_loc_sample)-100)*um_per_px*3

def close_vp_stages():
    close_vpx()
    close_vpy()

def initialise_vp_stages():
    initialise_vpx()
    initialise_vpy()

def _process_img(path):
    image = np.load(path)
    subtract_background_rolling_ball(image, 150, light_background=False, use_paraboloid=True, do_presmooth=True)
    np.save(path[:-4]+'_noBG.npy', image)
    return image

def position_in_list_of_lists(indices, lists):
    """Inputs: (list of indices, list of lists). First entry in each is the innermost 'for' loop.
    Output: 'global' index that the set of indices corresponds to."""
    for i, lst in enumerate(lists):
        for j in range(i+1, len(indices)):
            indices[j] *= len(lst)
    return sum(indices)

def indices_in_list_of_lists(global_index, lists):
    """Inputs: (int global_index, list of lists). First entry in the list is the innermost 'for' loop.
    Output: Set of indices in the lists that the global_index corresponds to."""
    x = global_index
    indices = []
    for i, lst in enumerate(lists):
        curr_index = x % len(lst)
        indices.append(curr_index)
        x = (x - curr_index) / len(lst)
    return indices
        