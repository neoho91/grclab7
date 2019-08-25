# -*- coding: utf-8 -*-
"""
Created on Thu May 02 23:17:21 2019

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
sys.path.append(os.path.abspath(r"D:\WMP_setup\Python_codes"))
sys.path.append(os.path.abspath(r"D:\Nonlinear_setup\Python_codes"))
from neo_common_code import *
from filter_rot_stages import *
from TS_CCS200 import *

def calibrate_single_filter(AOIs=np.arange(0,70.1,1),filter_number=1,with_OO=False,int_time_ms='auto',ave_num=100,log=''):
    """
    Turn on supercontinuum, put spectrometer after the filters, rotate 1 filter and acquire spectrum. The spectrum is to be normalized with filter at 90deg.
    """
    start_time=time.time()
    sample = create_date_str()
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\calibrate_single_filter',sample)
    os.makedirs(main_path)

    init_line = 'Started calibrate_single_filter (%s) on %s, filter number = %i, AOIs = \n%s'%(
            sample,time.strftime("%d%b%Y %H:%M", time.localtime()), filter_number,
            str(AOIs))
    print(init_line)
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    np.save(os.path.join(main_path,'AOIs'),AOIs)    
    
    def prepare_take_data():
        home_rot(1)
        home_rot(2)
        global specs, bg_specs
        specs = []
        bg_specs = []
        if with_OO:
            OO_initialize()
            np.save(os.path.join(main_path,'OO_wavelength'),OO_wavelengths)
            OO_set_int_time(int_time_ms)
        else:
            global CCS200
            CCS200=CCS(0)
            print 'Spectrometer CCS200 online.'
            np.save(os.path.join(main_path,'OO_wavelength'),CCS200.get_wavelengths())
            if int_time_ms == 'auto':
                CCS200.get_optimised_spec()
            else:
                CCS200.set_int_time(int_time_ms/1e3)
        
    def take_data(ang):
        set_AOI_rot1(90)
        set_AOI_rot2(90)
        if with_OO:
            time.sleep(OO_get_int_time()/1e3)
            bg_specs.append(copy.copy(OO_capture()))
        else:
            time.sleep(CCS200.get_int_time()/1e3)
            if int_time_ms == 'auto':
                CCS200.get_optimised_spec()
            s = []
            for i in range(ave_num):
                s.append(copy.copy(CCS200.get_spec()))
            bg_specs.append(np.mean(s,axis=0))
        
        if filter_number == 1:
            set_AOI_rot1(ang)
        else:
            set_AOI_rot2(ang)
        if with_OO:
            time.sleep(OO_get_int_time()/1e3)
            specs.append(copy.copy(OO_capture()))
        else:
            time.sleep(CCS200.get_int_time()/1e3)
            s = []
            for i in range(ave_num):
                s.append(copy.copy(CCS200.get_spec()))
            specs.append(np.mean(s,axis=0))
        
    def save_data():
        np.save(os.path.join(main_path,'specs'),np.array(specs))
        np.save(os.path.join(main_path,'bg_specs'),np.array(bg_specs))
    
    prepare_take_data()
    prev=''
    for i,AOI in enumerate(AOIs):
        curr_line = 'At AOI = %.1f deg (%.1f precent)'%(AOI,float(i)/len(AOIs)*100)
        prints(curr_line,prev)
        prev = curr_line
        
        take_data(AOI)
        save_data()
    
    if with_OO:
        OO_shutdown()
    else:
        CCS200.shutdown()
    final_line='\nDone! Time spent = %s'%sec_to_hhmmss(time.time()-start_time)
    print(final_line)
    log_txt.append(final_line)
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    try:
        anal_calibrate_single_filter(sample)
    except:
        pass
    
def calibrate_single_filter_manual_bg(AOIs=np.arange(0,70.1,1),filter_number=1,with_OO=False,int_time_ms='auto',ave_num=100,log=''):
    """
    Turn on supercontinuum, put spectrometer after 1 filter, rotate 1 filter and acquire spectrum. The spectrum is to be normalized without any filters.
    """
    start_time=time.time()
    sample = create_date_str()
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\calibrate_single_filter_manual_bg',sample)
    os.makedirs(main_path)

    init_line = 'Started calibrate_single_filter (%s) on %s, filter number = %i, AOIs = \n%s'%(
            sample,time.strftime("%d%b%Y %H:%M", time.localtime()), filter_number,
            str(AOIs))
    print(init_line)
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    np.save(os.path.join(main_path,'AOIs'),AOIs)    
    
    def prepare_take_bg():
        global bg_specs
        bg_specs = []
        if with_OO:
            OO_initialize()
            np.save(os.path.join(main_path,'OO_wavelength'),OO_wavelengths)
            OO_set_int_time(int_time_ms)
        else:
            global CCS200
            CCS200=CCS(0)
            print 'Spectrometer CCS200 online.'
            np.save(os.path.join(main_path,'OO_wavelength'),CCS200.get_wavelengths())
            if int_time_ms == 'auto':
                CCS200.get_optimised_spec()
            else:
                CCS200.set_int_time(int_time_ms/1e3)
                
    def take_background():
        if with_OO:
            time.sleep(OO_get_int_time()/1e3)
            bg_specs.append(copy.copy(OO_capture()))
        else:
            time.sleep(CCS200.get_int_time()/1e3)
            if int_time_ms == 'auto':
                CCS200.get_optimised_spec()
            s = []
            for i in range(ave_num):
                s.append(copy.copy(CCS200.get_spec()))
            bg_specs.append(np.mean(s,axis=0))
    
    def save_background():
        np.save(os.path.join(main_path,'bg_specs'),np.array(bg_specs))
        
    def prepare_take_data():
        home_rot(filter_number)
        time.sleep(2)
        global specs
        specs = []
        
    def take_data(ang):
        if filter_number == 1:
            set_AOI_rot1(ang)
        else:
            set_AOI_rot2(ang)
        if with_OO:
            time.sleep(OO_get_int_time()/1e3)
            specs.append(copy.copy(OO_capture()))
        else:
            time.sleep(CCS200.get_int_time()/1e3)
            s = []
            for i in range(ave_num):
                s.append(copy.copy(CCS200.get_spec()))
            specs.append(np.mean(s,axis=0))
        
    def save_data():
        np.save(os.path.join(main_path,'specs'),np.array(specs))
    
    raw_input('Remove all versachrome filters, then press ENTER.')
    print('Taking background...')
    prepare_take_bg()
    take_background()
    save_background()
    
    raw_input('Install versachrome %i, then press ENTER.'%filter_number)
    print('Taking data...')
    prepare_take_data()
    prev=''
    for i,AOI in enumerate(AOIs):
        curr_line = 'At AOI = %.1f deg (%.1f precent)'%(AOI,float(i)/len(AOIs)*100)
        prints(curr_line,prev)
        prev = curr_line
        
        take_data(AOI)
        save_data()
    
    if with_OO:
        OO_shutdown()
    else:
        CCS200.shutdown()
    final_line='\nDone! Time spent = %s'%sec_to_hhmmss(time.time()-start_time)
    print(final_line)
    log_txt.append(final_line)
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    try:
        anal_calibrate_single_filter(sample)
    except:
        pass

def anal_calibrate_single_filter(sample,fitting_start_wavelength = 650, fitting_end_wavelength = 830):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\calibrate_single_filter',sample)
    if not os.path.isdir(main_path):
        main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\calibrate_single_filter_manual_bg',sample)
    if not os.path.isdir(main_path):
        main_path = sample
    AOIs = np.load(os.path.join(main_path,'AOIs.npy'))
    OO_wavelength = np.load(os.path.join(main_path,'OO_wavelength.npy'))
    if len(OO_wavelength) == 1:
        OO_wavelength = OO_wavelength[0]
    specs = np.load(os.path.join(main_path,'specs.npy'))
    bg_specs = np.load(os.path.join(main_path,'bg_specs.npy'))
    
    fitting_start_idx = get_nearest_idx_from_list(fitting_start_wavelength,OO_wavelength)
    fitting_end_idx = get_nearest_idx_from_list(fitting_end_wavelength,OO_wavelength)
    OO_wavelength_crop = OO_wavelength[fitting_start_idx:fitting_end_idx]
    
    fitted_centrals, fitted_fwhms, fitted_T_peaks, fitted_slant = [],[],[],[]
    fitted_centrals_e, fitted_fwhms_e, fitted_T_peaks_e, fitted_slant_e = [],[],[],[]
    plt.figure(sample)
    for i,spec in enumerate(specs):
        baseline = np.median(spec)
        spec = spec - baseline
        spec_noise_max = np.max(spec[get_nearest_idx_from_list(490,OO_wavelength):get_nearest_idx_from_list(560,OO_wavelength)])
        spec = np.clip(spec,spec_noise_max,np.inf) - spec_noise_max
        if len(bg_specs) == 1:
            bg_spec = bg_specs[0] - baseline - spec_noise_max
        else:
            bg_spec = bg_specs[i] - baseline - spec_noise_max
        T = np.clip(spec/bg_spec,0,1)
        T_crop = T[fitting_start_idx:fitting_end_idx]
        T_crop = np.convolve(T_crop,np.ones(3)/3.)[1:-1].clip(0)
        
        popt, perr = fit_trapz_curve(OO_wavelength_crop,T_crop)
        curr_T_peak, curr_central, curr_slant, curr_fwhm = popt
        curr_T_peak_e, curr_central_e, curr_slant_e, curr_fwhm_e = perr
        
        fitted_centrals.append(curr_central)
        fitted_fwhms.append(curr_fwhm)
        fitted_T_peaks.append(curr_T_peak)
        fitted_slant.append(curr_slant)
        fitted_centrals_e.append(curr_central_e)
        fitted_fwhms_e.append(curr_fwhm_e)
        fitted_T_peaks_e.append(curr_T_peak_e)
        fitted_slant_e.append(curr_slant_e)
        
        plt.plot(OO_wavelength_crop,T_crop,'o',c=plt.cm.RdYlGn(255*i/(len(specs)-1)))
        plt.plot(OO_wavelength_crop,trapz_curve(OO_wavelength_crop,curr_T_peak,curr_central,curr_slant,curr_fwhm),c=plt.cm.RdYlGn(255*i/(len(specs)-1)))
    
    plt.savefig(os.path.join(main_path,'Fitting.png'))
    
    np.save(os.path.join(main_path,'fitted_centrals'),np.array(fitted_centrals))
    np.save(os.path.join(main_path,'fitted_fwhms'),np.array(fitted_fwhms))
    np.save(os.path.join(main_path,'fitted_T_peaks'),np.array(fitted_T_peaks))
    np.save(os.path.join(main_path,'fitted_slant'),np.array(fitted_slant))
    np.save(os.path.join(main_path,'fitted_centrals_e'),np.array(fitted_centrals_e))
    np.save(os.path.join(main_path,'fitted_fwhms_e'),np.array(fitted_fwhms_e))
    np.save(os.path.join(main_path,'fitted_T_peaks_e'),np.array(fitted_T_peaks_e))
    np.save(os.path.join(main_path,'fitted_slant_e'),np.array(fitted_slant_e))
    
    _fig = plt.figure('%s summary'%sample)
    _fig.suptitle('Fitting for %s'%sample)
    
    fig1 = _fig.add_subplot(221)
    fig1.errorbar(AOIs[:len(fitted_centrals)],fitted_centrals,marker='o',lw=0,elinewidth=1,yerr=fitted_centrals_e,capsize=2)
    fig1.set_title('Central wavelength')
    fig1.set_ylabel('nm')
    fig1.set_xlabel('AOI, deg')
    
    fig2 = _fig.add_subplot(222)
    fig2.errorbar(AOIs[:len(fitted_centrals)],fitted_fwhms,marker='o',lw=0,elinewidth=1,yerr=fitted_fwhms_e,capsize=2)
    fig2.set_title('FWHM')
    fig2.set_ylabel('nm')
    fig2.set_xlabel('AOI, deg')
    
    fig3 = _fig.add_subplot(223)
    fig3.errorbar(AOIs[:len(fitted_centrals)],fitted_T_peaks,marker='o',lw=0,elinewidth=1,yerr=fitted_T_peaks_e,capsize=2)
    fig3.set_title('Max transmission')
    fig3.set_ylabel('au')
    fig3.set_xlabel('AOI, deg')
    
    fig4 = _fig.add_subplot(224)
    fig4.errorbar(AOIs[:len(fitted_centrals)],fitted_slant,marker='o',lw=0,elinewidth=1,yerr=fitted_slant_e,capsize=2)
    fig4.set_title('slant')
    fig4.set_ylabel('nm')
    fig4.set_xlabel('AOI, deg')
    
    plt.pause(1e-6)
    _fig.tight_layout()
    _fig.savefig(os.path.join(main_path,'Fitted summary.png'))

#------------------------------------------------------------------------------
def trapz_curve(x,h,c,d,fwhm): #T peak, central, slant, fwhm
    sq = np.where(abs(x-c)<=(fwhm-d)/2., h, 0)
    rt = np.where(abs(x-c-fwhm/2.) <= d/2., h-(x-(c+(fwhm-d)/2.))*float(h)/d, 0)
    lt = np.where(abs(x-c+fwhm/2.) < d/2., (x-(c-(fwhm+d)/2.))*float(h)/d, 0)
    return sq + lt + rt

def fit_trapz_curve(Xs,Ys,fwhm = 10,d=2):
    h = np.max(Ys)
    c = Xs[list(Ys).index(h)]
    #2*np.abs(c-Xs[get_nearest_idx_from_list(h/2.,Ys)])
    p0 = (h, c, d, fwhm)
    popt, pcov = sp.optimize.curve_fit(trapz_curve, Xs, Ys, p0)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr