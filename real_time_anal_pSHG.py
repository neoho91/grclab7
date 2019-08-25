# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:52:44 2017

@author: Neo
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import csv
import sys
hero_pixel_wavelengths = np.load('D:/WMP_setup/Python_codes/hero_pixel_wavelengths.npy')

def real_time_anal_polarized_SHG(sample,time_sleep=11):
    while True:
        plt.cla()
        anal_polarized_SHG(sample,extend=False)
        plt.pause(time_sleep)

def anal_polarized_SHG(sample,extend=True,symmetry=6,gamma=0,correction=False,anal_390=True):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\polarized_SHG',sample)
    npy_files = list(filter(lambda name: name.endswith('.npy'),os.listdir(main_path)))
    spec_files = list(filter(lambda name: '%sspectrum'%sample in name,npy_files))
    spec_files = list(filter(lambda name: 'spectrum_b' not in name, spec_files))
    spec_bg_files = list(filter(lambda name: 'bck_gnd_spectrum_before' in name,npy_files))
    def get_pol_angle(spec_file_name):
        return 2*float(spec_file_name.split('spectrum')[1].split('.npy')[0])
    
    spec_files.sort(key=get_pol_angle)
    pol_angles = np.array(list(map(lambda name:get_pol_angle(name),spec_files)))
    bg_spec = np.load(os.path.join(main_path,spec_bg_files[0])) #just use 1 bg spec
    
    sum390s = []
    for spec_file in spec_files:
        spec = np.load(os.path.join(main_path,spec_file)) - bg_spec
#        smoothed = smooth_spec(spec)
#        sum390 = np.sum(smoothed[25:40])
        if anal_390:
            sum390s.append(np.sum(spec[60:90]))
        else:
            sum390s.append(np.sum(spec[550:620]))
        if get_pol_angle(spec_file) == 12:
            sc_power = np.max(spec[480:550])
    if max(pol_angles)-min(pol_angles) < 360 and extend:
        pol_angles,sum390s = extend_to_2pi(pol_angles,sum390s,symmetry=symmetry)
    pol_angles = np.array(pol_angles) + gamma
    if correction:
        sum390s = correcting_petals(pol_angles,np.array(sum390s))
    fig = plt.figure(sample)
    polarplot=fig.add_subplot(111, projection='polar')
    polarplot.plot(pol_angles/180.*np.pi,sum390s)
    polarplot.set_title(sample)
    return pol_angles,np.array(sum390s)

def extend_to_2pi(pol_angles,intensities,symmetry=6):
    max_ang = max(pol_angles)
    min_ang = min(pol_angles)
    incre = pol_angles[1]-pol_angles[0]
    ang_range = max_ang-min_ang
    sym_ang_range = 360/symmetry
    if ang_range < sym_ang_range:
        print('Measured angle range too small, unable to extend.')
        return
    residual_ang_range = ang_range % sym_ang_range
    if residual_ang_range == 0:
        num_of_rep = int(360/(ang_range-residual_ang_range))
    else:
        num_of_rep = int(360/(ang_range-residual_ang_range) + 1)
    if residual_ang_range%incre != 0:
        print('Angle increment size and duplicated data range cannot be divided equally.')
        len_of_duplicate = int(np.ceil(residual_ang_range/float(incre)))
    else:
        len_of_duplicate = int(residual_ang_range/incre + 1)
        
    rep_inten_1 = list(intensities[:-len_of_duplicate])*num_of_rep #preserve the small angle values
    rep_ang_1 = np.arange(pol_angles[0],pol_angles[-len_of_duplicate]+(num_of_rep-1)*(ang_range-residual_ang_range),incre)
    rep_inten_2 = list(intensities[len_of_duplicate:])*num_of_rep #preserve the large angle values
    rep_ang_2 = np.arange(pol_angles[len_of_duplicate],pol_angles[-1]+incre+(num_of_rep-1)*(ang_range-residual_ang_range),incre)

    rep_inten = (np.array(rep_inten_1[len_of_duplicate:]) + np.array(rep_inten_2[:-len_of_duplicate]))/2
    rep_ang = np.arange(rep_ang_2[0],rep_ang_1[-1]+incre,incre)
    return rep_ang,rep_inten

def anal_pSHG_VP2_sigle_delay(sample,delay_um,extend=True,symmetry=6,rel_sc_power_POL_ang=24,plot=True,mdeg=True):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\pSHG_VP2',sample)    
    npy_files = list(filter(lambda name: name.endswith('.npy'),os.listdir(main_path)))
    spec_files = list(filter(lambda name: 'spectrum' in name,npy_files))
    spec_bg_files = list(filter(lambda name: 'bck_gnd' in name,npy_files))
    spec_files = list(filter(lambda name: 'bck_gnd' not in name, spec_files))
    spec_files = list(filter(lambda name: delay_um == get_delay(name),spec_files))
    spec_c_files = list(filter(lambda name: is_control_spec(name),spec_files))
    spec_t_files = list(filter(lambda name: not is_control_spec(name),spec_files))
    
    spec_c_files.sort(key=lambda x: get_pol_angle(x,mdeg=mdeg))
    spec_t_files.sort(key=lambda x: get_pol_angle(x,mdeg=mdeg))
    pol_c_angles = np.array(list(map(lambda name:get_pol_angle(name,mdeg=mdeg),spec_c_files)))
    pol_t_angles = np.array(list(map(lambda name:get_pol_angle(name,mdeg=mdeg),spec_t_files)))
    bg_spec = np.load(os.path.join(main_path,spec_bg_files[0])) #just use 1 bg spec
    
    #control
    sum390s_c = []
    for spec_file in spec_c_files:
        spec = np.load(os.path.join(main_path,spec_file)) - bg_spec
        smoothed = smooth_spec(spec)
        sum390_c = np.sum(smoothed[24:40])
        if sum390_c < 0:
            sum390_c = 0
        sum390s_c.append(sum390_c)
    if max(pol_c_angles)-min(pol_c_angles) < 360 and extend:
        pol_c_angles,sum390s_c = extend_to_2pi(pol_c_angles,sum390s_c,symmetry=symmetry)
    
    #test
    sc_power = 'unmeasured'
    sum390s_t = []
    for spec_file in spec_t_files:
        spec = np.load(os.path.join(main_path,spec_file)) - bg_spec
        smoothed = smooth_spec(spec)
        sum390_t = np.sum(smoothed[24:40])
        if sum390_t < 0:
            sum390_t = 0
        sum390s_t.append(sum390_t)
        if round(get_pol_angle(spec_file,mdeg=mdeg),1) == round(rel_sc_power_POL_ang,1):
            sc_power = np.max(spec[480:550])
    if max(pol_t_angles)-min(pol_t_angles) < 360 and extend:
        pol_t_angles,sum390s_t = extend_to_2pi(pol_t_angles,sum390s_t,symmetry=symmetry)
        
    pol_c_angles = np.array(pol_c_angles)
    pol_t_angles = np.array(pol_t_angles)
    if plot:
        fig = plt.figure('%s %ium'%(sample,delay_um))
        polarplot=fig.add_subplot(111, projection='polar')
        polarplot.plot(pol_c_angles/180.*np.pi,sum390s_c,color='C0',label='control')
        polarplot.plot(pol_t_angles/180.*np.pi,sum390s_t,color='C1',label='test')
        polarplot.set_title('%s %ium, SC rel power %s'%(sample,delay_um,str(sc_power)))
        polarplot.legend(loc='best')
    return pol_c_angles,sum390s_c,sum390s_t, sc_power

def anal_pSHG_VP2(sample,extend=True,symmetry=6,rel_sc_power_POL_ang=24,plot=True,expected_amax=24,mdeg=True):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\pSHG_VP2',sample)  
    print('Loading data...')
    npy_files = list(filter(lambda name: name.endswith('.npy'),os.listdir(main_path)))
    spec_files = list(filter(lambda name: 'spectrum' in name,npy_files))
    spec_files = list(filter(lambda name: 'bck_gnd' not in name, spec_files))
    spec_c_files = list(filter(lambda name: is_control_spec(name),spec_files))
    
    delays = set(np.array(list(map(lambda name:get_delay(name),spec_c_files))))
    delays = sorted(delays)
    delays_ps = np.array(delays)*2/3*0.01
    
    prints('Extracting petals...')
    prev_completed=''
    total_delays_len = len(delays)
    controls = []
    tests = []
    sc_powers = []
    for i,delay in enumerate(delays):
        pol_angles,sum390s_c,sum390s_t, sc_power=anal_pSHG_VP2_sigle_delay(sample,delay,extend=extend,symmetry=symmetry,rel_sc_power_POL_ang=rel_sc_power_POL_ang,plot=plot,mdeg=mdeg)
        controls.append(sum390s_c)
        tests.append(sum390s_t)
        sc_powers.append(sc_power)
        
        completed = '%.1f %%'%((i+1.)/total_delays_len*100.)
        prints(completed,prev_completed)
        prev_completed = completed
    controls = list(map(lambda p: p/max(p),controls))
    tests = list(map(lambda p: p/max(p),tests))
    C_file = open(os.path.join(main_path,'%s_CONT.csv'%sample),'wb')
    C_writer = csv.writer(C_file)
    C_writer.writerow(pol_angles)
    for ele in controls:
        C_writer.writerow(ele)
    C_file.close()
    T_file = open(os.path.join(main_path,'%s_TEST.csv'%sample),'wb')
    T_writer = csv.writer(T_file)
    T_writer.writerow(pol_angles)
    for ele in tests:
        T_writer.writerow(ele)
    T_file.close()
    D_file = open(os.path.join(main_path,'%s_DELAYS.csv'%sample),'wb')
    D_writer = csv.writer(D_file)
    D_writer.writerow(delays)
    D_writer.writerow(delays_ps)
    D_file.close()
    P_file = open(os.path.join(main_path,'%s_SC_REL_P.csv'%sample),'wb')
    P_writer = csv.writer(P_file)
    P_writer.writerow(sc_powers)
    P_file.close()
    
    prints('\nFitting petals...')
    prev_completed=''
    controls_len = len(controls)
    tests_len = len(tests)
    total_data_len = controls_len+tests_len
    a_max_c = []
    a_max_c_err = []
    a_max_t = []
    a_max_t_err = []
    for i,control in enumerate(controls):
        (A,phi,offset),(A_err,phi_err,offset_err) = _six_petals_fitter(pol_angles,control)
        while phi > expected_amax + 360/symmetry*0.9:
            phi -= 360/symmetry
        while phi < expected_amax - 360/symmetry*0.9:
            phi += 360/symmetry
        a_max_c.append(phi)
        a_max_c_err.append(phi_err)
        
        completed = '%.1f %%'%((i+1.)/total_data_len*100.)
        prints(completed,prev_completed)
        prev_completed = completed
    for i,test in enumerate(tests):
        (A,phi,offset),(A_err,phi_err,offset_err) = _six_petals_fitter(pol_angles,test)
        while phi > expected_amax + 360/symmetry*0.9:
            phi -= 360/symmetry
        while phi < expected_amax - 360/symmetry*0.9:
            phi += 360/symmetry
        a_max_t.append(phi)
        a_max_t_err.append(phi_err)
        
        completed = '%.1f %%'%((controls_len+i+1.)/total_data_len*100.)
        prints(completed,prev_completed)
        prev_completed = completed
    AMAX_C_file = open(os.path.join(main_path,'%s_AMAX_C.csv'%sample),'wb')
    AMAX_C_writer = csv.writer(AMAX_C_file)
    AMAX_C_writer.writerow(a_max_c)
    AMAX_C_writer.writerow(a_max_c_err)
    AMAX_C_file.close()
    AMAX_T_file = open(os.path.join(main_path,'%s_AMAX_T.csv'%sample),'wb')
    AMAX_T_writer = csv.writer(AMAX_T_file)
    AMAX_T_writer.writerow(a_max_t)
    AMAX_T_writer.writerow(a_max_t_err)
    AMAX_T_file.close()
    _fig = plt.figure(sample)
    fig1 = _fig.add_subplot(111)
    fig3=fig1.twinx()
    fig3.plot(delays_ps,sc_powers,label='SC rel. power',color='C3')
    fig3.set_ylabel('SC rel. power, au',color='C3')
    fig2=fig1.twiny()
    fig1.errorbar(delays_ps,a_max_c,yerr=a_max_c_err,label='control',color='C0',capsize=2,fmt='o')
    fig1.errorbar(delays_ps,a_max_t,yerr=a_max_t_err,label='test',color='C1',capsize=2,fmt='o')
    fig1.cla()
    fig1.set_xlabel('Delay, ps')
    fig1.grid(axis=u'both')
    fig1.set_ylabel(r'$\alpha_{max}$, deg')
    fig2.errorbar(delays,a_max_c,yerr=a_max_c_err,label='control',color='C0',capsize=2,fmt='o')
    fig2.errorbar(delays,a_max_t,yerr=a_max_t_err,label='test',color='C1',capsize=2,fmt='o')
    fig2.set_xlabel('Delay/2, um')
    fig2.set_title(sample)
    fig2.legend(loc='best')
    _fig.tight_layout()
    _fig.savefig(os.path.join(main_path,'%s.png'%sample))
    
def real_time_anal_pSHG_VP2(sample,timesleep=11,mdeg=True):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\pSHG_VP2',sample)
    npy_files = list(filter(lambda name: name.endswith('.npy'),os.listdir(main_path)))
    spec_files = list(filter(lambda name: 'spectrum' in name,npy_files))
    spec_files = list(filter(lambda name: 'bck_gnd' not in name, spec_files))
    spec_c_files = list(filter(lambda name: is_control_spec(name),spec_files))
    delays = set(np.array(list(map(lambda name:get_delay(name),spec_c_files))))
    delays = sorted(delays)
    old_delays = set(delays)
    for delay in delays:
        anal_pSHG_VP2_sigle_delay(sample,delay,extend=False,mdeg=mdeg)
        plt.pause(1e-6)
        curr_delay = delay
    while os.path.getsize(os.path.join(main_path,'%s_powermeter.csv'%sample)) == 0:
        npy_files = list(filter(lambda name: name.endswith('.npy'),os.listdir(main_path)))
        spec_files = list(filter(lambda name: 'spectrum' in name,npy_files))
        spec_files = list(filter(lambda name: 'bck_gnd' not in name, spec_files))
        spec_c_files = list(filter(lambda name: is_control_spec(name),spec_files))
        delays = set(np.array(list(map(lambda name:get_delay(name),spec_c_files))))
        curr_delay_lst = list(delays - old_delays)
        if len(curr_delay_lst) == 0:
            plt.cla()
        else:
            curr_delay = curr_delay_lst[0]
        try:
            anal_pSHG_VP2_sigle_delay(sample,curr_delay,extend=False,mdeg=mdeg)
            old_delays = delays
        except ValueError:
            pass
        plt.pause(timesleep)
        
        
    
    
def smooth_spec(full_spec):
    s = np.convolve(full_spec[35:105],[.25,.5,0.25])
    return s[5:-5]

def fit_spec(full_spec):
    s = smooth_spec(full_spec)
    x = smooth_spec(hero_pixel_wavelengths)
    A, mu, sig, offset = _gauss_fitter(x,s)
    return A, mu, sig, offset

    
def gauss(x, A ,mu, sigma, offset):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2)) + offset

def _gauss_fitter(x_data, y_data):
    A = np.max(y_data)
    mu = x_data[y_data.argmax()]
    sigma = 3
    offset = np.min(y_data)
    
    p0 = (A, mu, sigma, offset)
    popt, pcov = sp.optimize.curve_fit(gauss, x_data, y_data, p0)
        
    return popt

def six_petals(ang,A,phi,offset):
    return A*np.square(np.cos(3*(ang-phi)/180.*np.pi))+offset

def six_petals_2(ang,C,S,offset):
    return C*np.cos(6*ang/180.*np.pi) + S*np.sin(6*ang/180.*np.pi) + offset

def _six_petals_fitter(angs,intens):
    A = np.max(intens)
    phi = angs[intens.argmax()]
    offset = np.min(intens)
    
    p0 = (A, phi, offset)
    popt, pcov = sp.optimize.curve_fit(six_petals, angs, intens, p0)
    perr = np.sqrt(np.diag(pcov))
    print popt
    return popt, perr

def _six_petals_fitter_2(angs,intens):
    C = np.max(intens)
    S = C
    offset = np.min(intens)
    p0 = (C, S, offset)
    popt, pcov = sp.optimize.curve_fit(six_petals_2, angs, intens, p0)
    perr = np.sqrt(np.diag(pcov))
    print popt
    return popt, perr

def sec_to_hhmmss(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dhr %2dmin %2ds" % (h, m, s)

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

def get_pol_angle(spec_file_name,mdeg):
    if mdeg:
        return 2*float(spec_file_name.split('spectrum')[1].split('.npy')[0])/1000.
    else:
        return 2*float(spec_file_name.split('spectrum')[1].split('.npy')[0])
    
def get_delay(spec_file_name):
    return int(spec_file_name.split('_d')[-1].split('spectrum')[0])

def is_control_spec(spec_file_name):
    return 'CONT' in spec_file_name

def multiple_petals_n7(ang,A0,A1,A2,A3,A4,A5,A6,A7,phi1,phi2,phi3,phi4,phi5,phi6,phi7):
    ans = A0 + A1*(np.cos(1*(ang-phi1)/180.*np.pi)) + A2*(np.cos(2*(ang-phi2)/180.*np.pi)) + A3*(np.cos(3*(ang-phi3)/180.*np.pi)) + A4*(np.cos(4*(ang-phi4)/180.*np.pi)) + A5*(np.cos(5*(ang-phi5)/180.*np.pi)) + A6*(np.cos(6*(ang-phi6)/180.*np.pi)) + A7*(np.cos(7*(ang-phi7)/180.*np.pi))
    return ans

def multiple_petals_n7_less6(ang,A0,A1,A2,A3,A4,A5,A7,phi1,phi2,phi3,phi4,phi5,phi7):
    ans = A0 + A1*(np.cos(1*(ang-phi1)/180.*np.pi)) + A2*(np.cos(2*(ang-phi2)/180.*np.pi)) + A3*(np.cos(3*(ang-phi3)/180.*np.pi)) + A4*(np.cos(4*(ang-phi4)/180.*np.pi)) + A5*(np.cos(5*(ang-phi5)/180.*np.pi)) + A7*(np.cos(7*(ang-phi7)/180.*np.pi))
    return ans

def multiple_petals_n7_fitter(angs,intens):
    As = [np.max(intens)]*8
    phis = [angs[intens.argmax()]]*7
    
    p0 = (As + phis)
    popt, pcov = sp.optimize.curve_fit(multiple_petals_n7, angs, intens, p0)
    return popt

def correcting_petals(angs,intens):
    A0,A1,A2,A3,A4,A5,A6,A7,phi1,phi2,phi3,phi4,phi5,phi6,phi7 = multiple_petals_n7_fitter(angs,intens)
    correction = multiple_petals_n7_less6(angs,A0,A1,A2,A3,A4,A5,A7,phi1,phi2,phi3,phi4,phi5,phi7)
    corrected_intens = np.array(intens)/correction
    return corrected_intens

def batch_anal(sample,gammas=[0,5,10,15,20,25,30,35,40,45],posi=1):
    amax = []
    amax_err = []
    pump_max = []
    for i,gamma in enumerate(gammas):
        angs780,intens780=anal_polarized_SHG('%s_%i'%(sample,gamma),True,gamma=posi*gamma,correction=False,anal_390=False)
        plt.close()
        angs,intens=anal_polarized_SHG('%s_%i'%(sample,gamma),False,gamma=posi*gamma,correction=True,anal_390=True)
        ((a,b,c),(d,e,f)) = _six_petals_fitter(angs,intens)
        if a < 0:
            b += 30
        if i ==0:
            pass
        else:
            while b < amax[i-1] - 50:
                b += 60
            while b > amax[i-1] + 50:
                b -= 60
        amax.append(b)
        amax_err.append(e)
        
        pump_max.append(max(intens780))
    amax = np.array(amax)
    amax_err = np.array(amax_err)
    pump_max = np.array(pump_max)
#    plt.close('all')
    gammas = np.array(gammas)
    gammas = np.square(np.cos(gammas/180.*np.pi))
    gammas = (1-gammas)*2
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.errorbar(gammas,np.tan((amax-amax[0])/180.*np.pi),yerr=amax_err/180.*np.pi/np.square(np.cos((amax-amax[0])/180.*np.pi)),capsize=2,color='C0')
    ax2=ax1.twinx()
    ax2.plot(gammas,pump_max,color='C1')
    ax1.set_xlabel('2*(1-cos($\gamma$)$^2$), |I($\sigma^-$)-I($\sigma^+$)|')
#    ax1.set_ylabel(r'$\Delta \alpha _{max}$, deg',color='C0')
    ax1.set_ylabel(r'tan($\Delta \alpha _{max}$)',color='C0')
    ax2.set_ylabel('Pump max, au',color='C1')
    ax1.axhline(0,color='k')
    return amax-amax[0],amax_err,pump_max