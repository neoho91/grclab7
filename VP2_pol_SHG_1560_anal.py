#%%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import sys
import csv
import time
import scipy as sp
import copy
sys.path.append(r'D:/WMP_setup/Python_codes')
from neo_common_code import *
hero_pixel_wavelengths = np.load(r'D:/WMP_setup/Python_codes/hero_pixel_wavelengths.npy')
try:
    reVP1test780coeff = np.load(r'D:\Nonlinear_setup\Python_codes\reVP1test780coeff.npy')
    reVP1test390coeff = np.load(r'D:\Nonlinear_setup\Python_codes\reVP1test390coeff.npy')
except:
    print('reVP1test calibration file(s) not found.')
    reVP1test780coeff = np.array([1])
    reVP1test390coeff = np.array([1])

#%%
#--------------------------------------------------#
#Code for analysis of the results

def VP2_pol_SHG_1560_anal(sample,pm_as_ref=True):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\VP2_pol_SHG',sample)
    npy_files = list(filter(lambda name: name.endswith('.npy'),os.listdir(main_path)))
    spec_files = list(filter(lambda name: 'SPECS_a' in name,npy_files))
    spec_bg_files = list(filter(lambda name: 'SPECS_BG_a' in name,npy_files))
    powers = np.load(os.path.join(main_path,'powers.npy'))*1e6 #uW
    global alphas,data780,datapm,datasc,max_alpha,d
    
    def get_a_from_name(name):
        return float(name.split('a')[1].split('.npy')[0])/100
           
    mean780s = []
    alphas = list(map(lambda name: get_a_from_name(name),spec_files))
    alphas.sort()
    alphas = np.array(alphas)
    max_alpha=[]

    for alpha in alphas: 
        curr_spec_file = list(filter(lambda name: alpha == get_a_from_name(name),spec_files))
        curr_spec_file = curr_spec_file[0]
        if len(spec_bg_files) > 0:
            curr_spec_bg_file = list(filter(lambda name: alpha == get_a_from_name(name),spec_bg_files))
            curr_spec_bg_file = curr_spec_bg_file[0]
            specs_bg = np.load(os.path.join(main_path,curr_spec_bg_file))
        else:
            specs_bg = np.load(os.path.join(main_path,'BCKGND_SPEC.npy'))
        specs = np.load(os.path.join(main_path,curr_spec_file)) - specs_bg
        specs780 = specs[:,550:625]
        specs780_bg = specs[:,625:700]
#        sums780 = np.sum(specs780,axis=1) - np.sum(specs780_bg,axis=1)
        sums780 = np.sum(specs780,axis=1) - np.median(specs780_bg,axis=1)*(700-625)
        mean780 = np.average(sums780)

        mean780s.append(mean780)
    
    mean780s = np.array(mean780s)
    alphas_spec,mean780s,alphas_spec_out,mean780s_out = remove_outlier(alphas,mean780s)
    alphas_power,powers,alphas_power_out,powers_out = remove_outlier(alphas,powers)
    if len(mean780s_out) > 0:
        print('Outlier(s) for spec sum removed:\n    alpha = %s\n    spec sum = %s'%(alphas_spec_out,mean780s_out))
    if len(powers_out) > 0:
        print('Outlier(s) for powers removed:\n    alpha = %s\n    powers = %s'%(alphas_power_out,powers_out))
    if pm_as_ref:
        mean780s = mean780s/np.square(powers)
         
    _fig = plt.figure(r'%s'%(sample))
    fig = _fig.add_subplot(121,projection='polar')
    fig.plot(alphas_spec[:len(mean780s)]/180.*np.pi,mean780s,'o',label='SHG')
    fig.set_title(r'SHG @ 780 nm')
    fig.set_xticks(np.arange(0,360,60)/180.*np.pi)
    
    fig2 = _fig.add_subplot(122,projection='polar')
    fig2.plot(alphas_power[:len(powers)]/180.*np.pi,powers,'o',label='pump, uW')
    if pm_as_ref:
        fig2.set_title('Pump as ref @ 1560 nm')
    else:
        fig2.set_title('Pump @ 1560 nm')
    fig2.set_xticks(np.arange(0,360,60)/180.*np.pi)
    
    if pm_as_ref:
        try:
            sc_powers = np.load(os.path.join(main_path,'powers_sc.npy'))*1e6 #uW
            datasc = np.array(sc_powers)
        except:
            pass
   
    _fig.suptitle(sample)
    
    data780 = np.array(mean780s)
    datapm = np.array(powers)
    alphas = np.array(alphas)
    
    try:
        fitted_X = np.linspace(0,360,300)
        fit_results_SHG = multiple_petals_n7_fitter(alphas_spec,mean780s)
        fig.plot(fitted_X/180.*np.pi,multiple_petals_n7(fitted_X,*fit_results_SHG[0]),label='fit n7 SHG')
        
        corrected_SHGs = correcting_petals(alphas_spec,mean780s)
        fit_results_SHG_6p = _six_petals_fitter(alphas_spec,corrected_SHGs)
        fig.plot(alphas_spec[:len(mean780s)]/180.*np.pi,corrected_SHGs,'v',label='6p SHG')
        fig.plot(fitted_X/180.*np.pi,six_petals(fitted_X,*fit_results_SHG_6p[0]),label='fit 6p SHG')
        SHG_max_alpha = fit_results_SHG_6p[0][1]
        SHG_max_alpha_e = fit_results_SHG_6p[1][1]
        t = 'A = %.2f $\pm$ %.2f\n$\\alpha_{max}$ = %.2f $\pm$ %.2f deg\nR = %.4f $\pm$ %.4f'%tuple(np.array(tuple(zip(fit_results_SHG_6p[0],fit_results_SHG_6p[1]))).flatten())
        fig.set_title('SHG @ 780 nm\n%s'%(t))
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01),ncol=2)
        _fig.tight_layout(rect=(0,0,1,0.95))
        
        fit_results_pump = multiple_petals_n7_fitter(alphas_power,powers)
        pump_max_alpha = fit_results_pump[0][9]
        pump_max_alpha_e = fit_results_pump[1][9]
        fig2.plot(alphas_power[:len(mean780s)]/180.*np.pi,multiple_petals_n7(alphas_power,*fit_results_pump[0]),label='fit n7 pump, uW')
        t = 'A0 + A2 = %.2f $\pm$ %.2f uW\n$\\alpha_{max}$ = %.2f $\pm$ %.2f deg'%(fit_results_pump[0][0]+fit_results_pump[0][2],np.sqrt(fit_results_pump[1][0]**2+fit_results_pump[1][2]**2),pump_max_alpha,pump_max_alpha_e)
        fig2.set_title('Pump @ 1560 nm\n%s'%(t))
        fig2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01))
        
        
        _fig2 = plt.figure('Fitting summary of %s'%sample)
        _fig2.suptitle('Fitting summary of %s'%sample)
        fig3 = _fig2.add_subplot(121)
        fig3.bar(range(8),fit_results_SHG[0][:8],yerr=fit_results_SHG[1][:8],tick_label=list(range(8)),capsize=5,color='C0')
        fig3.axhline(0,color='k')
        fig3.set_ylabel('Amplitude, au',color='C0')
        fig3.set_title('SHG @ 780 nm')
        fig3.set_xlabel('Harmonics')
        fig3_2 = fig3.twinx()
        fig3_2.errorbar(range(1,8),fit_results_SHG[0][8:],ls='None',marker='o',yerr=fit_results_SHG[1][8:],capsize=5,C='C1',ecolor='k')
        fig3_2.grid()
        fig3_2.set_ylabel('Angle, deg',color='C1')
        fig3_2.format_coord = make_format(fig3_2, fig3)
        
        fig4 = _fig2.add_subplot(122)
        fig4.bar(range(8),fit_results_pump[0][:8],yerr=fit_results_pump[1][:8],tick_label=list(range(8)),capsize=5,color='C0')
        fig4.axhline(0,color='k')
        fig4.set_ylabel('Amplitude, uW',color='C0')
        fig4.set_title('Pump @ 1560 nm')
        fig4.set_xlabel('Harmonics')
        fig4_2 = fig4.twinx()
        fig4_2.errorbar(range(1,8),fit_results_pump[0][8:],ls='None',marker='o',yerr=fit_results_pump[1][8:],capsize=5,C='C1',ecolor='k')
        fig4_2.grid()
        fig4_2.set_ylabel('Angle, deg',color='C1')
        fig4_2.format_coord = make_format(fig4_2, fig4)
        _fig2.tight_layout(rect=(0,0,1,0.95))
        

        
        return SHG_max_alpha, SHG_max_alpha_e, pump_max_alpha, pump_max_alpha_e, fit_results_SHG, fit_results_pump, fit_results_SHG_6p
    
        d = SHG_max_alpha  
        max_alpha.append(d)
    
    except Exception as e:
        print(e)
        print('Failed to fit %s'%sample)
#%%

def VP2_mapping_SHG_anal(sample,min_wl=757.9,max_wl=814.4,min_wl_bg=814.4,max_wl_bg=870.5,show_me_all_specs_to_choose_wl_range=False,normalize_SHG_by_pm_with_power=0):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\VP2_mapping_SHG',sample)
    npy_files = list(filter(lambda name: name.endswith('.npy'),os.listdir(main_path)))
    spec_files = list(filter(lambda name: 'SPECS' in name,npy_files))
    BCKGND_SPEC = np.load(os.path.join(main_path,'BCKGND_SPEC.npy'))
    all_Xs = np.load(os.path.join(main_path,'Xs.npy'))
    all_Zs = np.load(os.path.join(main_path,'Zs.npy'))
    resol=all_Xs[1]-all_Xs[0]
    global SHG_data, pm_data, all_specs_for_alpha
    
    if show_me_all_specs_to_choose_wl_range:
        plt.ion()
        all_specs = []
        for spec_file in spec_files:
            all_specs.append(np.load(os.path.join(main_path,spec_file))[0])
        sums = list(np.sum(all_specs,axis=1))
        fig = plt.figure('%s choose wavelength ranges'%sample)
        ax=fig.add_subplot(111)
        ax.set_xlabel('Wavelength, nm')
        ax.plot(hero_pixel_wavelengths,all_specs[sums.index(np.max(sums))])
        fig.canvas.mpl_connect('key_press_event', onclick)
        
        ax.set_title('select min wavelength to sum\nmouse over and press enter to choose')
        plt.pause(1e-6)
        while not plt.waitforbuttonpress():
            pass
        min_wl = _wl_chosen[0]
        ax.axvline(min_wl)
        
        ax.set_title('min = %.2f nm. Now select max wavelength to sum\nmouse over and press enter to choose'%min_wl)
        plt.pause(1e-6)
        while not plt.waitforbuttonpress():
            pass
        max_wl = _wl_chosen[0]
        ax.axvline(max_wl)
        
        ax.set_title('min = %.2f nm, max = %.2f nm. Now go to console'%(min_wl,max_wl))
        bg_min_plot = ax.axvline(min_wl_bg)
        bg_max_plot = ax.axvline(max_wl_bg)
        plt.pause(1e-6)
        ans = raw_input('Use default background (814 - 870 nm)? Enter "n" to choose. ')
        if ans == 'n':
            ax.set_title('select min background wavelength to sum\nmouse over and press enter to choose')
            plt.pause(1e-6)
            while not plt.waitforbuttonpress():
                pass
            min_wl_bg = _wl_chosen[0]
            bg_min_plot.set_xdata(min_wl_bg)
            
            ax.set_title('min = %.2f nm. Now select max background wavelength to sum\nmouse over and press enter to choose'%min_wl_bg)
            plt.pause(1e-6)
            while not plt.waitforbuttonpress():
                pass
            max_wl_bg = _wl_chosen[0]
            bg_max_plot.set_xdata(max_wl_bg)
            ax.set_title('min = %.2f nm, max = %.2f nm. Processing...'%(min_wl_bg,max_wl_bg))
            plt.pause(1e-6)
    
    print('min, max wavelength = %.2f, %.2f nm'%(min_wl,max_wl))
    print('min, max background wavelength = %.2f, %.2f nm'%(min_wl_bg,max_wl_bg))
    print('Processing... please wait...')
    
    min_idx=get_hero_idx(min_wl)
    max_idx=get_hero_idx(max_wl)
    min_idx_bg=get_hero_idx(min_wl_bg)
    max_idx_bg=get_hero_idx(max_wl_bg)
    
    def get_a_from_name(name):
        return float(name.split('a')[1].split('_')[0])/100
    def get_x_from_name(name):
        return float(name.split('x')[1].split('_')[0])/100
    def get_z_from_name(name):
        return float(name.split('z')[1].split('.npy')[0])/100
           
    alphas = list(map(lambda name: get_a_from_name(name),spec_files))
    alphas = list(set(alphas))
    alphas.sort()
    alphas = np.array(alphas)
    
    SHG_data = []
    pm_data = []
    for alpha in alphas: 
        curr_spec_files = list(filter(lambda name: alpha == get_a_from_name(name),spec_files))
        Zs = list(map(lambda name: get_z_from_name(name),curr_spec_files))
        Zs = list(set(Zs))
        Zs.sort()
        Xs = list(map(lambda name: get_x_from_name(name),curr_spec_files))
        Xs = list(set(Xs))
        Xs.sort()
        curr_data = np.zeros((len(all_Zs),len(all_Xs)))
        
        powers = np.load(os.path.join(main_path,'powers_a%i.npy'%(alpha*100)))
        curr_power_data = np.zeros((len(all_Zs),len(all_Xs)))
        
        all_specs_for_alpha_path = os.path.join(main_path,'all_specs_for_alpha_a%i.npy'%(alpha*100))
        if os.path.isfile(all_specs_for_alpha_path):
            all_specs_for_alpha = list(np.load(all_specs_for_alpha_path))
        else:
            all_specs_path_for_alpha = [os.path.join(main_path,r'SPECS_a%i_x%i_z%i.npy'%(alpha*100,X*100,Z*100)) for Z in Zs for X in Xs]
            all_specs_for_alpha = []
            for p in all_specs_path_for_alpha:
                if not os.path.isfile(p):
                    break
                all_specs_for_alpha.append(np.load(p))
            np.save(all_specs_for_alpha_path,np.array(all_specs_for_alpha))
        _i=0
        for i_z,Z in enumerate(Zs):
            for i_x,X in enumerate(Xs):
#                curr_spec_files_Z = list(filter(lambda name: Z == get_z_from_name(name),curr_spec_files))
#                curr_spec_file = list(filter(lambda name: X == get_x_from_name(name),curr_spec_files_Z))
#                if len(curr_spec_file) == 0:
#                    break
#                else:
#                    curr_spec_file = curr_spec_file[0]
                if _i < len(all_specs_for_alpha):
                    specs = all_specs_for_alpha[_i]
                else:
                    curr_spec_file = r'SPECS_a%i_x%i_z%i.npy'%(alpha*100,X*100,Z*100)
                    if not os.path.isfile(os.path.join(main_path,curr_spec_file)):
                        break
                    specs = np.load(os.path.join(main_path,curr_spec_file))# - BCKGND_SPEC
                    all_specs_for_alpha.append(specs)
                _i+=1
                
                try:
                    specs780 = specs[:,min_idx:max_idx]
                    specs780_bg = specs[:,min_idx_bg:max_idx_bg]
    #                sums780 = np.sum(specs780,axis=1) - np.sum(specs780_bg,axis=1)
                    sums780 = np.sum(specs780,axis=1) - np.median(specs780_bg,axis=1)*np.abs(max_idx-min_idx)
                    mean780 = np.average(sums780)
                    curr_data[i_z][i_x]=mean780
                    curr_power_data[i_z][i_x]=powers[i_z][i_x]*1e3#uW
                except IndexError:
                    pass
        
        curr_data[curr_data == 0] = np.median(curr_data[curr_data > 0])
        curr_power_data[curr_power_data == 0] = np.median(curr_power_data[curr_power_data > 0])
        
        curr_data = curr_data/(curr_power_data/100.)**normalize_SHG_by_pm_with_power
        SHG_data.append(curr_data)
        pm_data.append(curr_power_data)
        np.save(all_specs_for_alpha_path,np.array(all_specs_for_alpha))

        def format_coord(x, y):
            x += 1
            y += 1
            x = round(x*resol/resol)*resol-resol + all_Xs[0]
            y = round(y*resol/resol)*resol-resol + all_Zs[0]
            return 'x=%1.1f, z=%1.1f'%(x, y)
        xlb=list(np.arange(all_Xs[0],all_Xs[-1]+0.01,10,dtype=int))
        ylb=list(np.arange(all_Zs[0],all_Zs[-1]+0.01,10,dtype=int))

        _fig = plt.figure(r'%s: $\alpha$ = %.1f deg, sum %.2f to %.2f nm'%(sample,alpha,min_wl,max_wl))
        plt.clf()
        ax = _fig.add_subplot(121)
        cax = ax.imshow(curr_data, cmap='Reds_r',interpolation='none')
        _fig.colorbar(cax)
        ax.set_xticks(np.arange(0,len(all_Xs),10/resol))
        ax.set_yticks(np.arange(0,len(all_Zs),10/resol))
        ax.set_xticklabels(xlb)
        ax.set_yticklabels(ylb)
        ax.set_xlabel(r'$\mu$m')
        ax.set_ylabel(r'$\mu$m')
        ax.set_title('SHG, max/min = %.2f/%.2f'%(np.nanmax(curr_data),np.nanmin(curr_data)))
        ax.format_coord = format_coord
        plt.pause(1e-6)
        
        ax2 = _fig.add_subplot(122)
        cax2 = ax2.imshow(curr_power_data, cmap='Greys_r',interpolation='none')
        _fig.colorbar(cax2)
        ax2.set_xticks(np.arange(0,len(all_Xs),10/resol))
        ax2.set_yticks(np.arange(0,len(all_Zs),10/resol))
        ax2.set_xticklabels(xlb)
        ax2.set_yticklabels(ylb)
        ax2.set_xlabel(r'$\mu$m')
        ax2.set_ylabel(r'$\mu$m')
        ax2.set_title('Pump, max/min = %.2f/%.2f'%(np.nanmax(curr_power_data),np.nanmin(curr_power_data)))
        ax2.format_coord = format_coord
        plt.pause(1e-6)
        
        _fig.suptitle(r'%s: $\alpha$ = %.1f deg, sum %.2f to %.2f nm'%(sample,alpha,min_wl,max_wl))
        _fig.tight_layout(rect=(0,0,1,0.95))
        plt.pause(1e-6)
        print('Done.')
#%%
def VP2_pol_SHG_rotA_anal(sample,pm_as_ref=True,verbose=False):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\VP2_pol_SHG_rotA',sample)
    npy_files = list(filter(lambda name: name.endswith('.npy'),os.listdir(main_path)))
    spec_files = list(filter(lambda name: 'SPECS' in name,npy_files))
    BCKGND_SPEC = np.load(os.path.join(main_path,'BCKGND_SPEC.npy'))
    global alphas,betas,data780,datapm,datasc,max_alpha,d
    
    def get_a_from_name(name):
        try:
            return float(name.split('a')[1].split('.npy')[0])/100
        except:
            return float(name.split('a')[1].split('_')[0])/100
    def get_b_from_name(name):
        try:
            return float(name.split('b')[1].split('.npy')[0])/100
        except:
            return float(name.split('b')[1].split('_')[0])/100
    
    alphas = list(map(lambda name: get_a_from_name(name),spec_files))
    alphas = list(set(alphas))
    alphas.sort()
    alphas = np.array(alphas)
           
    betas = list(map(lambda name: get_b_from_name(name),spec_files))
    betas = list(set(betas))
    betas.sort()
    betas = np.array(betas)
    
    for alpha in alphas:
        print alpha
        mean780s = []
        max_alpha=[]
        spec_files_of_alpha = list(filter(lambda name: alpha == get_a_from_name(name),spec_files))
        powers = np.load(os.path.join(main_path,'powers_a%i.npy'%(alpha*100)))*1e6 #uW
        for beta in betas: 
            curr_spec_file = list(filter(lambda name: beta == get_b_from_name(name),spec_files_of_alpha))
            curr_spec_file = curr_spec_file[0]
            specs = np.load(os.path.join(main_path,curr_spec_file))# - BCKGND_SPEC
            specs780 = specs[:,550:625]
            specs780_bg = specs[:,625:700]
    #        sums780 = np.sum(specs780,axis=1) - np.sum(specs780_bg,axis=1)
            sums780 = np.sum(specs780,axis=1) - np.median(specs780_bg,axis=1)*(700-625)
            mean780 = np.average(sums780)
    
            mean780s.append(mean780)
        
        mean780s = np.array(mean780s)
        if pm_as_ref:
            mean780s = mean780s/np.square(powers)
             
        _fig = plt.figure(r'%s @ a = %ideg'%(sample,alpha))
        _fig.clf()
        fig = _fig.add_subplot(121,projection='polar')
        fig.plot(betas[:len(mean780s)]/180.*np.pi,mean780s,'o',label='SHG')
        fig.set_title(r'SHG @ 780 nm')
        fig.set_xticks(np.arange(0,360,60)/180.*np.pi)
        
        fig2 = _fig.add_subplot(122,projection='polar')
        fig2.plot(betas[:len(powers)]/180.*np.pi,powers,'o',label='pump, uW')
        if pm_as_ref:
            fig2.set_title('Pump as ref @ 1560 nm')
        else:
            fig2.set_title('Pump @ 1560 nm')
        fig2.set_xticks(np.arange(0,360,60)/180.*np.pi)
        
        if pm_as_ref:
            try:
                sc_powers = np.load(os.path.join(main_path,'powers_sc.npy'))*1e6 #uW
                datasc = np.array(sc_powers)
            except:
                pass
       
        _fig.suptitle(r'%s @ a = %ideg'%(sample,alpha))
        
        data780 = np.array(mean780s)
        datapm = np.array(powers)
        betas = np.array(betas)
        
        try:
            fit_results_SHG = multiple_petals_n7_fitter(betas,mean780s)
            fig.plot(betas[:len(mean780s)]/180.*np.pi,multiple_petals_n7(betas,*fit_results_SHG[0]),label='fit n7 SHG')
            SHG_max_alpha = fit_results_SHG[0][9]
            SHG_max_alpha_e = fit_results_SHG[1][9]
            t = 'A0 + A2 = %.2f $\pm$ %.2f cts\n$\\alpha_{max}$ = %.2f $\pm$ %.2f deg'%(fit_results_SHG[0][0]+fit_results_SHG[0][2],np.sqrt(fit_results_SHG[1][0]**2+fit_results_SHG[1][2]**2),SHG_max_alpha,SHG_max_alpha_e)
            fig.set_title('SHG @ 780 nm\n%s'%(t))
            fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01),ncol=2)
            _fig.tight_layout(rect=(0,0,1,0.95))
            
            fit_results_pump = multiple_petals_n7_fitter(betas,powers)
            pump_max_alpha = fit_results_pump[0][9]
            pump_max_alpha_e = fit_results_pump[1][9]
            fig2.plot(betas[:len(mean780s)]/180.*np.pi,multiple_petals_n7(betas,*fit_results_pump[0]),label='fit n7 pump, uW')
            t = 'A0 + A2 = %.2f $\pm$ %.2f uW\n$\\alpha_{max}$ = %.2f $\pm$ %.2f deg'%(fit_results_pump[0][0]+fit_results_pump[0][2],np.sqrt(fit_results_pump[1][0]**2+fit_results_pump[1][2]**2),pump_max_alpha,pump_max_alpha_e)
            fig2.set_title('Pump @ 1560 nm\n%s'%(t))
            fig2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01))
            plt.pause(0.1)
            
            _fig2 = plt.figure('Fitting summary of %s at a = %ideg'%(sample,alpha))
            _fig2.clf()
            _fig2.suptitle('Fitting summary of %s at a = %ideg'%(sample,alpha))
            fig3 = _fig2.add_subplot(121)
            fig3.bar(range(8),fit_results_SHG[0][:8],yerr=fit_results_SHG[1][:8],tick_label=list(range(8)),capsize=5,color='C0')
            fig3.axhline(0,color='k')
            fig3.set_ylabel('Amplitude, au',color='C0')
            fig3.set_title('SHG @ 780 nm')
            fig3.set_xlabel('Harmonics')
            fig3_2 = fig3.twinx()
            fig3_2.errorbar(range(1,8),fit_results_SHG[0][8:],ls='None',marker='o',yerr=fit_results_SHG[1][8:],capsize=5,C='C1',ecolor='k')
            fig3_2.grid()
            fig3_2.set_ylabel('Angle, deg',color='C1')
            
            fig4 = _fig2.add_subplot(122)
            fig4.bar(range(8),fit_results_pump[0][:8],yerr=fit_results_pump[1][:8],tick_label=list(range(8)),capsize=5,color='C0')
            fig4.axhline(0,color='k')
            fig4.set_ylabel('Amplitude, uW',color='C0')
            fig4.set_title('Pump @ 1560 nm')
            fig4.set_xlabel('Harmonics')
            fig4_2 = fig4.twinx()
            fig4_2.errorbar(range(1,8),fit_results_pump[0][8:],ls='None',marker='o',yerr=fit_results_pump[1][8:],capsize=5,C='C1',ecolor='k')
            fig4_2.grid()
            fig4_2.set_ylabel('Angle, deg',color='C1')
            _fig2.tight_layout(rect=(0,0,1,0.95))
            
    
            
#            return SHG_max_alpha, SHG_max_alpha_e, pump_max_alpha, pump_max_alpha_e, fit_results_SHG, fit_results_pump
        
            d = SHG_max_alpha  
            max_alpha.append(d)
        
        except Exception as e:
            if verbose:
                print(e)
                print('Failed to fit %s'%sample)

#%%
def anal_scan_delayline(npy_file,min_wl=500,max_wl=515):
    global poss, spec_sums, lockins
    data = np.load(os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline',npy_file+'.npy'))
    poss = list(map(lambda x: x[0][0],data))
    lockins = np.array(map(lambda x: x[0][1],data))*1e3
    specs = list(map(lambda x: x[1],data))
    min_idx = get_nearest_idx_from_list(min_wl,hero_pixel_wavelengths)
    max_idx = get_nearest_idx_from_list(max_wl,hero_pixel_wavelengths)
    spec_sums = list(map(lambda x: np.sum(x[min_idx:max_idx]),specs))
    min_y = np.min(specs[0][min_idx-5:max_idx+5])
    max_y = np.max(specs[0][min_idx-5:max_idx+5])
    _figall = plt.figure(npy_file+' all specs')
    figall = _figall.add_subplot(111)
    for spec in specs:
        figall.plot(hero_pixel_wavelengths,spec)
    figall.set_xlim(min_wl-5,max_wl+5)
    figall.set_ylim(min_y-10,max_y)
    figall.set_xlabel('Wavelength, nm')
    _fig = plt.figure(npy_file)
    fig = _fig.add_subplot(121)
    fig2 = _fig.add_subplot(122)
    line, = fig.plot(poss,spec_sums,'o')
    line2, = fig2.plot(poss,lockins,'o')
    _fig.suptitle(npy_file)
    fig.set_title('spec sum of %i to %i nm'%(min_wl,max_wl))
    fig2.set_title('Lock-in amplitude')
    fig.set_xlabel('fdl pos, mm')
    fig.set_ylabel('spec sum, au')
    fig2.set_xlabel('fdl pos, mm')
    fig2.set_ylabel('lock-in amp, mV')
    poss = np.array(poss)
    lockins = np.array(lockins)
    spec_sums = np.array(spec_sums)
    return (_fig,fig,line,figall,fig2,line2)
#%%
def real_time_anal_scan_delayline(npy_file,min_wl=500,max_wl=515,refresh_delay=1):
    min_idx = get_nearest_idx_from_list(min_wl,hero_pixel_wavelengths)
    max_idx = get_nearest_idx_from_list(max_wl,hero_pixel_wavelengths)
    _fig, fig, line, figall, fig2, line2 = anal_scan_delayline(npy_file,min_wl=min_wl,max_wl=max_wl)
    prev_data_len = 0
    num = 0
    global spec_sums
    while True:
        try:
            data = np.load(os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline',npy_file+'.npy'))
        except:
            plt.pause(0.1)
            data = np.load(os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline',npy_file+'.npy'))
        
        curr_data_len = len(data)
        if curr_data_len == prev_data_len:
            if num > 10:
                return
            plt.pause(refresh_delay)
            num += 1
            continue
        else:
            num = 0
        poss = list(map(lambda x: x[0][0],data))
        lockins = np.array(map(lambda x: x[0][1],data))*1e3
        specs = list(map(lambda x: x[1],data))
        spec_sums = list(map(lambda x: np.sum(x[min_idx:max_idx]),specs))
        
        line.set_xdata(poss)
        line.set_ydata(spec_sums)
        fig.relim(True)
        fig.autoscale_view(True,True,True)
        
        line2.set_xdata(poss)
        line2.set_ydata(lockins)
        fig2.relim(True)
        fig2.autoscale_view(True,True,True)
        
        while curr_data_len > len(figall.lines):
            figall.plot(hero_pixel_wavelengths,specs[len(figall.lines)])
        
        plt.draw_all()
        prev_data_len = curr_data_len
        plt.pause(refresh_delay)

#%%
def anal_scan_delayline_with_bg(npy_file,min_wl=770,max_wl=777.5,fdl_pos_start='all',fdl_pos_end='all',lockin_as_pump_ref_order=2,plot_scan_fdl=True):
    global poss, spec_sums, lockins
    try:
        data = np.load(os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline_with_bg',npy_file+'.npy'))
    except:
        supsample = npy_file.split('_a')[0]
        data = np.load(os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_multiple_alpha_with_bg',supsample,npy_file+'.npy'))
    info_line = data[0]
    for l in info_line:
        print l
    data = data[1:]
    poss = np.array(map(lambda x: x[0][0],data))
    if fdl_pos_start == 'all':
        min_pos_idx = 0
    else:
        min_pos_idx = get_nearest_idx_from_list(fdl_pos_start,poss)
    if fdl_pos_end == 'all':
        max_pos_idx = -1
    else:
        max_pos_idx = get_nearest_idx_from_list(fdl_pos_end,poss)
    data = data[min_pos_idx:max_pos_idx]
    poss = poss[min_pos_idx:max_pos_idx]
    lockins = np.nan_to_num(np.array(map(lambda x: x[0][1],data))*1e3)
    lockins_bg = np.nan_to_num(np.array(map(lambda x: x[0][2],data))*1e3)
    lockins = lockins - lockins_bg
    old_lockins = lockins
    poss_li,lockins,poss_li_out,lockins_out = remove_outlier(poss,lockins,5000)
    if len(lockins_out) > 0:
        print('Outlier(s) for lock-in removed:\n    poss = %s\n    lockins = %s'%(poss_li_out,lockins_out))
    specs = np.array(map(lambda x: x[1],data))
    specs_bg = np.array(map(lambda x: x[2],data))
    min_idx = get_nearest_idx_from_list(min_wl,hero_pixel_wavelengths)
    max_idx = get_nearest_idx_from_list(max_wl,hero_pixel_wavelengths)
    spec_sums = np.array(map(lambda x: np.sum(x[min_idx:max_idx]),specs))
    spec_sums_bg = np.array(map(lambda x: np.sum(x[min_idx:max_idx]),specs_bg))
    spec_sums = spec_sums - spec_sums_bg
    if lockin_as_pump_ref_order:
        spec_sums = spec_sums/(old_lockins**lockin_as_pump_ref_order)
    min_y = np.min((specs[0]-specs_bg[0])[min_idx-5:max_idx+5])
    max_y = np.max((specs[0]-specs_bg[0])[min_idx-5:max_idx+5])
    plt.ion()
    if plot_scan_fdl:
        _fig = plt.figure(npy_file,figsize=(15,9))
        figall = _fig.add_subplot(211)
        for i,spec in enumerate(specs):
            figall.plot(hero_pixel_wavelengths,spec-specs_bg[i])
        figall.set_xlim(min_wl-5,max_wl+5)
        figall.set_ylim(min_y-10,max_y+10)
        figall.set_ylabel('Intensity - bg, au')
        figall.set_xlabel('Wavelength, nm')
        figall.set_title('All specs')
        fig = _fig.add_subplot(212)
        fig2 = fig.twinx()
        line, = fig.plot(poss,spec_sums,'o',markersize=1,color='C0')
        line2, = fig2.plot(poss_li,lockins,'o',markersize=1,color='C1')
        _fig.suptitle(npy_file)
        fig.set_title('spec sum of %.1f to %.1f nm and lock-in amplitude'%(min_wl,max_wl))
        fig.set_xlabel('fdl pos, mm')
        fig.set_ylabel('spec sum - bg, au',color='C0')
        fig2.set_ylabel('lock-in amp - bg, mV',color='C1')
        fig2.format_coord = make_format(fig2, fig)
        plt.pause(.01)
        plt.get_current_fig_manager().window.showMaximized()
        plt.tight_layout(rect=[0, 0.00, 1, 0.95])
        plt.pause(.01)
        
        def on_xlims_change(a):
            fdl_pos_start,fdl_pos_end = a.get_xlim()
            min_pos_idx = get_nearest_idx_from_list(fdl_pos_start,poss)
            max_pos_idx = get_nearest_idx_from_list(fdl_pos_end,poss)
            figall.lines=[]
            plt.draw()
            for i in range(min_pos_idx,max_pos_idx):
                figall.plot(hero_pixel_wavelengths,specs[i]-specs_bg[i])
    
        fig.callbacks.connect('xlim_changed',on_xlims_change)
    
        return (_fig,fig,line,figall,fig2,line2)
#%%
def real_time_anal_scan_delayline_with_bg(npy_file,min_wl=770,max_wl=777.5,refresh_delay=10,lockin_as_pump_ref_order=2):
    min_idx = get_nearest_idx_from_list(min_wl,hero_pixel_wavelengths)
    max_idx = get_nearest_idx_from_list(max_wl,hero_pixel_wavelengths)
    _fig, fig, line, figall, fig2, line2 = anal_scan_delayline_with_bg(npy_file,min_wl=min_wl,max_wl=max_wl,lockin_as_pump_ref_order=lockin_as_pump_ref_order)
    prev_data_len = 0
    num = 0
    _data_path = os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline_with_bg',npy_file+'.npy')
    if not os.path.isfile(_data_path):
        supsample = npy_file.split('_a')[0]
        _data_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_multiple_alpha_with_bg',supsample,npy_file+'.npy')
    global spec_sums
    while True:
        try:
            data = np.load(_data_path)
            info_line = data[0]
            data = data[1:]
        except:
            plt.pause(0.1)
            data = np.load(_data_path)
            info_line = data[0]
            data = data[1:]
        
        curr_data_len = len(data)
        if curr_data_len == prev_data_len:
            if num > 10:
                return
            plt.pause(refresh_delay)
            num += 1
            continue
        else:
            num = 0
        poss = np.array(map(lambda x: x[0][0],data))
        lockins = np.nan_to_num(np.array(map(lambda x: x[0][1],data))*1e3)
        lockins_bg = np.nan_to_num(np.array(map(lambda x: x[0][2],data))*1e3)
        lockins = lockins - lockins_bg
        old_lockins = lockins
        poss_li,lockins,poss_li_out,lockins_out = remove_outlier(poss,lockins,5000)
        specs = np.array(map(lambda x: x[1],data))
        specs_bg = np.array(map(lambda x: x[2],data))
        spec_sums = np.array(map(lambda x: np.sum(x[min_idx:max_idx]),specs))
        spec_sums_bg = np.array(map(lambda x: np.sum(x[min_idx:max_idx]),specs_bg))
        spec_sums = spec_sums - spec_sums_bg
        if lockin_as_pump_ref_order:
            spec_sums = spec_sums/(old_lockins**lockin_as_pump_ref_order)
        
        line.set_xdata(poss)
        line.set_ydata(spec_sums)
        fig.relim(True)
        fig.autoscale_view(True,True,True)
        
        line2.set_xdata(poss_li)
        line2.set_ydata(lockins)
        fig2.relim(True)
        fig2.autoscale_view(True,True,True)
        
        while curr_data_len > len(figall.lines):
            figall.plot(hero_pixel_wavelengths,specs[len(figall.lines)]-specs_bg[len(figall.lines)])
        
        plt.draw_all()
        prev_data_len = curr_data_len
        plt.pause(refresh_delay)
#%%

def anal_scan_delayline_multiple_alpha_with_bg(sample,min_wl=770,max_wl=777.5,fdl_pos_start='all',fdl_pos_end='all',lockin_as_pump_ref_order=2,plot_scan_fdl=False):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_multiple_alpha_with_bg',sample)
    npy_files = list(filter(lambda name: name.endswith('.npy'),os.listdir(main_path)))
    data_files = list(filter(lambda name: sample in name,npy_files))
    def get_a_from_name(name):
        try:
            return float(name.split('_a')[1].split('.npy')[0])/100
        except:
            return float(name.split('_a')[1].split('_')[0])/100
    data_files.sort(key=get_a_from_name)
    alphas = list(map(get_a_from_name,data_files))
    all_spec_sums=[]
    for data_file in data_files:
        anal_scan_delayline_with_bg(data_file[:-4],min_wl=min_wl,max_wl=max_wl,fdl_pos_start=fdl_pos_start,fdl_pos_end=fdl_pos_end,lockin_as_pump_ref_order=lockin_as_pump_ref_order,plot_scan_fdl=plot_scan_fdl)
        all_spec_sums.append(copy.copy(spec_sums))
    all_spec_sums = np.array(all_spec_sums)
    all_spec_sums = all_spec_sums.transpose()
    
    print('\nFitting data...')
    As,A_es,phis,phi_es,Rs,R_es=[],[],[],[],[],[]
    for all_spec_sum in all_spec_sums:
        (A, phi, R),(A_e, phi_e, R_e) = _six_petals_fitter(alphas,all_spec_sum)
        As.append(A)
        A_es.append(A_e)
        phis.append(phi)
        phi_es.append(phi_e)
        Rs.append(R)
        R_es.append(R_e)
    prints('Done.')
    
    phis = put_angles_to_same_range(phis)
    _fig=plt.figure('Fitted %s'%sample)
    fig=_fig.add_subplot(111)
    fig2=fig.twinx()
    fig.errorbar(poss,As,yerr=A_es,ls='None',marker='o',capsize=5,C='C0',ecolor='royalblue',markersize=3)
    fig.set_xlabel('delay line position, mm')
    fig.set_ylabel('Amplitude, au',color='C0')
    fig2.errorbar(poss,phis,yerr=phi_es,ls='None',marker='o',capsize=5,C='C1',ecolor='darksalmon',markersize=3)
    fig2.set_ylabel('$\\beta_{max}$, deg',color='C1')
    fig2.format_coord = make_format(fig2, fig)
    plt.pause(.01)
    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()

#%%
#_______________________________________________________________________
def multiple_petals_n7(ang,A0,A1,A2,A3,A4,A5,A6,A7,phi1,phi2,phi3,phi4,phi5,phi6,phi7):
    ans = A0 + A1*(np.cos(1*(ang-phi1)/180.*np.pi)) + A2*(np.cos(2*(ang-phi2)/180.*np.pi)) + A3*(np.cos(3*(ang-phi3)/180.*np.pi)) + A4*(np.cos(4*(ang-phi4)/180.*np.pi)) + A5*(np.cos(5*(ang-phi5)/180.*np.pi)) + A6*(np.cos(6*(ang-phi6)/180.*np.pi)) + A7*(np.cos(7*(ang-phi7)/180.*np.pi))
    return ans
def multiple_petals_n7_fitter(angs,intens):
    As = [np.max(intens)]*8
    phis = [angs[intens.argmax()]]*7
    p0 = (As + phis)
    popt, pcov = sp.optimize.curve_fit(multiple_petals_n7, angs, intens, p0)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def six_petals(ang,A,phi,R):
    return A*(np.square(np.cos(3*(ang-phi)/180.*np.pi)) + R*np.square(np.sin(3*(ang-phi)/180.*np.pi)))
def _six_petals_fitter(angs,intens):
    A = np.max(intens)
    phi = angs[intens.argmax()]
    R = np.min(intens)
    p0 = (A, phi, R)
    popt, pcov = sp.optimize.curve_fit(six_petals, angs, intens, p0)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def multiple_petals_n7_less6(ang,A0,A1,A2,A3,A4,A5,A7,phi1,phi2,phi3,phi4,phi5,phi7):
    ans = A0 + A1*(np.cos(1*(ang-phi1)/180.*np.pi)) + A2*(np.cos(2*(ang-phi2)/180.*np.pi)) + A3*(np.cos(3*(ang-phi3)/180.*np.pi)) + A4*(np.cos(4*(ang-phi4)/180.*np.pi)) + A5*(np.cos(5*(ang-phi5)/180.*np.pi)) + A7*(np.cos(7*(ang-phi7)/180.*np.pi))
    return ans
def correcting_petals(angs,intens):
    popt, perr = multiple_petals_n7_fitter(angs,intens)
    A0,A1,A2,A3,A4,A5,A6,A7,phi1,phi2,phi3,phi4,phi5,phi6,phi7 = popt
    correction = multiple_petals_n7_less6(angs,A0,A1,A2,A3,A4,A5,A7,phi1,phi2,phi3,phi4,phi5,phi7)
#    corrected_intens = np.array(intens)/correction*A6
    corrected_intens = np.array(intens)-correction + A0
    return corrected_intens

def put_angles_to_same_range(angs):
    if len(angs) < 2:
        return angs
    else:
        ans = []
        for i,ang in enumerate(angs):
            if i == 0:
                while ang < 60:
                    ang += 60
                while ang > 60:
                    ang -= 60
                ans.append(ang)
            else:
                while ang < ans[-1] - 60*0.95:
                    ang += 60
                while ang > ans[-1] + 60*0.95:
                    ang -= 60
                ans.append(ang)
        return ans

def get_hero_idx(wl):
    diff = list(np.abs(hero_pixel_wavelengths-wl))
    return diff.index(np.min(diff))

global _wl_chosen
_wl_chosen = [0]
def onclick(event):
    _wl_chosen[0] = event.xdata
    

        