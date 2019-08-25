# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 21:48:27 2017

@author: Neo
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import sys
import csv
sys.path.append(r'D:/WMP_setup/Python_codes')
hero_pixel_wavelengths = np.load(r'D:/WMP_setup/Python_codes/hero_pixel_wavelengths.npy')
try:
    reVP1test780coeff = np.load(r'D:\Nonlinear_setup\Python_codes\reVP1test780coeff.npy')
    reVP1test390coeff = np.load(r'D:\Nonlinear_setup\Python_codes\reVP1test390coeff.npy')
except:
    print('reVP1test calibration file(s) not found.')
    reVP1test780coeff = np.array([1])
    reVP1test390coeff = np.array([1])

def anal_THG_1560(sample):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\THG_1560',sample)
    npy_files = list(filter(lambda name: name.endswith('.npy'),os.listdir(main_path)))
    spec_files = list(filter(lambda name: 'SPECS' in name,npy_files))
    BCKGND_SPEC = np.load(os.path.join(main_path,'BCKGND_SPEC.npy'))
    global alphas,data516,data780,comments
    data516 = [] #2d table to be used in originPro
    data780 = []
    comments = [] #to be used in header of originPro
    
    def get_a_from_name(name):
        return float(name.split('a')[1].split('.npy')[0])/100
    
    
    
   
    alphas = list(map(lambda name: get_a_from_name(name),spec_files))
    alphas.sort()
    alphas = np.array(alphas)
    
    mean516s = []
    mean780s = []
    for alpha in alphas:
        curr_spec_file = list(filter(lambda name: alpha == get_a_from_name(name),spec_files))
        curr_spec_file = curr_spec_file[0]
        
        specs = np.load(os.path.join(main_path,curr_spec_file)) - BCKGND_SPEC
        specs516 = specs[:,190:290]
        specs516_bg = specs[:,125:160]
        specs780 = specs[:,540:640]
        specs780_bg = specs[:,625:700]
        sums516 = np.sum(specs516,axis=1) - np.sum(specs516_bg,axis=1)
        sums780 = np.sum(specs780,axis=1)- np.sum(specs780_bg,axis=1)
        mean516 = np.average(sums516)
        mean780 = np.average(sums780)
#            mean390_dev = np.std(sums390)
#            mean780_dev = np.std(sums780)
        mean516s.append(mean516)
        mean780s.append(mean780)
           
    mean516s = np.array(mean516s)#*reVP1test390coeff
    mean780s = np.array(mean780s)#*reVP1test780coeff
        
    _fig = plt.figure('%s SnS THG @ 516 nm'%(sample))
    fig = _fig.add_subplot(111,projection='polar')
    fig.plot(alphas[:len(mean516s)]/180.*np.pi*2,mean516s)
    fig.legend()
    fig.set_title('%s SnS THG @ 516 nm'%(sample))
    
#    _fig = plt.figure('%s 780'%(sample))
#    fig = _fig.add_subplot(111,projection='polar')
#    fig.plot(alphas[:len(mean390s)]/180.*np.pi*2,mean780s)
#    fig.legend()
#    fig.set_title('%s 780'%(sample))
    
    comments.append("SnS THG @ 520 nm")
    data516.append(mean516s)
#    comments.append("780")
#    data780.append(mean780s)
        
    data516 = np.array(data516)
    data780 = np.array(data780)
    alphas = np.array(alphas)
    comments = np.array(comments)
    
def anal_polarized_SHG_singlebeam2(sample):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\reVP1',sample)
    npy_files = list(filter(lambda name: name.endswith('.npy'),os.listdir(main_path)))
    spec_files = list(filter(lambda name: 'SPECS' in name,npy_files))
    BCKGND_SPEC = np.load(os.path.join(main_path,'BCKGND_SPEC.npy'))
    pm_files = list(filter(lambda name: 'PM' in name,npy_files))
    global betas,gammas,data390,data780,comments,pm_data
    data780 = [] #2d table to be used in originPro
    data390 = []
    pm_data = []
    comments = [] #to be used in header of originPro
    
    def get_b_from_name(name):
        return float(name.split('b')[1].split('.npy')[0])/100
    def get_g_from_name(name):
        return float(name.split('g')[1].split('_b')[0])/100
    def get_g_from_name_pm(name):
        return float(name.split('g')[1].split('.npy')[0])/100
    
    gammas = list(map(lambda name: get_g_from_name(name),spec_files))
    gammas = list(set(gammas))
    gammas.sort()
    
    for i,gamma in enumerate(gammas):
        curr_spec_files = list(filter(lambda name: gamma == get_g_from_name(name),spec_files))
        betas = list(map(lambda name: get_b_from_name(name),curr_spec_files))
        betas.sort()
        betas = np.array(betas)
        
        if pm_files !=[]:
            curr_pm_file = list(filter(lambda name: gamma == get_g_from_name_pm(name),pm_files))
            curr_pm=np.load(os.path.join(main_path,curr_pm_file[0]))
            pm_data.append(curr_pm)
        
        mean390s = []
        mean780s = []
        for beta in betas:
            curr_spec_file = list(filter(lambda name: beta == get_b_from_name(name),curr_spec_files))
            curr_spec_file = curr_spec_file[0]
            
            specs = np.load(os.path.join(main_path,curr_spec_file)) - BCKGND_SPEC
            specs390 = specs[:,55:90]
            specs390_bg = specs[:,125:160]
            specs780 = specs[:,550:625]
            specs780_bg = specs[:,625:700]
            sums390 = np.sum(specs390,axis=1) - np.sum(specs390_bg,axis=1)
            sums780 = np.sum(specs780,axis=1) - np.sum(specs780_bg,axis=1)
            mean390 = np.average(sums390)
            mean780 = np.average(sums780)
#            mean390_dev = np.std(sums390)
#            mean780_dev = np.std(sums780)
            mean390s.append(mean390)
            mean780s.append(mean780)
        
        mean390s = np.array(mean390s)*reVP1test390coeff[:len(mean390s)]
        mean780s = np.array(mean780s)*reVP1test780coeff[:len(mean780s)]
            
        _fig = plt.figure('%s'%(sample))
        fig390 = _fig.add_subplot(131,projection='polar')
        fig390.plot(betas[:len(mean390s)]/180.*np.pi,mean390s,label='g%i'%gamma)
        fig390.legend()
        fig390.set_title('%s 390'%(sample))
        
        fig780 = _fig.add_subplot(132,projection='polar')
        fig780.plot(betas[:len(mean780s)]/180.*np.pi,mean780s,label='g%i'%gamma)
        fig780.legend()
        fig780.set_title('%s 780'%(sample))
        
        if pm_files != []:
            figpm = _fig.add_subplot(133,projection='polar')
            figpm.plot(betas[:len(curr_pm)]/180.*np.pi,curr_pm,label='g%i'%gamma)
            figpm.legend()
            figpm.set_title('%s Powermeter'%(sample))
        
        comments.append(gamma)
        data390.append(mean390s)
        data780.append(mean780s)
        
    data390 = np.array(data390)
    data780 = np.array(data780)
    betas = np.array(betas)
    gammas = np.array(gammas)
    comments = np.array(comments)
    pm_data = np.array(pm_data)