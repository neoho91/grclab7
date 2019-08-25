# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 17:47:00 2019

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
from pygame import mixer
mixer.init()

from laser_control import *
from yokogawa_waveform import *

def scan_pulse_duration_vs_toptica_prism(sample=None,prism_poss=np.arange(26000,33000.01,1000)):
    if sample == None:
        curr_time_str = create_date_str()
    else:
        curr_time_str = sample
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_pulse_duration_vs_toptica_prism',curr_time_str)
    os.mkdir(main_path)
    
    def prepare_measurement():
        global data
        data = []
        yoko_start()
        yoko_ave()
        timestamps = yoko_get_time_stamp()
        np.save(os.path.join(main_path,'timestamps'),timestamps)
    
    def take_data():
        yoko_ave()
        plt.pause(10)
        yoko_stop()
        plt.pause(0.1)
        datum = yoko_get_waveform('3')
        data.append(copy.copy(datum))
        yoko_start()
    
    def save_data():
        try:
            np.save(os.path.join(main_path,'data'),np.array(data))
        except:
            plt.pause(1)
            save_data()
    
    prev = ''
    n = 0
    total_len = len(prism_poss)
    prepare_measurement()
    for prism_pos in prism_poss:
        laser_prism(prism_pos,verbose=False)
        plt.pause(1)
        take_data()
        save_data()
        
        n += 1
        completed = u'prism pos at %i (%.2f percent)'%(prism_pos,100.0*(float(n)/total_len))
        prints(completed,prev)
        prev = completed
        