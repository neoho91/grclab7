# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:24:53 2019

@author: Millie
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

from fit_gaussian import *

def anal_scan_pulse_duration_vs_toptica_prism(sample,plot=False,timestamps_factor=28.7):
    """
    timestamps_factor = 28.7 (10 Hz), 14.35 (5 Hz), 7.18 (2.5 Hz)
    """
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_pulse_duration_vs_toptica_prism',sample)
    timestamps = np.load(os.path.join(main_path,'timestamps.npy'))*1e6*timestamps_factor
    data = np.load(os.path.join(main_path,'data.npy'))
    
    global fitted_data
    fitted_data = []
    for datum in data:
        fitted_data.append(gaussian_fitter(timestamps,datum,plot=plot)[0])
    fitted_data = np.array(fitted_data)