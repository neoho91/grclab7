# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:48:48 2017

@author: Giulio
"""
import time
import copy
#import TL_slider
import numpy as np
import matplotlib.pyplot as plt
from ThorlabsPM100 import ThorlabsPM100
import sys
import os
import threading

sys.path.append(os.path.abspath(r"D:\WMP_setup\Python_codes"))

#import avaspec
#import send_sms
from auto_delayline import *
from powermeter_analog import *

#TRANSLATION STAGE 1 - blade



step=np.linspace(19.5,9.5,101)
pma_wl(1550)
for n in range (10):
    
    blade_pos=np.array([], float)
    powermeter_read=np.array([], float)
    output=open(r'D:\Nonlinear_setup\Experimental_data\beam_area_ekspla\stage1_blade__vs_power%d.txt' % (n), 'w')

    move_gdl(20)
    time.sleep(1)

    for i in step:
        move_gdl(i)
        blade_pos=np.append(blade_pos, i)
        time.sleep(2)
        powermeter_read=np.append(powermeter_read, pma_power())
        

    for j in range(0, len(step)):
        output.write('%g %g\n' % (step[j], powermeter_read[j]))
    
    output.close()

