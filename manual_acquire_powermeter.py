# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:35:03 2017

@author: Jenny
"""

import os
import sys
sys.path.append(r'D:\Nonlinear_setup\Python_codes')
import numpy as np
import matplotlib.pyplot as plt

from powermeter_analog import *
from pygame import mixer
mixer.init()

#def play_sound(sound):
#    mixer.music.load(sound)
#    mixer.music.play()

#def play_done():
#    play_sound(r'D:/pthon_codes/done.mp3')

pma_wl(1560)
    
def acquire_pma():
    global data
    data = []
    for i in range(37):
        p=pma_power()*1000
        data.append(p)
        print('go to %i'%((i+1)*10))
        print(data)
#        play_done()
        raw_input()
    for datum in data:
        print(datum)
    return data
        
        