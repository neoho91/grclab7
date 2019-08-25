# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 16:43:41 2016

@author: Neo
"""

import visa
import time

rm=visa.ResourceManager()
slider2 = rm.open_resource(u'ASRL15::INSTR')
def _ask_slider_2(q):
#    slider.clear()
    ans = slider2.ask(q)
    if '001F' in ans:
        return 0
    return 1
    
def block_laser_2():
    return _ask_slider_2(u'0fw')
    
def unblock_laser_2():
    return _ask_slider_2(u'0bw')

print 'Slider shutter 2 online.'