# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 16:43:41 2016

@author: Neo
"""

import visa
import time

rm=visa.ResourceManager()
slider3 = rm.open_resource(u'ASRL16::INSTR')
def _ask_slider_3(q):
#    slider.clear()
    ans = slider3.ask(q)
    if '001F' in ans:
        return 0
    return 1
    
def block_laser_3():
    return _ask_slider_3(u'0fw')
    
def unblock_laser_3():
    return _ask_slider_3(u'0bw')

print 'Slider shutter 3 online.'