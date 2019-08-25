# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 16:43:41 2016

@author: TOPTICA
"""

import visa
import time

rm=visa.ResourceManager()
slider = rm.open_resource(u'ASRL9::INSTR')
def _ask_slider(q):
#    slider.clear()
    ans = slider.ask(q)
    if '001F' in ans:
        return 0
    return 1
    
def insert_DM():
    return _ask_slider(u'0fw')
    
def remove_DM():
    return _ask_slider(u'0bw')

print 'Dichroic mirror slider online.'