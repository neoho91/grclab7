# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 00:10:30 2017

@author: Neo
"""

import time
from win32api import keybd_event

def Press_enter():
    key = 13
    keybd_event(key, 0, 2, 0)
    time.sleep(0.05)
    keybd_event(key, 0, 1, 0)
    
def keep_pressing_enter():
    while True:
        Press_enter()
        time.sleep(5)