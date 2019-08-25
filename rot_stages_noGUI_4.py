# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:37:18 2017

@author: Neo
"""
import time
import sys
import os
sys.path.append(os.path.abspath('D:\WMP_setup\Python_codes\PyAPT-master'))

from PyAPT import APTMotor

#filter, rot4
try:
    rot4 = APTMotor(83847409,HWTYPE=31)
    rot4.setVelocityParameters(0,25,25)
    
    def move_rot4(ang=None):
        if ang == None:
            return rot4.getPos()
        else:
            rot4.mAbs(ang)  
            return move_rot4(ang=None)
    
    def home_rot4():
        rot4.go_home()

    print 'rot4 online.'
except:
    print('rot4 not connected')