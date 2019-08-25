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

#unused, rot1
try:
    rot1 = APTMotor(83845997,HWTYPE=31)
    rot1.setVelocityParameters(0,25,25)
    
    def move_rot1(ang=None):
        if ang == None:
            return rot1.getPos()
        else:
            rot1.mAbs(ang)  
            return move_rot1(ang=None)
    
    def home_rot1():
        rot1.go_home()
    
    print 'rot1 online.'
except:
    print('rot1 not connected')


#Rotation stage2  Analyzer
try:
    rot2 = APTMotor(83846250,HWTYPE=31)
    rot2.setVelocityParameters(0,25,25)

    def move_rot2(ang=None):
        if ang == None:
            return rot2.getPos()
        else:
            rot2.mAbs(ang)
            return move_rot2(ang=None)
        
    def home_rot2():
        rot2.go_home()
    
    print 'rot2 online.'
except:
    print('rot2 not connected')


#Rotation stage3  SC power control
try:
    rot3 = APTMotor(83846230,HWTYPE=31)
    rot3.setVelocityParameters(0,25,25)

    def move_rot3(ang=None):
        if ang == None:
            return rot3.getPos()
        else:
            rot3.mAbs(ang)
            return move_rot3(ang=None)
    
    def home_rot3():
        rot3.go_home()
    
    print 'rot3 online.'
except:
    print('rot3 not connected')
