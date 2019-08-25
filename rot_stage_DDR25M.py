# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:37:18 2017

@author: Neo
"""
import sys
import os
sys.path.append(os.path.abspath('D:\WMP_setup\Python_codes\PyAPT-master'))

from PyAPT import APTMotor

#analyzer, rot5
try:
    rot5 = APTMotor(28250988,HWTYPE=28)
    
    def move_rot5(ang=None):
        if ang == None:
            ans = rot5.getPos()*360.
            if ans > 64800:
                ans -= 129600
                ans %= -360
            else:
                ans %= 360
            return ans
        else:
            rot5.mAbs((ang/360.)%360) #/360 to convert to correct unit, %360 to keep in 1 rotation  
            return move_rot5(ang=None)
    
    def home_rot5():
        rot5.go_home()

    print(r'DDR25/M online.')
except:
    print(r'DDR25/M not connected')