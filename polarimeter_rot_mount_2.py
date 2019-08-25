# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:56:41 2019

@author: Neo
"""
import visa
import time
import numpy as np
import scipy as sp
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(r"D:\Nonlinear_setup\Python_codes"))

rm=visa.ResourceManager()
pol_rot_mount = rm.open_resource(u'ASRL21::INSTR')
pol_rot_mount.clear()
pol_rot_mount.timeout=5000
_deg_conv_pol = 360/143360.
pol_rot_mount_offset = 17.4

def _ask_rot_pol(q):
    _ans = pol_rot_mount.ask(q)
    try:
        return _ans[3:-2]
    except:
        if '001F' in _ans:
            return 0
        return 1

def _conv_to_deg_pol(h):
    ans = int(h,16)
    if ans > 143360:
        ans = -4294967296+ans
    return ans*_deg_conv_pol

def _conv_to_hex_pol(d):
    while d >= 360:
        d -= 360
    while d < -360:
        d += 360
    ans = d/_deg_conv_pol
    if d < 0:
        ans=0XFFFFFFFF+ans+2
    ans = '%08X'%ans
    return ans

def home_rot_pol():
    ans = _ask_rot_pol(u'0ho0')
    return _conv_to_deg_pol(ans)

def get_pola_mount():
    pos = _ask_rot_pol(u'0gp')
    try:
        return _conv_to_deg_pol(pos)
    except:
        time.sleep(1e-3)
        return get_pola_mount()

def get_pola_QWP_ang():
    return get_pola_mount()+pol_rot_mount_offset

def set_pola_QWP_ang(ang):
    ans = move_pola_mount(ang-pol_rot_mount_offset)+pol_rot_mount_offset
    return ans

def move_pola_mount(pos):
    try:
        pos = _ask_rot_pol(u'0ma%s'%_conv_to_hex_pol(pos))
    except visa.VisaIOError:
        time.sleep(1)
        pos = _ask_rot_pol(u'0ma%s'%_conv_to_hex_pol(pos))
    finally:
        return _conv_to_deg_pol(pos)

pol_rot_mount_info = _ask_rot_pol(u'0in')
print 'Polarimeter rotational mount (%s) online.'%(pol_rot_mount_info)