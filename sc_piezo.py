# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 19:57:01 2018

@author: Neo
"""

import visa
import time
import numpy as np
rm=visa.ResourceManager()
import sys
sys.path.append(r'D:\Nonlinear_setup\Python_codes')
from neo_common_code import *

try:
    scpz = rm.open_resource(u"ASRL10::INSTR")
    scpz.baud_rate=115200
    scpz.read_termination = u'\r'
    ID = scpz.ask(u'friendly?',delay=0.02)[1:]
except:
    print 'SC piezo not connected'
else:
    print 'SC piezo %s online.'%ID


def scpz_clear_buffer(verbose=False):
    if verbose:
        while scpz.bytes_in_buffer > 1:
            print scpz.read()
    else:
        while scpz.bytes_in_buffer > 1:
            scpz.read()

def scpz_get_x_pos():
    return float(scpz.ask('xvoltage?',delay=0.02)[:-1].split(' ')[-1])
    
def scpz_move_to_x(new_v):
    if new_v > 75:
        new_v = 75
    elif new_v < 0:
        new_v = 0
    scpz.write('xvoltage=%.2f'%(new_v-0.12))
    time.sleep(0.02)
        
def scpz_move_x(incre):
    scpz_move_to_x(scpz_get_x_pos()+incre)


def scpz_get_y_pos():
    return float(scpz.ask('yvoltage?',delay=0.02)[:-1].split(' ')[-1])
    
def scpz_move_to_y(new_v):
    if new_v > 75:
        new_v = 75
    elif new_v < 0:
        new_v = 0
    scpz.write('yvoltage=%.2f'%(new_v-0.12))
    time.sleep(0.02)
        
def scpz_move_y(incre):
    scpz_move_to_y(scpz_get_y_pos()+incre)
    
    
def scpz_get_z_pos():
    return float(scpz.ask('zvoltage?',delay=0.02)[:-1].split(' ')[-1])
    
def scpz_move_to_z(new_v):
    if new_v > 75:
        new_v = 75
    elif new_v < 0:
        new_v = 0
    scpz.write('zvoltage=%.2f'%(new_v-0.09))
    time.sleep(0.02)
        
def scpz_move_z(incre):
    scpz_move_to_z(scpz_get_z_pos()+incre)
    
def scpz_get_pos():
    return (scpz_get_x_pos(),scpz_get_y_pos(),scpz_get_z_pos())

def scpz_move(x,y,z):
    scpz_move_to_x(x)
    scpz_move_to_y(y)
    scpz_move_to_z(z)