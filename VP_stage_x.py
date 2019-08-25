# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 19:16:55 2015

@author: TOPTICA
"""

import visa
import time
import sys

def prints(s,prev_s=''):
    if prev_s == '':
        sys.stdout.write(s)
        sys.stdout.flush()
    else:
        last_len = len(prev_s)
        sys.stdout.write('\b' * last_len)
        sys.stdout.write(' ' * last_len)
        sys.stdout.write('\b' * last_len)
        sys.stdout.write(s)
        sys.stdout.flush()
        
def initialise_vpx():
    global vpx_state_dic,vpx
    vpx_state_dic = {'0A': 'NOT REFERENCED from RESET', '0B': 'NOT REFERENCED from HOMING','0C': 'NOT REFERENCED from CONFIGURATION',
     '0D': 'NOT REFERENCED from DISABLE','0E': 'NOT REFERENCED from READY','0F': 'NOT REFERENCED from MOVING','10': 'NOT REFERENCED - NO PARAMETERS IN MEMORY',
    '14': 'CONFIGURATION','1E': 'HOMING','28': 'MOVING','32': 'READY from HOMING','33': 'READY from MOVING','34': 'READY from DISABLE',
    '36': 'READY T from READY','37': 'READY T from TRACKING','38': 'READY T from DISABLE T','3C': 'DISABLE from READY','3D': 'DISABLE from MOVING',
    '3E': 'DISABLE from TRACKING','3F': 'DISABLE from READY T','46': 'TRACKING from READY T','47': 'TRACKING from TRACKING','35':'READY from TRACKING (?)'} 
    rm = visa.ResourceManager()
    vpx=rm.open_resource("ASRL18::INSTR")
    vpx.baud_rate=921600
    if 'NOT REFERENCED' in get_vpx_state():
        raw=vpx.ask('RS##',delay=0.1)
        address=raw.split('RSAddress #')[1].split('\r')[0]
        vpx.write('RS')
        time.sleep(0.4)
        prints("vpx (address #"+str(address)+"): I'm coming home, I'm coming home, tell the world that I'm coming home...\n")
        vpx.write('1OR')
        while get_vpx_state() == 'HOMING':
            time.sleep(1e-1)
    print 'VP stage x '+get_vpx_state()+'.'
    return vpx
   
def get_vpx_state():
    raw=vpx.ask('1TS')
    new_raw = raw.split('\r')[0].split('1TS')[1]
    err=new_raw[:4]
    state=new_raw[4:]
    return vpx_state_dic[state]
    
def get_vpx_pos():
    try:
        raw = vpx.ask('1TP')
    except:
        time.sleep(1)
        raw = vpx.ask('1TP')
    finally:
        new_raw=raw.split('1TP')[1].split('\r')[0]
        return float(new_raw)
    
def move_vpx_abs(new_pos):
    if new_pos < 0 or new_pos>12.01:
        raise Exception("New position exceeds vpx's limits (0 < pos < 12 mm): "+str(new_pos))
    state = get_vpx_state()
#    if 'READY' not in state:
#        prints('STATUS WARNING: vpx may not move to '+str(new_pos)+' mm.\n')
    vpx.write('1PA'+str(new_pos))
    start=time.time()
    while get_vpx_state() != 'READY from MOVING':
        time.sleep(1e-2)
        if time.time()-start > 60:
#            prints('TIME WARMING: vpx may not move to '+str(new_pos)+' mm.\n')
            break
    return get_vpx_pos()

def home_vpx():
    raw=vpx.ask('RS##',delay=0.1)
    address=raw.split('RSAddress #')[1].split('\r')[0]
    vpx.write('RS')
    time.sleep(0.4)
    prints("vpx (address #"+str(address)+"): I'm coming home, I'm coming home, tell the world that I'm coming home...\n")
    vpx.write('1OR')
    while get_vpx_state() == 'HOMING':
        time.sleep(1e-1)
    print 'VP stage x '+get_vpx_state()+'.'

def close_vpx():
    vpx.close()
    
vpx=initialise_vpx()