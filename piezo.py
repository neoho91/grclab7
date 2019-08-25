# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 08:40:02 2015

@author: TOPTICA
"""

import visa

# Initialization through visa
rm = visa.ResourceManager()
rm.list_resources()
piezo=rm.open_resource("ASRL5::INSTR")

piezo.baud_rate=19200
piezo.data_bits=8
piezo.read_termination = u'\r'

# The following commands set the piezo channels to close loop (um)
#cloop<channel>,<0|1>
piezo.write(u'cloop,0,1')
piezo.write(u'cloop,1,1')
piezo.write(u'cloop,2,1')

# Channel 0 is x
#Channel 1 is y, the focusing channel
#Channel 2 is z

# Sets the cpntrol for every channel

piezo.write(u'setk,0,1')
piezo.write(u'setk,1,1')
piezo.write(u'setk,2,1')

#The following command sets all channels to 0
#setall<channel0>,<channel1>,<channel2>
# the command set<channel><value> sets the position of the actuactor
# In closed loop in um and open in voltage
pos_aux=piezo.ask('measure').split(',')[1:]
pos=list(map(lambda x: float(x),pos_aux))

print 'Piezo online.'

def get_pos():
    pos_aux=piezo.ask('measure').split(',')[1:]
    pos=list(map(lambda x: float(x),pos_aux))
    return pos

def get_pos_raw():
    pos=piezo.ask('measure')
    return pos
        
def move_to_x(new_x):
    if new_x > 80 or new_x < -0.10:
        raise Exception('Piezo exceeded range: x = ' + str(new_x)+' um')
    else:
        try:
            piezo.ask('ERR?')
            piezo.write(u'set,0,'+str(new_x))
        except UnicodeDecodeError:
            piezo.ask('ERR?')
            piezo.write(u'set,0,'+str(new_x))
        
def move_x(incre):
    curr_x = get_pos()[0]
    new_x = curr_x + incre
    move_to_x(new_x)

def move_to_y(new_y):
    if new_y > 80 or new_y < -0.10:
        raise Exception('Piezo exceeded range: y = ' + str(new_y)+' um')
    else:
        try:
            piezo.ask('ERR?')
            piezo.write(u'set,1,'+str(new_y))
        except UnicodeDecodeError:
            piezo.ask('ERR?')
            piezo.write(u'set,1,'+str(new_y))
        
def move_y(incre):
    curr_y = get_pos()[1]
    new_y = curr_y + incre
    move_to_y(new_y)
    
def move_to_z(new_z):
    if new_z > 80 or new_z < -0.10:
        raise Exception('Piezo exceeded range: z = ' + str(new_z)+' um')
    else:
        try:
            piezo.ask('ERR?')
            piezo.write(u'set,2,'+str(new_z))
        except UnicodeDecodeError:
            piezo.ask('ERR?')
            piezo.write(u'set,2,'+str(new_z))
        
def move_z(incre):
    curr_z = get_pos()[2]
    new_z = curr_z + incre
    move_to_z(new_z)

def remote_piezo():
    piezo.write(u'setk,0,1')
    piezo.write(u'setk,1,1')
    piezo.write(u'setk,2,1')
    print 'Piezo in remote mode.'

def manual_piezo():
    piezo.write(u'setk,0,0')
    piezo.write(u'setk,1,0')
    piezo.write(u'setk,2,0')
    print 'Piezo in manual mode'