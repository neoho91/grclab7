#RemoteCheck_visa_Ether.py

import visa
import Tkinter as tk
from datetime import datetime

hostosa = '172.18.117.96'


rm = visa.ResourceManager()
rm.list_resources() 
osa = rm.open_resource('TCPIP::' + hostosa + '::INSTR')
print 'Target: '+ hostosa
print ''


#def teste():
#    device.write('SSI'+ chr(13) + chr(10))
#    a=device.ask('*OPC'+ chr(13) + chr(10))
#    b=device.ask('DMA'+ chr(13) + chr(10))


#Function


#CLS clears the status byte register
#osa.write('*CLS')

#Identification
#osa.ask('*IDN?')


#osa.timeout=3000
#osa.write("SSI");
#osa.ask("*OPC?");

#This command specifies the response data numeric format and queries
#the trace data sampling points, which is displayed on the screen
#ret = osa.ask("DMA?");
    
    
##This command sets and queries the center wavelength. 
#CNT <numeric_value>
#CNT?
#    
#    
#This command performs the measurement automatically.
#Bit 0 of the end event status register is set to 1 when measurement ends.
#This command queries the automatic measurement status.    
#    
#AUT
#
#The command sets the level scale to the linear and sets the Linear Level
#value.
#The command queries the Linear Level value.
#
#LLV <numeric_value> [MW|NW|PW|UW|W|PCT]
#LLV?
 

#This command sets the level scale to Log and scale division (dB/div)
#This command queries the Log scale.   
#LOG <numeric_value>
#LOG?

#This command sets the resolution.
#This command queries the set resolution.
#RES 0.03|0.05|0.07|0.1|0.2|0.5|1.0
#RES?

#This sets and queries the sweep width (nm).
#SPN <numeric_value>
#SPN?

#This command starts the repeat sweeping.
#SRT

#This command starts the single sweeping.
#When sweeping is completed, bit 1 (at sweeping end) of the end event
#status register (ESR2) is set to 1.
#SSI
#
#osa.write('SPC')
#SPC
#
#This command queries the measurement mode
#MOD?

 