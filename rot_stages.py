# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:37:18 2017

@author: Millie
"""

import wx
import wx.activex 
import wx.py
import wx.lib.activex
import comtypes.client as cc
from ctypes import byref, pointer, c_long, c_float, c_bool
#from textwrap import wrap
import ctypes
import ctypes.util
import ctypes.wintypes
import platform

cc.GetModule( ('{2A833923-9AA7-4C45-90AC-DA4F19DC24D1}', 1, 0) )
progID_motor = 'MGMOTOR.MGMotorCtrl.1'
import comtypes.gen.MG17MotorLib as APTMotorLib
channel1 = APTMotorLib.CHAN1_ID
channel2 = APTMotorLib.CHAN2_ID
break_type_switch = APTMotorLib.HWLIMSW_BREAKS

units_mm = APTMotorLib.UNITS_MM
units_deg=APTMotorLib.UNITS_DEG

home_rev = APTMotorLib.HOME_REV
homelimsw_rev = APTMotorLib.HOMELIMSW_REV_HW

#rotation_type_move=APTMotorLib.ROT_MOVE_POS
rotation_type_move=APTMotorLib.ROT_MOVE_SHORT
PosReportMode=APTMotorLib.ROT_POSDISP_360

class APTMotor( wx.lib.activex.ActiveXCtrl ):
    """The Motor class derives from wx.lib.activex.ActiveXCtrl, which
       is where all the heavy lifting with COM gets done."""
    
    def __init__( self, parent, HWSerialNum, id=wx.ID_ANY, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=0, name='Stepper Motor' ):
        wx.lib.activex.ActiveXCtrl.__init__(self, parent, progID_motor,
                                            id, pos, size, style, name)
        self.ctrl.HWSerialNum = HWSerialNum
        self.ctrl.StartCtrl()
        #self.ctrl.EnableHWChannel( channel1 )
        
        #""" Global variables:"""
        #self.StepSize = 0.05 # initial step size (mm)
        #self.PositionCh1 = 0.0
        #self.PositionCh2 = 0.0
    
    def __del__(self):
        self.ctrl.StopCtrl()
    

    
    def GetPosition( self, channel=channel1 ):
        position = c_float()
        self.ctrl.GetPosition(channel, byref(position))
        if channel==channel1: self.PositionCh1 = position.value
        return position.value
    
    def GetStageAxisInfo( self, channel=channel1 ):
        """Returns a tuple of:
            min position, max position, units, pitch, and direction"""
        min_position = c_float()
        max_position = c_float()
        units = c_long()
        pitch = c_float()
        direction = c_long()
        self.ctrl.GetStageAxisInfo(channel, byref(min_position),
                                          byref(max_position), byref(units),
                                          byref(pitch), byref(direction))
        return min_position.value, max_position.value, units.value, pitch.value, direction.value
    def GetStageAxisInfo_MaxPos( self, channel=channel1 ):
        """Get the maximum position of the stage that is accessible using
            the MoveAbsoluteEx or MoveRelativeEx commands, although you
            may be able to exceed it by Jogging. I think this is a
            user-settable quantity. For the small stepper we have,
            the max travel is like 18mm. (Or should be ~25mm?) """
        return self.ctrl.GetStageAxisInfo_MaxPos( channel )
    def GetStageAxisInfo_MinPos( self, channel=channel1 ):
        """Get the minimum position of the stage that is accessible using
            the MoveAbsoluteEx or MoveRelativeEx commands, although you
            may be able to exceed it by Jogging. I think this is a
            user-settable quantity. For the small stepper we have, if
            it's been "homed" then it sets 0 to be the minimum position."""
        return self.ctrl.GetStageAxisInfo_MinPos( channel )
    

    def GetStatusBits_Bits( self, channel=channel1 ):
        """ Returns the status bits. """
        return self.ctrl.GetStatusBits_Bits( channel )

    def GetRotStageModes( self, channel=channel1 ):
        MoveMode = c_long()
        PosReportMode = c_long()
        self.ctrl.GetRotStageModes(channel, byref(MoveMode),
                                          byref( PosReportMode))
        return MoveMode.value, PosReportMode.value
                                          
                                
    
    def MoveAbsoluteRot( self, position_ch1=0.0, position_ch2=0.0, MoveMode=rotation_type_move, channel=channel1, wait=True ):
        
        return self.ctrl.MoveAbsoluteRot( channel, position_ch1, position_ch2, MoveMode, wait )
        
    
    def MoveHome( self, channel=channel1,wait=True):
        return self.ctrl.MoveHome( channel, wait)

    def SetRotStageModes( self, MoveMode=rotation_type_move, PosMode=PosReportMode ): 

        return self.ctrl.SetRotStageModes( MoveMode, PosMode )

    def GetBLashDist(self,channel=channel1):
        pfBLashDist=c_float()
        
        self.ctrl.GetBLashDist(channel,byref(pfBLashDist))
        return pfBLashDist.value
        
    
    def SetBLashDist( self, channel=channel1, backlash=1.0001866817474365):
        """
        Sets the backlash distance in degrees.

        *channel*
           channel1 by default

        *backlash*
           distance in mm, 1.0001866817474365 by default
         """
        return self.ctrl.SetBLashDist( channel, backlash )
    
    
           
    def SetHomeParams( self, channel=channel1, direction=home_rev, switch=homelimsw_rev,
                       velocity=5.0, zero_offset=0.2 ):
        """
        Set the "home params". I forget what these actually mean.

        *channel*
            

        *direction*
          

        *switch*
           

        *velocity*
           

        *zero_offset*
           

        """
        return self.ctrl.SetHomeParams( channel, direction, switch, velocity, zero_offset )
        
    def SetStageAxisInfo(self,channel=channel1,MinPos=0,MaxPos=360,units=units_deg, fPitch=17.8700008392334,lDirSense=1):

        return self.ctrl.SetStageAxisInfo(channel,MinPos,MaxPos,units,fPitch,lDirSense)
            
    def SetVelParams(self,channel=channel1, fMinVel=0.0, fAccn=1.0, fMaxVel=20.0):

        return self.ctrl.SetVelParams(channel,fMinVel,fAccn,fMaxVel)



    def GetVelParams(self,channel=channel1):
      
        min_vel=c_float()
        max_vel=c_float()
        fAccn=c_float()
        self.ctrl.GetVelParams(channel,byref(min_vel),byref(fAccn),byref(max_vel))
        
        return min_vel.value,fAccn.value,max_vel.value


  
   

#Assigning the motors
class MyApp( wx.App ): 
    def __init__( self, redirect=False, filename=None, title='MG17MotorControl' ):
        wx.App.__init__( self, redirect, filename )
        self.frame = wx.Frame( None, wx.ID_ANY, title=title )
        self.panel = wx.Panel( self.frame, wx.ID_ANY )
         
#Rotation stage1  Polarizer
app1 = MyApp(title='Polarizer')
box1 = wx.BoxSizer( wx.VERTICAL )

polarizer = APTMotor( app1.panel, HWSerialNum=83845997, style=wx.SUNKEN_BORDER )
box1.Add( polarizer, proportion=1, flag=wx.EXPAND )

app1.panel.SetSizer( box1 )
app1.frame.Show()
app1.SetEvtHandlerEnabled(False)

polarizer.SetHomeParams()
polarizer.SetStageAxisInfo()
polarizer.SetVelParams(fAccn=25, fMaxVel=25)
polarizer.SetBLashDist()

def move_POL(ang=None):
    if ang == None:
        return polarizer.GetPosition()
    else:
        polarizer.MoveAbsoluteRot(ang)  
        return move_POL(ang=None)

def home_POL():
    move_POL(0.1)
    polarizer.MoveHome()

print 'Polarizer online.'


#Rotation stage2  Analyzer
app2 = MyApp(title='ANA')
box2 = wx.BoxSizer( wx.VERTICAL )

analyzer = APTMotor( app2.panel, HWSerialNum=83846250, style=wx.SUNKEN_BORDER )
box2.Add( analyzer, proportion=1, flag=wx.EXPAND )

app2.panel.SetSizer( box2 )
app2.frame.Show()
app2.SetEvtHandlerEnabled(False)

analyzer.SetHomeParams()
analyzer.SetStageAxisInfo()
analyzer.SetVelParams(fAccn=25, fMaxVel=25)
analyzer.SetBLashDist()

def move_ANA(ang=None):
    if ang == None:
        return analyzer.GetPosition()
    else:
        analyzer.MoveAbsoluteRot(ang)
        return move_ANA(ang=None)
    
def home_ANA():
    move_ANA(0.1)
    analyzer.MoveHome()

print 'Analyzer online.'

#Rotation stage3  Quarter Waveplate
app3 = MyApp(title='QWP')
box3 = wx.BoxSizer( wx.VERTICAL )

qwaveplate = APTMotor( app3.panel, HWSerialNum=83846230, style=wx.SUNKEN_BORDER )
box3.Add( qwaveplate, proportion=1, flag=wx.EXPAND )


app3.panel.SetSizer( box3 )
app3.frame.Show()
app3.SetEvtHandlerEnabled(False)

qwaveplate.SetHomeParams()
qwaveplate.SetStageAxisInfo()
qwaveplate.SetVelParams(fAccn=25, fMaxVel=25)
qwaveplate.SetBLashDist()

def move_QWP(ang=None):
    if ang == None:
        return qwaveplate.GetPosition()
    else:
        qwaveplate.MoveAbsoluteRot(ang)
        return move_QWP(ang=None)

def home_QWP():
    move_QWP(0.1)
    qwaveplate.MoveHome()
    
print 'Quarter Waveplate online.'

#app=MyApp(title='HWP QWP POL')
#box=wx.BoxSizer(wx.VERTICAL)
#
#waveplate = APTMotor( app.panel, HWSerialNum=83845997, style=wx.SUNKEN_BORDER )
#box.Add( waveplate, proportion=1, flag=wx.EXPAND )
#waveplate.SetHomeParams()
#waveplate.SetStageAxisInfo()
#waveplate.SetVelParams(fAccn=25, fMaxVel=25)
#waveplate.SetBLashDist()
#def move_HWP(ang=None):
#    if ang == None:
#        return waveplate.GetPosition()
#    else:
#        waveplate.MoveAbsoluteRot(ang)  
#        return move_HWP(ang=None)
#def home_HWP():
#    waveplate.MoveHome()
#print 'HWP online.'
#
#polarizer = APTMotor( app.panel, HWSerialNum=83846250, style=wx.SUNKEN_BORDER )
#box.Add( polarizer, proportion=1, flag=wx.EXPAND )
#polarizer.SetHomeParams()
#polarizer.SetStageAxisInfo()
#polarizer.SetVelParams(fAccn=25, fMaxVel=25)
#polarizer.SetBLashDist()
#def move_polarizer(ang=None):
#    if ang == None:
#        return polarizer.GetPosition()
#    else:
#        polarizer.MoveAbsoluteRot(ang)
#        return move_polarizer(ang=None)   
#def home_polarizer():
#    polarizer.MoveHome()
#print 'Polarizer online.'
#
#qwaveplate = APTMotor( app.panel, HWSerialNum=27000721, style=wx.SUNKEN_BORDER )
#box.Add( qwaveplate, proportion=1, flag=wx.EXPAND )
#qwaveplate.SetHomeParams()
#qwaveplate.SetStageAxisInfo()
#qwaveplate.SetVelParams(fAccn=25, fMaxVel=25)
#qwaveplate.SetBLashDist()
#def move_QWP(ang=None):
#    if ang == None:
#        return qwaveplate.GetPosition()
#    else:
#        qwaveplate.MoveAbsoluteRot(ang)
#        return move_QWP(ang=None)
#def home_QWP():
#    qwaveplate.MoveHome()
#print 'Quarter Waveplate online.'
#
#box.SetDimension(wx.Point(0,0),wx.Size(384,633))
#box.RecalcSizes()
#app.panel.SetSizer( box )
#app.frame.Show()
