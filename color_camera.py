
import ueye
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import time
import Tkinter as Tk
#import pyqtgraph as pg



def run_camera(exp, fps):
    """
    exp = exposure time, in ms
    fps = frame rate, /s
    """
    #imv = pg.ImageView()    
    
    
    root = Tk.Tk()

    cam = ueye.camera()
    cam.AllocImageMem()
    cam.SetImageMem()
    cam.SetImageSize()
    cam.SetColorMode()
    cam.Exposure(5)
    cam.CaptureVideo()
    cam.ExitCamera()
    cam = ueye.camera()
    cam.AllocImageMem()
    cam.SetImageMem()
    cam.SetImageSize()
    cam.SetColorMode()
    cam.Exposure(exp)
    cam.CaptureVideo()
    time.sleep(1.0/fps)
    
    
    def _quit():
        cam.FreeImageMem()
        cam.StopLiveVideo()
        cam.ExitCamera()
        plt.close()
        root.quit()
        root.destroy()
        
    button = Tk.Button(master=root, text='Stop', command=_quit)
    button.pack(side=Tk.BOTTOM)
    i =0
    while True:
        i+=1
        
        
        time.sleep(1.0/fps)
        cam.CopyImageMem()
            
        plt.clf()
           
        im = plt.imshow(cam.data)
        plt.title('Max of '+str(cam.data.max()))
        plt.draw()
        plt.pause(1.0/fps)
        #im.show()
        #im.setImage(cam.data)
        
      
        
            
    
   