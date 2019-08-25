# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 14:56:41 2017

@author: Neo
"""
import time
import sys
import numpy as np
import os
import psutil
from pynput import keyboard
histfile=os.path.join(r'C:\Users\Millie\.spyder\history.py')
import matplotlib.pyplot as plt
import copy
from scipy.optimize.minpack import curve_fit
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


def create_date_str():
    return time.strftime("%d%b%y_%H%M", time.localtime())

def create_date_str2():
    return time.strftime("%d%b%y_%H%M%S", time.localtime())

def sec_to_hhmmss(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dhr %2dmin %2ds" % (h, m, s)

def uniArray(array_unicode):
    items = [x.encode('utf-8') for x in array_unicode]
    array_unicode = np.array([items]) # remove the brackets for line breaks
    return array_unicode

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

def get_nearest_idx_from_list(val,lis):
    abs_diff_values = [abs(x-val) for x in lis]
    return abs_diff_values.index(min(abs_diff_values))

def get_nearest_values_from_list(val,lis):
    return lis[get_nearest_idx_from_list(val,lis)]

def make_format(current, other):
    """
    current and other are axes
    
    to use this code:
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax2.format_coord = make_format(ax2, ax1)
    """
    def format_coord(x, y):
        # x, y are data coordinates
        # convert to display coords
        display_coord = current.transData.transform((x,y))
        inv = other.transData.inverted()
        # convert back to data coords with respect to ax
        ax_coord = inv.transform(display_coord)
        coords = [ax_coord, (x, y)]
        return ('Left: {:<40}    Right: {:<}'
                .format(*['({:.3f}, {:.3f})'.format(x, y) for x,y in coords]))
    return format_coord

def remove_outlier(x,y,th='Auto'):
    x = list(x)
    y = list(y)
    if th == 'Auto':
        th = np.median(y)*10
    out_idxs = [y.index(ele) for ele in y if ele > th]
    in_idxs = [y.index(ele) for ele in y if ele < th]
    new_x = []
    new_y = []
    for idx in in_idxs:
        new_x.append(x[idx])
        new_y.append(y[idx])
    x_out = []
    y_out = []
    for idx in out_idxs:
        x_out.append(x[idx])
        y_out.append(y[idx])
    return np.array(new_x),np.array(new_y),np.array(x_out),np.array(y_out)

def remove_outlier2(x,y,th=1,cyclic=True,return_idx=False):
    x = list(x)
    y = list(y)
    in_idxs = []
    out_idxs = []
    for i in range(len(y)):
        if i == 0:
            if cyclic:
                prev_ele = y[i-1]
                curr_ele = y[i]
                next_ele = y[i+1]
            else:
                curr_ele = y[i]
                prev_ele = y[i+1]
                next_ele = y[i+2]
        elif i == len(y)-1:
            if cyclic:
                prev_ele = y[i-1]
                curr_ele = y[i]
                next_ele = y[0]
            else:
                curr_ele = y[i]
                prev_ele = y[i-1]
                next_ele = y[i-2]
        else:
            prev_ele = y[i-1]
            curr_ele = y[i]
            next_ele = y[i+1]
        if 2*abs(curr_ele - next_ele)/abs(curr_ele + next_ele) > th and 2*abs(curr_ele - prev_ele)/abs(curr_ele + prev_ele) > th:
            out_idxs.append(i)
        else:
            in_idxs.append(i)
    new_x = []
    new_y = []
    for idx in in_idxs:
        new_x.append(x[idx])
        new_y.append(y[idx])
    x_out = []
    y_out = []
    for idx in out_idxs:
        x_out.append(x[idx])
        y_out.append(y[idx])
    if return_idx:
        return np.array(new_x),np.array(new_y),np.array(in_idxs), np.array(x_out),np.array(y_out),np.array(out_idxs)
    else:
        return np.array(new_x),np.array(new_y),np.array(x_out),np.array(y_out)

def remove_outlier3(x,y,th=0.01,cyclic=False):
    # outliers can be one or two consecutive points, replaced with average of neighbours
    x = list(x)
    y = list(y)
    in_idxs = []
    out_idxs = []
    for i in range(len(y)):
        if i == 0:
            if cyclic:
                prev_ele = y[i-1]
                curr_ele = y[i]
                next_ele = y[i+1]
            else:
                curr_ele = y[i]
                prev_ele = y[i+1]
                next_ele = y[i+2]
        elif i == len(y)-1:
            if cyclic:
                prev_ele = y[i-1]
                curr_ele = y[i]
                next_ele = y[0]
            else:
                curr_ele = y[i]
                prev_ele = y[i-1]
                next_ele = y[i-2]
        else:
            prev_ele = y[i-1]
            curr_ele = y[i]
            next_ele = y[i+1]
        if 2*abs(curr_ele - next_ele)/abs(curr_ele + next_ele) > th or 2*abs(curr_ele - prev_ele)/abs(curr_ele + prev_ele) > th:
            out_idxs.append(i)
        else:
            in_idxs.append(i)
    new_x = []
    new_y = []
    for idx in range(len(x)):
        new_x.append(x[idx])
    for idx in in_idxs:
        new_y.append(y[idx])
    for idx in out_idxs:
        if idx == 0:
            new_y.insert(idx,new_y[idx+1])
        elif idx == len(new_y)-1:
            new_y.insert(idx,new_y[idx-1])
        else:
            try:
                fill_y = (new_y[idx-1] + new_y[idx+1]) / 2.
            except IndexError:
                fill_y = y[idx]
            new_y.insert(idx,fill_y)
            
    return np.array(new_x),np.array(new_y)


def put_angles_to_same_range(angs,r=60):
    if len(angs) < 2:
        return angs
    else:
        ans = []
        for i,ang in enumerate(angs):
            if i == 0:
                while ang < r:
                    ang += r
                while ang > r:
                    ang -= r
                ans.append(ang)
            else:
                while ang < ans[-1] - r*0.95:
                    ang += r
                while ang > ans[-1] + r*0.95:
                    ang -= r
                ans.append(ang)
        return ans
    
def get_last_input_line():
    with open(histfile) as hf:
        return (list(hf)[-1])

def start_keyboard_listening(keys,fxs):    
    def on_press(key):
        try: k = key.char # single-char keys
        except: k = key.name # other keys
        if key == keyboard.Key.esc: 
            print('Keyboard control stopped.')
            return False # stop listener
        for i,key in enumerate(keys):
            if k == key:
                fxs[i]()
    lis = keyboard.Listener(on_press=on_press)
    lis.start() # start to listen on a separate thread
    print('Keyboard control started.')
    
def find_peak_2(mover,position_getter,detector,increment,max_range,average=10,fluctuation='auto',timesleep=0,auto_max_range=True,plot=True,threshold_val=5e-5,stablize_time=0.5,poss=[],vals=[],save_plot_path=None,fig_data=None):
    if plot:    
        if fig_data == None:
            _fig_findpeak = plt.figure('find peak')
            fig_findpeak = _fig_findpeak.add_subplot(111)
            findpeak_plot, = fig_findpeak.plot([],[],marker='x',ls='',color='k')
        else:
            _fig_findpeak, fig_findpeak, findpeak_plot = fig_data

    
    def _detector():
        ans = 0
        for i in range(average):
            ans += detector()
            time.sleep(timesleep)
            plt.pause(1e-6)
        return ans/float(average)
    
    def parabola(x,a,tau,c):
        return a * ((x - tau) ** 2) + c
    
    def gauss(x, A ,mu, sigma):#gauss^2
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))**2
    
    def _parabola_fitter(x_data, y_data):
        tau = x_data[y_data.argmax()]
        c = np.mean(y_data)
        a = (y_data[2]-y_data[0])/(x_data[2]-x_data[0])/2./(x_data[1]-tau*.99)
        
        p0 = (a, tau, c)
        popt, pcov = curve_fit(parabola, x_data, y_data, p0)
            
        return popt
    
    def _gauss_fitter(x_data, y_data):
        A = np.max(y_data)
        mu = x_data[y_data.argmax()]
        sigma = (np.max(x_data)-np.min(x_data))/2
        
        p0 = (A, mu, sigma)
        popt, pcov = curve_fit(gauss, x_data, y_data, p0)
            
        return popt
    
    def sufficient_data(tau):
        max_pos = np.max(poss)        
        min_pos = np.min(poss)
        if tau < min_pos + 3*increment:
            return False
        if tau > max_pos - 3*increment:
            return False
        return True
    
    if fluctuation == 'auto':
        fluctuation = []
        for i in range(average):
            fluctuation.append(detector())
            time.sleep(timesleep)
        fluctuation = np.std(fluctuation)
    
    init_pos = position_getter()
    target_poss = np.arange(init_pos - max_range/2.0, init_pos + max_range/2.0+increment, increment)
    
    def plotting(title=''):
        if plot:
            findpeak_plot.set_xdata(poss)
            findpeak_plot.set_ydata(vals)
            fig_findpeak.relim()
            fig_findpeak.autoscale_view(True,True,True)
            fig_findpeak.set_title(title)
            plt.draw()
            plt.pause(1e-3)
        
    for pos in target_poss:
        mover(pos)
        plt.pause(stablize_time)
        curr_pos = position_getter()
        curr_val = _detector()
        if curr_val > threshold_val:
            break
        poss.append(curr_pos)
        vals.append(curr_val)
        
        plotting('At %f, signal %f.'%(curr_pos,curr_val))
        
        
    
    vals = [x for y, x in sorted(zip(poss,vals))]
    poss.sort()
    
#    poss,vals,nil,nil = remove_outlier2(poss,vals,th=0.5,cyclic=False)
    
    poss=list(poss)
    vals=list(vals)
#    return poss,vals
    result = False
    success_fit = False
    try:
        a,tau,c = _parabola_fitter(np.array(poss),np.array(vals)) #parabola fit
        plotting('Successfully fitted with %f, %f, %f'%(a,tau,c))
#        A, mu, sigma = _gauss_fitter(np.array(poss),np.array(vals)) #gaussian fit
        fitted_xs = np.linspace(poss[0],poss[-1],100)
        fig_findpeak.plot(fitted_xs,parabola(fitted_xs,a,tau,c),'k') #parabola fit
#        fig_findpeak.plot(fitted_xs,gauss(fitted_xs,A,mu,sigma),'k') #gaussian fit
        fig_findpeak.vlines(tau,np.min(vals),c) #parabola fit
#        fig_findpeak.vlines(mu,np.min(vals),A) #gaussian fit
        plt.draw()
        plt.pause(1e-3)
        success_fit = True
        
        if sufficient_data(tau): #parabola fit
#        if sufficient_data(mu): #gaussian fit
            max_pos = tau #parabola fit
            max_val = c #parabola fit
            mover(tau) #parabola fit
#            max_pos = mu #gaussian fit
#            max_val = A #gaussian fit
#            mover(mu) #gaussian fit
            plt.pause(stablize_time)
            fig_findpeak.set_title('Fitted max pos %f with val %f'%(max_pos,max_val))
            plt.draw()
            plt.pause(1e-3)
            if save_plot_path != None:
                if not os.path.exists(save_plot_path):
                    os.mkdir(save_plot_path)
                _fig_findpeak.savefig(os.path.join(save_plot_path,'%f.png'%tau)) #parabola fit
#                _fig_findpeak.savefig(os.path.join(save_plot_path,'%f.png'%mu)) #gaussian fit
#            fig_findpeak.cla()
            if fig_data != None:
                result = (_fig_findpeak, fig_findpeak, findpeak_plot)
            else:
                result = (max_pos,max_val,0)
            return result
    
    except RuntimeError:
        auto_max_range = True
#        print 'Unable to search for peak, increase max_range.'  
    finally:   
        if not result:
            result = False
            j = 1
            while not result:
                if success_fit:
                    mover(tau) #parabola fit
#                    mover(mu) #gaussian fit
                    plt.pause(stablize_time)
                else:
                    if len(vals) < 1:
                        k = 1
                        while detector() < threshold_val:
                            mover(init_pos+k*0.01*max_range)
                            plt.pause(stablize_time)
                            if detector() > threshold_val:
                                break
                            mover(init_pos-k*0.01*max_range)
                            plt.pause(stablize_time)
                            k += 1
                            
                    elif vals[0] < vals[-1]:
                        mover(init_pos+j*0.5*max_range)
                        plt.pause(stablize_time)
                    else:
                        mover(init_pos-j*0.5*max_range)
                        plt.pause(stablize_time)
                result = find_peak_2(mover,position_getter,detector,increment,max_range,average,fluctuation,timesleep,auto_max_range=False,plot=plot,stablize_time=stablize_time,poss=copy.copy(poss),vals=copy.copy(vals),threshold_val=threshold_val,save_plot_path=save_plot_path,fig_data=(_fig_findpeak, fig_findpeak, findpeak_plot))
                j += 1
            
        return result
        

    
def find_peak(mover,position_getter,detector,increment,max_range,overshoot_range,average=10,fluctuation='auto',timesleep=0,auto_max_range=True,plot=True,threshold_val=np.inf,stablize_time=0.5):
    if plot:    
        _fig_findpeak = plt.figure('find peak')
        fig_findpeak = _fig_findpeak.add_subplot(111)
        findpeak_plot, = fig_findpeak.plot([],[])
    
    posi='positive'
    nega='negative'
    if increment<0:
        posi,nega=nega,posi
        
    def _detector():
        ans = 0
        for i in range(average):
            ans += detector()
            time.sleep(timesleep)
            plt.pause(1e-6)
        return ans/float(average)
    
    poss = []
    vals = []
    
    if fluctuation == 'auto':
        fluctuation = []
        for i in range(average):
            fluctuation.append(detector())
            time.sleep(timesleep)
        fluctuation = np.std(fluctuation)
    
    def take_data(p,v):
        poss.append(p)
        vals.append(v)
    
    def is_increasing():
        return vals[-1] > vals[-2]
    
    def is_no_change():
        return np.abs((vals[-1]-vals[-2])/float(vals[-1])) < fluctuation
        
    def is_high_enough():
        return vals[-1] > threshold_val
        
    def move_positive_and_take_data(title=''):
        mover(position_getter()+increment)
        plt.pause(stablize_time)
        curr_pos = position_getter()
        curr_val = _detector()
        
        take_data(curr_pos,curr_val)
        plotting(title)
        
    def move_negative_and_take_data(title=''):
        mover(position_getter()-increment)
        plt.pause(stablize_time)
        curr_pos = position_getter()
        curr_val = _detector()
        
        take_data(curr_pos,curr_val)
        plotting(title)
    
    def get_max_pos_and_val():
        max_val = max(vals)
        max_pos = poss[vals.index(max_val)]
        
        return (max_pos,max_val)
    
    def plotting(title=''):
        if plot:
            findpeak_plot.set_xdata(poss)
            findpeak_plot.set_ydata(vals)
            fig_findpeak.relim()
            fig_findpeak.autoscale_view(True,True,True)
            fig_findpeak.set_title(title)
            plt.show()
            plt.pause(1e-6)
    
    
            
    init_pos = position_getter()
    init_val = _detector()    
    take_data(init_pos,init_val)
    
    consecutive_increase = 0
    _max_range = max_range
    #try positive positions first
    while np.abs(position_getter()-init_pos) < max_range:
        move_positive_and_take_data('Moving %s'%posi)
        if is_high_enough():
            return get_max_pos_and_val()
        
        while not is_no_change():
            if is_increasing():
                consecutive_increase += 1
                _max_range += increment
                move_positive_and_take_data('Signal raised, moving %s'%posi)
                if is_high_enough():
                    return get_max_pos_and_val()
            else:
                if consecutive_increase > 3:
                    move_positive_and_take_data('Signal dropped, peak suspected, moving %s'%posi)
                    if is_high_enough():
                        return get_max_pos_and_val()
                    if not is_no_change():
                        if is_increasing():
                            continue
                        else:
                            dec_init_pos = position_getter()
                            while np.abs(position_getter() - dec_init_pos) < overshoot_range:
                                move_positive_and_take_data('Peak suspected at %f, moving %s overshoot'%(dec_init_pos,posi))
                                if is_high_enough():
                                    return get_max_pos_and_val()
                            max_pos,max_val = get_max_pos_and_val()
                            mover(max_pos)
                            return (max_pos,max_val,1)
                else:
                    move_positive_and_take_data('Signal dropped, moving %s'%posi)
                    if is_high_enough():
                        return get_max_pos_and_val()
                    if not is_no_change():
                        if is_increasing():
                            consecutive_increase = 0
                            continue
                        else:
                            break
        
        if not is_no_change():
            break

    positive_len = len(poss)
    #cant find in positive positions, now try with negative positions
    consecutive_increase = 0
    poss = []
    vals = []
    take_data(init_pos,init_val)
    _max_range = max_range
    while np.abs(position_getter()-init_pos) < _max_range:
        move_negative_and_take_data('Moving %s'%nega)
        if is_high_enough():
            return get_max_pos_and_val()
        while not is_no_change():
            if is_increasing():
                consecutive_increase += 1
                _max_range += increment
                move_negative_and_take_data('Signal raised, moving %s'%nega)
                if is_high_enough():
                    return get_max_pos_and_val()
            else:
                if consecutive_increase > 3:
                    move_negative_and_take_data('Signal dropped, peak suspected, moving %s'%nega)
                    if is_high_enough():
                        return get_max_pos_and_val()
                    if not is_no_change():
                        if is_increasing():
                            continue
                        else:
                            dec_init_pos = position_getter()
                            while np.abs(position_getter() - dec_init_pos) < overshoot_range:
                                move_negative_and_take_data('Peak suspected at %f, moving %s overshoot'%(dec_init_pos,nega))
                                if is_high_enough():
                                    return get_max_pos_and_val()
                            max_pos,max_val = get_max_pos_and_val()
                            mover(max_pos)
                            return (max_pos,max_val,-1)
                else:
                    move_negative_and_take_data('Signal dropped, moving %s'%nega)
                    if is_high_enough():
                        return get_max_pos_and_val()
                    if not is_no_change():
                        if is_increasing():
                            consecutive_increase = 0
                            continue
                        else:
                            break
        
        if not is_no_change():
            return find_peak(mover,position_getter,detector,increment,max_range,overshoot_range,average,fluctuation,auto_max_range=True,plot=plot,stablize_time=stablize_time,timesleep=timesleep,threshold_val=threshold_val)

    negative_len = len(poss)
    if not auto_max_range:
#        mover(init_pos)
#        print 'Unable to search for peak. Moved back to initial position.'
        return False
    else:
        print 'Unable to search for peak, increase max_range.'        
        result = False
        j = 1
        if positive_len > negative_len:
            _j=1
        else:
            _j=-1
        while not result:
            mover(init_pos+_j*j*1.9*max_range)
            plt.pause(stablize_time)
            result = find_peak(mover,position_getter,detector,increment,max_range,overshoot_range,average,fluctuation,auto_max_range=False,plot=plot,stablize_time=stablize_time,timesleep=timesleep,threshold_val=threshold_val)
            if not result:
                mover(init_pos-_j*j*1.9*max_range)
                plt.pause(stablize_time)
                result = find_peak(mover,position_getter,detector,increment,max_range,overshoot_range,average,fluctuation,auto_max_range=False,plot=plot,stablize_time=stablize_time,timesleep=timesleep,threshold_val=threshold_val)
            j += 1
        
        return result
    
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def round_to_n(x,n):
    return round(x, get_leading_order(x) + (n - 1))

def get_leading_order(x):
    return -int(np.floor(np.log10(x)))

def round_to_error(val,err):
    new_err = round_to_n(err,1)
    new_val = round(val,get_leading_order(new_err))
    return new_val,new_err

def round_to_error_SI(val,err,init_prefix=' ',unit=''):
    precesion = -1-get_leading_order(val)
    val *= si_prefix.si_prefix_scale(init_prefix)
    err *= si_prefix.si_prefix_scale(init_prefix)
    val,err = round_to_error(val,err)
    val_str = si_prefix.si_format(val,precesion)
    val_str,val_prefix = val_str.split(' ')
    err_str = u'%g'%(err/si_prefix.si_prefix_scale(val_prefix))
    dp = 0
    if u'.' in err_str:
        dp = int(len(err_str.split('.')[-1]))
    val_str = ('%%.%if'%(dp))%(float(val_str))
    return u'(%s \xb1 %s) %s%s'%(val_str,err_str,val_prefix,unit)

def get_all_npy_files_in(folder_path):
    npy_files = list(filter(lambda x: x.endswith(".npy"),os.listdir(folder_path)))
    return npy_files

def get_only_npy_files_with_word(word, npy_files):
    return list(filter(lambda npy_file: word in npy_file, npy_files))
    
def imshow_npy_files_with_word_in(folder_path, word='image', sort_fx=lambda x: float(x.split('image')[1].split('.npy')[0])):
    npy_files = get_all_npy_files_in(folder_path)
    filtered_npy_files = get_only_npy_files_with_word(word, npy_files)
    try:
        filtered_npy_files = sorted(filtered_npy_files,key=sort_fx)
    except:
        print('.npy files not sorted.')
    
    def press(event):
        if event.key == 'left':
            curr_disp_img[0] -= 1
            if curr_disp_img[0] < 0:
                curr_disp_img[0] = len(filtered_npy_files)-1
        elif event.key == 'right':
            curr_disp_img[0] += 1
            if curr_disp_img[0] > len(filtered_npy_files):
                curr_disp_img[0] = 0

        curr_img = np.load(os.path.join(folder_path,filtered_npy_files[curr_disp_img[0]]))
        im.set_data(curr_img)
        curr_img_max = np.max(curr_img)
        curr_img_mean = np.mean(curr_img)
        im.set_data(curr_img)
        fig.set_title('%s\nMax = %f; Mean = %f'%(filtered_npy_files[curr_disp_img[0]],curr_img_max,curr_img_mean))
        im.set_clim(np.min(im.get_array()),np.max(im.get_array()))
        plt.pause(1e-3)
    
    _fig = plt.figure(folder_path)
    fig = _fig.add_subplot(111)
    im = fig.imshow(np.load(os.path.join(folder_path,filtered_npy_files[0])))
    curr_disp_img = [0]
    fig.set_title(filtered_npy_files[0])
    _fig.canvas.mpl_connect('key_press_event', press)
    im.set_cmap('Greys_r')
    plt.pause(1e-2)
    
    for i,npy_file in enumerate(filtered_npy_files):
        curr_img = np.load(os.path.join(folder_path,npy_file))
        curr_img_max = np.max(curr_img)
        curr_img_mean = np.mean(curr_img)
        im.set_data(curr_img)
        fig.set_title('%s\nMax = %f; Mean = %f'%(npy_file,curr_img_max,curr_img_mean))
        im.set_clim(np.min(im.get_array()),np.max(im.get_array()))
        
        curr_disp_img[0]=i
        plt.pause(1e-3)
        if curr_disp_img[0] != i:
            break

def put_polarimeter_angles_to_smallest_range(a,g):
    """
    putting 0 <= alpha < 180, -45 <= gamma <= 45 
    """
    if np.abs(g) > 45:
        g_new = np.sign(g)*(90 - np.abs(g))
        a_new = a + 90
    else:
        g_new = g
        a_new = a
    while a_new < 0:
        a_new += 180
    while a_new > 180:
        a_new -= 180
    return a_new, g_new

def put_polarimeter_angles_to_smallest_range2(a,g):
    """
    putting -90 <= alpha < 90, -45 <= gamma <= 45 
    """
    if np.abs(g) > 45:
        g_new = np.sign(g)*(90 - np.abs(g))
        a_new = a + 90
    else:
        g_new = g
        a_new = a
    while a_new < -90:
        a_new += 180
    while a_new > 90:
        a_new -= 180
    return a_new, g_new

def maximize_current_plt_window():
    plt.pause(1e-6)
    plt.get_current_fig_manager().window.showMaximized()
    
def active_process_names():
    return psutil.process_iter()

def kill_process(name):
    for proc in psutil.process_iter():
        if proc.name() == name:
            proc.kill()
    return