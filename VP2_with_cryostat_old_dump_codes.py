# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 18:47:56 2019

@author: Millie
"""

#%%
#---------------------------------------------------#
#codes for the actual experiments
#code #1: for polarized SHG experiment (6 petals)


def VP2_pol_SHG(sample,hero_int_time=1000,hero_avg_num=1,num_of_spec=1,alphas=np.arange(0,360,5),pump_wl=1560,sc_wl=750,sc_on=False,timesleep=0.5,log=''):
    """
    Obtain Hero spectrum for each alpha.
    Alpha is defined as the angle for the ANALYZER.
    The HWP controls the input polarization, and therefore its angle is Alpha/2.
    QWP is not needed in this experiment code, since circular poalrization parameters are fixed in the supercontinuum laser beam.
    SHG pump beam wavelength is 1560nm.
    The laser beam we are using comes from the 780nm output from Toptica, this is the reminiscent 1560nm beam.

    As carachterized on 21-nov-2017, 1560nm average power before 100x lens is ~3.2mW, and 100x lens transmittance is ~40%.

    for horizontal reference frame (0deg at horizontal, and counter-clockwise rotation in the view of the microscope)
    """
    main_path=r'D:\Nonlinear_setup\Experimental_data\VP2_pol_SHG\%s'%sample 
    os.makedirs(main_path)
    total_len = len(alphas)
    total_time = (2*total_len)*hero_avg_num*num_of_spec*hero_int_time/1000. + total_len*(0.5+timesleep+1) + total_len*1.5
    start_time = time.time()
    init_pos = get_pos()

    init_line = '\nStarted VP2_pol_SHG (%s) on %s, expected to complete on %s\nhero_int_time = %.2f ms, hero_avg_num = %i, num_of_spec = %i, timesleep = %.2f s, alphas = %s, piezo stage position at %s\n'%(sample,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+total_time)),hero_int_time,hero_avg_num,num_of_spec,timesleep,str(alphas),str(init_pos))
    print(init_line)
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    np.save(os.path.join(main_path,'alphas'),alphas)
    H.initialise_hero()
    H.hero_int_time(hero_int_time)
    H.hero_avg_num(hero_avg_num)

    if sc_on:
        unblock_sc()
    else:
        block_sc()

    def HOME_HWP_LOOP():
        home_H()
    def HOME_ANA_LOOP():
        home_A()
    def HOME_QWP_LOOP():
        home_Q()
    print('Homing HWP, QWP, ANA')
    HWP_th = threading.Thread(target=HOME_HWP_LOOP)
    ANA_th = threading.Thread(target=HOME_ANA_LOOP)
    QWP_th = threading.Thread(target=HOME_QWP_LOOP)
    HWP_th.start()
    plt.pause(0.01)
    ANA_th.start()
    plt.pause(0.01)
    QWP_th.start()

    block_laser()
    time.sleep(1)
    BCKGND_SPEC = H.hero_spec()
    unblock_laser()
    time.sleep(1)
    
    np.save(os.path.join(main_path,'BCKGND_SPEC'),BCKGND_SPEC)
     
    HWP_th.join()
    ANA_th.join()
    QWP_th.join()
    
    move_alpha(alphas[0])
    lockin_auto_gain()
    lockin_auto_phase()
    optimise_srs830_sensitivity()
    get_lockin_reading1()
    time.sleep(1)
    block_laser()

#    pma_wl(pump_wl)
#    pmd_wl(sc_wl)

    def prepare_take_data_inside_loop():
        global specs, specs_bg
        specs = []
        specs_bg = []
        
    def prepare_take_data_outside_loop():
        global powers, powers_dev, powers_sc, powers_sc_dev
        powers = []
#        powers_sc = []
        powers_dev = []
#        powers_sc_dev = []

    def take_data_specs():
        curr_spec = copy.copy(H.hero_spec())
        specs.append(curr_spec)
        block_laser()
        time.sleep(1)
        specs_bg.append(copy.copy(H.hero_spec()))
        unblock_laser()
        time.sleep(0.5)

    def take_data_powers():
        _powers=[]
#        _powers_sc=[]
        for i in range(10):
#            _powers.append(pma_power())
            _powers.append(get_lockin_reading1())
#        for i in range(10):
#            _powers_sc.append(pmd_power())
#            time.sleep(0.05)
        
        curr_power = np.mean(_powers)
        curr_power_dev = np.std(_powers)
#        curr_power_sc = np.mean(_powers_sc)
#        curr_power_sc_dev = np.std(_powers_sc)
        
        powers.append(curr_power)
        powers_dev.append(curr_power_dev)
#        powers_sc_dev.append(curr_power_sc_dev)

    def save_data_specs(a):
        try:
            np.save(os.path.join(main_path,'SPECS_a%s'%a),np.array(specs))
            np.save(os.path.join(main_path,'SPECS_BG_a%s'%a),np.array(specs_bg))
        except:
            time.sleep(1)
            save_data_specs(a)

    def save_data_powers():
        try:
            np.save(os.path.join(main_path,'powers'),np.array(powers))
            np.save(os.path.join(main_path,'powers_dev'),np.array(powers_dev))
#            np.save(os.path.join(main_path,'powers_sc'),np.array(powers_sc))
#            np.save(os.path.join(main_path,'powers_sc_dev'),np.array(powers_sc_dev))
        except:
            time.sleep(1)
            save_data_powers()

    prev_completed = ''
    _n = 0
    prints('\n')
    
    SP_filter_out()
    unblock_laser()
    time.sleep(1)
    
    start_time = time.time()
    prepare_take_data_outside_loop()
    for alpha in alphas:
        completed = 'alpha at %.1fdeg (%.2f percent)'%(alpha,_n*100./total_len)
        prints(completed,prev_completed)
        prev_completed = completed

        move_alpha(alpha)
        plt.pause(0.5)
        prepare_take_data_inside_loop()
        
        for i in range(num_of_spec):
            take_data_specs()
        take_data_powers()

        save_data_specs(a='%i'%(alpha*100))
        save_data_powers()

        plt.pause(timesleep)
        _n += 1
    end_time = time.time()
    np.save(os.path.join(main_path,'timestamps.npy'),np.array([start_time,end_time]))
    block_laser()
#    SP_filter_in()
    block_sc()
    
    print 'Done! Time spent = %is'%(time.time()-start_time)
    play_sound(complete)

    H.hero_shutdown()
    try:
        VP2_pol_SHG_1560_anal(sample)
        plt.pause(1e-6)
    except:
        pass

#%%
def pSHG_fix_alpha(sample,alpha=0,ana_angs=np.arange(0,360,5),sc_on=False,ave_num=10,timeconst=0.1,log='',):
    """
    Obtain lockin reading for each analyzer angle.
    Alpha is defined as the angle for the ANALYZER.
    SHG pump beam wavelength is 1560nm.
    """
    input_line = np.array([get_last_input_line()])
    main_path=r'D:\Nonlinear_setup\Experimental_data\pSHG_fix_alpha\%s'%sample 
    os.makedirs(main_path)
    timeconst=lockin_timeconst(timeconst)
    total_len = len(ana_angs)
    timesleep_full = timeconst*10
    timesleep = timeconst*5
    total_time = total_len*( timesleep_full + ave_num*timesleep )
    global start_time,end_time
    start_time = time.time()

    init_line = '\nStarted pSHG_fix_alpha (%s) on %s, expected to complete on %s.\n'%(sample,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+total_time)))
    init_line2 = 'alpha = %f deg, ana_angs = %s, sc_on = %s, ave_num = %i, timeconst = %f s.'%(alpha,str(ana_angs),str(sc_on),ave_num,timeconst)
    print(init_line)
    print(init_line2)
    log_txt = [unicode(input_line),unicode(init_line),unicode(init_line2),
               u'\n\n'+unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    unblock_laser()
    if sc_on:
        unblock_sc()
    else:
        block_sc()

    home_all_rot()
    move_1560_to_alpha(alpha)
        
    def prepare_take_data():
        global Xs, Xerrs, thetas, theta_errs, gains, freqs, data_path
        Xs, Xerrs, thetas, theta_errs, gains, freqs = [],[],[],[],[],[]
        lockin_disp_Xerr()
        data_path = os.path.join(main_path,'data.npz')

    def take_data():
        curr_Xs, curr_Xerrs, curr_thetas, curr_gains, curr_freqs = [],[],[],[],[]
        lockin_auto_gain()
        for i in range(ave_num):
            X, Xerr, theta, gain, freq = lockin_get_X_Xerr_theta_aux4in_freq()
            curr_Xs.append(X)
            curr_Xerrs.append(Xerr)
            curr_thetas.append(theta)
            curr_gains.append(gain)
            curr_freqs.append(freq)
            plt.pause(timesleep)
        Xs.append(np.mean(curr_Xs))
        Xerrs.append(np.sqrt( np.square(np.std(curr_Xs)/np.sqrt(ave_num)) + np.square(np.mean(curr_Xerrs)) )) #taking account of both signal fluctuation and its error
        thetas.append(np.mean(curr_thetas))
        theta_errs.append(np.std(curr_thetas))
        gains.append(np.mean(curr_gains))
        freqs.append(np.mean(curr_freqs))

    def save_data():
        try:
            np.savez(data_path,
             ana_angs=ana_angs,
             Xs=Xs,
             Xerrs=Xerrs,
             thetas=thetas,
             theta_errs=theta_errs,
             gains=gains,
             freqs=freqs)
        except:
            time.sleep(1)
            save_data()
    
    def finishing():
        lockin_disp_X()
        block_laser()
        block_sc()
        np.save(os.path.join(main_path,'timestamps.npy'),np.array([start_time,end_time]))

    prev_completed = ''
    _n = 0
    prints('\n')
    
    prepare_take_data()
    start_time = time.time()
    for ana_ang in ana_angs:
        completed = 'Analyzer at %.1fdeg (%.2f percent)'%(ana_ang,_n*100./total_len)
        prints(completed,prev_completed)
        prev_completed = completed

        move_A(ana_ang)
        plt.pause(timesleep_full)
        take_data()
        save_data()

        _n += 1
    end_time = time.time()
    finishing()
    
    print 'Done! Time spent = %is'%(time.time()-start_time)
    play_sound(complete)

    try:
        anal_pSHG_fix_alpha(sample)
        plt.pause(1e-6)
    except:
        print("%s not analyzed"%sample)


#%%
def rotA_meas_lockin(sample,ana_angs=np.arange(0,360,5),sc_on=False,ave_num=10,timeconst=0.1,log=''):
    """
    Obtain lockin reading for each analyzer angle.
    Alpha is defined as the angle for the ANALYZER.
    SHG pump beam wavelength is 1560nm.
    """
    input_line = np.array([get_last_input_line()])
    main_path=r'D:\Nonlinear_setup\Experimental_data\rotA_meas_lockin\%s'%sample 
    os.makedirs(main_path)
    timeconst=lockin_timeconst(timeconst)
    total_len = len(ana_angs)
    timesleep_full = timeconst*10
    timesleep = timeconst*5
    total_time = total_len*( timesleep_full + ave_num*timesleep )
    global start_time,end_time
    start_time = time.time()

    init_line = '\nStarted rotA_meas_lockin (%s) on %s, expected to complete on %s.\n'%(sample,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+total_time)))
    init_line2 = 'ana_angs = %s, sc_on = %s, ave_num = %i, timeconst = %f s.'%(str(ana_angs),str(sc_on),ave_num,timeconst)
    print(init_line)
    print(init_line2)
    log_txt = [unicode(input_line),unicode(init_line),unicode(init_line2),
               u'\n\n'+unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    unblock_laser()
    if sc_on:
        unblock_sc()
    else:
        block_sc()

    home_A()
        
    def prepare_take_data():
        global Xs, Xerrs, thetas, theta_errs, gains, freqs, data_path
        Xs, Xerrs, thetas, theta_errs, gains, freqs = [],[],[],[],[],[]
        lockin_disp_Xerr()
        data_path = os.path.join(main_path,'data.npz')

    def take_data():
        curr_Xs, curr_Xerrs, curr_thetas, curr_gains, curr_freqs = [],[],[],[],[]
        lockin_auto_gain()
        for i in range(ave_num):
            X, Xerr, theta, gain, freq = lockin_get_X_Xerr_theta_aux4in_freq()
            curr_Xs.append(X)
            curr_Xerrs.append(Xerr)
            curr_thetas.append(theta)
            curr_gains.append(gain)
            curr_freqs.append(freq)
            plt.pause(timesleep)
        Xs.append(np.mean(curr_Xs))
        Xerrs.append(np.sqrt( np.square(np.std(curr_Xs)/np.sqrt(ave_num)) + np.square(np.mean(curr_Xerrs)) )) #taking account of both signal fluctuation and its error
        thetas.append(np.mean(curr_thetas))
        theta_errs.append(np.std(curr_thetas))
        gains.append(np.mean(curr_gains))
        freqs.append(np.mean(curr_freqs))

    def save_data():
        try:
            np.savez(data_path,
             ana_angs=ana_angs,
             Xs=Xs,
             Xerrs=Xerrs,
             thetas=thetas,
             theta_errs=theta_errs,
             gains=gains,
             freqs=freqs)
        except:
            time.sleep(1)
            save_data()
    
    def finishing():
        lockin_disp_X()
        block_laser()
        block_sc()
        np.save(os.path.join(main_path,'timestamps.npy'),np.array([start_time,end_time]))

    prev_completed = ''
    _n = 0
    prints('\n')
    
    prepare_take_data()
    start_time = time.time()
    for ana_ang in ana_angs:
        completed = 'Analyzer at %.1fdeg (%.2f percent)'%(ana_ang,_n*100./total_len)
        prints(completed,prev_completed)
        prev_completed = completed

        move_A(ana_ang)
        plt.pause(timesleep_full)
        take_data()
        save_data()

        _n += 1
    end_time = time.time()
    finishing()
    
    print 'Done! Time spent = %is'%(time.time()-start_time)
    play_sound(complete)

    try:
        anal_rotA_meas_lockin(sample)
        plt.pause(1e-6)
    except:
        print("%s not analyzed"%sample)

#%%
def rotA_meas_lockin_XY(sample,ana_angs=np.arange(0,360,5),sc_on=False,ave_num=10,timeconst=0.1,log=''):
    """
    Obtain lockin readings X, Y, Xerr and Yerr for each analyzer angle.
    Alpha is defined as the angle for the ANALYZER.
    SHG pump beam wavelength is 1560nm.
    Xerr (and Yerr) are Xnoise (or Ynoise, unit Vrms/sqrt(Hz)) * sqrt(ENBW), units Vrms
    """
    SAMPLE_RATE = 512
    #get constants from lockin, define stuff
    TC = lockin_timeconst(timeconst)
    slpidx = get_lockin_slope_index()
    freq = get_lockin_freq()
    gain = lockin_aux4in()
    slope = SR830_SLOPE[slpidx]
    ENBW = SR830_ENBW_FACTOR_TC[slpidx] / np.sqrt(TC)
    sample_rate = lockin_sample_rate(SAMPLE_RATE)
    
    timesleep_full = TC*10
    timesleep = TC*5
    estimated_time = len(ana_angs) * (timesleep_full + ave_num * timesleep) # just to estimate completion time
    
    #prepare log file
    input_line = np.array([get_last_input_line()])
    main_path=r'D:\Nonlinear_setup\Experimental_data\rotA_meas_lockin\%s'%sample 
    os.makedirs(main_path)
    init_line = '\nStarted rotA_meas_lockin_XY (%s) on %s, expected to complete on %s.\n'%(sample,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+estimated_time)))
    init_line2 = 'ana_angs = %s, sc_on = %s, ave_num = %i, timeconst = %f s, lockin frequency = %f Hz, PMT gain = %f V, lockin slope = %f dB/oct, lockin sample rate = %f Hz.' \
        %(str(ana_angs),str(sc_on),ave_num,TC,freq,gain,slope,sample_rate)
    print(init_line)
    print(init_line2)
    log_txt = [unicode(input_line),unicode(init_line),unicode(init_line2),
               u'\n\n'+unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    #prepare data file location
    data_path = os.path.join(main_path,'data.npz')
    
    #prepare setup for measurements
    #unblock_laser()
    if sc_on:
        unblock_sc()
    else:
        block_sc()

    home_A()
    
    #prepare lockin for measurements
    lockin_sample_rate(SAMPLE_RATE)
    lockin_disp_Xerr()
    lockin_disp_Yerr()
    
    #how to take a single data point (average of ave_num readings), returns X, Xerr (in Vrms), Y, Yerr (in Vrms).
    def take_data_pt():
        readings = []
        lockin_auto_gain()
        for i in range(ave_num):
            readings.append(lockin_get_X_Xnoise_Y_Y_noise())
            time.sleep(timesleep)
        #for loop just keeps appending new readings to a list, ave_num times.
        #To return data, split the arrays in a different direction
        curr_Xs, curr_Xnoises, curr_Ys, curr_Ynoises = np.vsplit(np.transpose(np.vstack(tuple(readings))),1)[0] #hacky thing
        curr_Xerrs = curr_Xnoises*np.sqrt(ENBW)
        curr_Yerrs = curr_Ynoises*np.sqrt(ENBW)
        curr_Xerr = np.sqrt( np.square(np.std(curr_Xs)/np.sqrt(ave_num)) + np.square(np.mean(curr_Xerrs)) )
        curr_Yerr = np.sqrt( np.square(np.std(curr_Ys)/np.sqrt(ave_num)) + np.square(np.mean(curr_Yerrs)) )
        return np.mean(curr_Xs), curr_Xerr, np.mean(curr_Ys), curr_Yerr
    
    #actual measurement loop
    Xs, Xerrs, Ys, Yerrs = [],[],[],[] #where data will be stored
    _n = 0
    prev_completed = ''
    start_time = time.time()
    for ana_ang in ana_angs:
        #print message, set some throwaway variables
        completed = 'Analyzer at %.1fdeg (%.2f percent)'%(ana_ang,_n*100./len(ana_angs))
        prints(completed,prev_completed)
        prev_completed = completed
        _saved=False
        
        #move to angle and take data
        move_A(ana_ang)
        clear_lockin_buffer()
        start_lockin_buffer()
        time.sleep(timesleep_full)
        
        X, Xerr, Y, Yerr = take_data_pt()
        Xs.append(X)
        Xerrs.append(Xerr)
        Ys.append(Y)
        Yerrs.append(Yerr)
        
        #save the data
        while(_saved==False):
            try:
                np.savez(data_path,
                 ana_angs=ana_angs,
                 Xs=Xs,
                 Xerrs=Xerrs,
                 Ys=Ys,
                 Yerrs=Yerrs,
                )
                _saved=True
            except:
                time.sleep(1)
        _n += 1
    
    end_time = time.time()
    
    completed = 'Analyzer at %.1fdeg (%.2f percent)'%(ana_ang,_n*100./len(ana_angs))
    prints(completed,prev_completed)
    
    # Tidy up the setup
    lockin_disp_X()
    lockin_disp_Y()
    block_laser()
    block_sc()
    
    # Write stuff to log and save the data
    with open(os.path.join(main_path,'log.txt'),'a') as f:
        f.write('\nScan completed on %s.\n'%(time.strftime("%d%b%Y %H:%M", time.localtime())))
    np.save(os.path.join(main_path,'timestamps.npy'),np.array([start_time,end_time]))
    
    print 'Done! Time spent = %is'%(time.time()-start_time)
    play_sound(complete)

    # Run analysis function to show plot of results
    try:
        anal_rotA_meas_lockin_XY(sample)
        plt.pause(1e-6)
    except:
        print("%s not analyzed"%sample)
    
    
    return


#%%
def VP2_zscan_SHG(sample,center_z,distance_from_center,resol=1,hero_int_time=2000,hero_avg_num=1,num_of_spec=5,alphas=[233,180],ana_offset=-50.99,pump_wl=1560,timesleep=0.5,pump_with_1560=True,log=''):

    y0=center_z
    yl=distance_from_center
    Ys=np.arange(y0-yl,y0+yl+resol,resol)

    block_laser()
#    block_sc()
    total_len = len(Ys)*len(alphas)
    total_time = total_len*(hero_avg_num*num_of_spec*hero_int_time/1000. +2)
    start_time = time.time()

    init_line = '\nStarted VP2_zscan_SHG (%s) on %s, expected to complete on %s\nhero_int_time = %.2f ms, hero_avg_num = %i, num_of_spec = %i, timesleep = %.2f s, alphas = %s, central Z = %s, distance from central Z = %s, and resolution = %.1f um\n'%(
        sample,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+total_time)),hero_int_time,hero_avg_num,num_of_spec,timesleep,str(alphas),str(y0),str(yl),resol)
    print(init_line)
#    ans=raw_input('Continue? Input n to cancel. ')
#    if ans=='n' or ans=='N':
#        return
    main_path=r'D:\Nonlinear_setup\Experimental_data\VP2_zscan_SHG\%s'%sample 
    os.makedirs(main_path)
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    np.save(os.path.join(main_path,'alphas'),alphas)
    np.save(os.path.join(main_path,'Ys'),Ys)
    _alphas_csv=open(os.path.join(main_path, "alphas.csv"), "wb") 
    alphas_csv=csv.writer(_alphas_csv)
    alphas_csv.writerow(alphas)
    _alphas_csv.close()

    H.initialise_hero()
    H.hero_int_time(hero_int_time)
    H.hero_avg_num(hero_avg_num)

    def HOME_HWP_LOOP():
        home_HWP()
    def HOME_QWP_LOOP():
        home_QWP()
    def HOME_ANA_LOOP():
        home_ANA()
    print('Homing HWP, QWP, ANA')
    HWP_th = threading.Thread(target=HOME_HWP_LOOP)
    QWP_th = threading.Thread(target=HOME_QWP_LOOP)
    ANA_th = threading.Thread(target=HOME_ANA_LOOP)
    HWP_th.start()
    QWP_th.start()
    ANA_th.start()

    BCKGND_SPEC = H.hero_spec()
    
    np.save(os.path.join(main_path,'BCKGND_SPEC'),BCKGND_SPEC)
    _BCKGND_SPEC_csv=open(os.path.join(main_path, "BCKGND_SPEC.csv"), "wb")
    BCKGND_SPEC_csv=csv.writer(_BCKGND_SPEC_csv)
    BCKGND_SPEC_csv.writerow(BCKGND_SPEC)
    _BCKGND_SPEC_csv.close()
     
    HWP_th.join()
    QWP_th.join()
    ANA_th.join()

    pma_wl(pump_wl)

    def prepare_take_data_spec():
        global specs
        specs = []
        
    def prepare_take_data_pm():
        global powers, powers_dev
        powers = []
        powers_dev = []

    def take_data_specs():
        curr_spec = copy.copy(H.hero_spec())
        specs.append(curr_spec)

    def take_data_powers():
        _powers=[]
        for i in range(10):
            _powers.append(pma_power())
            time.sleep(0.05)
        
        curr_power = np.mean(_powers)
        curr_power_dev = np.std(_powers)
        
        powers.append(curr_power)
        powers_dev.append(curr_power_dev)

    def save_data_specs(a,y):
        try:
            np.save(os.path.join(main_path,'SPECS_a%s_y%s'%(a,y)),np.array(specs))
        except:
            time.sleep(1)
            save_data_specs(a,y)

    def save_data_powers(a):
        try:
            np.save(os.path.join(main_path,'powers_a%s'%a),np.array(powers))
            np.save(os.path.join(main_path,'powers_dev_a%s'%a),np.array(powers_dev))
        except:
            time.sleep(1)
            save_data_powers(a)
    
    prev_completed = ''
    _n = 0
    prints('\n')
    if pump_with_1560:
        unblock_laser()
        SP_filter_out()
    else:
#        unblock_sc()
        SP_filter_in()
    
    for alpha in alphas:
        move_alpha(alpha,ANA_off=ana_offset)
        prepare_take_data_pm()
        for Y in Ys:
            move_to_y(Y)
            completed = 'alpha at %.1fdeg, Y at %.1f um (%.2f percent)'%(alpha,Y,_n*100./total_len)
            prints(completed,prev_completed)
            prev_completed = completed
            
            prepare_take_data_spec()
            
#            unblock_laser()
            plt.pause(0.5)
            for i in range(num_of_spec):
                take_data_specs()
            take_data_powers()

#            block_laser()
            save_data_specs(a='%i'%(alpha*100),y='%i'%(Y*100))
            save_data_powers(a='%i'%(alpha*100))

            plt.pause(timesleep)
            _n += 1

    block_laser()
#    block_sc()
    print 'Done! Time spent = %is'%(time.time()-start_time)
    play_sound(complete)

    H.hero_shutdown()
    try:
        VP2_zscan_SHG_anal(sample)
        plt.pause(1e-6)
    except:
        pass
#%%
#---------------------------------------------------#
#code #3: for SHG power dependence, with defined alphas

def VP2_power_dep_SHG(sample,hero_int_time=2000,hero_avg_num=1,num_of_spec=5,alphas=np.arange(0,360,2),hwp_offset=-9.67,ana_offset=-50.99,pump_wl=1560,timesleep=0.5,log=''):
    """

    In this power dependence measurement, analyser holds its position.

    After aligning HWO and ANA to alpha for maximum SHG, a polazizer must be inserted after HWP,
    so by rotating HWP we change the power pumping SHG, and do not change the alpha.
    
    The objective is to observe how SHG intensity varies with varying pump power.

    Obtain Hero spectrum for each HWP position (alpha/2).
    Alpha is defined as the angle for the polarization after HWP.

    The HWP controls the input polarization, and therefore its angle is Alpha/2.

    SHG pump beam wavelength is 1560nm.
    The laser beam we are using comes from the 780nm output from Toptica, this is the reminiscent 1560nm beam.

    As carachterized on 21-nov-2017, 1560nm average power before 100x lens is ~3.2mW, and 100x lens transmittance is ~40%.

    for horizontal reference frame (0deg at horizontal, and counter-clockwise rotation in the view of the microscope)
    HWP = -9.67
    ANA = -50.99
    """
    main_path=r'D:\Nonlinear_setup\Experimental_data\VP2_power_dep_SHG\%s'%sample 
    os.makedirs(main_path)
    block_laser()
    total_len = len(alphas)
    total_time = total_len*hero_avg_num*num_of_spec*hero_int_time/1000. + total_len*(0.5+timesleep+1)
    start_time = time.time()
    init_pos = get_pos()

    init_line = '\nStarted VP2_power_dep_SHG (%s) on %s, expected to complete on %s\nhero_int_time = %.2f ms, hero_avg_num = %i, num_of_spec = %i, timesleep = %.2f s, alphas = %s, piezo stage position at %s\n'%(sample,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+total_time)),hero_int_time,hero_avg_num,num_of_spec,timesleep,str(alphas),str(init_pos))
    print(init_line)
    ans=raw_input('To run this experiment, a polarizer must be introduced after HWP. Have you done this modification? Press n to cancel, press any other key to proceed.')
    if ans=='n' or ans=='N':
        return
    
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')

    np.save(os.path.join(main_path,'alphas'),alphas)
    _alphas_csv=open(os.path.join(main_path, "alphas.csv"), "wb") 
    alphas_csv=csv.writer(_alphas_csv)
    alphas_csv.writerow(alphas)
    _alphas_csv.close()

    H.initialise_hero()
    H.hero_int_time(hero_int_time)
    H.hero_avg_num(hero_avg_num)

    def HOME_HWP_LOOP():
        home_HWP()
    print('Homing HWP')
    HWP_th = threading.Thread(target=HOME_HWP_LOOP)
    HWP_th.start()

    BCKGND_SPEC = H.hero_spec()

    np.save(os.path.join(main_path,'BCKGND_SPEC'),BCKGND_SPEC)
    _BCKGND_SPEC_csv=open(os.path.join(main_path, "BCKGND_SPEC.csv"), "wb")
    BCKGND_SPEC_csv=csv.writer(_BCKGND_SPEC_csv)
    BCKGND_SPEC_csv.writerow(BCKGND_SPEC)
    _BCKGND_SPEC_csv.close()

    HWP_th.join()


    pma_wl(pump_wl)

    def prepare_take_data_inside_loop():
        global specs
        specs = []

    def prepare_take_data_outside_loop():
        global powers, powers_dev
        powers = []
        powers_dev = []

    def take_data_specs():
        curr_spec = copy.copy(H.hero_spec())
        specs.append(curr_spec)

    def take_data_powers():
        _powers=[]
        for i in range(10):
            _powers.append(pma_power())
            time.sleep(0.05)

        curr_power = np.mean(_powers)
        curr_power_dev = np.std(_powers)

        powers.append(curr_power)
        powers_dev.append(curr_power_dev)

    def save_data_specs(a):
        try:
            np.save(os.path.join(main_path,'SPECS_a%s'%a),np.array(specs))
        except:
            time.sleep(1)
            save_data_specs(a)

    def save_data_powers():
        try:
            np.save(os.path.join(main_path,'powers'),np.array(powers))
            np.save(os.path.join(main_path,'powers_dev'),np.array(powers_dev))
        except:
            time.sleep(1)
            save_data_powers()

    prev_completed = ''
    _n = 0
    prints('\n')

    prepare_take_data_outside_loop()
    for alpha in alphas:
        completed = 'alpha at %.1fdeg (%.2f percent)'%(alpha,_n*100./total_len)
        prints(completed,prev_completed)
        prev_completed = completed

        move_HWP(alpha/2 + hwp_offset)
        prepare_take_data_inside_loop()

        unblock_laser()
        plt.pause(0.5)
        for i in range(num_of_spec):
            take_data_specs()
        take_data_powers()

        block_laser()
        save_data_specs(a='%i'%(alpha*100))
        save_data_powers()

        plt.pause(timesleep)
        _n += 1

    block_laser()
    print 'Done! Time spent = %is'%(time.time()-start_time)
    play_sound(complete)

    H.hero_shutdown()
#%%
#code #4: for polarized SHG experiment (2 petals)


def VP2_pol_SHG_rotA(sample,hero_int_time=2000,hero_avg_num=1,num_of_spec=5,alphas=np.arange(0,90.001,15),betas=np.arange(0,360,2),ana_offset=-50.99,pump_wl=1560,sc_wl=756,sc_on=False,timesleep=0.5,log=''):
    """
    Obtain Hero spectrum for each alpha.
    Beta is defined as the angle for the ANALYZER.
    Alpha is defined as the polarization angle for the 1560 beam.
    The laser beam we are using comes from the 780nm output from Toptica, this is the reminiscent 1560nm beam.

    As carachterized on 21-nov-2017, 1560nm average power before 100x lens is ~3.2mW, and 100x lens transmittance is ~40%.

    for horizontal reference frame (0deg at horizontal, and counter-clockwise rotation in the view of the microscope)
    HWP = -9.67
    ANA = -50.99
    """
    main_path=r'D:\Nonlinear_setup\Experimental_data\VP2_pol_SHG_rotA\%s'%sample 
    os.makedirs(main_path)
    block_laser()
    total_len = len(alphas)*len(betas)
    total_time = total_len*hero_avg_num*num_of_spec*hero_int_time/1000. + total_len*(0.5+timesleep+1)
    start_time = time.time()
    init_pos = get_pos()

    init_line = '\nStarted VP2_pol_SHG_rotA (%s) on %s, expected to complete on %s\nhero_int_time = %.2f ms, hero_avg_num = %i, num_of_spec = %i, timesleep = %.2f s, alphas = %s, betas = %s, piezo stage position at %s\n'%(sample,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+total_time)),hero_int_time,hero_avg_num,num_of_spec,timesleep,str(alphas),str(betas),str(init_pos))
    print(init_line)
    log_txt = [init_line,
               unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    np.save(os.path.join(main_path,'alphas'),alphas)
    np.save(os.path.join(main_path,'betas'),betas)

    H.initialise_hero()
    H.hero_int_time(hero_int_time)
    H.hero_avg_num(hero_avg_num)

    if sc_on:
        SP_filter_in()
        unblock_sc()
    else:
        SP_filter_out()
        block_sc()

    def HOME_HWP_LOOP():
        home_HWP()
    def HOME_ANA_LOOP():
        home_ANA()
    def HOME_QWP_LOOP():
        home_QWP()
    print('Homing HWP, QWP, ANA')
    HWP_th = threading.Thread(target=HOME_HWP_LOOP)
    ANA_th = threading.Thread(target=HOME_ANA_LOOP)
    QWP_th = threading.Thread(target=HOME_QWP_LOOP)
    HWP_th.start()
    plt.pause(0.01)
    ANA_th.start()
    plt.pause(0.01)
    QWP_th.start()

    block_laser()
    BCKGND_SPEC = H.hero_spec()
    unblock_laser()
    
    np.save(os.path.join(main_path,'BCKGND_SPEC'),BCKGND_SPEC)
     
    HWP_th.join()
    ANA_th.join()
    QWP_th.join()

    pma_wl(pump_wl)
#    pmd_wl(sc_wl)

    def prepare_take_data_inside_loop():
        global specs
        specs = []
        
    def prepare_take_data_outside_loop():
        global powers, powers_dev, powers_sc, powers_sc_dev
        powers = []
#        powers_sc = []
        powers_dev = []
#        powers_sc_dev = []

    def take_data_specs():
        curr_spec = copy.copy(H.hero_spec())
        specs.append(curr_spec)

    def take_data_powers():
        _powers=[]
#        _powers_sc=[]
        for i in range(10):
            _powers.append(pma_power())
#        for i in range(10):
#            _powers_sc.append(pmd_power())
#            time.sleep(0.05)
        
        curr_power = np.mean(_powers)
        curr_power_dev = np.std(_powers)
#        curr_power_sc = np.mean(_powers_sc)
#        curr_power_sc_dev = np.std(_powers_sc)
        
        powers.append(curr_power)
        powers_dev.append(curr_power_dev)
#        powers_sc_dev.append(curr_power_sc_dev)

    def save_data_specs(a,b):
        try:
            np.save(os.path.join(main_path,'SPECS_a%s_b%s'%(a,b)),np.array(specs))
        except:
            time.sleep(1)
            save_data_specs(a,b)

    def save_data_powers(a):
        try:
            np.save(os.path.join(main_path,'powers_a%s'%a),np.array(powers))
            np.save(os.path.join(main_path,'powers_dev_a%s'%a),np.array(powers_dev))
#            np.save(os.path.join(main_path,'powers_sc'),np.array(powers_sc))
#            np.save(os.path.join(main_path,'powers_sc_dev'),np.array(powers_sc_dev))
        except:
            time.sleep(1)
            save_data_powers(a)
    
    prev_completed = ''
    _n = 0
    prints('\n')
    
    for alpha in alphas:
        move_1560_to_alpha(alpha)
        prepare_take_data_outside_loop()
        for beta in betas:
            completed = 'alpha at %.1fdeg, beta at %.1fdeg (%.2f percent)'%(alpha,beta,_n*100./total_len)
            prints(completed,prev_completed)
            prev_completed = completed
    
            move_A(beta)
            plt.pause(0.5)
            prepare_take_data_inside_loop()
            
            for i in range(num_of_spec):
                take_data_specs()
            take_data_powers()
    
    
            save_data_specs(a='%i'%(alpha*100),b='%i'%(beta*100))
            save_data_powers(a='%i'%(alpha*100))
    
            plt.pause(timesleep)
            _n += 1

    block_laser()
#    SP_filter_in()
    block_sc()
    
    print 'Done! Time spent = %is'%(time.time()-start_time)
    play_sound(complete)

    H.hero_shutdown()
    try:
        VP2_pol_SHG_1560_anal(sample)
        plt.pause(1e-6)
    except:
        pass


#%%
def scan_delayline(sample=None,fdl_min=0,fdl_max=12,fdl_incre=0.01,hero_int_ms=1000,hero_ave_num=1):
    H.initialise_hero()
    if sample == None:
        curr_time_str = create_date_str()
    else:
        curr_time_str = sample
    lockin_auto_gain()
    lockin_auto_phase()
    optimise_srs830_sensitivity()
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline')
    H.hero_int_time(hero_int_ms)
    H.hero_avg_num(hero_ave_num)
    
    fdl_poss = np.arange(fdl_min,fdl_max+fdl_incre,fdl_incre)
    prev=''
    total_poss = len(fdl_poss)
    data = []
    for i,fdl_pos in enumerate(fdl_poss):
        move_fdl_abs(fdl_pos)
        curr_fdl_pos = get_fdl_pos()
        global curr_spec, curr_lockin
        curr_spec = []
        curr_lockin = []
        def hero_loop():
            curr_spec.append(H.hero_spec())
        hero_th = threading.Thread(target=hero_loop)
        def lockin_loop():
            while hero_th.isAlive():
                curr_lockin.append(get_lockin_reading1())
                time.sleep(0.1)
        lockin_th = threading.Thread(target=lockin_loop)
        hero_th.start()
        lockin_th.start()
        hero_th.join()
        lockin_th.join()
        data.append([(curr_fdl_pos,np.average(curr_lockin)),curr_spec[0]])
        np.save(os.path.join(main_path,curr_time_str+'.npy'),np.array(data))
            
        completed = u'%f (%.2f percent)'%(curr_fdl_pos,100.0*(float(i+1)/total_poss))
        prints(completed,prev)
        prev = completed
    
    print 'Done!'
    H.hero_shutdown()

#%%

def scan_delayline_with_andor2(sample=None,fdl_min=0,fdl_max=12,fdl_incre=0.01,andor_int_ms=100,andor_num=5,repeat=1,opt_sc=False):
    '''Scan with optimising SC for each run'''
    if sample == None:
        curr_time_str = create_date_str()
    else:
        curr_time_str = sample
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline')
    set_exposure_time(andor_int_ms)
    wls = get_wl_vector()
    
    fdl_poss = np.arange(fdl_min,fdl_max+fdl_incre,fdl_incre)
    prev=''
    total_poss = len(fdl_poss)
    total_poss_repeat = total_poss*repeat
    data = [None,wls]
    _i = 0
    for j in range(repeat):
        if opt_sc:
            opt_scpz()
        for i,fdl_pos in enumerate(fdl_poss):
            move_fdl_abs(fdl_pos)
            curr_fdl_pos = get_fdl_pos()
            
            curr_spec = []
            for k in range(andor_num):
                curr_spec.append(copy.copy(get_spectrum()))
            specs = np.array(curr_spec)
            
            data.append([fdl_pos,specs])
            np.save(os.path.join(main_path,curr_time_str+'.npy'),np.array(data))
                
            completed = u'run %i (%.2f percent): %f (%.2f percent)'%(j,100.0*(float(_i+1)/total_poss_repeat),curr_fdl_pos,100.0*(float(i+1)/total_poss))
            prints(completed,prev)
            prev = completed
            _i += 1
    
    print 'Done!'


#%%

def scan_delayline_with_andor3(sample=None,fdl_min=0,fdl_max=12,fdl_incre=0.01,andor_int_ms=100,andor_num=5,repeat=1,opt_sc=False,slow_save=True,sc_on=True,random_pos=False):
    '''Scan with options to optimise SC for each run, slow save, without SC and random scan order'''
    global curr_time_str
    if sample == None:
        curr_time_str = create_date_str()
    else:
        curr_time_str = sample
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_with_andor3')
    def prepare_take_data():
        global data,wls,fdl_poss
        set_exposure_time(andor_int_ms)
        fdl_poss = np.arange(fdl_min,fdl_max+fdl_incre,fdl_incre)
        if random_pos:
            np.random.shuffle(fdl_poss)
        wls = get_wl_vector()
        data = [None,wls]
        
        block_laser()
        block_sc()
        pma_wl(1550)
        time.sleep(1)
        pma_zero()
        time.sleep(5)
        
        unblock_laser()
        if sc_on:
            unblock_sc()
        else:
            block_sc()
    
    def take_data():
        curr_fdl_pos = get_fdl_pos()
        _specs = [0]
        _sc_power = [0]
        _laser_power = [0]
        def spec_loop():
            curr_spec = []
            for k in range(andor_num):
                curr_spec.append(copy.copy(get_spectrum()))
            specs = np.array(curr_spec)
            _specs[0] = specs
        spec_th = threading.Thread(target=spec_loop)
        def lockin_loop():
            curr_sc_power = []
            curr_laser_power = []
            while spec_th.isAlive():
                curr_sc_power.append(lockin_aux3in())
#                curr_laser_power.append(lockin_aux1in())
                curr_laser_power.append(pma_power())
                time.sleep(0.01)
            _sc_power[0] = np.mean(curr_sc_power)
            _laser_power[0] = np.mean(curr_laser_power)
        lockin_th = threading.Thread(target=lockin_loop)
        
        spec_th.start()
        lockin_th.start()
        spec_th.join()
        lockin_th.join()
        
        data.append([fdl_pos,_specs[0],_sc_power[0],_laser_power[0]])
    
    def save_data():
        np.save(os.path.join(main_path,curr_time_str+'.npy'),np.array(data))
    
    def finishing():
        block_laser()
        block_sc()
    
    prepare_take_data()
    prev=''
    total_poss = len(fdl_poss)
    total_poss_repeat = total_poss*repeat
    _i = 0
    for j in range(repeat):
        if opt_sc:
            opt_scpz(verbose=False)
        for i,fdl_pos in enumerate(fdl_poss):
            move_fdl_abs(fdl_pos)
            take_data()
            if not slow_save:
                save_data()
                
            completed = u'run %i (%.2f percent): %f (%.2f percent)'%(j,100.0*(float(_i+1)/total_poss_repeat),fdl_pos,100.0*(float(i+1)/total_poss))
            prints(completed,prev)
            prev = completed
            _i += 1
        if slow_save:
            save_data()
    
    finishing()
    print 'Done!'

#%%
def scan_delayline_with_andor4(sample=None,fdl_min=0,fdl_max=12,fdl_incre=0.01,andor_int_ms=100,andor_num=5,repeat=1,opt_sc=False,slow_save=True,random_pos=False,sc_wl=758.1,sc_power_nW=200,set_sc_power_every_n_point=1):
    '''Scan with options to optimise SC for each run, slow save, and random scan order.
    Also scans with laser only, SC only, and both beams.'''
    global curr_time_str
    if sample == None:
        curr_time_str = create_date_str()
    else:
        curr_time_str = sample
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_with_andor4')
    def prepare_take_data():
        global data,wls,fdl_poss
        set_exposure_time(andor_int_ms)
        fdl_poss = np.arange(fdl_min,fdl_max+fdl_incre,fdl_incre)
        if random_pos:
            np.random.shuffle(fdl_poss)
        wls = get_wl_vector()
        # data will be [[sc wavelength], wavelengths, [both on], [laser only], [sc only]]
        data = [[sc_wl],wls,[],[],[]]
        
        block_laser()
        block_sc()
        pma_wl(sc_wl)
        time.sleep(1)
        pma_zero()
        time.sleep(5)
    
    def take_data(laser_on,sc_on):
        curr_fdl_pos = get_fdl_pos()
        _specs = [0]
        _sc_power = [0]
        _laser_power = [0]
        def spec_loop():
            curr_spec = []
            for k in range(andor_num):
                curr_spec.append(copy.copy(get_spectrum()))
            specs = np.array(curr_spec)
            _specs[0] = specs
            
        spec_th = threading.Thread(target=spec_loop)
        def lockin_loop():
            curr_sc_power = []
            curr_laser_power = []
            while spec_th.isAlive():
                curr_sc_power.append(pma_power())
#                curr_sc_power.append(lockin_aux3in())
#                curr_laser_power.append(lockin_aux1in())
#                curr_laser_power.append(pma_power())
                time.sleep(0.01)
            if laser_on:
                #_laser_power[0] = np.mean(curr_laser_power)
                pass
            if sc_on:
                _sc_power[0] = np.mean(curr_sc_power)
        lockin_th = threading.Thread(target=lockin_loop)
        
        spec_th.start()
        lockin_th.start()
        spec_th.join()
        lockin_th.join()
        
        if laser_on:
            if sc_on:
                data[2].append([fdl_pos,_specs[0],_sc_power[0],_laser_power[0]])
            else:
                data[3].append([fdl_pos,_specs[0],_sc_power[0],_laser_power[0]])
        else: #only sc_on
            data[4].append([fdl_pos,_specs[0],_sc_power[0],_laser_power[0]])
    
    def save_data():
        np.save(os.path.join(main_path,curr_time_str+'.npy'),np.array(data))
    
    def finishing():
        block_laser()
        block_sc()
    
    prepare_take_data()
    prev=''
    total_poss = len(fdl_poss)
    total_poss_repeat = total_poss*repeat
    
    
    _i = 0
    for j in range(repeat):
        if opt_sc:
            opt_scpz(verbose=False)
            block_laser()
            unblock_sc()
            plt.pause(5)
            set_sc_power_nW(sc_power_nW,sc_wl)
        start_time = time.time()
        for i,fdl_pos in enumerate(fdl_poss):
            move_fdl_abs(fdl_pos)
            if i%set_sc_power_every_n_point == 0:
                set_sc_power_nW(sc_power_nW,sc_wl)
            
            block_laser()
            unblock_sc()
            plt.pause(1)
            take_data(laser_on=False,sc_on=True)
                
            unblock_laser()
            block_sc()
            plt.pause(1)
            take_data(laser_on=True,sc_on=False)
            
            unblock_laser()
            unblock_sc()
            plt.pause(1)
            take_data(laser_on=True,sc_on=True)
            
            if not slow_save:
                save_data()
            
            elapsed_time = time.time() - start_time
            time_left = elapsed_time*(1.*len(fdl_poss)/(i+1)-1)
            completed = u'run %i (%.2f percent): %f (%.2f percent) %s left for this run.'%(j,100.0*(float(_i+1)/total_poss_repeat),fdl_pos,100.0*(float(i+1)/total_poss),sec_to_hhmmss(time_left))
            prints(completed,prev)
            prev = completed
            _i += 1
        if slow_save:
            save_data()
    
    finishing()
    print ' Done!'
    

#%%
def scan_delayline_with_andor5(sample=None,
                               fdl_min=0,fdl_max=12,fdl_incre=0.01,
                               fdl_poss = None,
                               andor_int_ms=100,andor_num=5,andor_warm_up_sec=60,
                               repeat=1,
                               opt_sc=False,slow_save=True,
                               sc_wl=758.1,sc_power_nW=200,set_sc_power_take_img_every_n_point=1,
                               pco_exp_time_ms=500,led_power=0.017,num_imgs=5):
    '''Scan with options to optimise SC for each run, slow save.
    Also scans with laser only, SC only, and both beams.'''
    global curr_time_str
    if sample == None:
        curr_time_str = create_date_str()
    else:
        curr_time_str = sample
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_with_andor5',curr_time_str)
    os.makedirs(main_path)
    img_path = os.path.join(main_path,'images')
    os.makedirs(img_path)
    if fdl_poss == None:
        fdl_poss = np.arange(fdl_min,fdl_max+fdl_incre,fdl_incre)
    else:
        fdl_poss = np.array(fdl_poss)
    # write log file with parameters used for experiment
    input_line = np.array([get_last_input_line()])
    delay_params_line = 'Scanning from delay line position %s mm to %s mm with step size %s mm.'%(str(fdl_min), str(fdl_max), str(fdl_incre))
    andor_params_line = 'Andor integration time %s ms, take median of %i points, warming up for %s seconds, repeating %i times.'%(str(andor_int_ms), andor_num, str(andor_warm_up_sec), repeat)
    image_params_line = 'PCO camera integration time %s ms, averaging over %i images.'%(str(pco_exp_time_ms), num_imgs)
    misc_params_line = 'SC wavelength %s nm, power at sample %s nW.'%(str(sc_wl), str(sc_power_nW))
    log_txt = [unicode(input_line),u'\n'+unicode(delay_params_line),u'\n'+unicode(andor_params_line),
               u'\n'+unicode(image_params_line), u'\n'+unicode(misc_params_line)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    def prepare_take_data():
        global data,wls,img_info
        #TODO: COMMENTING THIS LINE OUT FOR DEBUGGING
        set_exposure_time(andor_int_ms)
        remove_DM()
        initialise_pco()
        set_pco_exposure_time(pco_exp_time_ms)
        
        wls = get_wl_vector()
        # data will be [[sc wavelength], wavelengths, [both on], [laser only], [sc only]]
        # each data point is appended in those last 3 lists, [fdl_pos, spectrum, sc_power, laser_power]
        data = [[sc_wl],wls,[],[],[]]
        # img info will be [[img npy name, rep, fdl pos, time since start]]
        img_info = []
        
        block_laser()
        block_sc()
#        pma_wl(sc_wl)
        pmd_wl(sc_wl)
        time.sleep(1)
#        pma_zero()
        pmd_zero()
        time.sleep(5)
    
    def take_data(laser_on,sc_on):
        curr_fdl_pos = get_fdl_pos()
        _specs = [0]
        _sc_power = [0]
        _laser_power = [0]
        def spec_loop():
            curr_spec = []
            for k in range(andor_num):
                curr_spec.append(copy.copy(get_spectrum()))
            specs = np.array(curr_spec)
            _specs[0] = specs
            
        spec_th = threading.Thread(target=spec_loop)
        def lockin_loop():
            curr_sc_power = []
            curr_laser_power = []
            while spec_th.isAlive():
#                curr_sc_power.append(pma_power())
                curr_sc_power.append(pmd_power())
#                curr_sc_power.append(lockin_aux3in())
#                curr_laser_power.append(lockin_aux1in())
#                curr_laser_power.append(pma_power())
                time.sleep(0.01)
            if laser_on:
                #_laser_power[0] = np.mean(curr_laser_power)
                pass
            if sc_on:
                _sc_power[0] = np.mean(curr_sc_power)
        lockin_th = threading.Thread(target=lockin_loop)
        
        spec_th.start()
        lockin_th.start()
        spec_th.join()
        lockin_th.join()
        
        if laser_on:
            if sc_on:
                data[2].append([fdl_pos,_specs[0],_sc_power[0],_laser_power[0]])
            else:
                data[3].append([fdl_pos,_specs[0],_sc_power[0],_laser_power[0]])
        else: #only sc_on
            data[4].append([fdl_pos,_specs[0],_sc_power[0],_laser_power[0]])
    
    def save_data():
        np.save(os.path.join(main_path,curr_time_str+'.npy'),np.array(data))
    
    def take_and_save_img(name,rep,fdl_pos):
        block_sc()
        unblock_laser()
        img = DMin_LEDon_TAKEimg(LED_p=led_power, num_imgs=num_imgs, DM_out_after=True)
        img_info_entry = (name,rep,fdl_pos,time.time()-experiment_start_time)
        img_info.append(img_info_entry)
        img_comp = equalize_histogram_and_8bit(img)
        np.save(os.path.join(main_path, 'img_info.npy'), img_info)
        np.save(os.path.join(img_path, name+'.npy'), img_comp)
    
    def finishing():
        block_laser()
        block_sc()
        close_pco()
        LED_power(0)
        remove_DM()
        
    def take_background_imgs():
        unblock_laser()
        block_sc()
        img_laser_only = DMin_LEDon_TAKEimg(LED_p=0, num_imgs=num_imgs, DM_out_after=False)
        img_laser_only = equalize_histogram_and_8bit(img_laser_only)
        
        block_laser()
        img_LED_only = DMin_LEDon_TAKEimg(LED_p=led_power, num_imgs=num_imgs, DM_out_after=True)
        img_LED_only = equalize_histogram_and_8bit(img_LED_only)
        
        np.save(os.path.join(main_path, 'img_LED_only.npy'), img_LED_only)
        np.save(os.path.join(main_path, 'img_laser_only.npy'), img_laser_only)
    
    def equalize_histogram_and_8bit(img):
        return ((img-np.min(img))/(np.max(img)-np.min(img)) * 255).astype(np.uint8)
        
    
    # Andor somehow is more sensitive when starting to take data after a long pause. This decays back to normal.
    # "timewasting" to get Andor to warm up and remove (?) this artifact.
    def warm_up_andor(warm_up_time):
        s = time.time()
        i = 0
        while (time.time() - s) < warm_up_time:
            get_spectrum()
            time.sleep(0.05)
            i += 1
        #print("Warmed up for "+ str(time.time()-s) + " seconds. ")
        #print("Took " + str(i) + " spectra. ")
        
        
    warm_up_th = threading.Thread(target=warm_up_andor, args=(andor_warm_up_sec,))
    warm_up_th.start()
    
    prepare_take_data()
    prev=''
    total_poss = len(fdl_poss)
    total_poss_repeat = total_poss*repeat
    take_background_imgs()
    
    #unblock lasers for more effective warmup
    unblock_laser()
    unblock_sc()
    warm_up_th.join()
    experiment_start_time = time.time()
    
    _i = 0
    for j in range(repeat):
        if opt_sc:
            opt_scpz(verbose=False)
            block_laser()
            unblock_sc()
            plt.pause(5)
            set_sc_power_nW(sc_power_nW,sc_wl,pm='pmd')
        start_time = time.time()
        _j = 0
        for i,fdl_pos in enumerate(fdl_poss):
            move_fdl_abs(fdl_pos)
            if i%set_sc_power_take_img_every_n_point == 0:
                set_sc_power_nW(sc_power_nW,sc_wl,pm='pmd')
                take_and_save_img('%i_%i'%(j,_j),j,fdl_pos)
                _j += 1
            
            block_laser()
            unblock_sc()
            plt.pause(1)
            take_data(laser_on=False,sc_on=True)
                
            unblock_laser()
            block_sc()
            plt.pause(1)
            take_data(laser_on=True,sc_on=False)
            
            unblock_laser()
            unblock_sc()
            plt.pause(1)
            take_data(laser_on=True,sc_on=True)
            
            if not slow_save:
                save_data()
            
            elapsed_time = time.time() - start_time
            time_left = elapsed_time*(1.*len(fdl_poss)/(i+1)-1)
            completed = u'run %i (%.2f percent): %f (%.2f percent) %s left for this run.'%(j,100.0*(float(_i+1)/total_poss_repeat),fdl_pos,100.0*(float(i+1)/total_poss),sec_to_hhmmss(time_left))
            prints(completed,prev)
            prev = completed
            _i += 1
            
        if slow_save:
            save_data()
    
    finishing()
    time_taken = time.time() - experiment_start_time
    print(u'\n' + 'Time taken for experiment = %s'%sec_to_hhmmss(time_taken))
    print(u'\n' + 'Done!')

#%%
def scan_delayline_with_bg(sample=None,fdl_min=0,fdl_max=12,fdl_incre=0.01,hero_int_ms=1000,hero_ave_num=1,chop_laser_as_bg=True,lockin_as_pump_ref=True):
    H.initialise_hero()
    if sample == None:
        curr_time_str = create_date_str()
    else:
        curr_time_str = sample
    global curr_spec, curr_lockin, curr_bg_spec, curr_lockin_bg
    unblock_laser()
    unblock_sc()
    time.sleep(1)
    lockin_auto_gain()
    lockin_auto_phase()
    optimise_srs830_sensitivity()
    time.sleep(1)
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_with_bg')
    H.hero_int_time(hero_int_ms)
    H.hero_avg_num(hero_ave_num)
    
    fdl_poss = np.arange(fdl_min,fdl_max+fdl_incre,fdl_incre)
    prev=''
    total_poss = len(fdl_poss)
    data = [(curr_time_str,'fdl_min = %f'%fdl_min,'fdl_max = %f'%fdl_max,'fdl_incre = %f'%fdl_incre,'hero_int_ms = %f'%hero_int_ms,'hero_ave_num = %i'%hero_ave_num,'chop_laser_as_bg = %s'%chop_laser_as_bg,'lockin_as_pump_ref = %s'%lockin_as_pump_ref)]
    np.save(os.path.join(main_path,curr_time_str+'.npy'),np.array(data))
    i = 0
    for fdl_pos in fdl_poss:
        move_fdl_abs(fdl_pos)
        curr_fdl_pos = get_fdl_pos()
        
        curr_spec = []
        curr_spec_bg = []
        curr_lockin = []
        curr_lockin_bg = []
        def hero_loop():
            curr_spec.append(H.hero_spec())
        hero_th = threading.Thread(target=hero_loop)
        def lockin_loop():
            while hero_th.isAlive():
                curr_lockin.append(get_lockin_reading1())
                time.sleep(0.1)
        
        def hero_bg_loop():
            curr_spec_bg.append(H.hero_spec())
        hero_bg_th = threading.Thread(target=hero_bg_loop)
        def lockin_bg_loop():
            while hero_bg_th.isAlive():
                curr_lockin_bg.append(get_lockin_reading1())
                time.sleep(0.1)
        
        if chop_laser_as_bg:
            block_laser()
            unblock_sc()
        else:
            unblock_laser()
            block_sc()
        time.sleep(1)
        lockin_bg_th = threading.Thread(target=lockin_loop)
        hero_bg_th.start()
        lockin_bg_th.start()
        hero_bg_th.join()
        lockin_bg_th.join()
                
        
        completed = u'%f BG (%.2f percent)'%(curr_fdl_pos,50.0*(float(i+1)/total_poss))
        prints(completed,prev+' '*4)
        prev = completed
        i+=1
        
        if chop_laser_as_bg:
            unblock_laser()
        else:
            unblock_sc()
        time.sleep(1)
        lockin_th = threading.Thread(target=lockin_loop)
        hero_th.start()
        lockin_th.start()
        hero_th.join()
        lockin_th.join()
        
        data.append([(curr_fdl_pos,np.average(curr_lockin),np.average(curr_lockin_bg)),curr_spec[0],curr_spec_bg[0]])
        np.save(os.path.join(main_path,curr_time_str+'.npy'),np.array(data))
            
        completed = u'%f DATA (%.2f percent)'%(curr_fdl_pos,50.0*(float(i+1)/total_poss))
        prints(completed,prev+' '*4)
        prev = completed
        i+=1
    
    print '\nDone!'
    H.hero_shutdown()

#%%
def scan_delayline_multiple_Aang_with_bg(sample=None,fdl_min=0,fdl_max=12,fdl_incre=0.01,ave_num=10,timeconst=0.1,SHG_pump_angle=0,ana_angs=np.arange(0,60,5),log='',theta_th=10):
    input_line = np.array([get_last_input_line()])
    if sample == None:
        curr_time_str = create_date_str()
    else:
        curr_time_str = sample
    timeconst=lockin_timeconst(timeconst)
    total_lenA = len(ana_angs)
    timesleep_full = timeconst*10
    timesleep = timeconst*5
    fdl_poss = np.arange(fdl_min,fdl_max+fdl_incre,fdl_incre)
    total_lenD = len(fdl_poss)
    total_time = total_lenA*(total_lenD*( timesleep_full + ave_num*timesleep + 3))
    total_start_time = time.time()
    ini_line = [('\nStarted scan_delayline_multiple_Aang_with_bg (%s) on %s, expected to complete on %s'%(
        curr_time_str,time.strftime("%d%b%Y %H:%M", time.localtime()),time.strftime("%d%b%Y %H:%M", time.localtime(time.time()+total_time))),
             'fdl_min = %f'%fdl_min,
             'fdl_max = %f'%fdl_max,
             'fdl_incre = %f'%fdl_incre,
             'ave_num = %f'%ave_num,
             'timeconst = %f'%timeconst,
             'ana_angs = %s'%str(ana_angs),
             'SHG_pump_angle = %f'%SHG_pump_angle
             )]
    for l in ini_line:
        print(l)
    log_txt = [unicode(input_line),str(ini_line),
               unicode(log)+u'\n\n']
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_multiple_Aang_with_bg',curr_time_str)
    os.makedirs(main_path)
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    
    def prepare_take_data_outside_loop():
        lockin_disp_Xerr()
        block_laser()
        block_sc()
        home_all_rot()
        move_A(ana_angs[0])
        move_1560_to_alpha(SHG_pump_angle)
    
    def prepare_take_data(suffix):
        global Xs, Xerrs, thetas, theta_errs, gains, freqs, data_path, curr_fdl_poss
        global Xs_sc, Xerrs_sc, thetas_sc, theta_errs_sc, gains_sc, freqs_sc
        Xs, Xerrs, thetas, theta_errs, gains, freqs, curr_fdl_poss = [],[],[],[],[],[],[]
        Xs_sc, Xerrs_sc, thetas_sc, theta_errs_sc, gains_sc, freqs_sc = [],[],[],[],[],[]
        data_path = os.path.join(main_path,'data_%s.npz'%suffix)

    def take_data():
        curr_fdl_poss.append(fdl_pos)
        
        block_sc()
        unblock_laser()
        plt.pause(1)
        curr_Xs, curr_Xerrs, curr_thetas, curr_gains, curr_freqs = [],[],[],[],[]
        lockin_auto_optimize()
        for i in range(ave_num):
            X, Xerr, theta, gain, freq = lockin_get_X_Xerr_theta_aux4in_freq(theta_th=theta_th)
            curr_Xs.append(X)
            curr_Xerrs.append(Xerr)
            curr_thetas.append(theta)
            curr_gains.append(gain)
            curr_freqs.append(freq)
            plt.pause(timesleep)
        Xs.append(np.mean(curr_Xs))
        Xerrs.append(np.sqrt( np.square(np.std(curr_Xs)/np.sqrt(ave_num)) + np.square(np.mean(curr_Xerrs)) )) #taking account of both signal fluctuation and its error
        thetas.append(np.mean(curr_thetas))
        theta_errs.append(np.std(curr_thetas))
        gains.append(np.mean(curr_gains))
        freqs.append(np.mean(curr_freqs))
        
        unblock_sc()
        unblock_laser()
        plt.pause(1)
        curr_Xs, curr_Xerrs, curr_thetas, curr_gains, curr_freqs = [],[],[],[],[]
        lockin_auto_optimize()
        for i in range(ave_num):
            X, Xerr, theta, gain, freq = lockin_get_X_Xerr_theta_aux4in_freq(theta_th=theta_th)
            curr_Xs.append(X)
            curr_Xerrs.append(Xerr)
            curr_thetas.append(theta)
            curr_gains.append(gain)
            curr_freqs.append(freq)
            plt.pause(timesleep)
        Xs_sc.append(np.mean(curr_Xs))
        Xerrs_sc.append(np.sqrt( np.square(np.std(curr_Xs)/np.sqrt(ave_num)) + np.square(np.mean(curr_Xerrs)) )) #taking account of both signal fluctuation and its error
        thetas_sc.append(np.mean(curr_thetas))
        theta_errs_sc.append(np.std(curr_thetas))
        gains_sc.append(np.mean(curr_gains))
        freqs_sc.append(np.mean(curr_freqs))

    def save_data():
        try:
            np.savez(data_path,
             fdl_poss=curr_fdl_poss,
             Xs=Xs,
             Xerrs=Xerrs,
             thetas=thetas,
             theta_errs=theta_errs,
             gains=gains,
             freqs=freqs,
             Xs_sc=Xs_sc,
             Xerrs_sc=Xerrs_sc,
             thetas_sc=thetas_sc,
             theta_errs_sc=theta_errs_sc,
             gains_sc=gains_sc,
             freqs_sc=freqs_sc)
        except:
            time.sleep(1)
            save_data()
    
    def finishing_outside_loop():
        lockin_disp_X()
        block_laser()
        block_sc()
    
    def finishing_inside_Aloop(suffix):
        np.save(os.path.join(main_path,'timestamps_%s.npy'%suffix),np.array([start_time,end_time]))

    prev_completed = ''
    _n = 0
    prints('\n')
    
    prepare_take_data_outside_loop()
    for ana_ang in ana_angs:
        move_A(ana_ang)
        prepare_take_data('A%i'%(ana_ang*10))
        global start_time,end_time
        start_time = time.time()
        for fdl_pos in fdl_poss:
            completed = 'Analyzer at %.1f deg, delayline at %.3f mm (%.3f percent)'%(ana_ang,fdl_pos,_n*100./(total_lenA*total_lenD))
            prints(completed,prev_completed)
            prev_completed = completed
    
            move_fdl_abs(fdl_pos)
            plt.pause(timesleep_full)
            take_data()
            save_data()
    
            _n += 1
        end_time = time.time()
        finishing_inside_Aloop('A%i'%(ana_ang*10))
    finishing_outside_loop()
    
    print 'Done! Time spent = %s'%(sec_to_hhmmss(time.time()-total_start_time))
    play_sound(complete)

    try:
        anal_scan_delayline_multiple_Aang_with_bg(curr_time_str)
        plt.pause(1e-6)
    except:
        print("%s not analyzed"%curr_time_str)

#%%
def scan_delayline_with_lockin(sample=None,fdl_min=0,fdl_max=12,fdl_incre=0.01,ave_num=10,timeconst=0.1,theta_th=360,log='',main_path=None):
    if sample == None:
        curr_time_str = create_date_str()
    else:
        curr_time_str = sample
    if main_path == None:
        main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_with_lockin',curr_time_str)
    os.makedirs(main_path)
    timeconst=lockin_timeconst(timeconst)
    timesleep_full = timeconst*10
    timesleep = timeconst*5
    fdl_poss = np.arange(fdl_min,fdl_max+fdl_incre,fdl_incre)
    total_len = len(fdl_poss)
    
    def prepare_take_data():
        global Xs, Xerrs, thetas, theta_errs, gains, freqs, data_path, curr_fdl_poss
        Xs, Xerrs, thetas, theta_errs, gains, freqs, curr_fdl_poss = [],[],[],[],[],[],[]
        lockin_disp_Xerr()
        data_path = os.path.join(main_path,'data.npz')

    def take_data():
        curr_Xs, curr_Xerrs, curr_thetas, curr_gains, curr_freqs = [],[],[],[],[]
        curr_fdl_poss.append(fdl_pos)
        lockin_auto_optimize()
        for i in range(ave_num):
            X, Xerr, theta, gain, freq = lockin_get_X_Xerr_theta_aux4in_freq(theta_th=theta_th)
            curr_Xs.append(X)
            curr_Xerrs.append(Xerr)
            curr_thetas.append(theta)
            curr_gains.append(gain)
            curr_freqs.append(freq)
            plt.pause(timesleep)
        Xs.append(np.mean(curr_Xs))
        Xerrs.append(np.sqrt( np.square(np.std(curr_Xs)/np.sqrt(ave_num)) + np.square(np.mean(curr_Xerrs)) )) #taking account of both signal fluctuation and its error
        thetas.append(np.mean(curr_thetas))
        theta_errs.append(np.std(curr_thetas))
        gains.append(np.mean(curr_gains))
        freqs.append(np.mean(curr_freqs))

    def save_data():
        try:
            np.savez(data_path,
             fdl_poss=curr_fdl_poss,
             Xs=Xs,
             Xerrs=Xerrs,
             thetas=thetas,
             theta_errs=theta_errs,
             gains=gains,
             freqs=freqs)
        except:
            time.sleep(1)
            save_data()
    
    def finishing():
        lockin_disp_X()
        block_laser()
        block_sc()
        np.save(os.path.join(main_path,'timestamps.npy'),np.array([start_time,end_time]))

    prev_completed = ''
    _n = 0
    prints('\n')
    
    prepare_take_data()
    global start_time,end_time
    start_time = time.time()
    for fdl_pos in fdl_poss:
        completed = 'Delayline at %.3f mm (%.2f percent)'%(fdl_pos,_n*100./total_len)
        prints(completed,prev_completed)
        prev_completed = completed

        move_fdl_abs(fdl_pos)
        plt.pause(timesleep_full)
        take_data()
        save_data()

        _n += 1
    end_time = time.time()
    finishing()
    
    print 'Done! Time spent = %is'%(time.time()-start_time)
    play_sound(complete)

    try:
        anal_scan_delayline_with_lockin(curr_time_str)
        plt.pause(1e-6)
    except:
        print("%s not analyzed"%curr_time_str)

#%%
def scan_delayline_with_lockin_XY(sample=None,fdl_min=0,fdl_max=12,fdl_incre=0.01,ave_num=10,timeconst=0.1,theta_th=360,log='',main_path=None):
    """
    Scans the delayline across a particular range (from fdl_min to fdl_max with increment fdl_incre).
    Measurements taken and saved to .npz are fdl_pos, X, Xerr, Y, Yerr from lockin, and theta, thetaerr calculated.
    Xerr and Yerr are the Xnoise and Ynoise from lockin, multiplied by sqrt(ENBW).
    theta and thetaerr are in degrees.
    Be sure to open the laser and SC as needed before running.
    """
    
    # get constants from lockin and otherwise, put in variables
    SAMPLE_RATE = 512
    TC = lockin_timeconst(timeconst)
    slpidx = get_lockin_slope_index()
    freq = get_lockin_freq()
    gain = lockin_aux4in()
    slope = SR830_SLOPE[slpidx]
    ENBW = SR830_ENBW_FACTOR_TC[slpidx] / np.sqrt(TC)
    ana_ang = move_A()
    sample_rate = lockin_sample_rate()
    
    timesleep_full = TC*10
    timesleep = TC*5
    
    fdl_poss = np.arange(fdl_min,fdl_max+fdl_incre,fdl_incre)
    total_len = len(fdl_poss)
    
    # Warning
    print("Make sure the laser covers are open as needed!")
    
    # create output folder
    if sample == None:
        curr_time_str = create_date_str()
    else:
        curr_time_str = sample
    if main_path == None:
        main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_with_lockin',curr_time_str)
    os.makedirs(main_path)
    
    # prepare log file
    input_line = np.array([get_last_input_line()])
    init_line = '\nStarted scan_delayline_with_lockin_XY (%s) on %s.\n'%(sample,time.strftime("%d%b%Y %H:%M", time.localtime()))
    init_line2 = 'Delay line range (min, max, step) = (%f,%f,%f), analyser angle = %s, ave_num = %i, timeconst = %f s, lockin frequency = %f Hz, PMT gain = %f V, lockin slope = %f dB/oct, lockin sample rate = %s Hz.' \
        %(float(fdl_min), float(fdl_max), float(fdl_incre), str(ana_ang),ave_num,TC,freq,gain,slope,sample_rate)
    print(init_line)
    print(init_line2)
    log_txt = [unicode(input_line),unicode(init_line),unicode(init_line2),
               u'\n\n'+unicode(log)+u'\n\n']
    np.savetxt(os.path.join(main_path,'log.txt'),uniArray(log_txt),fmt='%s')
    # prepare data file location
    data_path = os.path.join(main_path,'data.npz')
    
    # prepare setup for measurements
    #unblock_laser()
    #unblock_sc()

    # prepare lockin for measurements
    lockin_sample_rate(SAMPLE_RATE)
    lockin_disp_Xerr()
    lockin_disp_Yerr()
    
    # how to take a single data point (average of ave_num readings), returns X, Xerr (in Vrms), Y, Yerr (in Vrms), theta, thetaerr (in degrees).
    def take_data_pt():
        readings = []
        lockin_auto_gain()
        for i in range(ave_num):
            readings.append(lockin_get_X_Xnoise_Y_Y_noise())
            time.sleep(timesleep)
        # for loop just keeps appending new readings to a list, ave_num times.
        # To return data, split the arrays in a different direction
        curr_Xs, curr_Xnoises, curr_Ys, curr_Ynoises = np.vsplit(np.transpose(np.vstack(tuple(readings))),1)[0] #hacky thing
        curr_Xerrs = curr_Xnoises*np.sqrt(ENBW)
        curr_Yerrs = curr_Ynoises*np.sqrt(ENBW)
        curr_Xerr = np.sqrt( np.square(np.std(curr_Xs)/np.sqrt(ave_num)) + np.square(np.mean(curr_Xerrs)) )
        curr_Yerr = np.sqrt( np.square(np.std(curr_Ys)/np.sqrt(ave_num)) + np.square(np.mean(curr_Yerrs)) )
        curr_thetas = np.arctan(np.array(curr_Ys)/np.array(curr_Xs))
        curr_theta_err = np.std(curr_thetas) #standard deviation is appropriate because repeated measurements of same thing
        return np.mean(curr_Xs), curr_Xerr, np.mean(curr_Ys), curr_Yerr, (180./np.pi)*np.mean(curr_thetas), (180./np.pi)*curr_theta_err
    
    # actual measurement loop
    curr_fdl_poss, Xs, Xerrs, Ys, Yerrs, thetas, thetaerrs = [],[],[],[],[],[],[] #where data will be stored
    prev_completed = ''
    _n = 0
    prints('\n')
    
    start_time = time.time()
    for fdl_pos in fdl_poss:
        completed = 'Delayline at %.3f mm (%.2f percent)'%(fdl_pos,_n*100./total_len)
        prints(completed,prev_completed)
        prev_completed = completed
        _saved = False
        
        # take the data
        move_fdl_abs(fdl_pos)
        clear_lockin_buffer()
        start_lockin_buffer()
        time.sleep(timesleep_full)
        
        X, Xerr, Y, Yerr, theta, theta_err = take_data_pt()
        curr_fdl_poss.append(fdl_pos)
        Xs.append(X)
        Xerrs.append(Xerr)
        Ys.append(Y)
        Yerrs.append(Yerr)
        thetas.append(theta)
        thetaerrs.append(theta_err)
        
        # save the data
        while(_saved==False):
            try:
                np.savez(data_path,
                 fdl_poss = curr_fdl_poss,
                 Xs=Xs,
                 Xerrs=Xerrs,
                 Ys=Ys,
                 Yerrs=Yerrs,
                 thetas=thetas,
                 thetaerrs=thetaerrs
                )
                _saved=True
            except:
                time.sleep(1)
        
        _n += 1
        
    end_time = time.time()
    
    completed = 'Delayline at %.3f mm (%.2f percent)'%(fdl_pos,_n*100./total_len)
    prints(completed,prev_completed)
    
    # Tidy up the setup
    lockin_disp_X()
    lockin_disp_Y()
    block_laser()
    block_sc()
    
    # Write stuff to log and save the data
    with open(os.path.join(main_path,'log.txt'),'a') as f:
        f.write('\nScan completed on %s.\n'%(time.strftime("%d%b%Y %H:%M", time.localtime())))
    np.save(os.path.join(main_path,'timestamps.npy'),np.array([start_time,end_time]))
    
    
    print 'Done! Time spent = %is'%(time.time()-start_time)
    play_sound(complete)
    
    try:
        anal_scan_delayline_with_lockin_XY(curr_time_str)
        plt.pause(1e-6)
    except:
        print("%s not analyzed"%curr_time_str)