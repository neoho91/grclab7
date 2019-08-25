import visa
rm = visa.ResourceManager()
rm.list_resources()
ffpro=rm.open_resource("ASRL3::INSTR")
ffpro.baud_rate=115200
ffpro.data_bits=8
#ffpro.write(u'ffpro.arm0.level=1')
#ffpro.write(u'ffpro.arm0.axSi.target=30400')
#ffpro.write(u'ffpro.arm0.off()')
#ffpro.write(u'ffpro.hello()')

def laser_power(p):
    ffpro.write(u'ffpro.arm0.level='+str(p))
#    ffpro.write(u'ffpro.hello()')
    print('Power at '+str(p*100)+'%')
    
def laser_on():
    ffpro.write(u'ffpro.arm0.on()')
#    ffpro.write(u'ffpro.hello()')
    print('Laser is on! So are your glasses.')
    
def laser_off():
    ffpro.write(u'ffpro.arm0.off()')
#    ffpro.write(u'ffpro.hello()')
    print('Laser is off.')

def laser_prism(l,verbose=True):
    ffpro.write(u'ffpro.arm0.axSi.target='+str(l))
#    ffpro.write(u'ffpro.hello()')
    if verbose:
        print('prism at '+str(l))