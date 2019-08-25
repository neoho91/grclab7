This is some very simple code to use the Cypress FX2 chip as a digital I/O
port where individual bits of two ports can be set/reset, and a third port can
be used for digital inputs.

The target device is the mini USB board that hosts one FX2 chip, and a EEPROM
to keep the code.

The device emulates a CDC modem device, and should be usable with any OS
without any fancy driver.

LINUX: On a Linux system, the device shows up as a /dev/ttyACMx, or as a link
to it in /dev/serial/by-id/<some meaningful identifier>. Once the device is
plugged in, it may take a while for the device to show up, because some smart
modem mananging service may want to talk to it. In this case, it sends stuff
to the device, and the first read will return an "Unknown command"


WINDOWS: Apparently Windows systems need to make the device known to the
system using the .inf file. Then, the device should show up as a virtual com
port.


Issues:
The device does not honor the suspend/restart USB code as required per USB
specificiation. This may be the reason for having issues when rebooting the
host computer and keeping the device plugged in. The problem can be solved by
re-plugging it after a reboot of the computer. I will look into this at some
point.

State of this document: 11.2.2015, svn-2, c kurtsiefer


