/*  Firmware for a simple digital output unit.  Mimics a USB CDC serial
    interface device to have simple commands and be compabible with
    non-linux OS.

    USB end points for the four CDC connections:
    ep0: control, ep1in: interrupt, ep1out: data, ep8in: data

    
    Usage: Send plaintext commands, separated by newline/cr or semicolon. An
           eventual reply comes terminated with cr+lf. Parameters can be
	   separated by whitespace or by commas.

    Most important commands:

    *IDN?     Returns device identifier
    *RST      Resets device
    PORTB <value>
              sets the output port B to the 8 bit value (0..255).
    PORTB?    queries the current value of port B

    PORTD <value>
              sets the output port D to the 8 bit value (0..255).
    PORTD?    queries the current value of port D

    PORTA?    queries the current value of input port A

    HELP      returns some help text.

    ONB <value>
              switches on bit <value> of channel B
    OFFB <value>
              switches off bit <value> of channel B
    OND <value>
              switches on bit <value> of channel D
    OFFD <value>
              switches off bit <value> of channel D

   
    Port definitions:

    PA: input,  Pb, PD: output

    first code: 11.2.2015chk

*/

/* compilation control */
//#define UNKNOWN_DEBUG Yes

#define DEFAULT_LATENCY 1 /* Default timing for in command */
#define ONDELAYVALUE 100 /* start w 500msec delay before initializing */

/* do these need to show up here if they appear in the makefile? */
#include "fx2regs_c.h"

/*command requests over the management element for the cdc device */
#define CIC_SEND_ENCAPSULATED_COMMAND 0x00
#define CIC_GET_ENCAPSULATED_RESPONSE 0x01
#define CIC_SET_LINE_CODING        0x20
#define CIC_GET_LINE_CODING        0x21
#define CIC_SET_CONTROL_LINE_STATE 0x22 


/* forward declarations */
static char WriteEEPROM(int adr, __xdata char *source,int num);
static char ReadEEPROM(int adr, __xdata char *target, int num);
static void ClearEEPROM2();

/* static variables */
static __bit bitSUDAVSeen = 0; /* EP0 irq */
static __bit bitURESSeen = 0;  /*  bus reset */
static __bit bitOutdataSeen = 0;   /* out activity seen */
static __bit bitDoFlush = 0; /* Latency timer marker */
static __bit bitDoRead=0; /* read invitation in flowmode */

char configuration = 1; /* 1: full sp, 2: high sp */
char AlternateSetting = 0; /* do we have options? */

/* CDC parameters */
char LineCoding[8];

/* stuff for text input */
__xdata __at 0x1a00 char intext[256]; /* input buffer */
unsigned char inidx, innum; /* index to write to and presently stored inbytes. 
	 		   the index points no next free position in buffer */
__xdata __at 0x1b00 char outtext[256]; /* output buffer */
unsigned char outidx, outnum; /* index to write to and presently stored outbytes.
			     index points on next available char in buffer. */
__xdata __at 0x1c00 char stringbuf[10];
unsigned char  outidx2, outnum2;

/* static gobal variables for commands  */
char ep1command; /* parameter for status request */

/* generic delay */
static void SpinDelay(unsigned int count) {
    unsigned int c=count;
    while (c > 0) c -= 1;
}

#define OneNop __asm nop __endasm  
#define SYNCDELAY { OneNop; OneNop; OneNop; OneNop; } /* four clock cycles */

/* setup ports in a safe state. No other initialization is done, as this should
   take place in the PrepareCleanState command. */
static __code void initPorts() {
    IOA = 0x00;  /* default is off */
    OEA = 0x0;  /* PA is input */

    IOB = 0x00;   /* default is zero */
    OEB = 0xff;  /* port B output. */
 
    IOD = 0;    /* off */
    OED = 0xff; /* all Port D lines are output */
    
    /* prepare CTL line(s) */
    GPIFCTLCFG = 0;     /* CMOS, non-tristatable */
    GPIFIDLECTL = 0x00; /* idle state for CTL0=0 */
}


/*------------------------------------------------------------------------- */

/*  Latency timer code; runs at 1 msec */
static int current_ticks, ticks_refill;
static unsigned char zapdelay;
static unsigned char ondelay;

/* service routine timer2; */
static void isrt2(void) __interrupt (5) { /* should this use a bank? */
    TF2 = 0; /* reset timer overflow flag */
    /* checks autoflush and eventually forces a pktsend */
    current_ticks--;
    if (current_ticks == 0) { /* timer is full */
      current_ticks = ticks_refill;
      bitDoRead = 1; /* make us heared */
    }

}

/* init code; sets up timer 2 to 1msec; initializes refill values */
void latency_timer_init(int millisecs) {
    T2CON = 0; /* use clkout/12 for counter 2 and other config; counter off  */
    RCAP2L = (0xffff-4000) & 0xff;      /* 1 msec */
    RCAP2H = ((0xffff-4000) >> 8 ) & 0xff; 
    TR2 = 1; /* 4 normal enable cnt, 16bit autoreload, internal clk  */
    ticks_refill = millisecs;
    current_ticks = ticks_refill;
    ET2 = 1; /* Enable timer 2 interrupts */
}

static __code char replytext1[] = "CQT Digital IO gadget, svn-3\r\n";
//static code char replytext2[] = "Here should be some help text \r\n";
static __code char replytext3[] = "Unknown command\r\n";
//static code char replytext4[] = "Not yet implemented\r\n";
static __code char replytext5[] = "Value out of range\r\n";
static __code char replytext6[] = "Channel out of range\r\n";
static __code char replytext7[] = "Malformed float\r\n";
static __code char replytext8[] = "\r\n";
static __code char helptext[] = "Simple digital I/O gadget. Allows to use ports B and D as digital\r\noutputs, and read in port A (configured as input)\r\n\r\nUsage: Send plaintext commands, separated by newline/cr or semicolon.\r\n       An eventual reply comes terminated with cr+lf. Arguments can be\r\n       separated from parameters with space or commas or tab characters.\r\n\r\nImportant commands:\r\n\r\n*IDN?     Returns device identifier\r\n*RST      Resets device, outputs are 0V.\r\nPORTA?    Returns bit value of input port A (0..255)\r\nPORTB <value>\r\n          Sets bit value of output port B (0..255)\r\nPORTB?    Returns bit value of output port B (0..255)\r\nONB <bit>\r\n          Sets one specific bit on port B\r\nOFFB <bit>\r\n          Resets one specific bit of port B\r\nPORTD <value>, PORTD?, OND <bit>, OFFD <bit>\r\n          Similar to the port B commands but for port D\r\nHELP      Print this help text.\r\n\r\n";

#define TOKEN_NUMBER 12
__code char * __code tokenlist[] = {
  "*RST",
  "*IDN?",
  "PORTA?",
  "PORTB?",
  "PORTB",
  "PORTD?", /* 5 */
  "PORTD",
  "HELP",
  "ONB",
  "OFFB",
  "OND", /* 10 */
  "OFFD",
};

char find_token(int modlen) {
  unsigned char i; /* index into tokenlist */
  unsigned char j;
  __code char *token;
  for (i=0; i<TOKEN_NUMBER; i++) {
    token=tokenlist[i];
    
    for (j=0;j<=modlen;j++) {
      if (token[j]==0) {
	if (j==modlen) goto success; /* we got a hit */
	break; /* try next token */
      }
      if (token[j]!=outtext[(outidx+j)&0xff]) break;/* try next token */
    }
  }
  return -1; /* no success */
 success:
  return i;
}

#ifdef UNKNOWN_DEBUG
/* places a single character into the send queue */
void fill_char(char c) {
    intext[inidx]=c;
    inidx+=1; inidx &= 0xff;
}    

/* copy text from receive queue to send queue */
void fill_text(unsigned char startidx, unsigned char num) {
    unsigned char n   = num;
    unsigned char idx = startidx;
    while (n) {
	intext[inidx]=outtext[idx];
	idx++; idx &= 0xff;
	inidx++; inidx &=0xff; innum++;
	n--;
    }
}

#endif

/* sends out a value as decimal string and add decimal dot */
void fill_milli(unsigned long value) {
  unsigned long s1=value; /* keeps residue */
  char c;
  int i=0;
  /* start with last digit */
  do {
    if (i==5) {
       stringbuf[i]='.';
    } else {
      c = s1 % 10L ;
      s1 = s1/10;
      stringbuf[i]=c+'0';
    }
    i++;
  } while (((s1 != 0)|| (i<7)) && (i<10));
  stringbuf[i]=' '; i++;/* add space char */
  /* now reverse string and send to output */
  do {
    i--;
    intext[inidx]=stringbuf[i];
    inidx++; inidx &= 0xff; innum++;
  } while (i);
}

/* sends out a value as decimal string */
void fill_dec(unsigned int value) {
  unsigned int s1=value; /* keeps residue */
  char c;
  int i=0;
  /* start with last digit */
  do {
      c = s1 % 10 ;
      s1 = s1/10;
      stringbuf[i]=c+'0';
    i++;
  } while ((s1 != 0) && (i<3));
  stringbuf[i]=' '; i++;/* add space char */
  /* now reverse string and send to output */
  do {
    i--;
    intext[inidx]=stringbuf[i];
    inidx++; inidx &= 0xff; innum++;
  } while (i);
}


/* sends out a value as hex string */
void fill_hex(unsigned long value) {
  unsigned long s1=value; /* keeps residue */
  char c;
  int i=0;
  /* start with last digit */
  do {
      c = s1 &0xf ;
      s1 = s1>>4;
      if (c>9) c=c+'a'-'0'-10;
      stringbuf[i]=c+'0';    
    i++;
  } while ((s1 != 0)&&(i<5));
  stringbuf[i]=' '; i++;/* add space char */
  /* now reverse string and send to output */
  do {
    i--;
    intext[inidx]=stringbuf[i];
    inidx++; inidx &= 0xff; innum++;
  } while (i);
}

void eatwhitespace1(){
  char c;
  while (outnum2) {
    c=outtext[outidx2];
    if (c==' ' || c=='\t' || c==',' || c==',') {
     outidx2++; outnum2--;
    } else {
      break;
    }
  }
}
void eatwhitespace2(){
  char c;
  while (outnum2) {
    c=outtext[outidx2];
    if (c==' ' || c=='\t'|| c=='\r' || c=='\n' || c==';') {
     outidx2++; outnum2--;
    } else {
      break;
    }
  }
}
/* returns an integer number from instream */
int getint() {
  int u=0;
  char k=0; /* counts conversions */
  char c;
  while (outnum2) {
    c=outtext[outidx2];
    if (c<'0' || c>'9') break;
    u=10*u;
    u += (c-'0');
    k++; outidx2++; outnum2--;
  }
  if (k==0) return -1;
  return u;
}

__code char * textpointer;

/* main command parser. Interprets a command string in the out buffer */
void parse_command(){
  unsigned char modlen; /* length of remaining text in buffer, pointed to by idx */
  char c,t;
  unsigned int value; /* for timer value */

  outidx2 = outidx;
  outnum2 = outnum;
  
  /* remove leading whitespace */
  eatwhitespace2();

  if (outnum2==0) goto endparsing;
  /* find end of token */
  for (modlen=0; modlen<outnum2; modlen++) {
      c=outtext[(outidx2+modlen)&0xff];
    if (c=='*' || c=='?') continue;
    c &= 0xdf; /* uppercase characters */
    if (c>='A' && c<='Z') {
	outtext[(outidx2+modlen)&0xff]=c;
      continue;
    }
    break;
  }
  t=find_token(modlen);
  outidx2 = outidx2+modlen; outnum2 -= modlen;
  eatwhitespace1();

  switch (t) {
  case 0: /* *RST */
      /* just stop the counter and set timer interval to default */
      IOB=0; IOD=0;
      break;
  case 1: /* *IDN? */
      textpointer = replytext1;
      break;
      
  case 2: /* PORTA? */
      fill_dec(IOA);
      textpointer = replytext8;
      break;
  case 3: /* PORTB? */
      fill_dec(IOB);
      textpointer = replytext8;
      break;
  case 4: /* PORTB - set output value */
      value=getint();
      IOB=value & 0xff;
      break;
  case 5: /* PORTD? */
      fill_dec(IOD);
      textpointer = replytext8;
      break;
  case 6: /* PORTD - set output value */
      value=getint();
      IOD=value & 0xff;
      break;
  case 7: /* HELP */
      textpointer = helptext;
      break;
  case 8: /* ONB */
      value=getint();
      IOB = IOB | (1<<value);
      break;
  case 9: /* OFFB */
      value=getint();
      IOB = IOB & ~(1<<value);
      break;
  case 10: /* OND */
      value=getint();
      IOD = IOD | (1<<value);
      break;
  case 11: /* OFFD */
      value=getint();
      IOD = IOD & ~(1<<value);
      break;

  default: /* unknown command */
#ifdef UNKNOWN_DEBUG
      fill_char('>');
      //fill_text(outidx,outnum);
      fill_hex(outnum);
      tmp=outidx; 
      for (c=outnum;c;c--) {
	  fill_hex(outtext[tmp]&0xff);
	  tmp++; tmp &= 0xff;
      }
      fill_char('<');
#endif
    textpointer = replytext3;
  }
  
 endparsing:
  outidx += outnum ; outnum=0;
}

/* ------------------------------------------------------------------ */
/* Serial buffer handling */
void init_serbufs() {
  inidx=0; innum=0;
  outidx=0; outnum=0;
}

/* init CPU stuff */
static void initCPU() {
    CPUCS = bmCLKSPD48MHZ | bmCLKOE; /* output CPU clock */
    CKCON = 0;       /* have zero wait states for ext mem access */

    WAKEUPCS = bmWU | bmWU2;     /* disable wakeup machinery */
    IE = 0;   /* disable most irq */
    EIE = 0;  /* disable external irq */
    EXIF = 0; /* clear some IRQs */
    EICON = 0; /* disable resume irqs and others */
    IP = 0;    /* no high prio for most irq */

    REVCTL = bmENH_PKT | bmDYN_OUT; /* don't know if this helps. 
				       Should be after IFCONFIG??  */
}


/* at a few occasions we go for a....*/
void resetTogglebits(){
    TOGCTL = 0; TOGCTL = bmRESETTOGGLE; TOGCTL = 0; /* OUT 0 */
    /* IN 0 */
    TOGCTL = 0 | bmTOGCTL_IO; TOGCTL = 0 | bmTOGCTL_IO | bmRESETTOGGLE;
    TOGCTL = 0 | bmTOGCTL_IO;
    /* OUT 1 :*/
    TOGCTL = 1; TOGCTL = 1 | bmRESETTOGGLE; TOGCTL = 1;
    /* IN 1 : */
    TOGCTL = 1 | bmTOGCTL_IO; TOGCTL = 1 | bmTOGCTL_IO | bmRESETTOGGLE; 
    TOGCTL = 1 | bmTOGCTL_IO;
    /* IN 8 : */
    TOGCTL = 8 | bmTOGCTL_IO; TOGCTL = 8 | bmTOGCTL_IO | bmRESETTOGGLE; 
    TOGCTL = 8 | bmTOGCTL_IO;

}


/* switch on corresponding IRQ's and set data toggle bits for relevant EPs 
   for a given configuration and alternate setting */
static void initEndpoints() {
  /* use EP1 for in transfers */
  EP1INCFG =  bmVALID | bmTYPE1; /* bulk transfer, enable */
  SYNCDELAY;

  /* arm OUT endpoints */
  SUDPTRCTL = 0; /* manual */
  EP0BCH = 0; EP0BCL = 0x40; /* arm EP0 */


  EP1OUTCFG = bmVALID | bmTYPE1; /* bulk transfer */
  SYNCDELAY;

  EPIE = bmEPIE_EP1OUT; /* enable subset int 2 */
  EP1OUTBC = 0x40; /* arm EP1out */

  /* EP8 configured as input */
  FIFORESET = bmNAKALL; SYNCDELAY;
  FIFORESET = 0x08;     SYNCDELAY; /* reset FIFO in EP8 */
  FIFORESET = 0;        SYNCDELAY; /* normal op */

  EP8CFG = bmVALID | bmIN | bmBULK ; /* in endpoint */ 
  SYNCDELAY;
  EP8FIFOCFG = 0;  /* 8-bit, manual mode */
  SYNCDELAY;  	 	 

}


/* set whatever needs to be set for the USB engine before re-enumeration */
static void initUSB() {
    EP1INCFG  = bmTYPE1; /* disable for initial startup */

    /* configure EP1OUT for bulk transfer */
    initEndpoints();

    USBIRQ = 0xff; /* clear pending USB irqs */
    USBIE = bmURES | bmSUDAV;
    EUSB = 1;  /* enable USB irqs */    
}


/***********************************************************************/
/* USB utility stuff */
static void ReEnumberate() {
    USBCS &= ~bmNOSYNSOF;  /* allow synthesis of missing SOF */
    USBCS |=  bmDISCON;    /* disconnect */
    USBCS |=  bmRENUM;     /* RENUM = 1; use own device model */
    SpinDelay(0xf401);     /* wait a while for the host to detect */
    USBCS &= ~bmDISCON;    /* reconnect */
    USBCS |=  bmNOSYNSOF;  /* disallow synthesis of missing SOF */
}


/***********************************************************************/
/* usb IRQ stuff. There is some funny mixture between FX2 manual and
   SDCC interrupt labelling. Consult both manuals to verify numbering. */
static void isrUsb(void) __interrupt (8) __using (3)  {/* critical */
    if (  USBIRQ & bmSUDAV ) { /* Setup Data available */
	USBIRQ = bmSUDAV; bitSUDAVSeen = 1;
    }
    if (  USBIRQ & bmURES ) { /* USB bus reset */
	USBIRQ = bmURES; bitURESSeen = 1;
    }
    /* EP IRQ's */
    if (EPIRQ & bmEPIE_EP1OUT) {
      EPIRQ = bmEPIE_EP1OUT; bitOutdataSeen = 1; /* marker for later */
    }

    EXIF &= ~bmEXIF_USBINT; /* clear usb irq */
}

/***********************************************************************/
/* EP0 service routines */
/* Optional filler to keep the following table word-aligned */
//static __code char xuxu[]={0};
static __code char Descriptors[] = { /* only a full speed device */
    0x12, 0x01, 0x00, 0x02, 0x02, 0x00, 0x00, 0x40, // device, usb2.0,..
    0xb4, 0x04, 0x17, 0x47, 0x00, 0x00,  // cypress, test dev, rev 0

    0x01, 0x02, 0x03, 0x01, // some indx strings, 1 configuration

    0x0a, 0x06, 0x00, 0x02, 0x02, 0x00, 0x00, // device qualifier
    0x40, 0x01, 0x00,  //64 bytes pkts, 1 config

    0x09, 0x02, 0x43, 0x00, 0x02, // default config descriptor (len: 67 bytes)
    0x01, 0x00, 0x80, 0x12,  // config #0, bus power, 36 mA (bus: 01 00 80 4b)

    0x09, 0x04, 0x00, 0x00, 0x01, // interface0 (mgmt), alt set 0, #EP over 0
    0x02, 0x02, 0x00, 0x00, // if class=2 (comm), subclass=2 (ACM), proto: 1???
                            // not sure if proto 1 makes sense

    0x05, 0x24, 0x00, //Header descriptor: CS interface
    0x10, 0x01,       // bcdCDC for usb class definition for communication?

    0x04, 0x24, 0x02, // ACM descriptor: length, CS interface
    0x02,       // capabilities: Device supports
                // set_line_coding, get_line_coding, notification serial_state

    0x05, 0x24, 0x06, // union functional descriptor: CS interface
    0x00, 0x01,  // Master interface=0, slave interface0 = 1

    0x05, 0x24, 0x01,  // CM functional descriptor:  CS interface
    0x00, 0x01,  // capabilities, data interface
   
    0x07, 0x05, 0x81, 0x03, 0x10, 0x00, 0x09, // EP1in, irq, 16 byte, 9ms poll

    0x09, 0x04, //interface1 (data), length, interface descriptor
    0x01, 0x00, 0x02,    // index of if=1, altset=0, num endpoints=2
    0x0a, 0x00, 0x00, // interface class=0xa, subclass=0, protocol class=0
                    // 0x0a means data interface class
		    // no subclasses required, no class-specific data prot
    0x00,           // interface descriptor index

    0x07, 0x05, 0x01, 0x02, 0x40, 0x00, 0x00, // EP1out, bulk, 64 byte, no poll
    0x07, 0x05, 0x88, 0x02, 0x40, 0x00, 0x00, // EP8IN, bulk, 64 byte, no poll

    0x00,  // termination of descriptor list
};

//static __code char xuxu2[]={0};
static __code char Descriptors2[] = { /* high speed device */
    0x12, 0x01, 0x00, 0x02, 0x02, 0x00, 0x00, 0x40, // device, usb2.0,..
    0xb4, 0x04, 0x17, 0x47, 0x00, 0x00,  // cypress, test dev, rev 0

    0x01, 0x02, 0x03, 0x01, // some indx strings, 1 configuration

    0x0a, 0x06, 0x00, 0x02, 0x02, 0x00, 0x00, // device qualifier
    0x40, 0x01, 0x00,  //64 bytes pkts, 1 config

    0x09, 0x02, 0x43, 0x00, 0x02, // default config descriptor (len: 67 bytes)
    0x01, 0x00, 0x80, 0x12,  // config #0, self power, 36 mA (bus: 01 00 80 4b)

    0x09, 0x04, 0x00, 0x00, 0x01, // interface0 (mgmt), alt set 0, #EP over 0
    0x02, 0x02, 0x00, 0x00, // no strings

    0x05, 0x24, 0x00, //Header descriptor: CS interface
    0x10, 0x01,       // bcdCDC for usb class definition for communication?

    0x04, 0x24, 0x02, // ACM descriptor: length, CS interface
    0x02,       // capabilities: Device supports
                // set_line_coding, get_line_coding, notification serial_state

    0x05, 0x24, 0x06, // union functional descriptor: CS interface
    0x00, 0x01,  // Master interface=0, slave interface0 = 1

    0x05, 0x24, 0x01,  // CM functional descriptor:  CS interface
    0x00, 0x01,  // capabilities, data interface
   
    0x07, 0x05, 0x81, 0x03, 0x10, 0x00, 0x09, // EP1in, irq, 16 byte, 9ms poll

    0x09, 0x04, //virtual com data interface, length, interface descriptor
    0x01, 0x00, 0x02,    // index of if=1, altset=0, num endpoints=2
    0x0a, 0x00, 0x00, // interface class=0xa, subclass=0, protocol class=0
                    // 0x0a means data interface class
		    // no subclasses required, no class-specific data prot
    0x00,           // interface descriptor index

    0x07, 0x05, 0x01, 0x02, 0x00, 0x02, 0x00, // EP1out, bulk, 64 byte, no poll
    0x07, 0x05, 0x88, 0x02, 0x00, 0x02, 0x00, // EP8IN, bulk, 521 byte, no poll

    0x00,  // termination of descriptor list
};


/* some filler to keep stuff word-aligned */
//static __code char xuxu3[]= {0};  
static __code char StringDescriptors[] = { /* table for strings */
    0x04, 0x03, 'l',0,  // dummy but read??
    0x40, 0x03, 'C',0, 'e',0, 'n',0, 't',0, 'r',0, 'e',0, ' ',0, 'f',0, 
                'o',0, 'r',0, ' ',0, 'Q',0, 'u',0, 'a',0, 'n',0,
                't',0, 'u',0, 'm',0, ' ',0, 'T',0, 'e',0, 'c',0, 'h',0,
                'n',0, 'o',0, 'l',0, 'o',0, 'g',0, 'i',0, 'e',0, 's',0,
                // Manufacturer

    0x26, 0x03, 'D',0, 'i',0, 'g',0, 'i',0, 't',0, 'a',0, 'l',0, ' ',0,
                'I',0, '/',0, 'O',0, ' ',0, 'G',0, 'a',0, 'd',0, 'g',0,
                'e',0, 't',0,

    0x10, 0x03, 'D',0, 'G',0, '-',0, 'Q',0, 'O',0,
                '0',0, '2',0, //Serial number

    0x00, // termination of descriptor list

};


/* EP0 service routines */
static void ctrlGetStatus() {
    unsigned char a;
    SUDPTRCTL=1; /* simple send...just don't use SUDPTRL */
    EP0BCH = 0x00; EP0BUF[1] = 0x00; /* same for all requests */
    switch (SETUPDAT[0]) { /* bmRequest */
	case 0x80: // IN, Device (Remote Wakeup and Self Powered Bit)
	    EP0BUF[0] = 0x00; /* no Remote Wakeup, bus-powerd Device (self:1) */
            EP0BCL    = 0x02; /* 2 bytes */
	    break;
	case 0x81: // IN, Get Status / Interface
	    EP0BUF[0] = 0x00;
            EP0BCL    = 0x02; /* 2 bytes */
	    break;
	case 0x82: // IN, Endpoint (Stall Bits)
	    switch (SETUPDAT[4] & 0xf) { /* extract number */
		case 0: a=EP0CS; break;
		case 1: a=(SETUPDAT[4]&0x80)?EP1INCS:EP1OUTCS; break;
	        case 8: a=EP8CS; break;
		default: a=1; break; /* or better Stall? */
	    }
	    EP0BUF[0] = a & 1; /* return stall bit or 1 in case of err */
            EP0BCL = 0x02; /* 2 bytes */
	    break;
	default:  /* STALL indicating Request Error */
	    EP0CS = bmEPSTALL; 
	    break;
    }
}

/* combines clear or set feature; v=0: reset, v=1: set feature */
static void ctrlClearOrSetFeature(char v) {
    char a; /* to hold endpoint */
    switch (SETUPDAT[0]) { /* bmRequest */
	case 0x00: // Device Feature (Remote Wakeup)
	    if (SETUPDAT[2] != 0x01) { /* wValueL */
		EP0CS = bmEPSTALL;
	    }
	    break;
	case 0x02: // Endpoint Feature (Stall Endpoint)
	    if (SETUPDAT[2] == 0x00) { /* clear stall bit */
		a=SETUPDAT[4] & 0xf;
		switch (a) {
		    case 0: EP0CS=v; break;
		    case 1: 
			if (SETUPDAT[4] & 0x80) {
			    EP1INCS=v; 
			} else { 
			    EP1OUTCS =v;
			}
			break;
		    case 8:
		        EP8CS=v;
		        break;
		}
		/* in case of set feature clear toggle bit */
		if (v) { 
		    if (SETUPDAT[4] & 0x80) a |=bmTOGCTL_IO; /* set dir */
		    /* back to data stage 0 */
		    TOGCTL = a; TOGCTL = a | bmRESETTOGGLE; TOGCTL = a;
		}
		break;
	    } /* else follow stall... */ 
	default: 
	    EP0CS = bmEPSTALL; break;
    }
}
static void ctrlGetDescriptor() {
    char key   = SETUPDAT[3]; /* wValueH */
    char index = SETUPDAT[2]; /* wValueL */
    char count = 0;
    char seen = 0; /* have seen a string */
    static __code char *current_DP;
    current_DP = (USBCS & bmHSM)? Descriptors2:Descriptors; 
    /* try to make other speed config */
    if (key==7) {
      current_DP = (USBCS & bmHSM)? Descriptors:Descriptors2;
      key=2; /* go into 'retrieve configuration' state */
    }
    if (key==3) { /* get string table */
      current_DP = StringDescriptors;
    }
    SUDPTRCTL = bmSDPAUTO; /* allow for automatic device reply */
    for (; current_DP[0]; current_DP += current_DP[0])
	if ((current_DP[1] == key) && (count++ == index)) {
	    SUDPTRH = (char)(((unsigned int)current_DP)>>8)&0xff;
	    SUDPTRL = (char)( (unsigned int)current_DP    )&0xff;
	    seen=1;
	    break;
	}
      
    if (!seen) EP0CS = bmEPSTALL; /* did not find descriptor */
}


static void ctrlGetConfiguration() {
  SUDPTRCTL=1; /* simple send */
  EP0BUF[0] = configuration;
  EP0BCH = 0x0; EP0BCL = 0x1; /* 1 byte */
}

static void ctrlSetConfiguration() {
    if (SETUPDAT[2] & 0xfe) { /* not config 0 or 1 */
	EP0CS = bmEPSTALL;
    } else {
	configuration = SETUPDAT[2];
	resetTogglebits;
    }
}

static void ctrlGetInterface() {
  EP0BUF[0] = AlternateSetting;
  SUDPTRCTL=1; /* simple send */
  EP0BCH = 0x00; EP0BCL = 0x01; /* 1 byte */
}

static void ctrlSetInterface() {
    if (SETUPDAT[2] & 0xfe) { /* not config 0 or 1 */
	EP0CS = bmEPSTALL;
    } else {
        AlternateSetting=SETUPDAT[2];
	initEndpoints(); /* switch on/off end points */
    }
}

/* EP0 setup commands */
static void doSETUP() {
  unsigned char u;
  switch (SETUPDAT[0]& 0x60) { /* separate standard usb from class req */
  case 0: /* we have a standard usb request */
    switch  (SETUPDAT[1]) { /* bRequest */
	case 0x00: ctrlGetStatus();         break;
	case 0x01: ctrlClearOrSetFeature(0);      break; /* clear */
	    /*case 0x02: EP0CS = bmEPSTALL;       break; */
	case 0x03: ctrlClearOrSetFeature(1);        break; /* set */
	    /* case 0x04: EP0CS = bmEPSTALL;       break;  reserved */
	/* case 0x05:  SetAddress */
	case 0x06: ctrlGetDescriptor();     break;
	    /*  case 0x07:   SetDescriptor     break; */
	case 0x08: ctrlGetConfiguration();  break;
	case 0x09: ctrlSetConfiguration();  break;
	case 0x0a: ctrlGetInterface();      break;
	case 0x0b: ctrlSetInterface();      break;
	/* case 0x0c:  Sync: SOF           break; */ 
	default: EP0CS = bmEPSTALL ;         break;
    }
    break;
  case 0x20: /* class specific request: terminal control commands */
    switch  (SETUPDAT[1]) { /* bRequest */
    case CIC_SET_LINE_CODING: 
      EUSB=0; /* temporarily disable irq */
      SUDPTRCTL = 0x01;
      EP0BCL = 0x00;
      SUDPTRCTL = 0x00;
      EUSB = 1; /* some dirty trick, not sure if I understand */
      while (EP0BCL != 7);
      u=7; do {u--; LineCoding[u]= EP0BUF[u]; } while (u);    
      break;
    case CIC_GET_LINE_CODING:
      SUDPTRCTL=1; /* simple send */
      u=7; do {u--; EP0BUF[u] = LineCoding[u]; } while (u);
      EP0BCH = 0x0; EP0BCL = 0x7;
      while (EP0CS & 0x02); SUDPTRCTL=0;
      break;
    case CIC_SEND_ENCAPSULATED_COMMAND:
    case CIC_SET_CONTROL_LINE_STATE: /* ignore content */
      break;
    default: EP0CS = bmEPSTALL ;
    }
    break;
  }
  EP0CS = bmHSNAK; /* close hs phase */
  bitSUDAVSeen = 0;
}


/* ------------------------------------------------------------------- */
/* this is the notification element for the CDC. It replies status line
   information. At the moment, DSR/DTR lines are not implemented. */
static void fill_statusEP() {
  EP1INBUF[0]=10;               /* length */
  EP1INBUF[1]=0x20;             /* SERIAL_STATE notification */
  EP1INBUF[2]=0; EP1INBUF[3]=0; /* value */
  EP1INBUF[4]=0; EP1INBUF[5]=0; /* interface */
  EP1INBUF[6]=2; EP1INBUF[7]=0; /* length of data */
  EP1INBUF[8]=0; EP1INBUF[9]=0; /* actual line data */
  
  EP1INBC=10; /* commit package */

}

/* fill EP8in with data */
static void fill_dataEP() {
    unsigned char sendbytes; /* some buffer */
    if (EP2468STAT & bmEP8FULL) return; /* all buffers committed, wait */

    /* fill send buffer */
    sendbytes = 0;

    while (sendbytes<64 && innum>0) {
      EP8FIFOBUF[sendbytes] = intext[(inidx - innum)&0xff];
      innum--; sendbytes++;
    }

    /* commit packet */
    EP8BCH=0; SYNCDELAY; EP8BCL=sendbytes; SYNCDELAY;
}

#ifdef EEPROM

/* read EEprom into variable. return value is  0 on success, 1 on busy
   and <0 on error. does not work yet*/ 
static char ReadEEPROM(int adr, __xdata char *target, int num) {
  char z;

  z=5;
  while (I2CS & bmI2C_STOP); /* wait until we can have bus */
  do {
    I2CS = bmI2C_START; /* start engine */
    if ((I2CS & bmI2C_BERR) == bmI2C_BERR) goto bla;
    
    I2DAT = 0xa2; while (!(I2CS & bmI2C_DONE));
    
    if ((I2CS & bmI2C_BERR) != bmI2C_BERR) break;
  bla:
    z--;
    SpinDelay(10000);
  } while (z);
  if (z==0) return -6;

  /* check for ack condition */
  if ((I2CS & (bmI2C_BERR | bmI2C_ACK)) == 0) {
      I2CS = bmI2C_STOP;
      return 1;
  }
  if ((I2CS & bmI2C_BERR) == bmI2C_BERR) {return -4;}

  I2DAT = (adr >>8); while (!(I2CS & bmI2C_DONE));
  if ((I2CS & bmI2C_BERR) == bmI2C_BERR) {return -3;}

  I2DAT = adr;  while (!(I2CS & bmI2C_DONE));
  if ((I2CS & bmI2C_BERR) == bmI2C_BERR) {return -2;}
  I2CS = bmI2C_START;

  I2DAT = 0xa3; /* switch over to read */
  while (!(I2CS & bmI2C_DONE)); 
  if (num==1) I2CS=bmI2C_LASTRD;
  z=I2DAT; /* dummy read to initiate sequence */

  while (num) {
    if (num<2) I2CS=bmI2C_LASTRD;
    while (!(I2CS & bmI2C_DONE)) {
      if ((I2CS & bmI2C_BERR) == bmI2C_BERR) {
      I2CS = bmI2C_STOP;
      return -1;	
      }
    }
    *target = I2DAT;
    target++;
    num--;
  }
  I2CS = bmI2C_STOP;
  return 0;
}

/* write EEPROM page, param: addr. Data is in xdata space . return value is
   0 on success, 1 on busy and -1 on error. */
static char WriteEEPROMPage(int adr, __xdata char *source) {
  char z;
  while (I2CS & bmI2C_STOP); /* wait until we can have bus */
  I2CS = bmI2C_START; /* start engine */
  I2DAT = 0xa2; while (!(I2CS & bmI2C_DONE));
  /* check for ack condition */
  if ((I2CS & (bmI2C_BERR | bmI2C_ACK)) != bmI2C_ACK) {
      I2CS = bmI2C_STOP;
      return 1;
  }
  I2DAT = (adr >>8); while (!(I2CS & bmI2C_DONE));
  I2DAT = adr; while (!(I2CS & bmI2C_DONE));

  for (z=0;z<32;z++) { /* visit first few bytes */
    I2DAT = source[z];
    while (!(I2CS & bmI2C_DONE)); /* Wait for done */
    if ((I2CS & (bmI2C_BERR | bmI2C_ACK)) != bmI2C_ACK) {
      I2CS = bmI2C_STOP;
      return -1;
    }
  }
  I2CS = bmI2C_STOP;
  return 0;
}

/* write EEPROM range*/
static char WriteEEPROM(int adr, __xdata char *source,int num) {
  /* just assume 32 byte aligned the data */
  char c;
  while(num>0) {
    c=WriteEEPROMPage(adr, source);
    if (c==1) continue; /* loop until no ack */
    if (c) return c; /* some error msg */
    c=32; adr +=c; source +=c; num -=c;
  }
  return 0;
}

/* some code to wipe out the flash rom in case of an emergency */
__xdata __at 0x18f0 int zerolist; /* only first two bytes are zeroed */
static void ClearEEPROM() {
  zerolist=0;
  WriteEEPROMPage(0,(__xdata char *)(&zerolist)); /* this avoids initialization code */
}

#endif

/* some code to wipe out the flash rom in case of an emergency */
static void ClearEEPROM2() {
    unsigned char z;
       I2CS = bmI2C_START; /* start engine */
       I2DAT = 0xa2; while (!(I2CS & bmI2C_DONE));
       for (z=0;z<10;z++) { /* visit first few bytes */
	I2DAT = 0;
	while (!(I2CS & bmI2C_DONE)); /* Wait for done */
	if ((I2CS & (bmI2C_BERR | bmI2C_ACK)) != bmI2C_ACK) break; /* stop */
    }
    I2CS = bmI2C_STOP;
 
}

/* Receives input bytes into buffer for parsing and watches for start processing
   characters (line feed, semicolon)
*/
void swallow_data() {
  unsigned char bytecount, idx;
  char c; /* buffer */

  bytecount = EP1OUTBC;

  idx=0; //1; 
  while (idx<bytecount) {
    c = EP1OUTBUF[idx];
    outtext[(outidx+outnum)&0xff] = c;
    idx++; outnum++;
    if (c == '\r' || c== '\n' || c == ';') { /* we need to do something */
      parse_command();
    }
  }
  /* re-arm input */
  EP1OUTBC = 0x40;

  bitOutdataSeen=0;
}


void main() {
  char c=0;

    initPorts(); /* initialize out port */
    initCPU();   /* initialize CPU stuff */
    configuration = 0;
    initUSB();          /* initialize USB machine */

    init_serbufs();     /* initialize text buffers */
    textpointer = 0;

    /* load the delay table */
    latency_timer_init(DEFAULT_LATENCY); /* initialize latency timer */
    
    EA = 1; /* enable irqs */
    ReEnumberate();

    /* ------------ here we should have reached an idle state ------- */

    /* main loop: wait forever */
    for (;;) {
	if (bitSUDAVSeen) doSETUP();  /* Handle SUDAV Events */
	
	if (bitOutdataSeen) swallow_data(); /* Process command on EP1 */

	if (bitDoFlush || innum >62) fill_dataEP();  /* return accumulated text */

	/* see if we need to reload EP1 register */
	if (!(EP01STAT & bmEP1INBSY)) {
	  fill_statusEP();
	}
	/* see if there is text to submit */
	if (textpointer) {
	  while (innum<200 && (c=*textpointer)) {
	    intext[inidx]=c;
	    inidx++; inidx &= 0xff; innum++; textpointer++;
	  }
	  if (c==0) { textpointer=0; bitDoFlush=1; }
	}
    }    
}
