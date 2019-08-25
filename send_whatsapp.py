# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:55:50 2019

@author: Millie
"""

from twilio.rest import Client

client = Client('AC3426974cd8a9c703d34e9350311025ea','18d8c1da4c2e6160142574a38e5543d5')

def send_whatsapp_message(body,number='+6584231353'):
    client.messages.create(body=body,from_='whatsapp:+14155238886',to='whatsapp:%s'%number)