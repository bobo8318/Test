# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 19:59:05 2018

@author: My
"""

import requests
import random, datetime

def ident_generator():
    sheng = ('11', '12', '13', '14', '15', '21', '22', '23', '31', '32', '33', '34', '35', '36', '37', '41', '42', '43', '44', '45', '46', '50', '51', '52', '53', '54', '61', '62', '63', '64', '65', '66')
    birthdate = (datetime.date.today() - datetime.timedelta(days = random.randint(7000, 25000)))
    ident = sheng[random.randint(0, 31)] + '0101' + birthdate.strftime("%Y%m%d") + str(random.randint(100, 199))
    coe = {1: 7, 2: 9, 3: 10, 4: 5, 5: 8, 6: 4, 7: 2, 8: 1, 9: 6, 10: 3, 11:7, 12: 9, 13: 10, 14: 5, 15: 8, 16: 4, 17: 2}
    summation = 0
    for i in range(17):
        summation = summation + int(ident[i:i + 1]) * coe[i+1]#ident[i:i+1]
    key = {0: '1', 1: '0', 2: 'X', 3: '9', 4: '8', 5: '7', 6: '6', 7: '5', 8: '4', 9: '3', 10: '2'}
    return ident + key[summation % 11]
     
url = "http://218.92.108.33:5300/OpenCustomer/GetCardByIDCardNo?idCardNo="

for i in range(1,10):
    idcard = ident_generator()
    print
    response = requests.get(url+idcard)
    print(response.text)
    