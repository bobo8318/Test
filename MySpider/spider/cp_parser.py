# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 07:12:05 2018

@author: My
"""
import numpy as np
class Cp_parser(object):
    def _init_(self,html_content):
        self.out_data = []
  
    def parseTd(self,soup):
        if(soup.table is not None):
            trnodes = soup.table.find_all("tr")
            for trnode in trnodes:
                    if trnode.td is not None:
                        tdnodes = trnode.find_all("td")
                        cpno = tdnodes[0].string
                        cpdate = tdnodes[1].string
                        redballs = trnode.find_all("td", class_="redColor",limit=6)
                        redball_str = "";
                        for redball in redballs:
                            redball_str += redball.string+","
                        blueball = trnode.find("td", class_="blueColor").string
                        redball_str += blueball
                        td_data = [cpno,cpdate,redball_str]
                        print(td_data)
                    #self.out_data.append(td_data)
                    
    def out_data(self):
        return self.out_data