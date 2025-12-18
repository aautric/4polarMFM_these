# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 10:41:57 2025

@author: LOCCO_Louise
"""

import tifffile as tiff
import numpy as np

sqr1 = [[100,200],[50,150]]
sqr2 = [[320,420],[50,150]]
sqr3 = [[100,200],[200,300]]
sqr4 = [[320,420],[200,300]]
sqr5 = [[100,200],[390,490]]
sqr6 = [[320,420],[390,490]]

sqrzero = [[0,20],[0,200]]

def open_ome(path):
    data = tiff.imread(path)
    if len(data.shape)==3:
        data = np.mean(data, axis=0)
    return data

def extract_sqr(data):
    s1 = data[sqr1[0][0]:sqr1[0][1],sqr1[1][0]:sqr1[1][1]]
    s2 = data[sqr2[0][0]:sqr2[0][1],sqr2[1][0]:sqr2[1][1]]
    s3 = data[sqr3[0][0]:sqr3[0][1],sqr3[1][0]:sqr3[1][1]]
    s4 = data[sqr4[0][0]:sqr4[0][1],sqr4[1][0]:sqr4[1][1]]
    s5 = data[sqr5[0][0]:sqr5[0][1],sqr5[1][0]:sqr5[1][1]]
    s6 = data[sqr6[0][0]:sqr6[0][1],sqr6[1][0]:sqr6[1][1]]
    return np.mean(s1), np.mean(s2), np.mean(s3), np.mean(s4), np.mean(s5), np.mean(s6)

def zero(file):
    data = open_ome(file)
    sss = data[sqrzero[0][0]:sqrzero[0][1],sqrzero[1][0]:sqrzero[1][1]]
    return np.mean(sss)

def extract_calibr(file1):
    data = open_ome(file1)
    s1, s2, s3, s4, s5, s6 = extract_sqr(data)
    res1, res2, res3, res4, res5, res6 = [],[],[],[],[],[]
    res1.append(s1)
    res2.append(s2)
    res3.append(s3)
    res4.append(s4)
    res5.append(s5)
    res6.append(s6)
    for k in range(1,36):
        new_path = file1.rsplit("_0", 1)[0] + "_" + str(k) + file1.rsplit("_0", 1)[1]
        data = open_ome(new_path)
        s1, s2, s3, s4, s5, s6 = extract_sqr(data)
        res1.append(s1)
        res2.append(s2)
        res3.append(s3)
        res4.append(s4)
        res5.append(s5)
        res6.append(s6)
    return res1, res2, res3, res4, res5, res6

def polar_power(phi, delta, ex, ey):
    return (ex*np.cos(phi)+ey*np.sin(phi)*np.cos(delta))**2 + (ey*np.sin(phi)*np.sin(delta))**2