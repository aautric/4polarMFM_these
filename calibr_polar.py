# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 10:41:57 2025

@author: LOCCO_Louise
"""

import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

def extract_sqr(data, plot=True):
    squares = [sqr1, sqr2, sqr3, sqr4, sqr5, sqr6]

    means = []
    for sqr in squares:
        roi = data[sqr[0][0]:sqr[0][1], sqr[1][0]:sqr[1][1]]
        means.append(np.mean(roi))

    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(data, cmap="gray", origin="upper")

        for i, sqr in enumerate(squares):
            y0, y1 = sqr[0]
            x0, x1 = sqr[1]

            rect = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
            ax.add_patch(rect)

            ax.text(
                x0,
                y0 - 5,
                str(i),
                color="yellow",
                fontsize=12,
                weight="bold",
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),
            )

        ax.set_title("Selected ROIs")
        plt.show()

    return tuple(means)

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