#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:21:10 2026

@author: amaury, juliette
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os
from PIL import Image
import tifffile
import tkinter as tk
N = 9
omega = np.linspace(0, 2*np.pi, N+1)[:-1]

def rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])
    
def rotation_elt_opt(element, omega):
    rot1 = rotation(omega)
    rot2 = rotation(-omega)
    return rot1@element@rot2
look_up_folder = '/mnt/d/Amaury/DATA/'
polars = [0,40,80,120,160,200,240,280,320]
path_raw = ['/mnt/d/Amaury/DATA/2026_07_17_calibr_polar_analyser/nopolar/'+str(i)+'/images/RAW_DATA/image_Pos0.ome.tif' for i in polars]
path_raw2 = ['/mnt/d/Amaury/DATA/2026_07_20_calibr_polar_analyser_before_wollaston/polarY/'+str(i)+'/images/RAW_DATA/image_Pos0.ome.tif' for i in polars]

#%%
raw = []
raw2 = []
for i, pol in enumerate(polars):
    with tifffile.TiffFile(path_raw[i]) as tif:
             raw.append(np.mean(tif.asarray(), axis=0))
for i, pol in enumerate(polars):
    with tifffile.TiffFile(path_raw2[i]) as tif:
             raw2.append(np.mean(tif.asarray(), axis=0))
raw2 = np.array(raw2)
raw = np.array(raw)
#%%

carre1 = [[20,100],[100,180]]
carre2 = [[20,320],[100,390]]
carre3 = [[220,100],[310,180]]
carre4 = [[220,320],[310,390]]
carre5 = [[420,100],[500,180]]
carre6 = [[420,320],[500,390]]

carreA = [[20,290],[70,340]]
carreB = [[220,290],[300,340]]
carreC = [[450,290],[500,340]]
#%%
I1 = np.mean(raw[:,carre1[0][1]:carre1[1][1],carre1[0][0]:carre1[1][0]], axis=(1,2))
I2 = np.mean(raw[:,carre2[0][1]:carre2[1][1],carre2[0][0]:carre2[1][0]], axis=(1,2))
I3 = np.mean(raw[:,carre3[0][1]:carre3[1][1],carre3[0][0]:carre3[1][0]], axis=(1,2))
I4 = np.mean(raw[:,carre4[0][1]:carre4[1][1],carre4[0][0]:carre4[1][0]], axis=(1,2))
I5 = np.mean(raw[:,carre5[0][1]:carre5[1][1],carre5[0][0]:carre5[1][0]], axis=(1,2))
I6 = np.mean(raw[:,carre6[0][1]:carre6[1][1],carre6[0][0]:carre6[1][0]], axis=(1,2))

IA = np.mean(raw2[:,carreA[0][1]:carreA[1][1],carreA[0][0]:carreA[1][0]], axis=(1,2))
IB = np.mean(raw2[:,carreB[0][1]:carreB[1][1],carreB[0][0]:carreB[1][0]], axis=(1,2))
IC = np.mean(raw2[:,carreC[0][1]:carreC[1][1],carreC[0][0]:carreC[1][0]], axis=(1,2))

#%%
#calcul des coefs de Fourier
def polar_analysis(I_, plot=False):
    P4 = np.ones(N)
    Q2 = np.ones(N)
    Q4 = np.ones(N)
    for p in range(N):
        P4[p] = I_[p]*np.cos(4*omega[p])
        Q2[p] = I_[p]*np.sin(2*omega[p])
        Q4[p] = I_[p]*np.sin(4*omega[p])
    p_0 = (1/N)*sum(I_)
    q_2  = (2/N)*sum(Q2)
    q_4 = (2/N)*sum(Q4)
    p_4 = (2/N)*sum(P4)
    
    #calcul des angles 
    alpha = 0.5*np.arctan2(-q_4,-p_4)
    
    epsilon = 0.5*np.arctan(q_2*np.sin(2*alpha)/(2*q_4))
    
    #print(f"On obtient α = {alpha*180/np.pi} et ε = {epsilon*180/np.pi}")
    if plot:
        # demi-axes de l'ellipse
        a = np.cos(epsilon)
        b = np.sin(epsilon)
        
        # Tracé paramétrique 
        t = np.linspace(0, 2 * np.pi, 500)
        x = a * np.cos(t) * np.cos(alpha) - b * np.sin(t) * np.sin(alpha)
        y = a * np.cos(t) * np.sin(alpha) + b * np.sin(t) * np.cos(alpha)
        X = np.linspace(-1, 1, 500)
        
        plt.figure(figsize=(5, 5))
        plt.plot(x, y)
        if abs(epsilon)>1e-16:
            if epsilon<0:# position de la flèche sur la courbe
                i = 400
            if epsilon>=0:
                i = 100  # position de la flèche sur la courbe
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            plt.annotate('', xy=(x[i]+dx, y[i]+dy), xytext=(x[i], y[i]),
                         arrowprops=dict(arrowstyle='->', color='red', lw=2))
            plt.plot(X*np.cos(alpha), np.sin(alpha)*X)
            plt.plot(X*np.cos(alpha-np.pi/2), np.sin(alpha-np.pi/2)*X)
            plt.axhline(0, color='gray', linewidth=0.5)
            plt.axvline(0, color='gray', linewidth=0.5)
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            plt.title(f'Ellipse de polarisation  α={alpha*180/np.pi}°, ε={epsilon*180/np.pi}°')
            plt.xlabel('Ex')
            plt.ylabel('Ey')
            plt.tight_layout()
            plt.show()
    phi = np.arctan2(np.sin(2*epsilon), np.cos(2*epsilon)*np.sin(2*alpha))
    Ex = np.sqrt(0.5*(1+np.cos(2*epsilon)*np.cos(2*alpha)))
    Ey = np.sqrt(0.5*(1-np.cos(2*epsilon)*np.cos(2*alpha)))
    return epsilon, alpha, Ex, Ey, phi

#%%

epsilon, alpha, Ex, Ey, phi = polar_analysis(IB, plot=True)
print(epsilon, alpha, Ex, Ey, phi)

#%%

A = 0.8
B = 0.62
C = 0.59
D = 0.79
phiC = -0.23
phiDB = 2.98
phiB = np.linspace(0,2*np.pi, 100)
func = phiC+np.arctan2((D*np.sin(phiDB+phiB-phiC)),(C+D*np.cos(phiDB+phiB-phiC)))-np.arctan2((B*np.sin(phiB)),(A+B*np.cos(phiB)))
plt.plot(phiB, func)
plt.axhline(1.1)
plt.axvline(3.82)
plt.grid()
plt.show()
phiB = 3.82
phiD = phiDB+phiB
#%%
J1 = rotation(0*np.pi/180)@np.array([[A, B*np.exp(1j*phiB)],[C*np.exp(1j*phiC), D*np.exp(1j*phiD)]])
phia = np.angle(J1[0,0])
J1 = np.exp(-1j*phia)*J1
print(J1)
print(np.abs(J1[0,0])**2+np.abs(J1[0,1])**2)
print(np.abs(J1[1,0])**2+np.abs(J1[1,1])**2)
#%%
A = 0.38
B = 0.93
C = 0.92
D = 0.36
phiC = 2.49
phiDB = -0.55
phiB = np.linspace(0,2*np.pi, 100)
func = phiC+np.arctan2((D*np.sin(phiDB+phiB-phiC)),(C+D*np.cos(phiDB+phiB-phiC)))-np.arctan2((B*np.sin(phiB)),(A+B*np.cos(phiB)))
plt.plot(phiB, func)
plt.axhline(-1.05+2*np.pi)
plt.axvline(3.45)
plt.grid()
plt.show()
phiB = 3.45
phiD = phiDB+phiB

#%%
J2 = rotation(0*np.pi/180)@np.array([[A, B*np.exp(1j*phiB)],[C*np.exp(1j*phiC), D*np.exp(1j*phiD)]])
phia = np.angle(J2[0,0])
J2 = np.exp(-1j*phia)*J2
print(J2)
print(np.abs(J2[0,0])**2+np.abs(J2[0,1])**2)
print(np.abs(J2[1,0])**2+np.abs(J2[1,1])**2)
#%%
'''
# version -6
J1 = np.array(
    [[ 0.73570412-4.66978868e-20j ,-0.55993755-4.17129978e-01j],
     [ 0.65226294-1.46261776e-01j , 0.63913023+3.35387942e-01j]]
)

J2 = np.array([[ 0.30704435-1.92859416e-18j ,-0.95271127-9.24685391e-02j],
 [-0.64786141+6.90506597e-01j ,-0.22848831+1.61505073e-01j]]
)

#version 6
J1 = np.array(
    [[ 0.85578077-3.21326405e-19j ,-0.40253629-3.52810255e-01j],
 [ 0.48982534-1.25739435e-01j , 0.72637384+4.40874695e-01j]]
)

J2 = np.array([[ 0.45810932-7.52974492e-19j, -0.80096897-3.94932188e-01j],
 [-0.7528251 +4.62806894e-01j, -0.44382108-3.54001770e-04j]]
)
''' 
J1_fit = np.array([[ 0.77294344        ,             -0.37847298 + 1j*  -0.5097466 ],[
      -0.24436265 + 1j*  0.58565116  ,   -0.7503899 + 1j*  0.18626373 ]])
J2_fit = np.array([[ 0.22273345               ,      -0.8014731 + 1j*  -0.55417156 ],[
      0.48960716 + 1j*  0.84284395   ,  -0.017539864 + 1j*  0.2226117 ]])

#%%
for u in np.linspace(90, 180, 9):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    for ax, (J, label) in zip(axes, [(J1, 'J1'), (J1_fit, 'J1_fit')]):
        value = J @ rotation(u*np.pi/180) @ np.array([1, 0])
        Ex = np.abs(value[0])
        Ey = np.abs(value[1])
        phi = np.angle(value[1]) - np.angle(value[0])
        epsilon = 0.5 * np.arcsin((2*Ex*Ey*np.sin(phi)) / (Ex**2 + Ey**2))
        alpha = 0.5 * np.arctan2(2*Ex*Ey*np.cos(phi), Ex**2 - Ey**2)

        a = np.cos(epsilon)
        b = np.sin(epsilon)

        t = np.linspace(0, 2*np.pi, 500)
        x = a*np.cos(t)*np.cos(alpha) - b*np.sin(t)*np.sin(alpha)
        y = a*np.cos(t)*np.sin(alpha) + b*np.sin(t)*np.cos(alpha)
        X = np.linspace(-1, 1, 500)

        ax.plot(x, y)
        if abs(epsilon) > 1e-16:
            i = 400 if epsilon < 0 else 100
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            ax.annotate('', xy=(x[i]+dx, y[i]+dy), xytext=(x[i], y[i]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.plot(X*np.cos(alpha), np.sin(alpha)*X, 'b--', alpha=0.5)
        ax.plot(X*np.cos(alpha-np.pi/2), np.sin(alpha-np.pi/2)*X, 'g--', alpha=0.5)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{label} | input={u:.1f}°\nα={alpha*180/np.pi:.1f}°, ε={epsilon*180/np.pi:.1f}°')
        ax.set_xlabel('Ex')
        ax.set_ylabel('Ey')

    plt.tight_layout()
    plt.show()
#%%
for u in np.linspace(90, 180, 9):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    for ax, (J, label) in zip(axes, [(J2, 'J2'), (J2_fit, 'J2_fit')]):
        value = J @ rotation(u*np.pi/180) @ np.array([1, 0])
        Ex = np.abs(value[0])
        Ey = np.abs(value[1])
        phi = np.angle(value[1]) - np.angle(value[0])
        epsilon = 0.5 * np.arcsin((2*Ex*Ey*np.sin(phi)) / (Ex**2 + Ey**2))
        alpha = 0.5 * np.arctan2(2*Ex*Ey*np.cos(phi), Ex**2 - Ey**2)

        a = np.cos(epsilon)
        b = np.sin(epsilon)

        t = np.linspace(0, 2*np.pi, 500)
        x = a*np.cos(t)*np.cos(alpha) - b*np.sin(t)*np.sin(alpha)
        y = a*np.cos(t)*np.sin(alpha) + b*np.sin(t)*np.cos(alpha)
        X = np.linspace(-1, 1, 500)

        ax.plot(x, y)
        if abs(epsilon) > 1e-16:
            i = 400 if epsilon < 0 else 100
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            ax.annotate('', xy=(x[i]+dx, y[i]+dy), xytext=(x[i], y[i]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.plot(X*np.cos(alpha), np.sin(alpha)*X, 'b--', alpha=0.5)
        ax.plot(X*np.cos(alpha-np.pi/2), np.sin(alpha-np.pi/2)*X, 'g--', alpha=0.5)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{label} | input={u:.1f}°\nα={alpha*180/np.pi:.1f}°, ε={epsilon*180/np.pi:.1f}°')
        ax.set_xlabel('Ex')
        ax.set_ylabel('Ey')

    plt.tight_layout()
    plt.show()