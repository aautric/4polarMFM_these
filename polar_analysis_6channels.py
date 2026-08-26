#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:21:10 2026

@author: amaury, juliette
"""
#%%
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
path_raw2 = ['/mnt/d/Amaury/DATA/2026_08_24_calibr_polar_final_without_polarizer/45/'+str(i)+'/images/RAW_DATA/image_Pos0.ome.tif' for i in polars]
#path_raw2 = ['/mnt/d/Amaury/DATA/2026_07_20_calibr_polar_analyser_before_wollaston/45/'+str(i)+'/images/RAW_DATA/image_Pos0.ome.tif' for i in polars]

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

carreA = [[30,310],[40,330]]
carreB = [[200,280],[250,340]]
carreC = [[450,290],[500,340]]
#%%
I1 = np.mean(raw[:,carre1[0][1]:carre1[1][1],carre1[0][0]:carre1[1][0]], axis=(1,2))
I2 = np.mean(raw[:,carre2[0][1]:carre2[1][1],carre2[0][0]:carre2[1][0]], axis=(1,2))
I3 = np.mean(raw[:,carre3[0][1]:carre3[1][1],carre3[0][0]:carre3[1][0]], axis=(1,2))
I4 = np.mean(raw[:,carre4[0][1]:carre4[1][1],carre4[0][0]:carre4[1][0]], axis=(1,2))
I5 = np.mean(raw[:,carre5[0][1]:carre5[1][1],carre5[0][0]:carre5[1][0]], axis=(1,2))
I6 = np.mean(raw[:,carre6[0][1]:carre6[1][1],carre6[0][0]:carre6[1][0]], axis=(1,2))
Iback1 = np.mean(raw[0,0:200,50:150])

Iback = np.mean(raw2[0,0:200,0:200])
IA = np.mean(raw2[:,carreA[0][1]:carreA[1][1],carreA[0][0]:carreA[1][0]], axis=(1,2))-Iback
IB = np.mean(raw2[:,carreB[0][1]:carreB[1][1],carreB[0][0]:carreB[1][0]], axis=(1,2))-Iback
IC = np.mean(raw2[:,carreC[0][1]:carreC[1][1],carreC[0][0]:carreC[1][0]], axis=(1,2))-Iback

# %% Calcul de phi

#données constructeur
lbda_list = list(range(400, 801))
phi_list = [
0.19896, 0.19991, 0.20086, 0.20180, 0.20274, 0.20368, 0.20461, 0.20553,
0.20645, 0.20737, 0.20828, 0.20919, 0.21009, 0.21099, 0.21188, 0.21277,
0.21365, 0.21452, 0.21539, 0.21625, 0.21711, 0.21796, 0.21880, 0.21964,
0.22047, 0.22130, 0.22212, 0.22293, 0.22373, 0.22453, 0.22532, 0.22610,
0.22688, 0.22765, 0.22841, 0.22917, 0.22992, 0.23066, 0.23140, 0.23213,
0.23285, 0.23356, 0.23427, 0.23496, 0.23566, 0.23634, 0.23702, 0.23769,
0.23835, 0.23901, 0.23965, 0.24029, 0.24093, 0.24155, 0.24217, 0.24278,
0.24339, 0.24399, 0.24458, 0.24516, 0.24574, 0.24630, 0.24687, 0.24742,
0.24797, 0.24851, 0.24904, 0.24957, 0.25009, 0.25060, 0.25111, 0.25161,
0.25210, 0.25258, 0.25306, 0.25353, 0.25400, 0.25446, 0.25491, 0.25535,
0.25579, 0.25623, 0.25665, 0.25707, 0.25748, 0.25789, 0.25829, 0.25868,
0.25907, 0.25945, 0.25983, 0.26020, 0.26056, 0.26092, 0.26127, 0.26162,
0.26195, 0.26229, 0.26262, 0.26294, 0.26325, 0.26357, 0.26387, 0.26417,
0.26446, 0.26475, 0.26504, 0.26531, 0.26559, 0.26585, 0.26611, 0.26637,
0.26662, 0.26687, 0.26711, 0.26735, 0.26758, 0.26780, 0.26802, 0.26824,
0.26845, 0.26866, 0.26886, 0.26906, 0.26925, 0.26944, 0.26962, 0.26980,
0.26998, 0.27015, 0.27031, 0.27047, 0.27063, 0.27078, 0.27093, 0.27107,
0.27121, 0.27135, 0.27148, 0.27161, 0.27174, 0.27186, 0.27197, 0.27208,
0.27219, 0.27230, 0.27240, 0.27249, 0.27259, 0.27268, 0.27276, 0.27285,
0.27293, 0.27300, 0.27308, 0.27314, 0.27321, 0.27327, 0.27333, 0.27339,
0.27344, 0.27349, 0.27353, 0.27358, 0.27362, 0.27365, 0.27369, 0.27372,
0.27375, 0.27377, 0.27379, 0.27381, 0.27383, 0.27384, 0.27385, 0.27386,
0.27387, 0.27387, 0.27387, 0.27387, 0.27386, 0.27386, 0.27385, 0.27383,
0.27382, 0.27380, 0.27378, 0.27376, 0.27373, 0.27371, 0.27368, 0.27365,
0.27361, 0.27358, 0.27354, 0.27350, 0.27346, 0.27341, 0.27337, 0.27332,
0.27327, 0.27321, 0.27316, 0.27310, 0.27305, 0.27298, 0.27292, 0.27286,
0.27279, 0.27272, 0.27266, 0.27258, 0.27251, 0.27244, 0.27236, 0.27228,
0.27220, 0.27212, 0.27204, 0.27195, 0.27187, 0.27178, 0.27169, 0.27160,
0.27151, 0.27141, 0.27132, 0.27122, 0.27112, 0.27102, 0.27092, 0.27082,
0.27072, 0.27061, 0.27051, 0.27040, 0.27029, 0.27018, 0.27007, 0.26996,
0.26985, 0.26973, 0.26962, 0.26950, 0.26938, 0.26926, 0.26914, 0.26902,
0.26890, 0.26877, 0.26865, 0.26852, 0.26840, 0.26827, 0.26814, 0.26801,
0.26788, 0.26775, 0.26762, 0.26749, 0.26735, 0.26722, 0.26708, 0.26695,
0.26681, 0.26667, 0.26653, 0.26639, 0.26625, 0.26611, 0.26597, 0.26582,
0.26568, 0.26554, 0.26539, 0.26525, 0.26510, 0.26495, 0.26481, 0.26466,
0.26451, 0.26436, 0.26421, 0.26406, 0.26391, 0.26375, 0.26360, 0.26345,
0.26329, 0.26314, 0.26299, 0.26283, 0.26267, 0.26252, 0.26236, 0.26220,
0.26205, 0.26189, 0.26173, 0.26157, 0.26141, 0.26125, 0.26109, 0.26093,
0.26077, 0.26061, 0.26045, 0.26028, 0.26012, 0.25996, 0.25979, 0.25963,
0.25947, 0.25930, 0.25914, 0.25897, 0.25881, 0.25864, 0.25847, 0.25831,
0.25814, 0.25797, 0.25781, 0.25764, 0.25747, 0.25731, 0.25714, 0.25697,
0.25680, 0.25663, 0.25646, 0.25629, 0.25612, 0.25595, 0.25578, 0.25561,
0.25544, 0.25527, 0.25510, 0.25493, 0.25476, 0.25459, 0.25442, 0.25425,
0.25408, 0.25391, 0.25373, 0.25356, 0.25339, 0.25322, 0.25305, 0.25288,
0.25270, 0.25253, 0.25236, 0.25219, 0.25201, 0.25184, 0.25167, 0.25150,
0.25132, 0.25115, 0.25098, 0.25080, 0.25063, 0.25046, 0.25029, 0.25011,
0.24994, 0.24977, 0.24959, 0.24942, 0.24925, 0.24907, 0.24890, 0.24873,
0.24856, 0.24838, 0.24821, 0.24804, 0.24786, 0.24769, 0.24752, 0.24734,
0.24717, 0.24700, 0.24683, 0.24665, 0.24648, 0.24631, 0.24614, 0.24596,
0.24579, 0.24562, 0.24545, 0.24527, 0.24510, 0.24493, 0.24476, 0.24459,
0.24441
] #en unité de lambda

#%% functions
#dephasage lame d'onde
lbda = 620 #arrondi à l'unité
phi_val = phi_list[lbda_list.index(lbda)]*2*np.pi
def polar_analysis(I_, plot=False, verif_fit=False):
    P2 = np.ones(N)
    P4 = np.ones(N)
    Q2 = np.ones(N)
    Q4 = np.ones(N)
    for p in range(N):
        P2[p] = I_[p]*np.cos(2*omega[p])
        P4[p] = I_[p]*np.cos(4*omega[p])
        Q2[p] = I_[p]*np.sin(2*omega[p])
        Q4[p] = I_[p]*np.sin(4*omega[p])
    p_0 = (1/N)*sum(I_)
    p_2 = (2/N)*sum(P2)
    q_2  = (2/N)*sum(Q2)
    q_4 = (2/N)*sum(Q4)
    p_4 = (2/N)*sum(P4)

    #-------correction décalage lame/polariseur
    p_2 = round(p_2, 5)
    print(f'P2 = {p_2}')
    print(f'Q2 = {q_2}')
    decalage_pola = 0.5*np.arctan2(p_2, q_2)
    if np.abs(decalage_pola)> np.abs((decalage_pola-np.pi/2)):
        decalage_pola = decalage_pola-np.pi/2
    #decalage_pola = np.radians(-6)
    #print('first measurements had a 90 error on the axis of the quarter')
    #decalage_pola-=np.pi/2 # first measurements had a 90 error on the axis of the quarter
    print(f'decalage du polariseur : {np.degrees(decalage_pola):.2f}')

    omega_corr = np.linspace(0, 2*np.pi, N+1)[:-1] + decalage_pola
    #calcul des coefs de Fourier expérimentaux
    P2 = np.ones(N)
    P4 = np.ones(N)
    Q2 = np.ones(N)
    Q4 = np.ones(N)
    for p in range(N):
        P2[p] = I_[p]*np.cos(2*omega_corr[p])
        P4[p] = I_[p]*np.cos(4*omega_corr[p])
        Q2[p] = I_[p]*np.sin(2*omega_corr[p])
        Q4[p] = I_[p]*np.sin(4*omega_corr[p])
    p_0 = (1/N)*sum(I_)
    p_2  = (2/N)*sum(P2)
    q_2  = (2/N)*sum(Q2)
    q_4 = (2/N)*sum(Q4)
    p_4 = (2/N)*sum(P4)

    print(f'P2" = {p_2}')
    print(f'Q2" = {q_2}')
    print(f'P4" = {p_4}')
    print(f'Q4" = {q_4}')

    #--------------dépolarisation
    #paramètres de stokes
    S3 = -q_2 / np.sin(phi_val)
    S1 = 2*p_4 / (np.cos(phi_val)-1)
    S2 = 2*q_4 / (np.cos(phi_val)-1)
    S0 = p_0 + S1*(1 + np.cos(phi_val))/2

    q = S1/S0
    u = S2/S0
    v = S3/S0

    depola = np.sqrt(q**2+u**2+v**2)

    alpha = 0.5*np.arctan2(S2, S1)
    epsilon = 0.5*np.arctan2(S3, np.sqrt(S1**2 + S2**2))
    #alpha = 0.5*np.arctan2(-q_4,-p_4)
    #epsilon = 0.5*np.arctan2(-(q_2*np.sin(2*alpha)*(np.cos(phi_val)-1)),-(2*q_4*np.sin(phi_val)))
    
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
            plt.title(f'Ellipse de polarisation  α={alpha*180/np.pi}°, ε={epsilon*180/np.pi}°, depola={depola}')
            plt.xlabel('Ex')
            plt.ylabel('Ey')
            plt.tight_layout()
            plt.show()
    if verif_fit:
        N_ = 1000
        omega__ = np.linspace(0, 2*np.pi, N_+1)[:-1] + decalage_pola
        omega_exp = np.linspace(0, 2*np.pi, 9+1)[:-1]  + decalage_pola
        # "mesures" d'intensité en sortie du montage

        I__ = np.ones(N_)
        '''
        for p in range (N_):
            E = polariseur_y@rotation_elt_opt(quart_onde,omega[p])@pola_inc
            I[p] = abs(E[1,0])**2
        '''

        for p in range(N_):
            I__[p] = p_0 + p_4*np.cos(4*omega__[p]) + q_2*np.sin(2*omega__[p])+ p_2*np.cos(2*omega__[p])+q_4*np.sin(4*omega__[p])

        plt.plot(omega__, I__)
        plt.scatter(omega_exp,I_)
        plt.xlabel('omega')
        plt.ylabel('I(omega)')
        plt.show()
    phi = np.arctan2(np.sin(2*epsilon), np.cos(2*epsilon)*np.sin(2*alpha))
    Ex = np.sqrt(0.5*(1+np.cos(2*epsilon)*np.cos(2*alpha)))
    Ey = np.sqrt(0.5*(1-np.cos(2*epsilon)*np.cos(2*alpha)))
    return epsilon, alpha, Ex, Ey, phi

#%%

epsilon, alpha, Ex, Ey, phi = polar_analysis(IB, plot=True, verif_fit=True)
print(f'epsilon" = {epsilon*180/np.pi}')
print(f'alpha" = {alpha*180/np.pi}')
print(f'Ex" = {Ex}')
print(f'Ey" = {Ey}')
print(f'phi" = {phi}')

#%%
'''
A = 0.8
B = 0.62
C = 0.59
D = 0.79
phiC = -0.23
phiDB = 2.98

A =  0.8668759124839902 # best version
B = 0.5870852690046182
C = 0.4985239737014151
D = 0.8095250996218555
phiC = 0.2889154491430255
phiDB = -2.9532601513428656
''' 
A =  0.8909791204551392
B = 0.505643433201802
C = 0.4540442785819315
D = 0.8627425563051209
phiC = -0.32460758222046554
phiDB = 2.8342197809269853


phiB = np.linspace(0,2*np.pi, 100)
func = phiC+np.arctan2((D*np.sin(phiDB+phiB-phiC)),(C+D*np.cos(phiDB+phiB-phiC)))-np.arctan2((B*np.sin(phiB)),(A+B*np.cos(phiB)))
plt.plot(phiB, func)
#plt.axhline(1.1)
#plt.axvline(3.82)
plt.axhline(1.009256885667573)
plt.axvline(4.195)
plt.grid()
plt.show()
phiB = 4.195
phiD = phiDB+phiB
#%%
J1 = rotation(0*np.pi/180)@np.array([[A, B*np.exp(1j*phiB)],[C*np.exp(1j*phiC), D*np.exp(1j*phiD)]])
phia = np.angle(J1[0,0])
J1 = np.exp(-1j*phia)*J1
print(J1)
print(np.abs(J1[0,0])**2+np.abs(J1[1,0])**2)
print(np.abs(J1[0,1])**2+np.abs(J1[1,1])**2)
#%%
'''
A = 0.38
B = 0.93
C = 0.92
D = 0.36
phiC = 2.49
phiDB = -0.55 
A =  0.3156127707159146
B = 0.9591632828542673
C = 0.9488880750441664
D = 0.2828529597232894
phiC = -2.330143009967272
phiDB = 0.7466871004606931
'''
A =  0.3167651506500153
B = 0.9358510462414047
C = 0.9485040006946059
D = 0.3523958275134203
phiC = 2.2338342800407056
phiDB = -0.801892200962413

phiB = np.linspace(0,2*np.pi, 100)
func = phiC+np.arctan2((D*np.sin(phiDB+phiB-phiC)),(C+D*np.cos(phiDB+phiB-phiC)))-np.arctan2((B*np.sin(phiB)),(A+B*np.cos(phiB)))
plt.plot(phiB, func)
plt.axhline(-0.9716145618923928+2*np.pi)
plt.axvline(3.22)
plt.grid()
plt.show()
phiB = 3.22
phiD = phiDB+phiB

#%%
J2 = rotation(0*np.pi/180)@np.array([[A, B*np.exp(1j*phiB)],[C*np.exp(1j*phiC), D*np.exp(1j*phiD)]])
phia = np.angle(J2[0,0])
J2 = np.exp(-1j*phia)*J2
print(J2)
print(np.abs(J2[0,0])**2+np.abs(J2[1,0])**2)
print(np.abs(J2[0,1])**2+np.abs(J2[1,1])**2)
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
# %%
print(np.conj(J1.T))
print(np.conj(J2.T))
j1iinv = np.conj(J1.T)
j2iinv = np.conj(J2.T)
# %%
fig, ax = plt.subplots()
Ex = np.abs(j1iinv[0,0])
Ey = np.abs(j1iinv[1,0])
phi = np.angle(j2iinv[1,0]) - np.angle(j2iinv[0,0])
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
plt.show()
# %%
