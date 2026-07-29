# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:30:31 2026

@author: LOCCO
"""

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk


def rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])
    
    
def rotation_elt_opt(element, omega):
    rot1 = rotation(omega)
    rot2 = rotation(-omega)
    return rot1@element@rot2


# %%                         SIMULATION

#définition de la polarisation choisie (en radian)
alpha_th = 2*np.pi/4
epsilon_th = 1*np.pi/8

pola_inc = rotation(alpha_th)@np.array([[np.cos(epsilon_th)],
                                        [1j*np.sin(epsilon_th)]])

#elements de base 
polariseur_y = np.array([[0,0],[0,1]])
quart_onde = np.array([[1,0],[0, 1j]])

N = 9
omega = np.linspace(0, 2*np.pi, N+1)[:-1]

# "mesures" d'intensité en sortie du montage
I = np.ones(N)
for p in range (N):
    E = polariseur_y@rotation_elt_opt(quart_onde,omega[p])@pola_inc
    I[p] = abs(E[1,0])**2

#%%                        MESURE
N = 9
omega = np.linspace(0, 2*np.pi, N+1)[:-1]

entries = {}
valeurs = {}

def valider():
    global valeurs
    for angle in omega:
        valeurs[angle] = float(entries[angle].get())
    print(valeurs)
    root.destroy()

root = tk.Tk()
root.title("Mesures d'intensités")

tk.Label(root, text="Entrer les valeurs d'intensité mesurées correspondant aux orientations suivante de la lame quart d'onde : ",
         font=("Arial", 11), fg="gray").grid(row=0, columnspan=2, pady=15)

for i, angle in enumerate(omega):
    tk.Label(root, text=f"I(Ω = {180*angle/np.pi:.3f}°)").grid(row=i+1, column=0, padx=10, pady=4, sticky="e")
    entries[angle] = tk.Entry(root)
    entries[angle].grid(row=i+1, column=1, padx=10, pady=4)

#pour pouvoir cliquer sur entrée quand on rentre les valeurs
champs = list(entries.values())  # liste ordonnée des Entry
        
for i, entry in enumerate(champs):
    if i < len(champs) - 1:
        next_entry = champs[i+1]
        entry.bind('<Return>', lambda e, n=next_entry: n.focus_set())
        entry.bind('<KP_Enter>', lambda e, n=next_entry: n.focus_set())  # pavé num
    else:
        entry.bind('<Return>', lambda e: valider())
        entry.bind('<KP_Enter>', lambda e: valider())  # pavé num

tk.Button(root, text="Valider", command=valider).grid(row=len(omega)+1, columnspan=3, pady=10)
root.mainloop()

I = [valeurs[angle] for angle in omega]


# %%                 CALCULS

#calcul des coefs de Fourier
P4 = np.ones(N)
Q2 = np.ones(N)
Q4 = np.ones(N)
for p in range(N):
    P4[p] = I[p]*np.cos(4*omega[p])
    Q2[p] = I[p]*np.sin(2*omega[p])
    Q4[p] = I[p]*np.sin(4*omega[p])
p_0 = (1/N)*sum(I)
q_2  = (2/N)*sum(Q2)
q_4 = (2/N)*sum(Q4)
p_4 = (2/N)*sum(P4)

print(p_0)
print(q_2)
print(q_4)
print(p_4)

#calcul des angles 
alpha_exp = 0.5*np.arctan2(-q_4,-p_4)

epsilon_exp = 0.5*np.arctan(q_2*np.sin(2*alpha_exp)/(2*q_4))

print(f"On obtient α = {alpha_exp*180/np.pi} et ε = {epsilon_exp*180/np.pi}")

# demi-axes de l'ellipse
a = np.cos(epsilon_exp)
b = np.sin(epsilon_exp)

# Tracé paramétrique 
t = np.linspace(0, 2 * np.pi, 500)
x = a * np.cos(t) * np.cos(alpha_exp) - b * np.sin(t) * np.sin(alpha_exp)
y = a * np.cos(t) * np.sin(alpha_exp) + b * np.sin(t) * np.cos(alpha_exp)
X = np.linspace(-1, 1, 500)

plt.figure(figsize=(5, 5))
plt.plot(x, y)
if abs(epsilon_exp)>1e-16:
    if epsilon_exp<0:# position de la flèche sur la courbe
        i = 400
    if epsilon_exp>=0:
        i = 100  # position de la flèche sur la courbe
    dx = x[i+1] - x[i]
    dy = y[i+1] - y[i]
    plt.annotate('', xy=(x[i]+dx, y[i]+dy), xytext=(x[i], y[i]),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
    plt.plot(X*np.cos(alpha_exp), np.sin(alpha_exp)*X)
    plt.plot(X*np.cos(alpha_exp-np.pi/2), np.sin(alpha_exp-np.pi/2)*X)
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.title(f'Ellipse de polarisation  α={alpha_exp*180/np.pi}°, ε={epsilon_exp*180/np.pi}°')
plt.xlabel('Ex')
plt.ylabel('Ey')
plt.tight_layout()
plt.show()