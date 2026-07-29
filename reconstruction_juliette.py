# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:35:13 2026

@author: LOCCO
"""
import matplotlib
matplotlib.use('TkAgg')#pour ouvrir des fenêtres externes
import numpy as np
import matplotlib.pyplot as plt
import os
import tifffile 
import tkinter as tk
import cv2
import statistics
from tkinter import filedialog, messagebox
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from scipy.interpolate import RBFInterpolator
from pathlib import Path
from scipy import ndimage
import csv
from scipy.ndimage import uniform_filter, gaussian_filter
from tkinter import simpledialog
from tqdm import tqdm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd

def extract_raw_ome(path_raw, first_frame=None, last_frame=None):
    if not os.path.exists(path_raw):
        raise FileNotFoundError(f"File not found: {path_raw}")
    try:
        with tifffile.TiffFile(path_raw) as tif:
            if first_frame is None:
                first_frame=0
                last_frame = len(tif.pages)
            raw = np.zeros((last_frame-first_frame, 512, 512))
            for i in range(last_frame-first_frame):
                raw[i] = tif.pages[first_frame+i].asarray()
        return raw
    except Exception as e:
        print(f"Error reading {path_raw}: {e}")
        return None
    
def detect_plateau_0(img):
    sigma = 3
    line = np.mean(img, axis=0)
    der = line[1:]-line[:-1]
    smooth = gaussian_filter1d(der, sigma=sigma)
    max1 = np.argmax(smooth[0:50])
    min2 = 150+np.argmin(smooth[150:200])
    max2 = 150+np.argmax(smooth[150:200])
    min3 = 300+np.argmin(smooth[300:400])
    max3 = 300+np.argmax(smooth[300:400])
    min4 = 420+np.argmin(smooth[420:-1])
    return [max1-1, min2+1, max2-1, min3+1, max3-1, min4+1]
    
def detect_plateau_1(img):
    sigma = 3
    line = np.mean(img, axis=1)
    der = line[1:]-line[:-1]
    smooth = gaussian_filter1d(der, sigma=sigma)
    
    max1 = np.argmax(smooth[0:80])
    min2 = 200+np.argmin(smooth[200:300])
    max2 = 200+np.argmax(smooth[200:300])
    min3 = 420+np.argmin(smooth[420:-1])
    return [max1-1, min2+1, max2-1, min3+1]
    
def gauss2d(coords, A, mu1, mu2, s, c):
    x, y = coords
    return (16**2)*A*(1/(2*np.pi*(s**2)))*np.exp(-0.5*(((x-mu1)**2+(y-mu2)**2)/((s**2)))) + c

def gauss1d(z, A, mu, sigma, c):
    return A * np.exp(-0.5 * ((z - mu) / sigma)**2) + c

def asymmetric_gauss(z, A, mu, sigma_left, sigma_right, c):
    result = np.where(
        z < mu,
        A * np.exp(-0.5 * ((z - mu) / sigma_left)**2),
        A * np.exp(-0.5 * ((z - mu) / sigma_right)**2)
    )
    return result + c

def affine(z,a,b):
    return(a*z+b)

plt.ion()
def select_points_subimage(img, x1, x2, y0, y1):

    # Extract the subimage
    subimg = img[y0:y1, x1:x2]

    # Display the subimage
    fig, ax = plt.subplots()
    ax.imshow(subimg, cmap='gray', norm='log') #en log pour voir plus de billes et mieux reconstruire
    ax.set_title("Click to select points, press Enter when done")
    plt.pause(0.1)

    # Let the user select points interactively
    points = plt.ginput(n=-1, timeout=0)  # n=-1 allows unlimited points
    plt.close(fig)

    # Convert to coordinates in the original image
    points = np.array([(x + x1, y + y0) for x, y in points])

    return points

def refine_centers(img, points, window=3):
    """
    Précise les coordonnées des centres en prenant le max dans une fenêtre 
    autour de chaque point selectionné par l'utilisateur
    points : array (N, 2) avec colonnes [x, y]
    window : demi-taille de la fenêtre de recherche (en pixels)
    """
    refined = []
    for i in range(len(points)):
        x_c, y_c = int(points[i, 0]), int(points[i, 1])
        
        # Définit la zone sondée (sans sortir de l'image)
        x_min, x_max = max(0, x_c - window), min(img.shape[1], x_c + window)
        y_min, y_max = max(0, y_c - window), min(img.shape[0], y_c + window)
        
        sous_matrice = img[y_min:y_max, x_min:x_max]
        # Trouve les coordonnes du max dans la sous-matrice
        idx_local = np.unravel_index(np.argmax(sous_matrice), sous_matrice.shape)
        # idx_local = (row, col) = (y_local, x_local)
        
        # Reconvertit en coordonnées globales
        y_refined = y_min + idx_local[0]
        x_refined = x_min + idx_local[1]
        
        refined.append([x_refined, y_refined])
    return np.array(refined)

def fit_one_image(img, points, n=512, m=512):
    X, Y = np.meshgrid(np.arange(n), np.arange(m))  # n x m grids
    xdata = np.vstack((X.ravel(), Y.ravel()))  # shape (2, 262144)
    zdata = img.ravel()
    offset_estime = np.percentile(img, 5)  # min sur l'image sauf 0
    p_list = []
    points = refine_centers(img, points)
    for i in range(len(points)):
        p, pcov = curve_fit(gauss2d, xdata, zdata, p0=(200, points[i,0], points[i,1], 2,  offset_estime),
                            bounds=([0, -np.inf, -np.inf,  1, -np.inf],  # min
                                    [ np.inf,  np.inf,  np.inf, 5,  np.inf]))# max 
        p_list.append(p)
    return np.array(p_list)

def mean_intensity_per_z_moy(zstack, coord, taille=7):
    """
    calcule la moyenne d'intensité sur un carré autour de chaque bille 
    et moyenne sur toutes les billes, ce pour chaque plan
    zstack  : shape (Z, H, W)
    coord : shape (N, 2) → colonnes (x0, y0)
    taille : côté du carré sur lequel on moyenne chaque bille
    """
    taille = taille//2
    intensities = np.zeros(zstack.shape[0]) #liste de la taille du nb de plans en z
    
    for z in range(zstack.shape[0]):
        vals = []
        for cx, cy in coord:
            cx, cy = int(cx), int(cy)
            tache = zstack[z, max(0, cy-taille):cy+taille+1, max(0, cx-taille):cx+taille+1]
            vals.append(tache.mean())
        intensities[z] = np.mean(vals)
        
    return intensities  # shape (Z)


def mean_intensity_per_z(zstack, coord, taille=7):
    """
    calcule la moyenne d'intensité sur un carré autour de chaque bille 
    et renvoie toutes ces valeurs, ce pour chaque plan
    zstack  : shape (Z, H, W)
    coord : shape (N, 2) → colonnes (x0, y0)
    taille : côté du carré sur lequel on moyenne chaque bille
    """
    taille = taille//2
    intensities = np.zeros((coord.shape[0],zstack.shape[0])) #liste de la taille du nb de plans en z
    
    for z in range(zstack.shape[0]):
        for i, (cx, cy) in enumerate(coord):
            cx, cy = int(cx), int(cy)
            tache = zstack[z, max(0, cy-taille):cy+taille+1, max(0, cx-taille):cx+taille+1]
            intensities[i,z] = tache.mean()
    return intensities  # shape (N,Z)


def find_transfo_locale(pts_deformes, pts_ref):
    """
    Renvoie les coordonnées à fournir à map_coordinates pour reconstruire l'image
    pts_deformes : points connus de l'image déformée (N,2)
    pts_ref : points de référence de l'image qui doivent correspondre à pts_deformes (N,2)

    """
    # déplacement entre l'image déformée et l'image de référence
    dx = pts_ref[:, 0] - pts_deformes[:, 0]
    dy = pts_ref[:, 1] - pts_deformes[:, 1]

    # interpolation : estime l'expression de (dx,dy) en fonction de (x,y)
    rbf_dx = RBFInterpolator(pts_deformes, dx, kernel="thin_plate_spline",smoothing=0.0)
    rbf_dy = RBFInterpolator(pts_deformes, dy, kernel="thin_plate_spline", smoothing=0.0)
    
    return (rbf_dx, rbf_dy)

    
def transfo_im_locale(image_def, rbf_dx, rbf_dy):
    """
    Renvoie l'image reconstruite, de la même taille que l'image de départ
    image_def : image à reconstruire
    rbf_dx, rbf_dy:: matrices contenant les déplacement dx et dy pour chaque pixel (x,y),
    déterminés avec find_transfo
    """
    h, w = image_def.shape[:2]
    
    # grille de pixels
    X, Y = np.meshgrid(np.arange(w), np.arange(h))   # shape (215, 160)
    grid = np.column_stack([X.ravel(), Y.ravel()])  # shape (34400, 2)

    # champ de déplacement interpolé : DX[i,j] = dx  
    DX = rbf_dx(grid).reshape(h, w)  # shape (215, 160)
    DY = rbf_dy(grid).reshape(h, w)
    
    # remap : on soustrait les vecteurs car on veut chercher 
    #la position du pixel source (X-DX) pour mettre sa valeur en x 
    map_x = (X - DX).astype(np.float32) # shape (215, 160)
    map_y = (Y - DY).astype(np.float32) # shape (2, 34400)
    
    coords = np.vstack([map_y.ravel(), map_x.ravel()])
    
    #lit les valeurs dans image aux coordonnées coord pour reconstruire 
    #result (interpolation ORDRE3)
    result = map_coordinates(
        image_def,
        coords,
        order=3
    ).reshape(h, w)
    
    return result


#-----------------------------------------------------------------------

def delta_z(data_beads, X, Y, vecteurs, nb_billes, coord_beads_loc):
    """
    Renvoie l'écart entre les plans de focus calculé à partir des z-stacks
    ----------
    data_beads : z-stack de données de billes
    X,Y: coordonnées des délimitations des différents canaux de l'image
    vecteurs : liste des vecteurs de passage du canal de référence (canal 1) aux autres
    nb_billes : nb de billes par canal
    coord_beads_loc : liste des coordonnées des billes pour chaque canal. shape (6,N,2)

    """
    liste_z = np.arange(0, data_beads.shape[0], 1) * 100

    # calcul de l'intensité des billes pour chaque z du stack (moyennée sur plusieurs billes et sur un carré de 7x7 dans chacune)
    mean_intensities_plane1 = (mean_intensity_per_z_moy(data_beads[:, Y[0]+int(round(vecteurs[0][1])):Y[1]+int(round(vecteurs[0][1])),
                        X[2]+int(round(vecteurs[0][0])):X[3]+int(round(vecteurs[0][0]))], coord_beads_loc[0])
                              + mean_intensity_per_z_moy(data_beads[:, Y[0]+int(round(vecteurs[3][1])):Y[1]+int(round(vecteurs[3][1])),
                        X[2]+int(round(vecteurs[3][0])):X[3]+int(round(vecteurs[3][0]))], coord_beads_loc[3])) / 2

    mean_intensities_plane2 = (mean_intensity_per_z_moy(data_beads[:, Y[0]:Y[1], X[2]:X[3]], coord_beads_loc[1])
                              + mean_intensity_per_z_moy(data_beads[:, Y[0]+int(round(vecteurs[4][1])):Y[1]+int(round(vecteurs[4][1])),
                        X[2]+int(round(vecteurs[4][0])):X[3]+int(round(vecteurs[4][0]))], coord_beads_loc[4])) / 2

    mean_intensities_plane3 = (mean_intensity_per_z_moy(data_beads[:, Y[0]+int(round(vecteurs[2][1])):Y[1]+int(round(vecteurs[2][1])),
                        X[2]+int(round(vecteurs[2][0])):X[3]+int(round(vecteurs[2][0]))], coord_beads_loc[2])
                              + mean_intensity_per_z_moy(data_beads[:, Y[0]+int(round(vecteurs[5][1])):Y[1]+int(round(vecteurs[5][1])),
                        X[2]+int(round(vecteurs[5][0])):X[3]+int(round(vecteurs[5][0]))], coord_beads_loc[5])) / 2

    # calcul de l'intensité de chaque bille pour chaque z du stack
    intensities_plane1 = (mean_intensity_per_z(data_beads[:, Y[0]+int(round(vecteurs[0][1])):Y[1]+int(round(vecteurs[0][1])),
                    X[2]+int(round(vecteurs[0][0])):X[3]+int(round(vecteurs[0][0]))], coord_beads_loc[0])
                        + mean_intensity_per_z(data_beads[:, Y[0]+int(round(vecteurs[3][1])):Y[1]+int(round(vecteurs[3][1])),
                    X[2]+int(round(vecteurs[3][0])):X[3]+int(round(vecteurs[3][0]))], coord_beads_loc[3])) / 2

    intensities_plane2 = (mean_intensity_per_z(data_beads[:, Y[0]+int(round(vecteurs[1][1])):Y[1]+int(round(vecteurs[1][1])),
                    X[2]+int(round(vecteurs[1][0])):X[3]+int(round(vecteurs[1][0]))], coord_beads_loc[1])
                        + mean_intensity_per_z(data_beads[:, Y[0]+int(round(vecteurs[4][1])):Y[1]+int(round(vecteurs[4][1])),
                    X[2]+int(round(vecteurs[4][0])):X[3]+int(round(vecteurs[4][0]))], coord_beads_loc[4])) / 2

    intensities_plane3 = (mean_intensity_per_z(data_beads[:, Y[0]+int(round(vecteurs[2][1])):Y[1]+int(round(vecteurs[2][1])),
                    X[2]+int(round(vecteurs[2][0])):X[3]+int(round(vecteurs[2][0]))], coord_beads_loc[2])
                        + mean_intensity_per_z(data_beads[:, Y[0]+int(round(vecteurs[5][1])):Y[1]+int(round(vecteurs[5][1])),
                    X[2]+int(round(vecteurs[5][0])):X[3]+int(round(vecteurs[5][0]))], coord_beads_loc[5])) / 2
    
    #fit gaussienne normale
    p_focus1 = np.zeros((nb_billes,4))
    p_focus2 = np.zeros((nb_billes,4))
    p_focus3 = np.zeros((nb_billes,4))
    for i in range(nb_billes):
        p_focus1[i], pcov_focus1 = curve_fit(gauss1d, liste_z, intensities_plane1[i],
                                             p0 = (mean_intensities_plane1.max(),
                                                   np.argmax(mean_intensities_plane1)*100,500,
                                                   mean_intensities_plane1.min()))
        p_focus2[i], pcov_focus2 = curve_fit(gauss1d, liste_z, intensities_plane2[i],
                                             p0 = (mean_intensities_plane2.max(),
                                                   np.argmax(mean_intensities_plane2)*100, 500,
                                                   mean_intensities_plane2.min()))
        p_focus3[i], pcov_focus3 = curve_fit(gauss1d, liste_z, intensities_plane3[i],
                                             p0 = (mean_intensities_plane3.max(),
                                                   np.argmax(mean_intensities_plane3)*100, 500,
                                                   mean_intensities_plane3.min()))
    focus = []
    plans = []
    for i in range(nb_billes):
        focus.append(p_focus1[i,1])
        plans.append(1)
    for i in range(nb_billes):
        focus.append(p_focus2[i,1])
        plans.append(2)
    for i in range(nb_billes):
        focus.append(p_focus3[i,1])
        plans.append(3)
    aff, affcov = curve_fit(affine, plans,  focus, p0 = (3 , 0))
    deltaz = aff[0]
    print(f"δz = {aff[0]:.1f} nm")
    return deltaz

def subcanals(image, X, Y, vecteurs):
    """
    ----------
    image : image 2D à découper en sous canaux
    X : coordonnées des frontières en x
    Y : coordonnées des frontières en y
    vecteurs : vecteurs de passage d'un cadrant à l'autre

    Returns
    -------
    liste des images des canaux séparés

    """
    canal0 = image[Y[0]+int(round(vecteurs[0][1])):Y[1]+int(round(vecteurs[0][1])),
                       X[2]+int(round(vecteurs[0][0])):X[3]+int(round(vecteurs[0][0]))]
    canal1 = image[Y[0]:Y[1],X[2]:X[3]]
    canal2 = image[Y[0]+int(round(vecteurs[2][1])):Y[1]+int(round(vecteurs[2][1])),
                       X[2]+int(round(vecteurs[2][0])):X[3]+int(round(vecteurs[2][0]))]
    canal3 = image[Y[0]+int(round(vecteurs[3][1])):Y[1]+int(round(vecteurs[3][1])),
                       X[2]+int(round(vecteurs[3][0])):X[3]+int(round(vecteurs[3][0]))]
    canal4 = image[Y[0]+int(round(vecteurs[4][1])):Y[1]+int(round(vecteurs[4][1]))
                       , X[2]+int(round(vecteurs[4][0])):X[3]+int(round(vecteurs[4][0]))]
    canal5= image[Y[0]+int(round(vecteurs[5][1])):Y[1]+int(round(vecteurs[5][1])),
                      X[2]+int(round(vecteurs[5][0])):X[3]+int(round(vecteurs[5][0]))]
    return [canal0, canal1, canal2, canal3, canal4,  canal5]


def save_calibration(deformation, X, Y, vecteurs, normalisation, deltaz, fichiers):
    # remonter de N niveaux
    dossier = Path(fichiers[0]).parents[4]  # 0 = dossier du fichier, 1 = dossier parent, 2 = grand-parent...
    save_path = os.path.join(dossier, "calibration.npz")
    np.savez(
        save_path,
        X=X, Y=Y, vecteurs=vecteurs, normalisation=normalisation, deltaz=deltaz,
        **{f"M_{i}": deformation[i] for i in range(len(deformation))}
    )
    print(f"Calibration sauvegardée : {save_path}")

def load_calibration(cwd):
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    save_path = filedialog.askopenfilename(
        title="Choisir le fichier de calibration",
        filetypes=[("Calibration", "*.npz"), ("Tous", "*.*")], initialdir=cwd
    )
    cal = np.load(save_path, allow_pickle=True)
    X, Y, vecteurs, normalisation, deltaz = cal["X"], cal["Y"], cal["vecteurs"].tolist(), cal["normalisation"], cal["deltaz"]
    n = len(cal.files) - 5
    deformation = [cal[f"M_{i}"] for i in range(n)]
    return deformation, X, Y, vecteurs, normalisation, deltaz

def beads_location(X, Y, vecteurs, data_beads):
    """
    divise les données en 6 canaux selon les coordonnées données en entrée,
    demande à l'utilisateur de sélectionner les billes, 
    et renvoie les localisations des billes obtenues par fit gaussiens sur les canaux
    
    """
    max_beads = np.max(data_beads, axis=0)
    canaux = subcanals(max_beads, X, Y, vecteurs)
    
    #selection billes
    points = select_points_subimage(max_beads, X[2], X[3], Y[0], Y[1])
    nb_billes = points.shape[0]
    
    # coordonnées locales des centres des billes
    coord_beads_approx = points - [X[0], Y[0]] + vecteurs[0]
    m,n = canaux[0].shape
    #print(n,m)
    P_list = [fit_one_image(canal, coord_beads_approx, n, m) for canal in canaux]

    coord_beads_loc = [np.array(p_list[:,1:3].astype(np.float64)) for p_list in P_list]
    return coord_beads_loc, nb_billes


def find_normalization(data_beads,X,Y):
    beads_tot = np.concatenate(data_beads, axis = 0)
    max_beads_tot = np.max(beads_tot, axis=0)
    # normalisation par les moyennes de chaque canal billes et soustraction du bruit
    read_noise = np.mean(max_beads_tot[1:Y[0]-2,:])
    norm0 = np.mean(max_beads_tot[Y[0]:Y[1],X[0]:X[1]])-read_noise
    norm1 = np.mean(max_beads_tot[Y[0]:Y[1],X[2]:X[3]])-read_noise
    norm2 = np.mean(max_beads_tot[Y[0]:Y[1],X[4]:X[5]])-read_noise
    norm3 = np.mean(max_beads_tot[Y[2]:Y[3],X[0]:X[1]])-read_noise
    norm4 = np.mean(max_beads_tot[Y[2]:Y[3],X[2]:X[3]])-read_noise
    norm5 = np.mean(max_beads_tot[Y[2]:Y[3],X[4]:X[5]])-read_noise

    mn = (norm0+norm1+norm2+norm3+norm4+norm5)/6
    normalization_coeff = np.array([norm0/mn, norm1/mn, norm2/mn, norm3/mn, norm4/mn, norm5/mn])
    print( normalization_coeff)
    return normalization_coeff

def check_calib(cwd):
    deformation, X, Y, vecteurs, normalisation, deltaz = load_calibration(cwd)
    
    #pour parcourir les fichiers
    root = tk.Tk() #crée fenêtre principale
    root.withdraw()  # cache la fenêtre principale
    root.call('wm', 'attributes', '.', '-topmost', True)
    # choix des données à utiliser
    data_raw = extract_raw_ome(filedialog.askopenfilename(
    title="Choisir le fichier billes à reconstruire comme test",
    filetypes=[("Images", "*.tif"), ("Tous", "*.*")], initialdir=cwd
    ))
    max_data_raw = np.max(data_raw, axis=0)
    canaux = subcanals(max_data_raw, X, Y, vecteurs)
    canaux_corriges = [cv2.warpPerspective(canaux[i],
                                           deformation[i],
                                           canaux[i].shape[:2][::-1],
                                           flags=cv2.INTER_CUBIC) for i in range(6)]
    #selection billes
    points = select_points_subimage(max_data_raw, X[2], X[3], Y[0], Y[1])
    # coordonnées locales des centres des billes
    coord_beads_approx = points - [X[0], Y[0]] + vecteurs[0]
    m,n = canaux[0].shape
    #fit sur les images transformées pour retrouver les centres des billes de l'image reconstruite
    Pr_list = [fit_one_image(canal_cor, coord_beads_approx, n, m) for canal_cor in canaux_corriges]
    new_coord_beads = [pr_list[:,1:3] for pr_list in Pr_list]
    #calculs des erreurs par rapport au canal de référence
    Erreurs = [np.linalg.norm((new_coord_beads[1] - coord) * 120, axis=1) for coord in new_coord_beads]
    erreurs_tot = np.concatenate(Erreurs)
    #print(Erreurs)
    print(f'standard deviation : {statistics.stdev(erreurs_tot)}')
    print(f'interquartile : {np.percentile(erreurs_tot, 75)-np.percentile(erreurs_tot, 25)}')
    matplotlib.use("module://matplotlib_inline.backend_inline")
    hh = plt.hist(erreurs_tot, bins=15)
    plt.show()
    
    

def calibration(cwd):
    """
    demande à l'utilisateur de choisir les fichiers à utiliser puis 
    se sert des autres fonctions pour renvoyer:
        -l'écart entre les plans en z
        -les données nécessaire à la reconstruction d'images:
            dx,dy,coeffs de normalisation
    enregistre ces données dansun fichier 'calibration' dans le dossier des données
    """
    root = tk.Tk()
    root.withdraw()
    already = messagebox.askyesno("Calibration", "Do you already have a calibration?")
    if already:
        deformation, X, Y, vecteurs, normalisation, deltaz = load_calibration(cwd)
        return deltaz
    else:
        root.destroy()
        #pour parcourir les fichiers
        root = tk.Tk() #crée fenêtre principale
        root.withdraw()  # cache la fenêtre principale
        root.call('wm', 'attributes', '.', '-topmost', True)
        
        # choix des données à utiliser
        fichiers = []
        while True:
            nouveaux = filedialog.askopenfilenames(
                title="Choisir les fichiers billes (Annuler pour terminer)",
                filetypes=[("Images", "*.tif"), ("Tous", "*.*")], initialdir=cwd
                )
            if not nouveaux:
                break
            fichiers.extend(nouveaux)
        print(f"{len(fichiers)} fichiers sélectionnés au total")
        data_beads = [extract_raw_ome(f) for f in fichiers]
        
        #certaine parties du codes sont pas souples vis-à-vis des variations de tailles d'images
        for i, data in enumerate(data_beads):
            if data.shape[1:] != (512, 512):
                raise ValueError(f"Fichier {fichiers[i]} : taille inattendue {data.shape[1:]}, attendu (512, 512)")
            
        data_intensity = extract_raw_ome(filedialog.askopenfilename(
            title="Choisir un fichier intensité",
            filetypes=[("Images", "*.tif"), ("Tous", "*.*")]
            ))[0]
        
        #detection des différent canaux
        X = detect_plateau_0(data_intensity)
        Y = detect_plateau_1(data_intensity)
    
        # vecteurs approximatifs(!) pour passer d'une image à l'autre :calculés à partir des déctections des bords, pas des billes
        v0 = [X[0]-X[2], 0]
        v1 = [0,0]
        v2 = [X[4]-X[2], 0]
        v3 = [X[0]-X[2], Y[2]-Y[0]]
        v4 = [0, Y[2]-Y[0]]
        v5 = [X[4]-X[2], Y[2]-Y[0]]
        vecteurs = [v0, v1, v2, v3, v4, v5]
        
        all_locs = []
        Deltaz = []
        for data in data_beads:
            loc, nb_billes = beads_location(X, Y, vecteurs, data)
            all_locs.append(loc) #shape
            Deltaz.append(delta_z(data, X, Y, vecteurs, nb_billes, loc))
        #print(np.array(all_locs).shape)
        deltaz = np.mean(np.array(Deltaz))
        print(f"δz_moy = {deltaz:.1f} nm")
    
        # concaténer les billes de tous les fichiers, canal par canal
        # coord_par_canal[i] = array (N_total, 2) de toutes les billes du canal i
        coord_par_canal = [np.concatenate([all_locs[f][i] for f in range(len(fichiers))], axis=0)
            for i in range(6)]
        normalisation = find_normalization(data_beads,X,Y)
        deformation = [cv2.findHomography(coord_par_canal[i], coord_par_canal[1], method=0)[0]
                           for i in range(6)]
        save_calibration(deformation, X, Y, vecteurs, normalisation, deltaz, fichiers)
        return deltaz
    
def reconstruction(cwd):
    """
    Demande à l'utilisateur le fichier à reconstruire et le fichier de calibration
    renvoyé par la fonction calibration.
    Renvoie le fichier reconstruit

    """
    deformation, X, Y, vecteurs, normalisation, deltaz = load_calibration(cwd)
    
    #pour parcourir les fichiers
    root = tk.Tk() #crée fenêtre principale
    root.withdraw()  # cache la fenêtre principale
    root.call('wm', 'attributes', '.', '-topmost', True)
    
    # choix des données à utiliser
    data_raw = extract_raw_ome(filedialog.askopenfilename(
    title="Choisir un fichier à reconstruire",
    filetypes=[("Images", "*.tif"), ("Tous", "*.*")]
    ))
    max_data_raw = np.max(data_raw, axis=0)
    canaux = subcanals(max_data_raw, X, Y, vecteurs)
    canaux_corriges = [cv2.warpPerspective(canaux[i],
                                           deformation[i],
                                           canaux[i].shape[:2][::-1],
                                           flags=cv2.INTER_CUBIC) for i in range(6)]
    canaux_normalises = [canaux_corriges[i]/normalisation[i] for i in range(6)]
    data_corr = np.stack(canaux_normalises, axis=0)
    data_corr = np.stack([canaux_normalises[0], canaux_normalises[3], canaux_normalises[1], canaux_normalises[4], canaux_normalises[2], canaux_normalises[5]], axis=0)
    data_corr = data_corr / data_corr.max() * 65535  # remettre à l'échelle uint16 pour éviter de tronquer des valeurs
    tifffile.imwrite("data_corr.tif", data_corr.astype(np.uint16))

#@njit(parallel=True) 
def reconstruction_stack(cwd):
    """
    Demande à l'utilisateur le fichier à reconstruire et le fichier de calibration
    renvoyé par la fonction calibration.
    Renvoie le fichier reconstruit

    """
    deformation, X, Y, vecteurs, normalisation, deltaz = load_calibration(cwd)
    
    #pour parcourir les fichiers
    root = tk.Tk() #crée fenêtre principale
    root.withdraw()  # cache la fenêtre principale
    root.call('wm', 'attributes', '.', '-topmost', True)
    
    multiple = messagebox.askyesno(
        "File selection",
        "Do you want to process several files?"
    )
    
    if multiple:
        input_files = [
            Path(f) for f in filedialog.askopenfilenames(
                title="Select the files to reconstruct",
                filetypes=[("Images", "*.tif"), ("All files", "*.*")], initialdir=cwd
            )
        ]
    else:
        filename = filedialog.askopenfilename(
            title="Select the file to reconstruct",
            filetypes=[("Images", "*.tif"), ("All files", "*.*")], initialdir=cwd
        )
        input_files = [Path(filename)] if filename else []
    
    root.destroy()
    
    if not input_files:
        return  # User cancelled
    
    # Create output folder next to the first selected file
    parent = input_files[0].parent
    output_dir = parent / "reconstruction"
    
    i = 1
    while output_dir.exists():
        output_dir = parent / f"reconstruction{i}"
        i += 1
    
    output_dir.mkdir()
    count = 0
    for jj, input_file in enumerate(input_files):
        # choix des données à utiliser
        #data_raw = extract_raw_ome(input_file)
        
        print("Input file:", input_file)
        with tifffile.TiffFile(input_file) as tif:
            for ii in tqdm(range(len(tif.pages))):
                canaux = subcanals(tif.pages[ii].asarray(), X, Y, vecteurs)
                canaux_corriges = [cv2.warpPerspective(canaux[i],
                                                       deformation[i],
                                                       canaux[i].shape[:2][::-1],
                                                       flags=cv2.INTER_CUBIC) for i in range(6)]
                canaux_normalises = [canaux_corriges[i]/normalisation[i] for i in range(6)]
                data_corr = np.stack([canaux_normalises[0], canaux_normalises[3], canaux_normalises[1], canaux_normalises[4], canaux_normalises[2], canaux_normalises[5]], axis=0)
                data_corr = data_corr #/ data_corr.max() * 65535  # remettre à l'échelle uint16 pour éviter de tronquer des valeurs
                #print(data_corr[0])
                #data_corr[data_corr>=65530.] = 0
                output_file = output_dir / f"{count}.tif"
                tifffile.imwrite(output_file, data_corr)#.astype(np.uint16))
                count+=1
    
def find_pre_loc(img, vmin, connectivity=8, min_size=4, max_size=15, weighted=True, plot=False, ax=None):
    img = np.asarray(img)
    mask = img > vmin
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=int)
    elif connectivity == 4:
        structure = ndimage.generate_binary_structure(2, 1)
    else:
        raise ValueError("connectivity must be 4 or 8")
    labeled, n_islands = ndimage.label(mask, structure=structure)
    if n_islands == 0:
        return np.empty((0, 2)), labeled

    if min_size > 1:
        sizes = ndimage.sum(mask, labeled, index=np.arange(1, n_islands + 1))
        keep_labels = np.where((sizes >= min_size) & (sizes <= max_size))[0] + 1
        new_mask = np.isin(labeled, keep_labels)
        labeled, n_islands = ndimage.label(new_mask, structure=structure)
        if n_islands == 0:
            return np.empty((0, 2)), labeled

    if weighted:
        centers = ndimage.center_of_mass(img, labeled, index=np.arange(1, n_islands + 1))
    else:
        centers = ndimage.center_of_mass(mask, labeled, index=np.arange(1, n_islands + 1))

    centers = np.array(centers)

    centers_corrected = []
    for c in centers:
        if not ((c[0] < 5) | (c[0] > 211) | (c[1] < 5) | (c[1] > 151)):
            centers_corrected.append(c)
    centers_corrected = np.array(centers_corrected)

    if plot:
        sizes = ndimage.sum(mask, labeled, index=np.arange(1, n_islands + 1))
        means = ndimage.mean(img, labeled, index=np.arange(1, n_islands + 1))
        raw_centers = ndimage.center_of_mass(img, labeled, index=np.arange(1, n_islands + 1))
        print(f"{'Island':>8} {'Size':>8} {'Mean intensity':>16} {'Row':>8} {'Col':>8}")
        for i, (s, m, c) in enumerate(zip(sizes, means, raw_centers)):
            print(f"{i+1:>8} {int(s):>8} {m:>16.1f} {c[0]:>8.1f} {c[1]:>8.1f}")

        if ax is not None:
            # draw onto the canvas the caller already owns - no new window, no plt.show()
            ax.clear()
            ax.imshow(img, cmap="gray")
            if len(raw_centers) > 0:
                rows, cols = zip(*raw_centers)
                ax.scatter(cols, rows, s=40, facecolors="none", edgecolors="red")
            ax.set_title(f"{n_islands} islands detected (vmin={vmin})")
        else:
            # fallback: standalone window, only used when NOT called from the Tk dialog
            fig, ax2 = plt.subplots()
            ax2.imshow(img, cmap="gray")
            if len(raw_centers) > 0:
                rows, cols = zip(*raw_centers)
                ax2.scatter(cols, rows, s=40, facecolors="none", edgecolors="red")
            ax2.set_title(f"{n_islands} islands detected (vmin={vmin})")
            plt.show()

    return centers_corrected, labeled
    
def pre_loc_single_frame(frame, vmin, connectivity=8, min_size=4, max_size=15, weighted=True,
                          bg_subtract=True, bg_kernel_size=100, plot=False, ax=None):
    plan0 = frame[0] + frame[1]
    plan1 = frame[2] + frame[3]
    plan2 = frame[4] + frame[5]
    mip = np.maximum(np.maximum(plan0, plan1), plan2)
    if bg_subtract:
        background = uniform_filter(mip, size=bg_kernel_size)
        mip = mip - background
        mip = np.clip(mip, 0, None)
    c0, l = find_pre_loc(mip, vmin, connectivity, min_size, max_size, weighted, plot, ax=ax)
    '''
    plt.imshow(mip)
    for uu in range(c0.shape[0]):
        plt.scatter(c0[uu,1], c0[uu,0],
            marker='o',
            s=100,
            c='none',
            edgecolors='red')
    plt.show()
    '''
    return c0 * 120

def ask_frame_vmin(directory, current_vmin, connectivity, min_size, max_size, weighted, bg_subtract, bg_kernel_size, on_result):
    """
    Opens a dialog asking for a frame number and a vmin value, with an
    embedded plot (not a separate window) that updates on every "Try" click.
    """
    directory = Path(directory)
    result = {}

    root = tk.Tk()
    root.title("Frame / vmin selection")

    tk.Label(root, text="Frame number:").grid(row=0, column=0, padx=8, pady=6, sticky="e")
    frame_entry = tk.Entry(root)
    frame_entry.insert(0, str(0))
    frame_entry.grid(row=0, column=1, padx=8, pady=6)

    tk.Label(root, text="vmin:").grid(row=1, column=0, padx=8, pady=6, sticky="e")
    vmin_entry = tk.Entry(root)
    vmin_entry.insert(0, str(current_vmin))
    vmin_entry.grid(row=1, column=1, padx=8, pady=6)

    # one figure/canvas, created once, reused for every Try click
    fig = Figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, padx=8, pady=8)

    def _run_detection(frame_number, vmin):
        frame = tifffile.imread(directory / f"{frame_number}.tif")
        centers = pre_loc_single_frame(
            frame, vmin,
            connectivity=connectivity,
            min_size=min_size,
            max_size=max_size,
            weighted=weighted,
            bg_subtract=bg_subtract,
            bg_kernel_size=bg_kernel_size,
            plot=True,
            ax=ax,
        )
        canvas.draw_idle()
        return centers

    def _read_values():
        try:
            frame_number = int(frame_entry.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Frame number must be an integer.")
            return None
        try:
            vmin = float(vmin_entry.get())
        except ValueError:
            messagebox.showerror("Invalid input", "vmin must be a number.")
            return None
        return frame_number, vmin

    def _on_try():
        values = _read_values()
        if values is None:
            return
        frame_number, vmin = values
        try:
            centers = _run_detection(frame_number, vmin)
        except FileNotFoundError:
            messagebox.showerror("File not found", f"No tiff found for frame {frame_number}.")
            return
        result["frame_number"], result["vmin"], result["centers"] = frame_number, vmin, centers
        if on_result is not None:
            on_result(frame_number, vmin, centers)

    def _on_stop():
        values = _read_values()
        if values is None:
            return
        frame_number, vmin = values
        try:
            centers = _run_detection(frame_number, vmin)
        except FileNotFoundError:
            messagebox.showerror("File not found", f"No tiff found for frame {frame_number}.")
            return
        result["frame_number"], result["vmin"], result["centers"] = frame_number, vmin, centers
        if on_result is not None:
            on_result(frame_number, vmin, centers)
        root.destroy()

    button_frame = tk.Frame(root)
    button_frame.grid(row=2, column=0, columnspan=2, pady=10)
    tk.Button(button_frame, text="Try", width=10, command=_on_try).pack(side="left", padx=6)
    tk.Button(button_frame, text="Stop", width=10, command=_on_stop).pack(side="left", padx=6)

    root.mainloop()

    return result.get("frame_number"), result.get("vmin"), result.get("centers")

def find_param_preloc(nb_frames_threshold=1, connectivity=8, min_size=4, max_size=15, weighted=True, bg_subtract=True, bg_kernel_size=100, cwd='/mnt/d/Amaury/DATA'):
    directory = (
    Path(filedialog.askdirectory(initialdir=cwd, title="Select the directory containing the tiff files"))
    / "reconstruction"
)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a valid directory")
 
    tiff_files = sorted(
        directory.glob("*.tif"),
        key=lambda p: int(p.stem)
    )
    if not tiff_files:
        raise FileNotFoundError(f"No .tif files found in {directory}")
    
    frame_idx = []
    avg = []
    std = []

    for ii in tqdm(range(len(tiff_files)//50)):
        tiff_path = tiff_files[ii*50]
        frame_idx.append(int(tiff_path.stem))
        dat = tifffile.imread(tiff_path)
        avg.append(np.mean(dat))
        std.append(np.std(dat))
    plt.plot(frame_idx, avg)
    plt.xlabel('Frame')
    plt.ylabel('Background')
    plt.grid()
    plt.show()
    vmin = avg[0] + 4*std[0]
    def show_try(frame_number, vmin):
       print(f"[Try]  frame={frame_number}, vmin={vmin}")
    def show_stop(frame_number, vmin):
        print(f"[Stop] final frame={frame_number}, vmin={vmin}")
    for i in range(nb_frames_threshold):
        final_frame, final_vmin, final_centers = ask_frame_vmin(
    directory=directory, current_vmin=vmin, connectivity=connectivity, min_size=min_size, max_size=max_size, weighted=weighted,
    bg_subtract=bg_subtract, bg_kernel_size=bg_kernel_size,
    on_result=lambda f, v, c: print(f"frame={f}, vmin={v}, {len(c)} spots"))
    return final_vmin

def pre_loc_dataset(vmin, deltaz, first_frame=0, last_frame=None,
                    connectivity=8, min_size=4, max_size=15,
                     weighted=True, plot=False, bg_kernel_size=100, cwd='/mnt/d/Amaury/DATA'):
    """
    Prompts for a directory, then runs pre_loc_single_frame on every "{n}.tif"
    file inside it (n = 0, 1, 2, ...), writing the detected spot locations to
    a CSV as they're found. pre_loc_single_frame is expected to handle the
    max-intensity projection over the 6 channels internally.
 
    Parameters
    ----------
    vmin, connectivity, min_size, max_size, weighted, plot :
        Passed straight through to pre_loc_single_frame.
    output_name : str
        Name of the output CSV, written inside `directory`. Default "pre_loc.csv".
 
    Output
    ------
    Writes `directory/pre_loc.csv` with columns: frame, row, col
    (one row per detected spot; a frame with no detections contributes no rows).
    Results are flushed to disk after every frame, so the file stays valid/usable
    even if the run is interrupted partway through.
 
    Returns
    -------
    output_path : Path
        Path to the written CSV.
    """
    directory = (
    Path(filedialog.askdirectory(initialdir=cwd, title="Select the directory containing the tiff files"))
    / "reconstruction"
)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a valid directory")
    upper_bound = last_frame if last_frame is not None else float("inf")
    tiff_files = sorted(
    (p for p in directory.glob("*.tif")
     if p.stem.isdigit() and first_frame <= int(p.stem) <= upper_bound),
    key=lambda p: int(p.stem)
     )
    if not tiff_files:
        raise FileNotFoundError(
            f"No .tif files found in {directory} for frames in [{first_frame}, {last_frame}]"
        )
    output_name="pre_loc_deltaz_"+str(int(deltaz))+".csv"
    output_path = directory / output_name
 
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "row", "col"])
 
        for tiff_path in tqdm(tiff_files):
            frame_idx = int(tiff_path.stem)
            frame = tifffile.imread(tiff_path)
 
            centers = pre_loc_single_frame(
                frame, vmin,
                connectivity=connectivity,
                min_size=min_size,
                max_size=max_size,
                weighted=weighted,
                bg_kernel_size=bg_kernel_size,
                plot=plot)
 
            for row, col in centers:
                writer.writerow([frame_idx, row, col])
 
            f.flush()
 
    return output_path
    
def verify_preloc(directory, frame_number, cwd):
    csv_files = list(directory.glob("*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No csv file found in {directory}")
    if len(csv_files) > 1:
        raise ValueError(f"Expected exactly one csv file in {directory}, found {len(csv_files)}: {csv_files}")
    csv_path = csv_files[0]

    # first row is "deltaz,<value>"; the real column header is the second row
    df = pd.read_csv(csv_path)
    frame_centers = df[df["frame"] == frame_number]
    print(frame_centers)
    tiff_path = directory / f"{frame_number}.tif"
    if not tiff_path.is_file():
        raise FileNotFoundError(f"No tiff found for frame {frame_number} in {directory}")
    frame = tifffile.imread(tiff_path)

    plan0 = frame[0] + frame[1]
    plan1 = frame[2] + frame[3]
    plan2 = frame[4] + frame[5]
    mip = np.maximum(np.maximum(plan0, plan1), plan2)

    fig, ax = plt.subplots()
    ax.imshow(mip, cmap="gray")
    if len(frame_centers) > 0:
        # stored values are scaled by 120 (see pre_loc_single_frame's `return c0*120`),
        # so divide back down to pixel coordinates to overlay correctly
        rows = frame_centers["row"].to_numpy() / 120
        cols = frame_centers["col"].to_numpy() / 120
        ax.scatter(cols, rows, marker="o", s=100, facecolors="none", edgecolors="red", linewidths=2)    
    ax.set_title(f"Frame {frame_number} - {len(frame_centers)} spots")
    plt.show()
    