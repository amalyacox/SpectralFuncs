### SHG Functions

from numpy import load
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os
from General_Functions_ACJ import * 

def load_data(fn, bg_subtract=False, fit=True):
    data = load(fn)
    lst = data.files

    data = load(fn)
    lst = data.files

    angle_data = data['angle']
    intensity = data['intensity']
    spectra = data['spectra']

    if bg_subtract: 
        intensity = intensity - bkg 
        
    return angle_data, intensity 


def plot_multiple(fns, labels, plot_polar=True, plot_ang_raw=False, plot_fit=True, average=False, bg=False, **kwargs):
    
    fig = plt.figure()
    
    
    if plot_polar: 
        ax = fig.add_subplot(111, projection = 'polar')
    for i, fn in enumerate(fns): 
        if plot_fit == True: 
            angle_data, intensity = load_data(fn, fit=True)
        else: 
            angle_data, intensity = load_data(fn, fit=False)

        if plot_ang_raw: 
            ax.plot(angle_data, intensity, **kwargs)

        if plot_polar: 
            ax.plot(angle_data*(1/360)*(2*np.pi), intensity, label=labels[i], **kwargs)
            ax.set_yticklabels([])
        if average == True: 
            if i == 0: 
                av_spec = intensity
            else: 
                av_spec += intensity 
    av_spec = av_spec/(i+1)
    
    if bg != False: 
        angle_data, intensity = load_data(bg, fit=False)
        bg_sub_spec = av_spec - intensity 
        return angle_data, av_spec, bg_sub_spec
    else: 
        return angle_data, av_spec 

# plot_multiple([f1, f2], ['ML1', 'ML4'])
# plt.legend()

def sine_func(x, a, b,c,d):
    return a * np.sin((b * x) + c) + d

def angle_dif(fns, labels, plot_raw_polar = False, plot_fit_polar=True, **kwargs): 
    fig = plt.figure(figsize=(7.5, 10))
    ax = fig.add_subplot(211, projection = 'polar')
    ax2 = fig.add_subplot(212)

    offsets = []
    for i, fn in enumerate(fns): 
        angle_data, intensity = load_data(fn, fit=False)
        
        angle_dat_radian = angle_data *((2*np.pi)/360)
        params, params_covariance = optimize.curve_fit(sine_func, angle_dat_radian, intensity, [160000,6,-(2*np.pi)/9,85000])

        fit = sine_func(np.linspace(0,2*np.pi,1000), *params)
        maxid = np.argmax(fit)
        offset_rad = np.linspace(0,2*np.pi,1000)[maxid] 
        offset_rad = offset_rad % (np.pi/3)
        offset_deg = offset_rad * 180/np.pi

        print(offset_deg)
        # offset = offset % 60
        # newid = np.argmin(np.abs(np.linspace(0,2*np.pi,1000) * 180/np.pi - offset))

        if plot_fit_polar: 
            ax.plot(angle_data*(1/360)*(2*np.pi), intensity) #'o'
            ax.plot(np.linspace(0,2*np.pi,1000),sine_func(np.linspace(0,2*np.pi,1000), *params), label=labels[i] + 'fit')
            ax.plot(offset_rad, fit[maxid], 'k*')
            ax.set_yticklabels([])
            ax.legend()
        
        if plot_raw_polar: 
            ax.plot(angle_data*(1/360)*(2*np.pi), intensity, label=labels[i], **kwargs)
            ax.set_yticklabels([])
            ax.set_title('Raw Data')
            ax.legend()

        ax2.plot(np.linspace(0,2*np.pi,1000),sine_func(np.linspace(0,2*np.pi,1000), *params))
        ax2.plot(offset_rad, fit[maxid], 'k*')
        offsets.append(offset_rad)
    
    deg_dif = np.abs(offsets[1] - offsets[0]) * 180/np.pi
    ax2.set_title(f'twist angle:{round(deg_dif, 2)}deg')
        



# angle_dif([f1, f2], ['ML1', 'ML2'], plot_fit_polar=False, plot_raw_polar=True)
# ax.legend()


def strain_inten(phi, A, B, theta, delta, i_0): #i_0
#     A = (1- nu) * (p1 + p2) * (exx + eyy) + 2*chi2_0
#     B = (1 + nu) * (p1 - p2) * (exx - eyy)
#     I think I need the intensity background because my plots don't go to 0. Keeping this or not changes the offset angle best fit 
    I = (A * np.cos(3*phi - 3*delta) + B*np.cos(2*theta + phi - 3*delta))**2 + i_0
    
    return I 

def fit_strain_inten(inten, phi, p0=[150, 0.1, np.pi/3, np.pi/2, 1e4]): #i_0 = 1e4
    ang_rad = phi * np.pi/180
    popt, pcov = optimize.curve_fit(strain_inten, ang_rad, inten, p0)
    fit_x = np.linspace(0,2*np.pi, 1000)
    fit_y = strain_inten(fit_x, *popt)
    return fit_x, fit_y, popt

def AX0(exx, eyy, p1=-0.13, p2=-0.58, nu=0.25):
    """
    A/X0
    """ 
    return (1-nu)*(p1 + p2)*(exx + eyy) + 2
    
def BX0(exx, eyy, p1=-0.13, p2=-0.58, nu=0.25): 
    """
    B/X0
    """ 
    return (1+nu)*(p1 - p2)*(exx - eyy)

def offset(inten): 
    maxid = np.argmax(inten)
    offset_rad = np.linspace(0,2*np.pi,len(inten))[maxid] 
    offset_rad = offset_rad % (np.pi/3)
    return offset_rad

def ey_from_fit(A, B, p1=-0.13, p2=-0.58, nu=0.25): 
    """"
    A = A/X0
    B = B/X0
    """
    KA = (1-nu)*(p1 + p2)
    KB = (1+nu)*(p1 - p2)
    return 0.5 * ((A-2)/KA - B/KB)

def ex_from_fit(A, B, p1=-0.13, p2=-0.58, nu=0.25): 
    """"
    A = A/X0
    B = B/X0
    """
    KA = (1-nu)*(p1 + p2)
    KB = (1+nu)*(p1 - p2)
    return B/KB + ey_from_fit(A,B, p1, p2, nu)