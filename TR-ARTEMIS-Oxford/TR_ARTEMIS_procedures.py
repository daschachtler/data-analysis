# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 13:40:55 2018

@author: dasch
"""


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import numpy.core.multiarray
from pylab import *
from cmath import *
import glob 
import os
import re
from scipy.special import wofz
from scipy.optimize import curve_fit
import numpy.ma as ma
from numpy.random import uniform, seed
from cmath import *
from scipy.ndimage.filters import gaussian_filter
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import gc
import scipy.special as sse

nums = re.compile(r"[+-]?\d+(?:\.\d+)?")

def gauss(x_axis, G, mu, sig):
    gauss= G*1/np.sqrt(2*pi*sig**2)*np.exp(-(x_axis-mu)**2/(2*sig**2))
    return gauss
    
def EMG(x, G, mu, sig, lam): # exponential modified gaussian
    erfc_value = (mu+lam*sig*sig-x)/(1.41*sig)
    emg=G*0.5*lam*np.exp(0.5*lam*(2*mu+lam*sig*sig-2.*x))*sse.erfc(erfc_value) 
    return emg 

def fit_time_93(t_axis, b_left, m_left, t0_left,
                m_mid, 
                b_right, m_right, t0_right,
                G, mu, sig, lam):

    func=np.heaviside(-t_axis+t0_left, 1)*(t_axis*m_left+b_left) \
    + np.heaviside(t_axis-t0_left, 0.)*np.heaviside(-t_axis+t0_right, 0.)*(m_mid*t_axis+(b_right+b_left)/2.) \
    + np.heaviside(t_axis-t0_right, 1.)*(t_axis*m_right+b_right) \
    + EMG(t_axis, G, mu, sig, lam)

    return func
    
def fit_time_143(t_axis, b_left, m_left, t0_left,
                m_mid, 
                b_right, m_right, t0_right,
                G, mu, sig, lam):

    func=np.heaviside(-t_axis+t0_left, 1)*(t_axis*m_left+b_left) \
    + np.heaviside(t_axis-t0_left, 0.)*np.heaviside(-t_axis+t0_right, 0.)*(m_mid*t_axis+(b_right-t0_right*m_mid)) \
    + np.heaviside(t_axis-t0_right, 1.)*(t_axis*m_right+b_right) \
    + EMG(t_axis, G, mu, sig, lam)

    return func
    
def fit_function2(x_axis, const_bg, lin_bg,
                 G1, mu1, sig1,
                 G1_2, mu1_2, sig1_2,
                 G2, mu2, sig2,
                 G3, mu3, sig3):
    fit_func=const_bg+lin_bg*x_axis+gauss(x_axis, G1, mu1, sig1)\
    + gauss(x_axis, G1_2, mu1_2, sig1_2)\
    + gauss(x_axis, G2, mu2, sig2)\
    + gauss(x_axis, G3, mu3, sig3)
    return fit_func
    
def fit_function2Ef(x_axis, const_bg, lin_bg,
                 G1, mu1, sig1,
                 G2, mu2, sig2,
                 G3, mu3, sig3,
                 G4, mu4, sig4,
                 G5, mu5, sig5):
    fit_func=const_bg+lin_bg*x_axis\
    + gauss(x_axis, G1, mu1, sig1)\
    + gauss(x_axis, G2, mu2, sig2)\
    + gauss(x_axis, G3, mu3, sig3)\
    + gauss(x_axis, G4, mu4, sig4)\
    + gauss(x_axis, G5, mu5, sig5)
    return fit_func
    
def fit_function1(x_axis, const_bg, lin_bg,
                 G1, mu1, sig1,
                 G2, mu2, sig2,
                 G3, mu3, sig3):
    fit_func=const_bg+lin_bg*x_axis+gauss(x_axis, G1, mu1, sig1)\
    + gauss(x_axis, G2, mu2, sig2)\
    + gauss(x_axis, G3, mu3, sig3)
    return fit_func
    
def fit_functionEf(x_axis, const_bg, lin_bg,
                 G1, mu1, sig1):
    fit_func=const_bg+lin_bg*x_axis+gauss(x_axis, G1, mu1, sig1)
    return fit_func
    
def fit_function12(x_axis, const_bg, lin_bg,
                 G1, mu1, sig1,
                 G1_2, mu1_2, sig1_2,
                 G2, mu2, sig2,
                 G3, mu3, sig3):
    fit_func=const_bg+lin_bg*x_axis+gauss(x_axis, G1, mu1, sig1)\
    + gauss(x_axis, G1_2, mu1_2, sig1_2)\
    + gauss(x_axis, G2, mu2, sig2)\
    + gauss(x_axis, G3, mu3, sig3)
    return fit_func
    
def timelist(DL):
    t1=float(nums.search(DL[0]).group(0))
    t2=float(nums.search(DL[1]).group(0))
    dt=float(nums.search(DL[2]).group(0))
   
    t=frange(t1,t2,dt)
    return t
    
def load_files2(path):
    files=glob.glob(path) 
    files=np.flip(files,0)

    data=zeros((shape(loadtxt(files[0]))[0],shape(loadtxt(files[0]))[1],timesteps))
    i=0
    for file in files[:timesteps]:
        data[:,:,i]=flipud(loadtxt(file))
        print file

        i+=1
    return data
    
def load_files(path):
    files=glob.glob(path) 

    data=zeros((shape(loadtxt(files[0]))[0],shape(loadtxt(files[0]))[1],timesteps))
    i=0
    for file in files[:timesteps]:
        data[:,:,i]=flipud(loadtxt(file))
        print file

        i+=1
    return data
    
def EnA_vs_t(data, angle_1,angle_2,E1,E2):
    
    int_HOMO=zeros((timesteps))
    for i in range(timesteps):
        int_HOMO[i]=np.sum(data[E1:E2,angle_1:angle_2,i])
        
#    imshow(data[E1:E2,angle_1:angle_2,1])
    int_max=max(int_HOMO)
    int_HOMO/=int_max
    
    return int_HOMO
    
def clean_spec(bg_position, bg_len, data, angle_1, angle_2, E1, E2):
    nE=344
    data_cropped=zeros((nE, angle_2-angle_1))
    data_cropped[:,:]=data[:,angle_1:angle_2,0]
    data_cropped/=np.amax(np.abs(data_cropped))
    
    fig = plt.figure()
#    title('%(measurement_name)s' %vars())
    fig.set_size_inches((9, 9), forward=False)
    ax = fig.add_subplot(1, 1, 1)
    
    imshow(data_cropped,aspect='auto', extent=[angles[angle_1],
                                               angles[angle_2],energy[0],energy[-1]])
                                               
    ax.tick_params(axis='both', labelsize=20)                                          
    xlim(angles[angle_1],angles[angle_2])
    ylim(energy[0],energy[-1])
    
    xlabel(r'angle', fontsize=20)
    ylabel(r'E-E$_{f}$ (eV)', fontsize=20)
    colorbar()         
    
    directory=measurement_folder+'sbg_pos=%(bg_position)s, bg_len=%(bg_len)s/sclean spec/' %vars()
    imgname='E1=%(E1)s, E2=%(E2)s.png' %vars()
    if not os.path.exists(directory):
            os.makedirs(directory)
    savefig(directory+imgname, dpi=200, bbox_inches='tight')                                      
    
def energy_vs_t(data, angle_1,angle_2, E1_t, E2_t, energy_range, bg=True, relative=True):
    
    energy_t=zeros((energy_range,timesteps))
    background=zeros((energy_range))
    data=data[:,angle_1:angle_2,:]
    energy_t_diff=zeros((energy_range,timesteps))
    
    for j in range(timesteps):
        for i in range(energy_range):
          
            energy_t[i,j]=np.sum(data[E2_t+i,:,j])
    
    if bg==True:
        
        for i in range(bg_len):
            background+=energy_t[:,bg_position+i]
            
        background/=bg_len
        for i in range(timesteps):
            energy_t_diff[:,i]=energy_t[:,i]-background
            
    if relative==True:
        for i in range(timesteps):
            energy_t_diff[:,i]/=background
            
    if relative==False and bg==False:
        energy_t_diff=energy_t
#    energy_t_diff=energy_t

    return energy_t_diff
    
def angle_vs_t(data, angle_1,angle_2, E1_at, E2_at, angle_range, bg=True):
    
    angle_t=zeros((angle_range,timesteps))
    background=zeros((angle_range))
    data=data[E1_at:E2_at,:,:]
    
    for j in range(timesteps):
        for i in range(angle_range):
          
            angle_t[i,j]=np.sum(data[:,angle_1+i,j])
    
    if bg==True:
        for i in range(bg_len):
            background+=angle_t[:,bg_position+i]
            
        background/=bg_len
        for i in range(timesteps):
            angle_t[:,i]-=background
    angle_t=np.flipud(angle_t.T)  
    return angle_t
    
def difference_plot(data, angle_1,angle_2, E1, E2, sigma=[5,5,0]):
    nE=E2-E1
    data_cropped=zeros((nE, angle_2-angle_1,timesteps))
    data_cropped[:,:,:]=data[(344-E2):(344-E1),angle_1:angle_2,:]
    data_cropped/=np.amax(np.abs(data_cropped))
    
    background=zeros((nE, angle_2-angle_1))
    
    for i in range(bg_len):
        background+=data_cropped[:,:,bg_position+i]
        
    background/=bg_len
    
    for i in range(timesteps):
        data_cropped[:,:,i]-=background
        
    data_cropped=gaussian_filter(data_cropped, sigma)
    data_cropped /= np.amax(np.abs(data_cropped))
    
    return data_cropped, sigma
    
def save_diffpics(bg_position, bg_len, sigma, E1_d):
    data_diff, sigma_r=difference_plot(data, angle_1_d, angle_2_d, E1_d, E2_d, sigma)
    
#    vmax=np.amax(data_diff)
#    vmin=np.amin(data_diff)
    vmax=1.
    vmin=-1.
    for i in range(timesteps):

        f, (a0, a1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[1, 15]})
        y_val=zeros((timesteps))
        a0.plot(DL, y_val, 'o',  color='k', markersize=5)
        a0.plot((DL[0],DL[-1]), (0,0), color='k', ls='-')
        a0.plot(DL[i], y_val[i], 'o', color='g', markersize=10)
        a0.set_xlabel('pump-probe delay (fs)', fontsize=15)
        a0.set_yticklabels([])
#        a0.xaxis.set_ticks([])
        a0.yaxis.set_ticks([])
        a0.set_xticks(a0.get_xticks()[::2])
        a0.set_xlim(DL[0]-50, DL[-1]+50)
        for spine in a0.spines.itervalues():
            spine.set_visible(False)
        
#        ax = fig.add_subplot(212)
#        axis('on')
        img=a1.imshow(data_diff[:,:,i], aspect='auto',extent=[angles[angle_1],
                         angles[angle_2],energy[E1_d],energy[E2_d]], vmin=vmin,vmax=vmax)
        t=DL[i]
        a1.set_title(r'time=%(t)s fs' %vars(), fontsize=15)
        a1.set_xlabel('angle', fontsize=20)
        a1.set_ylabel(r'E-E$_{f}$', fontsize=20)
        colorbar(img)
        
        f.tight_layout()
        
        directory=measurement_folder+'sbg_pos=%(bg_position)s, bg_len=%(bg_len)s/smovie/E1=%(E1_d)s/%(sigma_r)s/' %vars()
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        f.savefig(directory + r'%(i)s, %(t)s.png' %vars())
        plt.clf()
        plt.close('all')
        
def make_video(bg_position, bg_len, E1_d, outimg=None, fps=1, size=None,
               is_color=True, format="XVID",outvid='', sigma=[4,4,1]):
    sigma_r=difference_plot(data, angle_1_d, angle_2_d, E1_d, E2_d, sigma)[1]
    outvid=measurement_folder+'sbg_pos=%(bg_position)s, bg_len=%(bg_len)s/smovie/E1=%(E1_d)s/%(sigma_r)s.avi' %vars()
    images=measurement_folder+'sbg_pos=%(bg_position)s, bg_len=%(bg_len)s/smovie/E1=%(E1_d)s/%(sigma_r)s/*.png' %vars()

    images=images.replace('/[', '/[[]')
    images=images.replace(']/', '[]]/')

    images=sorted(glob.glob(images), key=os.path.getmtime)
    
    fourcc = VideoWriter_fourcc(*format)
    
    vid = None
    
    for image in images:
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        
        vid.write(img)
        
    vid.release()
    return vid
    
def make_video_Espec(images, outvid, outimg=None, fps=1, size=None,
               is_color=True, format="XVID"):

    images=images.replace('=[', '=[[]')
    images=images.replace(']/', '[]]/')
    
    images=sorted(glob.glob(images), key=os.path.getmtime)
    
    fourcc = VideoWriter_fourcc(*format)
    
    vid = None
    
    for image in images:
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        
        vid.write(img)
        
    vid.release()
    return vid
    
def video_series(bg_position, bg_len, s, r, E1_d, fps=1):
    for i in range(s):
        for j in range(r):
            sigma=[i*4+4,i*4+4,0.3*j+0.3]
            save_diffpics(bg_position, bg_len, sigma, E1_d)
           
            make_video(bg_position, bg_len, E1_d, outimg=None, fps=fps, size=None,
                   is_color=True, format="XVID", sigma=sigma)
            plt.clf()
            plt.close('all')
                   
def smth_vs_t_plot(bg_position, bg_len, angle_1, angle_2,
           a1_at, a2_at,E1_at,E2_at, angle_range,
           a1_et, a2_et, E1_et, E2_et, energy_range,
           angle_1_d, angle_2_d, E1_d, E2_d,
           a1_eat, a2_eat, E1_eat, E2_eat,
           s, energy, angles, measurement_name, relative,
           n_energy, n_angle, measurement_folder, n=10
           ):
                       
    f, (a0, a1) = plt.subplots(1,2, gridspec_kw = {'width_ratios':[1, 2]})
#    f.set_size_inches((21,7), forward=True)
    f.set_size_inches((21,10.5), forward=True)
    
    fontsize=25
    
    data_cropped=zeros((nE, angle_2-angle_1))
    data_cropped[:,:]=data[:,angle_1:angle_2,0]
    data_cropped/=np.amax(np.abs(data_cropped))
    img=a0.imshow(data_cropped,aspect='auto', extent=[angles[angle_1],
                                               angles[angle_2],energy[0],energy[-1]])
    
#    a0.plot((angles[a1_et],angles[a1_et]),(energy[E1_et+1],energy[E2_et]),ls='--',color='k')
#    a0.plot((angles[a2_et],angles[a2_et]),(energy[E1_et+1],energy[E2_et]),ls='--',color='k')
#    a0.plot((angles[a1_et],angles[a2_et]),(energy[E1_et+1],energy[E1_et+1]),ls='--',color='k')
#    a0.plot((angles[a1_et],angles[a2_et]),(energy[E2_et],energy[E2_et]),ls='--',color='k')
#    
#    a0.plot((angles[a1_at],angles[a1_at]),(energy[nE-E1_at],energy[nE-E2_at]),ls='--',color='k')
#    a0.plot((angles[a2_at],angles[a2_at]),(energy[nE-E1_at],energy[nE-E2_at]),ls='--',color='k')
#    a0.plot((angles[a1_at],angles[a2_at]),(energy[nE-E1_at],energy[nE-E1_at]),ls='--',color='k')
#    a0.plot((angles[a1_at],angles[a2_at]),(energy[nE-E2_at],energy[nE-E2_at]),ls='--',color='k')
    
    a0.plot((angles[a1_et],angles[a1_et]),(energy[nE-E1_at],energy[nE-E2_at]),ls='-',color='k', lw=1.5)
    a0.plot((angles[a2_et],angles[a2_et]),(energy[nE-E1_at],energy[nE-E2_at]),ls='-',color='k', lw=1.5)
    a0.plot((angles[a1_et],angles[a2_et]),(energy[nE-E1_at],energy[nE-E1_at]),ls='-',color='k', lw=1.5)
    a0.plot((angles[a1_et],angles[a2_et]),(energy[nE-E2_at],energy[nE-E2_at]),ls='-',color='k', lw=1.5)
    
#    a0.set_yticks(np.arange(-15,2+0.5,0.5))
    a0.set_xticks(np.arange(-15,15+15, 15))
    a0.set_xlim(angles[angle_1],angles[angle_2])
    a0.set_ylim(energy[0],energy[-1])
    
    a0.set_xlabel(r'angle', fontsize=fontsize)
    a0.set_ylabel(r'E-E$_{f}$ (eV)', fontsize=fontsize)
    a0.tick_params(labelsize=fontsize)
    
    cbar = colorbar(img, ax=a0)
    cbar.set_label('normalized intensity (au)', size=fontsize)
    cbar.ax.tick_params(labelsize=fontsize) 
    
    #############################  energy vs t  ##########################
    
    energy2=energy[E1_et:E2_et]

    t, E=np.meshgrid(DL/1000., energy2)
    energy_t=np.flipud(energy_vs_t(data=data,angle_1=a1_et,angle_2=a2_et, 
                                   E1_t=nE-E1_et, E2_t=nE-E2_et, energy_range=energy_range, relative=relative))
    
    sigma=[s,0.]
    np.savetxt(measurement_folder+'E_axis.dat', energy2)
    np.savetxt(measurement_folder+'rel_Spec_vst.dat', energy_t)
    np.savetxt(measurement_folder+'time_stamps.dat', DL)
    
    energy_t2=gaussian_filter(energy_t[:,:], sigma)
    
    if (relative==True):
        energy_t2*=100.
    else:
        energy_t2/=np.amax(energy_t2)
#    levels = MaxNLocator(nbins=1000).tick_values(np.amin(energy_t2), np.amax(energy_t2))
    vmin=np.amin(energy_t2)
#    vmin=-10
    vmax=np.amax(energy_t2)
#    vmax=0.7
    img2=a1.contourf(t[:,:],E[:,:], energy_t2, np.linspace(energy_t2.min(), energy_t2.max(), 100), vmin=vmin, vmax=vmax)
    
#    a1.plot((-0.1, -0.1), (-2, 1.5), ls='--')
    a1.set_xlabel(r'pump-probe delay (ps)', fontsize=fontsize)
    a1.set_ylabel(r'E-E$_{f}$ (eV)', fontsize=fontsize)
    a1.tick_params(labelsize=fontsize)
    
    a1.grid(axis='both', lw=1, color='k')
    cbar2 = plt.colorbar(img2, ax=a1, format='%.1f')
    if (relative==True):
        cbar2.set_label('relative difference in %', size=fontsize)
    else: 
        cbar2.set_label('absolute difference (normalized)', size=fontsize)
#    cbar2.set_label('intensity normalized', size=fontsize)
    cbar2.ax.tick_params(labelsize=fontsize) 
    
    tight_layout()
    ############################  angle vs t  ##########################
    
#    ax = fig.add_subplot(2, 2, 3)
#    
#    angle_x=angles[a1_at:a2_at]
#    
#    A, t=meshgrid(angle_x, DL)
#    angle_t=np.flipud(angle_vs_t(data=data,angle_1=a1_at,angle_2=a2_at, 
#                                   E1_at=E1_at, E2_at=E2_at, angle_range=angle_range))
#    
#    sigma=[1., s+4]
#    angle_t = gaussian_filter(angle_t, sigma)
#    angle_t /= np.amax(np.abs(angle_t))
#    
#    levels = MaxNLocator(nbins=30).tick_values(-1, 1)
#                               
#    contourf(A[:,:], t[:,:], angle_t, levels=levels)
#    
#    xlabel(r'angle', fontsize=15)
#    ylabel(r'pump-probe delay (fs)', fontsize=15)
#    
#    
#    colorbar()    
    
    #############################  (E & a) vs t  ##########################
#    ax = fig.add_subplot(2, 2, 4)
#    
#    ena_t=EnA_vs_t(data=data, angle_1=a1_eat,angle_2=a2_eat,E1=E1_eat,E2=E2_eat)
#    ax.grid(color='k', linestyle='--', linewidth=1, axis='x')
#    plot(DL, ena_t, lw=2)
#    
#    xlabel(r'pump-probe delay (fs)', fontsize=15)
#    ylabel(r'counts (a.u.)', fontsize=15)
#    
#    xlim(DL[0], DL[-1])
#    ylim(np.min(ena_t)-0.001, np.max(ena_t)+0.001)
    
    ############################ savefig ##################################
    E=(E2_at+E1_at)/2
    E=energy[nE-E]
    E=round(E,2)
    
    directory=measurement_folder+'sfit, sbg_pos=%(bg_position)s, bg_len=%(bg_len)s/smth_vs_t/E1=%(E1_et)s, a1=%(a1_at)s, a2=%(a2_at)s, nE=%(n_energy)s, nA=%(n_angle)s/' %vars()
    imgname='%(E)s eV, a1=%(a1_et)s, a2=%(a2_et)s, E1=%(E1_at)s, E2=%(E2_at)s.png' %vars()
    if not os.path.exists(directory):
            os.makedirs(directory)
    savefig(directory+imgname, dpi=200, bbox_inches='tight')
    
    if n_energy>1:
        plt.clf()
        plt.close('all')
    