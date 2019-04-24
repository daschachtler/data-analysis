# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 13:29:37 2018

@author: dasch
"""
import procedures_fit
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
from scipy import integrate
import scipy

from scipy.special import wofz
from scipy.optimize import curve_fit
import numpy.ma as ma
from numpy.random import uniform, seed
from cmath import *
from scipy.ndimage.filters import gaussian_filter
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import gc
from datetime import datetime
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.ticker as ticker

startTime = datetime.now()

nums = re.compile(r"[+-]?\d+(?:\.\d+)?")

smth_vs_t_series=False
E_spec_t=False
Espec_fit=False
vid_E_spec_t_var=False
video_series_var=False
clean_spec_var=False
Espec_fit=False

smth_vs_t_series=True
#E_spec_t=True
#Espec_fit=True
#vid_E_spec_t_var=True
#video_series_var=True
#clean_spec_var=True

HOMO_2=False
        
i=10
s=2.
relative=True
whole=True

fontsize=25
    
guess=(0.88-0.023*Ef, #constant bg
       -0.023,    #slope of lin bg
       0.5, 35.2-Ef, 0.41, #HOMO 
       1.15, 33.9-Ef, 0.51, #HOMO-1
       .1, 33.-Ef, 0.1) #Substrate

if (i==0): #
    EHomo=34.86
    Ef=EHomo+1.35
    guess=(0.88-0.023*Ef, #constant bg
       -0.023,    #slope of lin bg
       0.5, 35.-Ef, 0.41, #HOMO 
       1.15, 33.-Ef, 0.51, #HOMO-1
       0.01, 32.7-Ef, 0.1) #Substrate
    guess_bounds=([0.6-0.023*Ef,-0.0231, 0.5, 34.2-Ef, 0.2, 1., 33.-Ef, 0.3, 0.01, 32.7-Ef, 0.1],
                  [0.881-0.023*Ef,-0.023, 0.9, 35.4-Ef, 0.5, 1.4, 33.6-Ef, 0.7, 0.0101, 32.71-Ef, .101])
if (i==1):
    EHomo=25.75 #scan 50
    Ef=EHomo+1.35
if (i==3):
    EHomo=35.2    
    Ef=EHomo+1.35
    guess=(0.88-0.023*Ef, #constant bg
       -0.023,    #slope of lin bg
       0.5, 35.-Ef, 0.41, #HOMO 
       1.15, 33.5-Ef, 0.51, #HOMO-1
       1., 33.-Ef, 0.2) #Substrate
    guess_bounds=([0.88-0.023*Ef,-0.0231, 0.3, 35.-Ef, 0.2, .8, 32.5-Ef, 0.3, 0.9, 33.-Ef, 0.2],
                  [0.881-0.023*Ef,-0.023, 0.9, 35.5-Ef, 0.6, 1.4, 34.-Ef, 0.7, 1.01, 33.1-Ef, .21]) 
if (i==4):
    EHomo=35.42    
    Ef=EHomo+1.35
    guess_bounds=([0.88-0.023*Ef,-0.0231, 0.5, 35.0-Ef, 0.35, 1., 33.9-Ef, 0.3, 0.1, 33.-Ef, 0.1],
                  [0.881-0.023*Ef,-0.023, 0.9, 35.8-Ef, 0.5, 1.4, 34.3-Ef, 0.7, 0.101, 33.01-Ef, .101]) 
if (i==5):
    EHomo=35.28 
    Ef=EHomo+1.35
    guess_bounds=([0.88-0.023*Ef,-0.0231, 0.5, 35.0-Ef, 0.35, 1., 33.9-Ef, 0.3, 0.1, 33.-Ef, 0.1],
                  [0.881-0.023*Ef,-0.023, 0.9, 35.6-Ef, 0.5, 1.4, 34.3-Ef, 0.7, 0.101, 33.01-Ef, .101]) 
if (i==6):
    EHomo=35.49       
    Ef=EHomo+1.35
    guess_bounds=([0.88-0.023*Ef,-0.0231, 0.5, 35.0-Ef, 0.35, 1.1, 34.-Ef, 0.5, 0.1, 33.-Ef, 0.1],
                  [0.881-0.023*Ef,-0.023, 0.9, 35.6-Ef, 0.5, 1.45, 34.5-Ef, 0.7, 1, 33.5-Ef, .3]) 
if (i==7):
    EHomo=35.39       
    Ef=EHomo+1.35
    guess_bounds=([0.88-0.023*Ef,-0.0231, 0.5, 35.0-Ef, 0.35, 1.1, 34.-Ef, 0.5, 0.1, 33.-Ef, 0.1],
                  [0.881-0.023*Ef,-0.023, 0.9, 35.6-Ef, 0.5, 1.3, 34.3-Ef, 0.7, 1, 33.5-Ef, .3]) 

  
fps=1
#n_measurements=11
#for i in range(n_measurements):
main_folder='C:/Studium/Artemis 2018/measurements/580,660/'
if i==0:
    main_folder='C:/Studium/Artemis 2018/measurements/550/'
    
if i==8 or i==2:
    main_folder='C:/Studium/Artemis 2018/measurements/400/'
measurements=os.listdir(main_folder)
measurement_folder=main_folder+measurements[i]+'/'
if (i==1):
    measurement_folder='C:/Studium/Artemis 2018/measurements/400/050 ml PnTiO2 30min evap x/'
file_path=measurement_folder+os.listdir(measurement_folder)[2]+'/*.tsv'
info_path=measurement_folder+'info.tsv'
measurement_name=measurements[i]
print file_path

#EXTRACT INFO
infofile=loadtxt(info_path,dtype='string',delimiter=';',comments='$')

#CONSTANTS FROM INFO FILE
Ekin=float(nums.search(infofile[1]).group(0))
Epass=float(nums.search(infofile[2]).group(0))
t0=nums.search(infofile[7]).group(0)
nE=344
nA=256
dE=Epass*0.2/nE
BW=Epass*0.2/2.
Ekin-=Ef
energy=frange(Ekin-BW,Ekin+BW-dE,dE)
angles=frange(-30,30,60./nA)

#CREATING LIST WITH TIMESTAMPS
Delaylist=infofile[5]
Delaylist=Delaylist.split('#')
DL1=Delaylist[0].split('_')
if not (i==2):
    DL2=Delaylist[1].split('_')
    DL3=Delaylist[2].split('_')
    
    DL2=timelist(DL2)
    DL3=timelist(DL3)   

DL1=timelist(DL1)

DL=zeros((0))
DL=np.append(DL,DL1,0)
if not (i==2):
    DL=np.append(DL,DL2,0)
    DL=np.append(DL,DL3,0)
if (i==7 or i==10 or i==13):
    DL4=Delaylist[3].split('_')
    DL4=timelist(DL4)
    DL=np.append(DL,DL4,0)

timesteps=len(DL)
if (i==3 or i==4 or i==5 or i==8 or i==2 or i==1):
    data=load_files2(file_path)
else:
    data=load_files(file_path)
data.flags.writeable = False

#background settings##########################################

bg_position=0
bg_len=1
if (i==3 or i==4 or i==5):
    bg_len=2
if (i==6 or i==7):
    bg_len=1
if i==0:
    bg_len=4

if i==13:
    bg_len=3
    
#normal plot parameters
angle_1, angle_2=40,210

#angle vs t paramters (smth vs t: angle here)
a1_at, a2_at,E1_at,E2_at=50, 200 ,210 ,245
angle_range=a2_at-a1_at

#energy_vs_t parameters (smth vs t: Energy here)
a1_et, a2_et, E1_et, E2_et=110, 150, 100, 334
energy_range=E2_et-E1_et

#Diff_plot parameters
angle_1_d, angle_2_d, E1_d, E2_d=40, 210, 10, 330

#EnA_vs_t_plot parameters
a1_eat, a2_eat, E1_eat, E2_eat=a1_et, a2_et, E1_at, E2_at

#time parameters as indices of list DL
t0_depl=6
t0_peak=9

#whole spectrum, only homo angle


if (Espec_fit==False):
    a1_at, a2_at=60, 190
    E1_et, E2_et=20, 330
   
else:
    a1_at, a2_at=70, 150
#    a1_at, a2_at=60, 190
    E1_et, E2_et=20, 250
    if (i==3):
        a1_at, a2_at=110, 150
    if (i==0):
        a1_at, a2_at=100, 160
        E1_et, E2_et=50, 250
        


#Start HOMO - 38.5 eV
#a1_at, a2_at=55, 195
#E1_et, E2_et= 94, 334

Ef=False

#~EF - 38.5 eV
#a1_at, a2_at=50, 200
#E1_et, E2_et= 214, 334
#Ef=True

"""E_spec"""################################################################
#Raster settings E_spectra
n_energyE=1
n_angleE=1

#angle and energy 
#std-angle values: full window: angle= 50, 200
#std-energy values: full window: energy= 64, 334)
#Ef and above:energy= 190,330
#peak area: energy= 80, 220

#Gauss filter
sE, rE=1,1

if E_spec_t==True:
    if (Espec_fit==False):
        log=2
    else:
        log=1
    for i in range(log):
        log_Espec=True
        if (i==0):
            log_Espec=False
        else:
            log_Espec=True
        for s in range(sE):
            for r in range(rE):
                if (Espec_fit==True):
                    s_Et=[2.+s*3,0.3*r]
                else: 
                    s_Et=[2.+s*3,0.3*r]
                for i in range(n_energyE):
                    for j in range(n_angleE):
                        bE=(E2_et-E1_et)/n_energyE
                        bA=(a2_at-a1_at)/n_angleE
                        E1_at, E2_at=nE-E2_et+i*bE, nE-E2_et+(i+1)*bE
                        a1_et, a2_et=a1_at+j*bA, a1_at+(j+1)*bA
                        energy_range=E2_at-E1_at
                        E_spec_t=energy_vs_t(data, angle_1=a1_et,angle_2=a2_et, E1_t=E1_at, E2_t=E1_at, 
                                             energy_range=energy_range, bg=False, relative=False)
                                                              
                        E_axis=energy[nE-E2_at:nE-E1_at]
                        E_spec_t=np.flipud(E_spec_t)
                        E_spec_t=gaussian_filter(E_spec_t, s_Et)
                        E_spec_t/=np.amax(E_spec_t)
                        
                        diff_Espec=zeros((len(E_axis), timesteps))
                        
                        bg_Espec=zeros((len(E_axis)))
                        
                        for i in range(bg_len):
                            bg_Espec[:]+=E_spec_t[:,bg_position+i]
                        bg_Espec/=bg_len
                        
                        for l in range(timesteps):
                            diff_Espec[:,l]=E_spec_t[:,l]-bg_Espec[:]
    
                        diff_Espec/=np.amax(np.abs(diff_Espec))

                        fit_error=zeros((timesteps))
                        area_HOMO=zeros((timesteps))
                        popt_t=zeros((len(guess), timesteps))
                        pcov_t=zeros((len(guess), len(guess), timesteps))
                        
                        for k in range(timesteps):                
                            f, (a0, a1) = plt.subplots(1,2, gridspec_kw = {'width_ratios':[1, 2.]})
                            f.set_size_inches((21, 10.5), forward=False)
                            
                            data_cropped=zeros((nE, angle_2-angle_1))
                            data_cropped[:,:]=data[:,angle_1:angle_2,0]
                            data_cropped/=np.amax(np.abs(data_cropped))
                            vmin, vmax=0., 1.
                            img=a0.imshow(data_cropped,aspect='auto', extent=[angles[angle_1],
                                                                   angles[angle_2],energy[0],energy[-1]])
                            
                            ##### Ang_spec as orientation #####                                      
                            a0.set_xlabel('angle', fontsize=fontsize)
                            a0.set_ylabel(r'E-E$_{f}$ (eV)', fontsize=fontsize)
                            a0.tick_params(labelsize=fontsize)
                            a0.set_xlim(angles[angle_1],angles[angle_2])
                            a0.set_ylim(energy[0],energy[-1])
            #                colorbar(img)
                            
                            a0.plot((angles[a1_et],angles[a1_et]),(E_axis[0],E_axis[-1]),ls='-',color='k', lw=1.5)
                            a0.plot((angles[a2_et],angles[a2_et]),(E_axis[0],E_axis[-1]),ls='-',color='k', lw=1.5)
                            a0.plot((angles[a1_et],angles[a2_et]),(E_axis[0],E_axis[0]),ls='-',color='k', lw=1.5)
                            a0.plot((angles[a1_et],angles[a2_et]),(E_axis[-1],E_axis[-1]),ls='-',color='k', lw=1.5)
                        
                            tb=DL[0]
                            t=DL[k]
                            
                            #### E-spec########                        
                            
                            if (log_Espec==True):
                                a1.semilogy(E_axis, bg_Espec, label='ref. spectrum' %vars(), color='k', lw=2, ls=':')
                                a1.semilogy(E_axis, E_spec_t[:,k], label='spectrum %(t)s' %vars(), color='k', lw=2, ls='-')
                                np.savetxt(measurement_folder+'E_axis.dat', E_axis)
                                np.savetxt(measurement_folder+'diff_Espec.dat', diff_Espec)
                                np.savetxt(measurement_folder+'Espec.dat', E_spec_t)
                            else:
                                a1.plot(E_axis, bg_Espec, label='ref. spectrum' %vars(), color='k', lw=3, ls=':')
                                a1.plot(E_axis, E_spec_t[:,k], label='spectrum %(t)s' %vars(), color='k', lw=3, ls='-')
                            
                            if (Espec_fit==True):
                                if (HOMO_2==True):
                                    fit_function=fit_function2
                                else:
                                    fit_function=fit_function1
                                    
                                popt, pcov = curve_fit(fit_function, E_axis, E_spec_t[:, k], guess, bounds=guess_bounds,
                                                       maxfev=100000)
                                fit_spec=fit_function(E_axis, *popt)
                                
                                a1.plot(E_axis, fit_spec, 
                                        color='red', label='fit to spectrum', ls='--', lw=3)                        
                                
                                bg=popt[0]+E_axis*popt[1]
                                
                                HOMO=gauss(E_axis, popt[2], popt[3], popt[4])
                                HOMO_1=gauss(E_axis, popt[5], popt[6], popt[7])
                                sub=gauss(E_axis, popt[8], popt[9], popt[10])
                                #lin bg
                                a1.plot(E_axis, bg, ls='-', color='grey')
                                
                                #plot HOMO                          
                                a1.plot(E_axis, HOMO, ls='-', color='green')
                                
                                #HOMO-1,2                           
                                a1.plot(E_axis, HOMO_1, color='blue')
                                
                                #plot sub
                                if not(i==0):
                                    a1.plot(E_axis, sub, color='orange')
                                
                                #fit calculations
                                fit_error[k]+=np.sum(np.sqrt((E_spec_t[:,k]-fit_spec)**2))                                    
                                popt_t[:,k]=popt
                                pcov_t[:,:,k]=pcov
                                
                                gauss_HOMO = lambda E: gauss(E, popt[2], popt[3], popt[4])
                                area_HOMO[k]= integrate.quad(gauss_HOMO, 34, 38)[0]                        
                                
                            a1.set_xlabel(r'E-E$_{f}$ (eV)', fontsize=fontsize)
                            a1.set_ylabel(r'counts (au)', fontsize=fontsize)
                            a1.tick_params(labelsize=fontsize)
                            a1.set_xlim(E_axis[0], E_axis[-1])
                            a1.set_ylim(np.amin(E_spec_t)-0.05,np.amax(E_spec_t)+0.05)
                            a1.grid(color='k', linestyle='--', linewidth=1, axis='x')
                            legend = a1.legend(loc='upper right', shadow=True, fontsize=fontsize)
                            
                            if (Espec_fit==False):
                                a1t = a1.twinx()
                                a1t.plot(E_axis, diff_Espec[:,k], '-', color='blue')
                                a1t.plot((E_axis[0], E_axis[-1]),(0,0), ls='--', color='blue')
                                a1t.set_xlim(E_axis[0], E_axis[-1])
                                a1t.set_ylim(-1.05,1.04)
                            
                            ### time pos graph###
#                            y_val=zeros((timesteps))
#                            a2.plot(y_val, DL, 'o',  color='k', markersize=5)
#                            a2.plot((0,0), (DL[0],DL[-1]), color='k', ls='-')
#                            a2.plot(0 , DL[k], 'o', color='g', markersize=10)
#                            a2.set_ylabel('pump-probe delay (fs)', fontsize=15)
#                            a2.set_xticklabels([])
#                    #        a0.xaxis.set_ticks([])
#                            a2.xaxis.set_ticks([])
#            #                a2.set_ticks(a0.get_xticks()[::2])
#                            a2.set_ylim(DL[0]-50, DL[-1]+50)
#                            for spine in a2.spines.itervalues():
#                                spine.set_visible(False)
                            ######################
                                
                            E_len=int(len(E_axis)/2)
                            E_mid=round(E_axis[E_len],1)
                            A_mid=angles[a1_et]+angles[a2_et]
                            A_mid/=2
                            A_mid=round(A_mid,0)
                            if (Espec_fit==True):                            
                                if (log_Espec==True):
                                    directory_Espec=measurement_folder+'sfit, sbg_pos=%(bg_position)s, bg_len=%(bg_len)s/sE_specs log/a1=%(a1_at)s, a2=%(a2_at)s, E1=%(E1_et)s, E2=%(E2_et)s, sEt=%(s_Et)s, HOMO-2G=%(HOMO_2)s/' %vars()
                                else: 
                                    directory_Espec=measurement_folder+'sfit, sbg_pos=%(bg_position)s, bg_len=%(bg_len)s/sE_specs/a1=%(a1_at)s, a2=%(a2_at)s, E1=%(E1_et)s, E2=%(E2_et)s, sEt=%(s_Et)s, HOMO-2G=%(HOMO_2)s/' %vars()
                            else:
                                if (log_Espec==True):
                                    directory_Espec=measurement_folder+'sbg_pos=%(bg_position)s, bg_len=%(bg_len)s/sE_specs log/a1=%(a1_at)s, a2=%(a2_at)s, E1=%(E1_et)s, E2=%(E2_et)s, sEt=%(s_Et)s, HOMO-2G=%(HOMO_2)s/' %vars()
                                else: 
                                    directory_Espec=measurement_folder+'sbg_pos=%(bg_position)s, bg_len=%(bg_len)s/sE_specs/a1=%(a1_at)s, a2=%(a2_at)s, E1=%(E1_et)s, E2=%(E2_et)s, sEt=%(s_Et)s, HOMO-2G=%(HOMO_2)s/' %vars()

                            if not os.path.exists(directory_Espec):
                                os.makedirs(directory_Espec)
                            
                            f.tight_layout()
                            count=round(float(k),1)
                            img_path=directory_Espec + 'n=%(count)s, t=%(t)s fs.png' %vars()
                            
                            f.savefig(img_path)
                            plt.clf()
                            plt.close('all')
    
                        if (Espec_fit==True):
                            #plots from fit stuff
                            perr=zeros((len(guess), timesteps))
                            for i in range(timesteps):
                                perr[:,i] = np.sqrt(np.diag(pcov_t[:,:,i]))
                            print popt_t[8,:], popt_t[10,:]                               
                            A_HOMO=popt_t[2,:]
                            A_HOMO=gaussian_filter(A_HOMO,1)
                            A_HOMO_err=perr[2,:]
                           
                            E_HOMO=popt_t[3,:]
                            E_HOMO=gaussian_filter(E_HOMO,1)
                            E_HOMO_err=perr[3,:]
                            
                            FWHM_HOMO=2.3548*popt_t[4]
                            FWHM_HOMO=gaussian_filter(FWHM_HOMO,1)
                            FWHM_HOMO_err=2.3548*perr[4,:]
                            
                            var_HOMO=np.multiply(popt_t[4],popt_t[4])
                            var_HOMO=gaussian_filter(var_HOMO,1)
                            var_HOMO_err=perr[4,:]
                            
                            A_HOMO_1=popt_t[5]
                            A_HOMO_1=gaussian_filter(A_HOMO_1,1)
                            A_HOMO_1_err=perr[5,:]
                            
                            E_HOMO_1=popt_t[6,:]
                            E_HOMO_1=gaussian_filter(E_HOMO_1,1)
                            E_HOMO_1_err=perr[6,:]
                            
                            FWHM_HOMO_1=2.3548*popt_t[7]
                            FWHM_HOMO_1=gaussian_filter(FWHM_HOMO_1,1)
                            FWHM_HOMO_1_err=2.3548*perr[7,:]
                            
                            var_HOMO_1=np.multiply(popt_t[7,:],popt_t[7,:])                                                               
                            var_HOMO_1=gaussian_filter(var_HOMO_1,1)
                            var_HOMO_1_err=perr[7,:]
                            
                            tot_A=popt_t[5,:]+popt_t[2,:]
                            tot_A=gaussian_filter(tot_A,1)
                            tot_A_err=perr[2,:]+perr[5,:]
                                                      
                            #HOMO plots (A & V) #####
                            fits_HAV, HA = plt.subplots(1,1)
                            fits_HAV.set_size_inches((11.5, 10.5), forward=True)
                            
                            labHA = HA.errorbar(DL/1000, A_HOMO, yerr=A_HOMO_err, color='k', lw=2, 
                                        ecolor='k', elinewidth=1, label='area')   
                            HV=HA.twinx()
                            labHV = HV.errorbar(DL/1000, FWHM_HOMO, yerr=FWHM_HOMO_err, ls='--', color='k', lw=2, 
                                        ecolor='k', elinewidth=1, label='FWHM')  

                            plt.legend(handles=[labHA, labHV], loc='upper left', numpoints=1, fontsize=20)
                            
                            HA.tick_params(labelsize=fontsize)
                            HA.set_xlim(DL[0]/1000, DL[-1]/1000)
                            HA.set_xlabel('pump-probe delay (ps)', fontsize=fontsize)
                            HA.set_ylabel('area (au)', fontsize=fontsize)
                            HV.set_ylabel('FWHM (eV)', fontsize=fontsize)
                            HV.tick_params(labelsize=fontsize)
                            HA.set_ylim(np.amin(popt_t[2,:])-0.02,np.amax(popt_t[2,:])+0.06) 
                            HV.set_ylim(np.amin(FWHM_HOMO)-0.05,np.amax(FWHM_HOMO)+0.02) 
                            HA.grid(axis='both', ls='--', color='gray', lw=1.) 
                            img_path_HAV=directory_Espec + 'A&V HOMO.png'
                            fits_HAV.savefig(img_path_HAV)
                            
                            #HOMO plot energy ####
                            fits_HE, HE = plt.subplots(1,1)
                            fits_HE.set_size_inches((10.5, 10.5), forward=True)
                            
                            minE_HOMO=np.amin(E_HOMO)
                            dE=(E_HOMO-minE_HOMO)*10**3 # E shift from minimum in meV
                            HE.errorbar(DL/1000, np.round(dE, 0), yerr=E_HOMO_err*10**3, color='k', lw=2, 
                                        ecolor='k', elinewidth=1, label='relative energy shift (meV)' %vars())
                                        
                            HE.tick_params(labelsize=fontsize)
                            HE.grid(axis='both', lw=1., ls='--', color='gray') 
                            HE.set_xlim(DL[0]/1000, DL[-1]/1000)
                            HE.set_xlabel('pump-probe delay (ps)', fontsize=fontsize)
                            HE.set_ylabel('relative energy shift  (meV)' %vars(), fontsize=fontsize)
                            HE.tick_params(labelsize=fontsize)            
                            img_path_HE=directory_Espec + 'E HOMO.png'
                            fits_HE.savefig(img_path_HE)
                            
                            #HOMO-1,2 plots (A & V) #####
                            fits_H2AV, H2A = plt.subplots(1,1)
                            fits_H2AV.set_size_inches((9, 7), forward=False)
                            
                            labH2A = H2A.errorbar(DL, A_HOMO_1, yerr=A_HOMO_1_err, color='blue', lw=2, 
                                        ecolor='k', elinewidth=1, label='area')   
                            H2V=H2A.twinx()
                            labH2V = H2V.errorbar(DL, FWHM_HOMO_1, yerr=FWHM_HOMO_1_err, ls='--', color='blue', lw=2, 
                                        ecolor='k', elinewidth=1, label='FWHM')  

                            plt.legend(handles=[labH2A, labH2V], loc='upper left', numpoints=1, fontsize=20)
                            
                            H2A.tick_params(labelsize=14)
#                            print popt_t[7,:]
                            H2A.set_xlim(DL[0], DL[-1])
                            H2A.set_xlabel('pump-probe delay (fs)', fontsize=20)
                            H2A.set_ylabel('amplitude (au)', fontsize=20)
                            H2V.set_ylabel('FWHM (eV)', fontsize=20)
                            H2V.tick_params(labelsize=14)
                            H2A.set_ylim(np.amin(popt_t[5,:])-0.02,np.amax(popt_t[5,:])+0.1) 
                            H2V.set_ylim(np.amin(FWHM_HOMO_1)-0.09,np.amax(FWHM_HOMO_1)+0.03) 

                            img_path_H2AV=directory_Espec + 'A&V HOMO-1,2.png'
                            fits_H2AV.savefig(img_path_H2AV)
                            
                            #HOMO-1,2 plot energy ####
                            fits_H2E, H2E = plt.subplots(1,1)
                            fits_H2E.set_size_inches((9, 7), forward=False)
                            
                            minE_HOMO_1=np.round(np.amin(E_HOMO_1),2)
                            dE_1=(E_HOMO_1-minE_HOMO_1)*10**3 # E shift from minimum in meV
                            H2E.errorbar(DL, np.round(dE_1,0), yerr=E_HOMO_1_err*10**3, color='blue', lw=2, 
                                        ecolor='k', elinewidth=1, label='energy shift rel. to %(minE_HOMO_1)s eV (meV)' %vars())
                            H2E.grid(axis='x', ls='--', color='gray')          
                            H2E.tick_params(labelsize=14)

                            H2E.set_xlim(DL[0], DL[-1])

                            H2E.set_xlabel('pump-probe delay (fs)', fontsize=20)
                            H2E.set_ylabel('relative energy shift (meV)' %vars(), fontsize=20)
                                        
                            img_path_H2E=directory_Espec + 'E HOMO-1.png'
                            fits_H2E.savefig(img_path_H2E)
                            
                            # plot A _tot ###
                            
                            fits_A_tot, A_tot = plt.subplots(1,1)
                            fits_A_tot.set_size_inches((9, 7), forward=False)
                            
                            A_tot.errorbar(DL, tot_A, yerr=tot_A_err, color='black', lw=2, 
                                        ecolor='k', elinewidth=1, label='total amplitude (HOMO+HOMO-1,2)' %vars())
                            
                            A_tot.tick_params(labelsize=14)
                            A_tot.set_xlim(DL[0], DL[-1])
                            A_tot.set_xlabel('pump-probe delay (fs)', fontsize=20)
                            A_tot.set_ylabel('total amplitude (HOMO+HOMO-1,2)', fontsize=20)
                                        
                            img_path_A_tot=directory_Espec + 'A_tot.png'
                            fits_A_tot.savefig(img_path_A_tot)
                            
                            
                            plt.clf()
                            plt.close('all')
                            np.savetxt(measurement_folder+'popt_t.dat', popt_t)
                            np.savetxt(measurement_folder+'perr_t.dat', perr)
                            np.savetxt(measurement_folder+'time_stamps.dat', DL)
                            ### Correlations ###
                            t1=3
                            t2=18
                            optpar = pd.DataFrame({'A-HOMO':popt_t[2,:],'E-HOMO':popt_t[3,:], 'S-HOMO':popt_t[4,:],
                                                   'A-HOMO-1,2':popt_t[5,:],'E-HOMO-1,2':popt_t[6,:], 'S-HOMO-1,2':popt_t[7,:]}, 
                                                    index=DL[:])    
#                            optpar = pd.DataFrame({'A-HOMO':popt_t[2,t1:t2],'E-HOMO':popt_t[3,t1:t2], 'S-HOMO':popt_t[4,t1:t2],
#                                                   'A-HOMO-1,2':popt_t[5,t1:t2],'E-HOMO-1,2':popt_t[6,t1:t2], 'S-HOMO-1,2':popt_t[7,t1:t2]}, 
#                                                    index=DL[t1:t2])  
                            
                            attributes = ["A-HOMO", "E-HOMO", "S-HOMO", 
                                          "A-HOMO-1,2", 'E-HOMO-1,2', 'S-HOMO-1,2']
                            ax = pd.scatter_matrix(optpar[attributes], figsize=(10, 12), diagonal='kde')
                            ax[0,0].xaxis.set_major_formatter(FormatStrFormatter('%3.2f'))
                            ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%3.2f'))
#                            ax.tick_params(axis='x', which='major', pad=10)
#                            ax.tick_params(axis='y', which='major', pad=10)
#                            ax[:,:].tick_params(axis='both', which='major', pad=10)
                                                      
                            
                            
    print 'completed E_spec at', datetime.now() - startTime
                        

"""Video Espec"""########################################################################

if vid_E_spec_t_var==True:
    for s in range(sE):
        for r in range(rE):
            s_Et=[4+s*2,0.3*r]
            for i in range(n_energyE):
                for j in range(n_angleE):
                    bE=(E2_et-E1_et)/n_energyE
                    bA=(a2_at-a1_at)/n_angleE
                    E1_at, E2_at=nE-E2_et+i*bE, nE-E2_et+(i+1)*bE
                    a1_et, a2_et=a1_at+j*bA, a1_at+(j+1)*bA
                    energy_range=E2_at-E1_at
                    E_len=int(len(E_axis)/2)
                    E_mid=round(E_axis[E_len],1)
                    A_mid=angles[a1_et]+angles[a2_et]
                    A_mid/=2
                    A_mid=round(A_mid,0)
            
                    images=measurement_folder+'sbg_pos=%(bg_position)s, bg_len=%(bg_len)s/sE_specs/E=%(E_mid)s, bE=%(bE)s, bA=%(bA)s/E=%(E_mid)s, A=%(A_mid)s, sEt=%(s_Et)s/*.png' %vars()
                  
                    outvid=measurement_folder+'sbg_pos=%(bg_position)s, bg_len=%(bg_len)s/sE_specs/E=%(E_mid)s, bE=%(bE)s, bA=%(bA)s/E=%(E_mid)s, A=%(A_mid)s, sEt=%(s_Et)s.avi' %vars()
                    
                    make_video_Espec(images=images, outvid=outvid, outimg=None, fps=fps, size=None,
                           is_color=True, format="XVID")
                           
                    plt.clf()
                    plt.close('all')
                    
                    print 'completed Video E_spec at', datetime.now() - startTime

angle_1_d, angle_2_d, E1_d, E2_d=40, 210, 10, 330

if clean_spec_var==True:
    clean_spec(bg_position, bg_len, data=data, angle_1=angle_1, angle_2=angle_2, E1=E1_d, E2=E2_d)
    plt.clf()
    plt.close('all') 
    
    print 'completed clean_spec at', datetime.now() - startTime        
    
print 'script ended at', datetime.now() - startTime
        

""" smth vs t"""############################################


if smth_vs_t_series==True:
    #Gauss conv.
#    s=1.
    
    #Raster settings 
    n_energy=10
    if (Espec_fit==False):
        n_energy=1
    n_angle=1
    
    #Window settings
    a1_at, a2_at=50, 200
    E1_et, E2_et=180, 280
    if i==6:
        E1_et, E2_et=92, 334
    if i==7:
        E1_et, E2_et=92, 334
    
    if i==4:
        #-2 - 1.5 eV
        E1_et, E2_et=155, 341
        E1_et, E2_et=86, 330
    if i==5:
        #-2 - 1.5 eV
        E1_et, E2_et=77, 320
    
    if i==3:
        E1_et, E2_et=210, 314
        #-2 - 1.5 eV        
        if (whole==True):
            E1_et, E2_et=72, 314
    if i==0:
        E1_et, E2_et=213, 300
        #-2 - 1.5 eV
        if (whole==True):
            E1_et, E2_et=97, 300
    if i==10:
        #-2 - 1.5 eV
        E1_et, E2_et=79, 322
    if i==13:
        #-2 - 1.5 eV
        E1_et, E2_et=79, 322
    if i==1:
        #-2 - 1.5 eV
        E1_et, E2_et=91, 294
        #over Ef
#        E1_et, E2_et=200, 294
    angle_range=a2_at-a1_at
    energy_range=E2_et-E1_et
    
    for i in range(n_energy):
        for j in range(n_angle):
            n=i*n_angle+j
            bE=(E2_et-E1_et)/n_energy
            bA=(a2_at-a1_at)/n_angle
            E1_at, E2_at=nE-E2_et+i*bE, nE-E2_et+(i+1)*bE
            a1_et, a2_et=a1_at+j*bA, a1_at+(j+1)*bA
            E1_eat, E2_eat= E1_at, E2_at
            a1_eat, a2_eat= a1_at, a2_et
            
            smth_vs_t_plot(bg_position=bg_position, bg_len=bg_len,
                       angle_1=angle_1, angle_2=angle_2,
                       a1_at=a1_at, a2_at=a2_at,E1_at=E1_at,E2_at=E2_at, angle_range=angle_range,
                       a1_et=a1_et, a2_et=a2_et, E1_et=E1_et, E2_et=E2_et, energy_range=energy_range,
                       angle_1_d=angle_1_d, angle_2_d=angle_2_d, E1_d=E1_d, E2_d=E2_d,
                       a1_eat=a1_eat, a2_eat=a2_eat, E1_eat=E1_eat, E2_eat=E2_eat,
                       s=s, energy=energy, angles=angles, measurement_name=measurement_name, relative=relative,
                       n_energy=n_energy, n_angle=n_angle, measurement_folder=measurement_folder, n=n
                       )
    print 'completed smth_vs_t at', datetime.now() - startTime

"""Video"""#################################################################
#Video parameters:
#window size:
angle_1_d, angle_2_d, E1_d, E2_d=40, 210, 214, 330

if video_series_var==True:
    s=2
    r=2
    
    video_series(bg_position, bg_len, s, r, E1_d=E1_d, fps=1)
    print 'completed Diff map video at', datetime.now() - startTime
            
            
                              
            
            
            
