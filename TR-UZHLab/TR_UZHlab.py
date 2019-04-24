# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:30:00 2018

@author: dasch
"""

from pylab import *
from cmath import *
from operator import itemgetter
#import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import misc
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy.ma as ma
from numpy.random import uniform, seed
import re
from scipy.optimize import curve_fit
import numpy as np

"""LOAD DATA"""
#Input File path
path="WAL_20180214_UZH_KW_dScan016.dat"

data=loadtxt(path,skiprows=15,dtype=float)
data=np.fliplr(data)
data=data.T
steps=shape(data)[1]

#Scaninfo, header of file
scaninfo=genfromtxt(path,skip_footer=steps,dtype=None,delimiter='\n',comments='$')
#Scanparameters from scaninfo
date=scaninfo[1]
Ek=float(re.findall(r"[-+]?\d*\.\d+|\d+",scaninfo[5])[0])
Ep=float(re.findall(r"[-+]?\d*\.\d+|\d+",scaninfo[4])[0])
dt=float(re.findall(r"[-+]?\d*\.\d+|\d+",scaninfo[2])[0])
bw=0.12*Ep
dE=bw/348


def process_data(start, t0, nt, E0,nE, right=2):
    t1=t0-nt*dt
    t2=t0+right*nt*dt
    E1=E0-nE*dE
    E2=E0+nE*dE
    tl=int(round((t1-start)/dt,0))
    tr=int(round((t2-start)/dt,0))
    Eup=int((Ek+bw/2.-E2)/dE)
    Elow=int((Ek+bw/2.-E1)/dE)
    data_c=data[Eup:Elow,tl:tr]
    data_cropped=data_c
    
    #axis (energy and time)
    axis_E=frange(E1,E2-dE,dE)
    axis_t=frange(t1,t2-dt,dt)
        
    #Integrated E vs t
    n_seg=3
    E_seg=len(axis_E)/n_seg
    
    I_t=zeros((len(axis_t),n_seg))
#    if (shape(I_t)[0]!=shape(data_c)[1]):
#        data_c=data[Eup:Elow,tl:tr-1]
#        data_cropped=data_c
    
    for i in range(len(axis_t)):
        for j in range(n_seg):
            I_t[i,j]=sum(data_c[E_seg*j:E_seg*(j+1),i])
    I_t/=amax(I_t)
    
    #Integrated t vs E
    
    t_seg=len(axis_t)/n_seg
    
    I_E=zeros((len(axis_E),n_seg))
    
    if (shape(I_E)[0]!=shape(data_c)[0]):
        data_c=data[Eup:Elow-1,tl:tr]
        data_cropped=data_c
    
    
    I_E_bg=zeros((len(axis_E),n_seg))
    
    for i in range(len(axis_E)):
        for j in range(n_seg):
            I_E[i,j]=sum(data_c[len(axis_E)-1-i,(t_seg*j):(t_seg*(j+1))])
            I_E_bg[i,j]=sum(data_c[i,(t_seg*j):(t_seg*(j+1))])/(len(axis_t)/3)
            
    I_E/=amax(I_E)
    
    difft0t1=I_E[:,0]-I_E[:,1]
    difft0t2=I_E[:,2]-I_E[:,1]
    
    data_full=data/amax(data)
    return t1, t2, E1, E2, data_full, data_cropped,t_seg, E_seg, I_E, axis_E, axis_t, I_E_bg, difft0t1, difft0t2, data_c, I_t

def full_plot(start=-79,t0=-77.8,nt=steps/2,E0=Ek,nE=120, right=2):
    t1, t2, E1, E2, data_full, data_cropped, t_seg, E_seg, I_E, axis_E, axis_t, I_E_bg, difft0t1, difft0t2, data_c, I_t= process_data(start,t0, nt, E0, nE, right)
    path2=path
    subplot(2,3,1)
    
    plot((t1,t1),(E1,E2),color='k',lw='2')
    plot((t2,t2),(E1,E2),color='k',lw='2')
    plot((t1,t2),(E1,E1),color='k',lw='2')
    plot((t1,t2),(E2,E2),color='k',lw='2')
    
    imshow(data_full,extent=[start,start+steps*dt,Ek-bw/2,Ek+bw/2],aspect='auto')
    colorbar()
    title('%(path2)s' %vars(), fontsize=10)
    xlabel('Timedelay (ps)', fontsize=15)
    ylabel('Energy (eV)', fontsize=15)
    
    subplot(2,3,2)
    
    data_cropped=data_cropped/amax(data_cropped)
    imshow(data_cropped,extent=[t1,t2,E1,E2],aspect='auto')
    
    plot((t1+t_seg*dt,t1+t_seg*dt),(E1,E2),color='k',lw='2',ls='--')
    plot((t2-t_seg*dt,t2-t_seg*dt),(E1,E2),color='k',lw='2',ls='--')
    plot((t1,t2),(E1+E_seg*dE,E1+E_seg*dE),color='k',lw='1',ls='--')
    plot((t1,t2),(E1+2*E_seg*dE,E1+2*E_seg*dE),color='k',lw='1',ls='--')
    xlim(t1,t2)
    ylim(E1,E2)
    xlabel('Timedelay (ps)', fontsize=15)
    ylabel('Energy (eV)', fontsize=15)
    
    colorbar()
    
    subplot(2,3,3)
    
    #Time integrated 
    plt.plot(I_E[:,0],axis_E, label='left',color='blue', lw='2')
    plt.plot(I_E[:,1],axis_E, label='mid',color='red',lw='2')
    plt.plot(I_E[:,2],axis_E, label='right',color='orange',lw='2')
    plt.plot(difft0t1,axis_E, label='$\Delta$ (left, mid)',color='blue')
    plt.plot(difft0t2,axis_E, label='$\Delta$ (right, mid)',color='orange')
    legend2 = legend(loc='upper right', shadow=True)
    plt.plot((0,0),(E1,E2),ls='--')
    xlabel('Time integrated (au)', fontsize=15)
    ylabel('Energy (eV)', fontsize=15)
    ylim(E1,E2)
    
    subplot(2,3,4)
    
    #diffrence plot
    
    diff=zeros((shape(data_c)[0],shape(data_c)[1]))
    for i in range(len(axis_t)):
        diff[:,i]=data_c[:,i]-I_E_bg[:,0]
    diff=diff/amax(diff)
    imshow(diff,extent=[t1,t2,E1,E2],aspect='auto')
    xlabel('Timedelay (ps)', fontsize=15)
    ylabel('Energy integrated (eV)', fontsize=15)
    colorbar()
    
    subplot(2,3,5)
    
    #Fit
    
    def GaussConv(t, C, A, tc, var):
        return C+A*np.exp(-(t-tc)**2/(2*(var**2)))
    
#    C, A, tc, var
    guess=(0.5, 0.1,-77.8, 0.1)
    popt0, pcov0 = curve_fit(GaussConv, axis_t, I_t[:,0], guess)
    popt1, pcov1 = curve_fit(GaussConv, axis_t, I_t[:,1], guess)
    popt2, pcov2 = curve_fit(GaussConv, axis_t, I_t[:,2], guess)
    
    plt.plot(axis_t,GaussConv(axis_t,*popt0),color='orange')
    plt.plot(axis_t,GaussConv(axis_t,*popt1),color='r')
    plt.plot(axis_t,GaussConv(axis_t,*popt2),color='b')
    
    tau=zeros((3))
    tau0,tau1,tau2=(int(round(popt0[3]*10**3,0)),int(round(popt1[3]*10**3,0)),int(round(popt2[3]*10**3,0)))

#    Energy integrated over time in n_segments
    
#    \n' r'$\tau$=%(tau0)s
    
    plt.plot(axis_t,I_t[:,0], label='top \n' r'$\tau$ = %(tau0)s' %vars(),color='orange')
    plt.plot(axis_t,I_t[:,1], label='middle \n' r'$\tau$ = %(tau1)s' %vars(),color='red')
    plt.plot(axis_t,I_t[:,2], label='bottom \n' r'$\tau$ = %(tau2)s' %vars(),color='blue')
    xlim(t1,t2)
    xlabel('Time delay [ps]',fontsize=15)
    ylabel('Intensity [au]', fontsize=15)
    
    legend3 = legend(loc='center right', shadow=True)

#Plot input: start, t0, nt, E0,nE

full_plot(start=-79, t0=-77.87, nt=40, E0=Ek, nE=140, right=1.5)



