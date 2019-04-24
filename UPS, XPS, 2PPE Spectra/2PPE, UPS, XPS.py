# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:23:02 2018

@author: dasch
"""

import numpy as np
from matplotlib import pyplot as plt
#import cv2
from pylab import *
from cmath import *
import matplotlib.pyplot as plt
import glob 

h_bar=10**-34
c=3*10**8
e=1.602*10**-19
def process_data(filepath, finalpath, Ek, n=40):
    Ek_max=4.8
    
    data_raw=loadtxt(filepath)
    VpC=len(data_raw[:,0])/120
    
    n=40
    
    data_n=zeros((2*n*VpC,2))
    
    """Crop WAL data to only +- 5°"""
    data_n[:,0]=data_raw[(120/2-n)*VpC:(120/2+n)*VpC,0]
    data_n[:,1]=data_raw[(120/2-n)*VpC:(120/2+n)*VpC,1]
    
    """Integrate all energies over +-5°"""
    data_ns=zeros((VpC,2))
    data_ns[:,0]=data_n[0:VpC,0]
    data_ns[:,0]-=Ek_max+10+0.2
    
    for i in range(VpC):
        for j in range(2*n):
            data_ns[i,1]+=data_n[j*VpC+i,1]
            
    savetxt(finalpath, data_ns)

def derivative(x):
    dE=(x[1,0]-x[0,0])
    
    nE=len(x)
    dfdx=zeros((nE-1,2))
    dfdx[:,0]=x[:(nE-1),0]+dE/2
    
    for i in range(nE-1):
        dfdx[i,1]=(x[i+1,1]-x[i,1])/dE
    return dfdx

def plot(loadpath1, loadpath2, deriv=False):
    fig, ax = plt.subplots()
    fig.set_facecolor('white')
    ax2 = ax.twinx()
    
    #data1
    data_p=loadtxt(loadpath1)
    data_p[:,1]=data_p[:,1]/amax(data_p[:,1])
    data_p_dfdx = derivative(data_p)
    data_p_dfdx[:,1]/=amax(data_p_dfdx[:,1])
    data_p_dfdx2= derivative(derivative(data_p))
    data_p_dfdx2[:,1]/=amax(data_p_dfdx2[:,1])
    data_p_log=np.log(data_p[:,1])
    #data2
    data_p2=loadtxt(loadpath2)
    data_p2[:,1]=data_p2[:,1]/amax(data_p2[:,1])
    data_p2_dfdx = derivative(data_p2)
    data_p2_dfdx[:,1]/=amax(data_p2_dfdx[:,1])
    data_p2_dfdx2= derivative(derivative(data_p2))
    data_p2_dfdx2[:,1]/=amax(data_p2_dfdx2[:,1])
    data_p2_log=np.log(data_p2[:,1])
    
    ax.plot([-100,100],[0,0],color='k',ls='--')
    
    
    #plot 1
    ax1_n = ax.plot(data_p[:,0],data_p[:,1],color='k',lw=1.5,label='480 nm, 4.5 mW, $\Theta$=206$^{\circ}$, V$_{b}$=10 V')
    if(deriv==True):
        ax1_dx = ax.plot(data_p_dfdx[:,0],data_p_dfdx[:,1],color='orange',lw=1.5,label='1st derivative')
        ax1_dxdx = ax.plot(data_p_dfdx2[:,0],data_p_dfdx2[:,1],color='red',lw=1.5,label='2nd derivative')
    ax2_log = ax2.plot(data_p[:,0],data_p_log,color='green',lw=1.5,label='Ln(intensities)')
    
    #plot 2
    
    ax1_n2 = ax.plot(data_p2[:,0],data_p2[:,1],color='k',ls='--',lw=1.5,label='480 nm, 3 mW, $\Theta$=206$^{\circ}$, V$_{b}$=10 V')
    if(deriv==True):
        ax1_dx2 = ax.plot(data_p2_dfdx[:,0],data_p2_dfdx[:,1],color='orange',ls='--',lw=1.5,label='1st derivative')
        ax1_dxdx2 = ax.plot(data_p2_dfdx2[:,0],data_p2_dfdx2[:,1],color='red',ls='--',lw=1.5,label='2nd derivative')
    ax2_log2 = ax2.plot(data_p2[:,0],data_p2_log,color='green',ls='--',lw=1.5,label='Ln(intensities)')
    
    
    if(deriv==True):
        plots=ax1_n+ax1_dx+ax1_dxdx+ax2_log+ax1_n2+ax1_dx2+ax1_dxdx2+ax2_log2
    else:
        plots=ax1_n+ax2_log+ax1_n2+ax2_log2
    labs=[l.get_label() for l in plots]
    
    ax.legend(plots, labs,loc='upper right', shadow=True)
    
    ax.set_ylabel('Intensity (a.u.)',fontsize=20)
    ax.set_xlabel('binding energy (eV), (0 eV = E$_{F}$)',fontsize=20)
    
    ax.set_xlim(np.amin(data_p[:,0]),np.amax(data_p[:,0]))

    tick_params(axis='x', labelsize=15)
    tick_params(axis='y', labelsize=15)

def plot_all(deriv=False,log=False,low=False,norm=True,Eb=False):
    path='data/20180612 Ag_clean 2PPE/processed/480-650/*.dat'
    files=glob.glob(path) 
    
    fig, ax = plt.subplots()
    fig.set_facecolor('white')
    ax2 = ax.twinx()
    i=0
    amount=len(files)/2
    for file in files: 
        m=i%(amount*2)/2+1
        scale=m*1./amount
        lw=scale*3
        lam=480+10*(i/2)
        E=round(1.2398/(lam)*1000,2)

        
        
        data_p=loadtxt(file)
        data_p[:,0]+=15
        if(Eb==True):
#            
            ax.set_xlabel('binding energy (eV), (0 eV = E$_{F}$)',fontsize=20)
            if(2*E>4.44):
                data_p[:,0]-=10.2+E*2
            else:
                data_p[:,0]-=10.2+E*3
        else:
            data_p[:,0]-=14.68
            ax.set_xlabel(r'E$_{kin}$',fontsize=20)
        data_p_dfdx = derivative(data_p)
        data_p_dfdx2= derivative(derivative(data_p))
        if(norm==True):
            data_p[:,1]=data_p[:,1]/amax(data_p[:,1])
            data_p_dfdx[:,1]/=amax(data_p_dfdx[:,1])
            data_p_dfdx2[:,1]/=amax(data_p_dfdx2[:,1])
        data_p_log=np.log(data_p[:,1])
        
        if((i%2)==0):
            lstyle='--'
        else:
            lstyle='-'

        if(low==False and (i%2)==0):
            pass
        else:
            ax1_n = ax.plot(data_p[:,0],data_p[:,1],color=(scale,0,1-scale),ls=lstyle,lw=2,label='%(lam)s nm, %(E)s eV' %vars())
            if((i%2)==1):
                if(deriv==True):
    #                ax1_dx = ax.plot(data_p_dfdx[:,0],data_p_dfdx[:,1],color='orange',lw=1.5,label='1st derivative')
                    ax1_dxdx = ax.plot(data_p_dfdx2[:,0],data_p_dfdx2[:,1],color=(scale,0,0),lw=1.5,label='%(lam)s nm, %(E)s eV, 2nd derivative' %vars())
            if(log==True):
                ax2_log = ax2.plot(data_p[:,0],data_p_log,color=(scale,scale,0),ls=lstyle,lw=1.5,label='%(lam)s nm, %(E)s eV, Ln(intensities)' %vars())
            
            ax.legend(loc='upper right', shadow=True)
            ax2.legend(loc='upper left', shadow=True)
            
            ax.set_ylabel('Intensity (a.u.)',fontsize=20)
            
            ax.set_xlim(np.amin(data_p[:,0]),np.amax(data_p[:,0]))
        
            tick_params(axis='x', labelsize=15)
            tick_params(axis='y', labelsize=15)
            
        i+=1
    
#filepath1='data/20180612 Ag_clean 2PPE/2PPE 650 nm 6.2mW.xy'
#finalpath1='data/20180612 Ag_clean 2PPE/processed/2PPE 650 nm 6.2mW n=40.dat'
#Ek_max1=2.58*2.
#
#filepath2='data/20180612 Ag_clean 2PPE/2PPE 600 nm 5mW.xy'
#finalpath2='data/20180612 Ag_clean 2PPE/processed/2PPE 600 nm 5mW n=40.dat'
#Ek_max2=2.58*2.
#
#process_data(filepath1,finalpath1, Ek_max1, n=40)
#process_data(filepath2,finalpath2, Ek_max2, n=40)

#plot(finalpath1, finalpath2, False)        

plot_all(low=True, norm=True)