# -*- coding: utf-8 -*-
"""
Created on Mon Apr 09 11:07:49 2018

@author: dasch
"""

from pylab import *
from cmath import *
from operator import itemgetter
#import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import misc
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy.ma as ma
from numpy.random import uniform, seed
from os.path import join, dirname, abspath
from matplotlib import pyplot
from matplotlib.cbook import get_sample_data


#substrate lattice constants
#TiO2
#a1a=2.98
#a2a=6.51
#Ag110
a1a=2.89
a2a=4.08

a1sa=2*pi/a1a
a2sa=2*pi/a2a

pn_img = pyplot.imread(get_sample_data(join(dirname(abspath(__file__)), "pentacene.png")))
pn2_img = pyplot.imread(get_sample_data(join(dirname(abspath(__file__)), "pentacene2.png")))

def inv(Gs,T=False):
    G11s=Gs[0,0]
    G12s=Gs[0,1]
    G21s=Gs[1,0]
    G22s=Gs[1,1]
    
    det_Gs=G11s*G22s-G21s*G12s
    
    adj_Gs=array([[G22s,-G12s],[-G21s,G11s]])
    
    Gs_inv=1/det_Gs*adj_Gs
    
    G=Gs_inv.T
    if (T==True):
        return G
    else:
        return Gs_inv
        
#TiO2
#Gs=array([[1/6.,0.],[0.0,1.]])
#Ag110
Gs=1/11*array([[4.,1.],[1,3.]])

def real_l(n=10,lim=40, BL=False):
    
    fig, ax = pyplot.subplots()
    ax_width = ax.get_window_extent().width
    fig_width = fig.get_window_extent().width
    fig_height = fig.get_window_extent().height
    ax.set_aspect('equal')
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_xlabel(r'$\AA$',fontsize=20)
    ax.set_ylabel(r'$\AA$',fontsize=20)

    ngrid=2*n
    a1=frange(-a1a*ngrid,a1a*ngrid,a1a)
    a2=frange(-a2a*ngrid,a2a*ngrid,a2a)
    
    a1_g, a2_g=meshgrid(a1,a2)
    
    #Gao paper
    b1=array([3*a1a,-1*a2a])
    b2=array([-1*a1a,4*a2a])
#    TiO2
#    b1=array([6*a1a,0.*a2a])
#    b2=array([0.*a1a,1*a2a])
    
    G=inv(Gs,T=True)
    c1=array([G[0,0]*a1a,G[0,1]*a2a])
    c2=array([G[1,0]*a1a,G[1,1]*a2a])
    
    b=zeros((2))
    c=zeros((2))

#    plot=ax.plot(a1_g, a2_g, 'o', color='red', markersize=7/(lim/40.), alpha=0.8)
    plot=ax.plot(a1_g, a2_g, 'o', color='grey', markersize=12/(lim/40.), alpha=0.8)
    n1 = n
    m1 = n
    m2 = n
    n2 = n
    centre=array([0,0])
    #TiO2
#    centre=array([0.,0.3*4.086])
#    plot = ax.plot((0+centre[0],b1[0]+centre[0]),(0+centre[1],b1[1]+centre[1]), linewidth=3, color='blue')
#    plot = ax.plot((0+centre[0],b2[0]+centre[0]),(0+centre[1],b2[1]+centre[1]), linewidth=3, color='blue')
    for i in range(n1):
        i-=n1/2
        for j in range(m1):
            j-=m1/2
            b=i*b1+j*b2+centre
            #TIO2
#            pn_size = ax_width*2.3/(fig_width*m1)
            #Ag110
            pn_size = ax_width*2.7/(fig_width*m1)
            pn_axs = [None for j in range(m1)]
            #TiO2
#            loc = ax.transData.transform((b[0]/1.3, b[1]))
#            Ag110
            loc = ax.transData.transform((b[0]/1.3, b[1]))
            pn_axs[j] = fig.add_axes([loc[0]/fig_width-pn_size/2, loc[1]/fig_height-pn_size/2,
                                       pn_size, pn_size], anchor='C')
                                
#            pn_axs[j].imshow(pn_img)
            pn_axs[j].axis("off")
                        
            
            
    if(BL==True):
        for i in range(n2):
            i-=n2/2
            for j in range(m2):
                j-=m2/2
                c=i*c1+j*c2+centre
                
                pn2_size = ax_width*2.4/(fig_width*m2)
                pn2_axs = [None for j in range(m2)]
                
                loc = ax.transData.transform((c[0]/1.9, c[1]))
                pn2_axs[j] = fig.add_axes([loc[0]/fig_width-pn2_size/2, loc[1]/fig_height-pn2_size/2,
                                           pn2_size, pn2_size], anchor='C')
                pn2_axs[j].imshow(pn2_img)
                pn2_axs[j].axis("off")
#                plot(c[0],c[1],'d',markersize=20,color='red')
                plot = ax.plot((0+centre[0],c1[0]+centre[0]),(0+centre[1],c1[1]+centre[1]), linewidth=1, color='red')
                plot = ax.plot((0+centre[0],c2[0]+centre[0]),(0+centre[1],c2[1]+centre[1]), linewidth=1, color='red')
                
    fig.savefig("pnML.png")

#real_l(20,60, True)
        
def reci_l(n=10, lim=1, BL=False):
    
    bg_color='black'
    fg_color='white'
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.patch.set_facecolor(fg_color)
    axes.set_aspect('equal')
    axes.set_xlim(-lim,lim)
    axes.set_ylim(-lim,lim)
    axes.set_xlabel(r'$\AA^{-1}$',fontsize=15)
    axes.set_ylabel(r'$\AA^{-1}$',fontsize=15)
    
    ngrid=n/2
    a1s=frange(-a1sa*ngrid,a1sa*ngrid,a1sa)
    a2s=frange(-a2sa*ngrid,a2sa*ngrid,a2sa)
    
    a1s_g, a2s_g=meshgrid(a1s,a2s)

    c1s=array([Gs[0,0]*a1sa,Gs[0,1]*a2sa])
    c2s=array([Gs[1,0]*a1sa,Gs[1,1]*a2sa])
    
    bs=zeros((2))
    cs=zeros((2))

    plt.plot(a1s_g, a2s_g, 'o',mew=0, markersize=15, color=bg_color)
    n1 = n
    m1 = n
    m2 = n+2
    n2 = n+2
    centre=array([0,0])
#   centre=array([1*2.889,1*4.086])
    
    for k in range(2):
        #Gao paper
#        b1s=1/11.*array([4*a1sa,(-1)**k*a2sa])
#        b2s=1/11.*array([1*a1sa,(-1)**k*3*a2sa])
#        b1s=array([0.27272727*a1sa,(-1)**k*-0.18181818*a2sa])
#        b2s=array([0.09090909*a1sa,(-1)**k*0.27272727*a2sa])
        b1s=array([1./6.*a1sa,(-1)**k*0.*a2sa])
        b2s=array([0.0*a1sa,(-1)**k*1.*a2sa])
        
#        if(k==0):
#            plt.plot((0,b1s[0]),(0,b1s[1]), linewidth=3, color='blue')
#            plt.plot((0,b2s[0]),(0,b2s[1]), linewidth=3, color='blue')
        for i in range(n1):
            i-=n1/2
            for j in range(m1):
                j-=m1/2
                bs=i*b1s+j*b2s
                r=sqrt(bs[0]**2+bs[1]**2)
                r_max=n1/2*b1s+m1/2*b2s
                r_maxa=sqrt(r_max[0]**2+r_max[1]**2)
                d=real((r/r_maxa)**(1/1.5))
#                if(k==0):
#                    plt.plot(bs[0],bs[1],'o', markersize=10,color=(0,0,1), mew=0)
#                else:
#                    plt.plot(bs[0],bs[1],'o', markersize=10,color=(1,0,0), mew=0)
                plt.plot(bs[0],bs[1],'o', markersize=10,markerfacecolor='none',markeredgecolor='blue', mew=2)
    
    if(BL==True):
        for k in range(2):
            c1s=array([Gs[0,0]*a1sa,(-1)**k*Gs[0,1]*a2sa])
            c2s=array([Gs[1,0]*a1sa,(-1)**k*Gs[1,1]*a2sa])
            
#            if(k==0):
#                 plt.plot((0,c1s[0]),(0,c1s[1]), linewidth=1.5, color='red')
#                 plt.plot((0,c2s[0]),(0,c2s[1]), linewidth=1.5, color='red')
            for i in range(n2):
                i-=n2/2
                for j in range(m2):
                    j-=m2/2
                    cs=i*c1s+j*c2s
                    r=sqrt(cs[0]**2+cs[1]**2)
                    r_max=n2/2*c1s+m2/2*c2s
                    r_maxa=sqrt(r_max[0]**2+r_max[1]**2)
                    d=real((r/r_maxa)**(1/2.))
#                    plt.plot(cs[0],cs[1],'o', markersize=10,color=(1,d,d), mew=0)
                    plt.plot(cs[0],cs[1],'o', markersize=10,markerfacecolor='none',markeredgecolor='red',mew=2)
                
    
#reci_l(20,3.,False)
real_l(8,25.,False)
