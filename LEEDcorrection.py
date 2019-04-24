# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:27:16 2017

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

#Read image (Crop image to LEED screensize before)
imagefull=misc.imread("LEED_correction/BD_corrected/LEED_137eV_17nm_20150904.png")
image=imagefull[:,:,0]
image=image/(amax(image)*1.)

#Parameters of image and LEED experimental setup
nx=shape(image)[1]
ny=shape(image)[0]

z_real=5.4*10**-2
ds=7.00*10**-2
dx=ds/nx
dy=ds/ny

#imshow(image,cmap=cm.gray_r)

#xlabel('cm',fontsize=15)
#ylabel('cm',fontsize=15)
#show()

#Center E-gun/screen and LEED image in px and realspace coordinates [x,y]

ncs=[244,165]   #### ENTRY ####
ncl=[251,189]   #### ENTRY ####


cs=[ncs[0]*dx,ncs[1]*dy]
cl=[ncl[0]*dx,ncl[1]*dy]

dsl=[cl[0]-cs[0],cl[1]-cs[1]]
#distance (screen center) - (LEED secular spot)
dsl_abs=(dsl[0]**2+dsl[1]**2)**(1/2.)

#Tilt angle of sample w.r.t. screen
theta_i=real(math.atan2(dsl_abs,z_real))

#Coordinate system with (0,0) at screen center (cs)
xs=linspace(0,dx*nx,nx)
ys=linspace(0,dy*ny,ny)

[gx_realpxl,gy_realpxl]=meshgrid(xs,ys)

gx_realpxl-=cs[0]
gy_realpxl-=cs[1]

#Matrix with distances of pixels from cs
q_measured=gx_realpxl**2+gy_realpxl**2
q_measured=q_measured**(1./2.)

#Matrix with azimutal angles of pixels, 0 is set at vector cl-cs
phi_LEED=arccos((gx_realpxl*dsl[0]+gy_realpxl*dsl[1])/(q_measured*dsl_abs))
phi_LEED=real(phi_LEED)

q_tiltedplane_cs=zeros((ny,nx))

 #Central projection of LEED image to a plane tilted through center of E-gun at an angle making the plane parallel to the sample surface
for i in range(ny):
    for j in range(nx):
        a=math.atan2(z_real,q_measured[i,j])
        b=-1*math.atan2(abs(tan(theta_i))*q_measured[i,j]*cos(phi_LEED[i,j]),q_measured[i,j])-1*math.atan2(z_real,q_measured[i,j])
        #q_tiltedplane_cs has distances from central projected image to screen center        
        q_tiltedplane_cs[i,j]=-1*q_measured[i,j]*sin(a)/sin(b)



#matrices for griddata function, some inputs need to be lists (length ny*nx), not only meshgrids
#Ex.: imageintensity is a list with all the image values
phi_real=zeros((ny,nx))
xy_vec_tilted=zeros((ny*nx,2))
imageintensity=zeros((ny*nx,1))

#ny x nx matrix with azimut angle of realpxl grid, phi=0 direction of x-axis in image
for i in range(ny):
    for j in range(nx):
        phi_real[i,j]=math.atan2(gy_realpxl[i,j],gx_realpxl[i,j])
        imageintensity[i*nx+j]=image[i,j]
       
#Coordinates of central projected image, center at screen and 0-azimutal at x-axis
gx_tiltedplane=q_tiltedplane_cs*np.cos(phi_real)
gy_tiltedplane=q_tiltedplane_cs*np.sin(phi_real)

for i in range(ny):
    for j in range(nx):
        xy_vec_tilted[i*nx+j,0]=gx_tiltedplane[i,j]
        xy_vec_tilted[i*nx+j,1]=gy_tiltedplane[i,j]

#shift LEED center to middle, for following inverse stereographic projection

gx_tiltedplane-=dsl[0]*np.cos(theta_i)
gy_tiltedplane-=dsl[1]*np.cos(theta_i)

#shortest distance of LEED center of tilted plane to sample
z_tilted=z_real*np.cos(theta_i)

#inverse stereographic projection of tilted plane centered at LEED pattern and circle with radius z_tilted
tg_theta=(gx_tiltedplane**2+gy_tiltedplane**2)**(1./2.)/z_tilted

q_tiltedplane_cl=(gx_tiltedplane**2+gy_tiltedplane**2)**(1/2.)

q_ew=2.*q_tiltedplane_cl*((((tg_theta**2+1)**(1./2)-1)/(2.*(tg_theta**2+1)**(1./2)))**(1./2)/tg_theta)

#Parallel projection of Ewald vectors onto sample surface
q_ew_p=q_ew*(1-q_ew**2/(4*z_tilted**2))**(1/2)

xy_vec_stereo=zeros((ny*nx,2))

#ny x nx matrix with azimutal angle of vectors of grid_tiltedplane
phi_tilted=zeros((ny,nx))

for i in range(ny):
    for j in range(nx):
        phi_tilted[i,j]=math.atan2(gy_tiltedplane[i,j],gx_tiltedplane[i,j])
#Grid of stereo and parallel projection
gx_stereo=q_ew_p*np.cos(phi_tilted)
gy_stereo=q_ew_p*np.sin(phi_tilted)

#center grid back to E-gun

dsl_tilted=[dsl[0]*np.cos(theta_i),dsl[1]*np.cos(theta_i)]
gx_stereo+=dsl_tilted[0]
gy_stereo+=dsl_tilted[1]

for i in range(ny):
    for j in range(nx):
        xy_vec_stereo[i*nx+j,0]=gx_stereo[i,j]
        xy_vec_stereo[i*nx+j,1]=gy_stereo[i,j]

#Map image onto original Grid and Show image

gridcentral=griddata(xy_vec_tilted,imageintensity,(gx_realpxl,gy_realpxl),method='nearest',fill_value=0)
gridcentralstereopara=griddata(xy_vec_stereo,imageintensity,(gx_realpxl,gy_realpxl),method='nearest',fill_value=0)

#colorimage (comparison corrected and original)
#colorim=zeros((ny,nx,4))
#
#colorim[:,:,1]=gridcentralstereopara[:,:,0]
#colorim[:,:,0]=gridcentral[:,:,0]
##colorim[:,:,3]=image
#colorim[:,:,3]=abs(image-1.)
#
#plt.imshow(colorim,extent=[0,7.,0,7.])
#gridcentralstereopara[:,:,0]=gaussian_filter(gridcentralstereopara[:,:,0],1)

imshow(gridcentralstereopara[:,:,0],extent=[0,7.,0.,7.],cmap=cm.gray)

xlabel('cm',fontsize=15)
ylabel('cm',fontsize=15)
show()






