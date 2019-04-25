# -*- coding: utf-8 -*-
# -*- coding: iso-8859-1 -*-
"""
Created on Wed Apr 01 13:22:01 2015

@author: saturas
"""

from pylab import *
from cmath import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

lam=650.*10**-9

#Auflösung des SLM (x*y)[Px]
Nx=624
Ny=624
o=1
Nrx=Nx*o
Nry=Ny*o


#Pitch der "pixel" der Projektion [m]

dx=32*10**-6
dy=32*10**-6
#Abmessungen des aktiven Bereichs des SLM [m]
Sx=Nx*dx
Sy=Ny*dy
#Wellenvektor des Lasers [m**-1]
k=2*pi/lam

dX=1/Sx
dY=1/Sy

SX=1/dx
SY=1/dy



def C():

    C=zeros((Nry,Nrx),dtype=complex)

    for i in range(Nry):
        for j in range(Nrx):
            C[i,j]=exp(1j*pi*(i+j))
    return C
            
def C2():
    C2=zeros((Ny,Nx),dtype=complex)

    for i in range(Ny):
        for j in range(Nx):
            C2[i,j]=exp(1j*pi*(i+j))
    return C2
    
def rectapperture(d):
    """
    Erstellt eine Quadratische Appertur in der Mitte des Bildes mit der Breite/Höhe d
    """
    a=zeros((Ny,Nx))
    sx=(Nx-d)/2
    sy=(Ny-d)/2
    for i in range(d):
        for j in range(d):
            a[sy-1+i,sx-1+j]=1.
    return a
    
def sinusgrating(T):
    """
    Erstellt ein sinusförmiges vertikales Streifenmuster mit der Frequenz f.
    f=1 bedeutet von schwarz nach weiss.
    """
    
    a=zeros((Ny,Nx),dtype=complex)
    for i in range(Nx):
        a[:,i]=0.5*(cos(2*pi*dx*i/T)+1.)
    
    return a
    
def stripepattern(d,l):
    a=zeros((Ny,Nx),dtype=complex)
    sx=0.5*(Nx-l)
    sy=0.5*(Ny-l)
    f=l/d
    for i in range(f):
        for j in range(d):
            a[(sy-1):(sy-1+l),sx-1+i*d+j]=0.5*(1+(-1)**i)
    return a


def linse(f,t=0):
    """
    Erstellung einer Matrix welche die Phasenvercshiebung einer Linse in der Paraxialen Näherung
    in eine 832x832 Matrix speichert. Pixelabmessung dx x dy
    t=0, nur phase -> exp(1j*..)
    t=1. nur transmission -> phase(exp(1j*..))
    """
    a=zeros((Ny,Nx),dtype=complex)
    my=0.5*(Ny-1)
    mx=0.5*(Nx-1)
    for i in range(Ny):
        for j in range(Nx):
            #xy=x**2+y**2 vom Zentrum aus (mx,my)
            xy=((mx-j)*dx)**2+((my-i)*dy)**2
            if (t==1.):
                a[i,j]=phase(exp(-1j*k*xy/(2*f)))
            else:
                a[i,j]=exp(-1j*k*xy/(2*f))
    return a
    
def shiftlinse(f,sx,sy,t=0):
    """
    Erstellung einer Matrix welche die Phasenvercshiebung einer Linse in der Paraxialen Näherung
    in eine 832x832 Matrix speichert. Pixelabmessung dx x dy
    sx,sy verschiebung des Linsenmittelpunktes in Pixel
    t=0, nur phase -> exp(1j*..)
    t=1. nur transmission -> phase(exp(1j*..))
    """
    a=zeros((Ny,Nx),dtype=complex)
    my=0.5*(Ny-1)+sy
    mx=0.5*(Nx-1)+sx
    for i in range(Ny):
        for j in range(Nx):
            #xy=x**2+y**2 vom Zentrum aus (mx,my)
            xy=((mx-j)*dx)**2+((my-i)*dy)**2
            if (t==1.):
                a[i,j]=phase(exp(-1j*k*xy/(2*f)))
            else:
                a[i,j]=exp(-1j*k*xy/(2*f))
            
    return a
    
def intensity(c=1.):
    '''
    t=1-c*i/Nx
    '''
    a=zeros((Ny,Nx),complex)
     
    for i in range(Nx):
        a[:,i]=1-c*i/624
    
    return a

def gauss(b):
    a=zeros((Ny,Nx),dtype=complex)
    my=0.5*(Ny-1)
    mx=0.5*(Nx-1)
    for i in range(Ny):
        for j in range(Nx):
            #xy=x**2+y**2 vom Zentrum aus (mx,my)
            xy=((mx-j)*dx)**2+((my-i)*dy)**2
            a[i,j]=exp(-b*xy)
            
    return a

def cubicphase(bx,by,t=0):
    """
    Erstellung einer Matrix welche die Phasenvercshiebung einer Linse in der Paraxialen Näherung
    in eine 832x832 Matrix speichert. Pixelabmessung dx x dy
    input=bx,by,t=0
    """
    a=zeros((Ny,Nx),dtype=complex)
    my=0.5*(Ny-1)
    mx=0.5*(Nx-1)
    
    for i in range(Ny):
        for j in range(Nx):
            #xy=x**2+y**2 vom Zentrum aus (mx,my)
            
            xy=((mx-j)*dx*bx)**3+((my-i)*dy*by)**3
            if (t==1.):
                a[i,j]=phase(exp(1j*xy)*exp(1j*pi))
            else:
                a[i,j]=exp(1j*xy)*exp(1j*pi)
            
    return a

def cubiclens(f,b,t=0):
    """
    Cubische Phase + spährische Linsen Phase
    """
    a=zeros((Ny,Nx),dtype=complex)
    my=0.5*(Ny-1)
    mx=0.5*(Nx-1)
    
    for i in range(Ny):
        for j in range(Nx):
            XY=((mx-j)*dx)**2+((my-i)*dy)**2
            xy=((mx-j)*dx*b)**3+((my-i)*dy*b)**3
            if (t==1.):
                a[i,j]=phase(exp(1j*xy)*exp(1j*pi)*exp(-1j*k*XY/(2*f)))
            else:
                a[i,j]=exp(1j*xy)*exp(1j*pi)*exp(-1j*k*XY/(2*f))
            
    return a
    
def cubiclens2(f,bx,sx=0,sy=0,t=0):
    """
    Cubische Phase + spährische Linsen Phase
    sx,sy in Px
    input: f,by,bx,sx=0,sy=0,t=0
    """
    by=bx
    Nx=Nrx
    Ny=Nx
    a=zeros((Ny,Nx),dtype=complex)
    my=0.5*(Ny-1)
    mx=0.5*(Nx-1)
    
    for i in range(Ny):
        for j in range(Nx):
            XY=((mx+sx-j)*dx)**2+((my+sy-i)*dy)**2
            xy=((mx-j)*dx*bx)**3+((my-i)*dy*by)**3
            if (t==1.):
                a[i,j]=phase(exp(1j*xy)*exp(1j*pi)*exp(-1j*k*XY/(2*f)))
            else:
                a[i,j]=exp(1j*xy)*exp(1j*pi)*exp(-1j*k*XY/(2*f))
            
    return a




def realpixel(a):
    to=0
    r=zeros((Nry,Nrx),dtype=complex)
    r[:,:]=to
    
    for i in range(Ny):
        for j in range(Nx):
            for k in range(4):
                for l in range(5):
                    r[i*o+k+2,j*o+l+1]=a[i,j]
    return r
    
def pixel(a):
    to=0
    r=zeros((Nry,Nrx),dtype=complex)
    r[:,:]=to
    
    for i in range(Ny):
        for j in range(Nx):
            r[i*o:(i*o+o),j*o:(j*o+o)]=a[i,j]
    return r
    

def propagator(z):
    p=zeros((Nry,Nrx),dtype=complex)
    my=(Nry-1.)/2
    mx=(Nrx-1.)/2
    for i in range(Nry):
        for j in range(Nrx):
            c=(lam*(my-i)*dY)**2+(lam*(mx-j)*dX)**2
            p[i,j]=exp(1j*k*z*sqrt(1-c))
    
    return p
    
def propagator2(z):
    p=zeros((Nx,Ny),dtype=complex)
    m=(Nx-1.)/2
    for i in range(Nx):
        for j in range(Ny):
            c=(lam*(m-i)*dY)**2+(lam*(m-j)*dX)**2
            p[i,j]=exp(1j*k*z*sqrt(1-c))
    
    return p
    
def Uf(Ui,z):
    Uf=fft2(fft2(Ui/(Nrx*Nry)*C())*C()*propagator(z)*C())*C()
    return Uf
    
def Uf2(Ui,z):
    Uf=fft2(fft2(Ui/(Nx*Ny)*C2())*C2()*propagator2(z)*C2())*C2()
    return Uf

def linse2(f,t=0):
    """
    Erstellung einer Matrix welche die Phasenvercshiebung einer Linse in der Paraxialen Näherung
    in eine 832x832 Matrix speichert. Pixelabmessung dx x dy
    t=0, nur phase -> exp(1j*..)
    t=1. nur transmission -> phase(exp(1j*..))
    """
    Nx=624*4
    Ny=Nx
    dx=(3.2*10**-5)/4.
    dy=dx
    a=zeros((Nx,Nx),dtype=complex)
    my=0.5*(Ny-1)
    mx=0.5*(Nx-1)
    for i in range(Ny):
        for j in range(Nx):
            #xy=x**2+y**2 vom Zentrum aus (mx,my)
            xy=((mx-j)*dx)**2+((my-i)*dy)**2
            if (t==1.):
                a[i,j]=phase(exp(-1j*k*xy/(2*f)))
            else:
                a[i,j]=exp(-1j*k*xy/(2*f))
    return a
    
def circ(d,t=1.):
    A=zeros((Ny,Nx),complex)
    my=0.5*(Ny-1)
    mx=0.5*(Nx-1)
    
    for i in range(Nx):
        for j in range(Ny):
            if (t==1.):
                if (((mx-i)**2*dx**2+(my-j)**2*dy**2)<(d/2)**2):
                    A[j,i]=1.
            else:
                if (((mx-i)**2*dx**2+(my-j)**2*dy**2)<(d/2)**2):
                    A[j,i]=exp(1j*pi)
                else:
                    A[j,i]=1.
            
    return A
    
def rs(A):
    rs=zeros((2))
    Asum=sum(A)
    for i in range(shape(A)[0]):
        for j in range(shape(A)[1]):
            rs[0]+=A[i,j]/Asum*i
            rs[1]+=A[i,j]/Asum*j
    return rs

f=1.8 #focal length of lens
z=3 #plane of Uf, Ui at 0
b=420 #cubic phase param

Ui=cubiclens2(f,b)
Uf=Uf2(Ui,z)
imshow(abs(Uf), cmap=cm.hot_r)
