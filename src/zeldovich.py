import numpy as np
import scipy.interpolate as I

# This is the main function of the program, it runs a Zeldovich realization to a given redshift
# and returns the final density grid, and arrays of the particle positions (x, y, z)
# redshift: float 
# pkinit: array of [k, pk] for the initial power spectrum
# boxsize: float (Mpc/h)
# ngrid: integer

def run_wn(redshift,wn,pkinit, boxsize=512.0, ngrid=512, exactpk=True, smw=.0, seed=314159, return_particles=False):
    
    #make initial Gaussian density field
    dens0=make_wn2gauss_init(pkinit,wn, boxsize=boxsize, ngrid=ngrid, seed=seed, exactpk=exactpk, smw=smw)
    #get displacements on the grid
    fx, fy, fz=get_disp(dens0, boxsize=boxsize, ngrid=ngrid)
    #get final positions of particles at a given redshift
    x, y, z, period_mask=get_pos(fx, fy, fz, redshift, boxsize=boxsize, ngrid=ngrid)
    
    if(return_particles):
        return density(z,y,x,boxsize,ngrid), dens0 ,x ,y, z
    else:
        return density(z,y,x,boxsize,ngrid), dens0
        
    
def make_wn2gauss_init(pkinit,u,boxsize=512.0,ngrid=512,seed=314159,exactpk=True, smw=2.0):
    
    thirdim = ngrid//2+1
    kmin = 2*np.pi/np.float64(boxsize)
    sk = (ngrid,ngrid,thirdim)
    
    #print(seed)
    a = np.fromfunction(lambda x,y,z:x, sk).astype(np.float64)
    a[np.where(a > ngrid//2)] -= ngrid
    b = np.fromfunction(lambda x,y,z:y, sk).astype(np.float64)
    b[np.where(b > ngrid//2)] -= ngrid
    c = np.fromfunction(lambda x,y,z:z, sk).astype(np.float64)
    c[np.where(c > ngrid//2)] -= ngrid
    
    kgrid = kmin*np.sqrt(a**2+b**2+c**2).astype(np.float64)
    
    rs = np.random.default_rng(seed)
    
    dk     = np.fft.rfftn(u)/ngrid**1.5 #normalize by Fourier convention
    
    filt=np.exp(-kgrid*kgrid*smw*smw)
    
    pkinterp=I.interp1d(pkinit[:,0], pkinit[:,1])
    
    
    if (kgrid[0,0,0]==0):
        dk[0,0,0]=0
        wn0=np.where(kgrid!=0)
        
        dk[wn0] *= np.sqrt(filt[wn0]*pkinterp(kgrid[wn0]))*ngrid**3/boxsize**1.5
    else:
        dk *= np.sqrt(filt*pkinterp(kgrid.flatten())).reshape(sk)*ngrid**3/boxsize**1.5
    
    dens = np.fft.irfftn(dk)
    return dens    
        
def get_disp(dens, boxsize=512.0, ngrid=512):

    cell_len=np.float64(boxsize)/np.float64(ngrid)
    thirdim=ngrid//2+1
    kmin = 2*np.pi/np.float64(boxsize)
    dens = np.fft.rfftn(dens)
    sk = (ngrid,ngrid,thirdim)
    
    a = np.fromfunction(lambda x,y,z:x, sk).astype(np.float64)
    a[np.where(a > ngrid//2)] -= ngrid
    b = np.fromfunction(lambda x,y,z:y, sk).astype(np.float64)
    b[np.where(b > ngrid//2)] -= ngrid
    c = np.fromfunction(lambda x,y,z:z, sk).astype(np.float64)
    c[np.where(c > ngrid//2)] -= ngrid
    
    
    xp=np.zeros((ngrid, ngrid, ngrid//2+1), dtype=np.complex128)
    yp=np.zeros((ngrid, ngrid, ngrid//2+1), dtype=np.complex128)
    zp=np.zeros((ngrid, ngrid, ngrid//2+1), dtype=np.complex128)
    
    kgrid = kmin*np.sqrt(a**2+b**2+c**2).astype(np.float64)
    
    kgrid[0,0,0]=1e-20
    
    xp.real =-kmin*a*(dens.imag)/(kgrid*kgrid)
    xp.imag=kmin*a*(dens.real)/(kgrid*kgrid)
    xp[0,0,0]=0.
    yp.real =-kmin*b*(dens.imag)/(kgrid*kgrid)
    yp.imag=kmin*b*(dens.real)/(kgrid*kgrid)
    yp[0,0,0]=0.
    zp.real =-kmin*c*(dens.imag)/(kgrid*kgrid)
    zp.imag=kmin*c*(dens.real)/(kgrid*kgrid)
    zp[0,0,0]=0.
    
    a=0
    b=0
    c=0
    kgrid=0
    
    xp=np.fft.irfftn(xp)
    yp=np.fft.irfftn(yp)
    zp=np.fft.irfftn(zp)
    
    return xp, yp, zp

def get_pos(fx, fy, fz, redshift, boxsize=512.0, ngrid=512):
    cell_len=np.float64(boxsize)/np.float64(ngrid)

    #setup particles on a uniform grid
    sk = (ngrid,ngrid,ngrid)
    a = np.fromfunction(lambda x,y,z:x+0.5, sk).astype(np.float64)
    b = np.fromfunction(lambda x,y,z:y+0.5, sk).astype(np.float64)
    c = np.fromfunction(lambda x,y,z:z+0.5, sk).astype(np.float64)
    a=cell_len*a.flatten()
    b=cell_len*b.flatten()
    c=cell_len*c.flatten()
    
    #displacements, scaled by the growth function at the redshift we want
    d1=growthfunc(1./(1+redshift))/growthfunc(1.)
    x=fx*d1
    y=fy*d1
    z=fz*d1
    
    #assuming ngrid=nparticles, displace particles from the grid
    a+=x.flatten()
    b+=y.flatten()
    c+=z.flatten()
    
    #periodic boundary conditions
    period_mask =np.ones((ngrid,ngrid,ngrid)).flatten()
    
    foo = np.where(a<0)
    a[foo]+=boxsize
    period_mask[foo]=0
    
    foo = np.where(a>boxsize)
    a[np.where(a>boxsize)]-=boxsize
    period_mask[foo]=0
    
    foo = np.where(b<0)
    b[foo]+=boxsize
    period_mask[foo]=0
    
    foo = np.where(b>boxsize)
    b[np.where(b>boxsize)]-=boxsize
    period_mask[foo]=0
    
    foo = np.where(c<0)
    c[foo]+=boxsize
    period_mask[foo]=0
    
    foo = np.where(c>boxsize)
    c[np.where(c>boxsize)]-=boxsize
    period_mask[foo]=0
 
    return a, b, c, period_mask

def growthfunc(a, omega_m=0.289796, omega_l=0.710204):
    da=a/10000.
    integral = 0.
    for i in range(10000):
        aa=(i+1)*da
        integral+=da/((aa*np.sqrt(omega_m/(aa**3)+omega_l))**3)
    return 5*omega_m/2*np.sqrt(omega_m/a**3+omega_l)*integral

# ensuring arrays are hermitian
def nyquist(xp):
    ngrid=xp.shape[0]
    xp[ngrid//2+1:,1:,0]= np.conj(np.fliplr(np.flipud(xp[1:ngrid//2,1:,0])))
    xp[ngrid//2+1:,0,0] = np.conj(xp[ngrid//2-1:0:-1,0,0])
    xp[0,ngrid//2+1:,0] = np.conj(xp[0,ngrid//2-1:0:-1,0])
    xp[ngrid//2,ngrid//2+1:,0] = np.conj(xp[ngrid//2,ngrid//2-1:0:-1,0])
    
    xp[ngrid//2+1:,1:,ngrid//2]= np.conj(np.fliplr(np.flipud(xp[1:ngrid//2,1:,ngrid//2])))
    xp[ngrid//2+1:,0,ngrid//2] = np.conj(xp[ngrid//2-1:0:-1,0,ngrid//2])
    xp[0,ngrid//2+1:,ngrid//2] = np.conj(xp[0,ngrid//2-1:0:-1,ngrid//2])
    xp[ngrid//2,ngrid//2+1:,ngrid//2] = np.conj(xp[ngrid//2,ngrid//2-1:0:-1,ngrid//2])
    return xp

def density(x_in,y_in,z_in,boxsize,Ngrid):
    
    cell_len=np.float64(boxsize)/np.float64(Ngrid)
    
    x_dat=x_in/cell_len
    y_dat=y_in/cell_len
    z_dat=z_in/cell_len

    #Create a new grid which will contain the densities
    grid = np.zeros([Ngrid, Ngrid, Ngrid], dtype='float64')
    
    #Find cell center coordinates
    x_c = np.floor(x_dat).astype(int)
    y_c = np.floor(y_dat).astype(int)
    z_c = np.floor(z_dat).astype(int)
    
    #Calculating contributions for the CIC interpolation
    d_x = x_dat - x_c
    d_y = y_dat - y_c
    d_z = z_dat - z_c
    
    t_x = 1. - d_x
    t_y = 1. - d_y
    t_z = 1. - d_z
    
    #Enforce periodicity for cell center coordinates + 1                
    X = (x_c+1)%Ngrid
    Y = (y_c+1)%Ngrid
    Z = (z_c+1)%Ngrid
    
    #Populate the density grid according to the CIC scheme
    
    
    aux, edges = np.histogramdd(np.array([z_c,y_c,x_c]).T, weights=t_x*t_y*t_z, bins=(Ngrid,Ngrid,Ngrid), range=[[0,Ngrid],[0,Ngrid],[0,Ngrid]])
    grid += aux
   
    aux, edges = np.histogramdd(np.array([z_c,y_c,X]).T, weights=d_x*t_y*t_z, bins=(Ngrid,Ngrid,Ngrid), range=[[0,Ngrid],[0,Ngrid],[0,Ngrid]])
    grid += aux
    
    aux, edges = np.histogramdd(np.array([z_c,Y,x_c]).T, weights=t_x*d_y*t_z, bins=(Ngrid,Ngrid,Ngrid), range=[[0,Ngrid],[0,Ngrid],[0,Ngrid]])
    grid += aux
    
    aux, edges = np.histogramdd(np.array([Z,y_c,x_c]).T, weights=t_x*t_y*d_z, bins=(Ngrid,Ngrid,Ngrid), range=[[0,Ngrid],[0,Ngrid],[0,Ngrid]])
    grid += aux
    
    aux, edges = np.histogramdd(np.array([z_c,Y,X]).T, weights=d_x*d_y*t_z, bins=(Ngrid,Ngrid,Ngrid), range=[[0,Ngrid],[0,Ngrid],[0,Ngrid]])
    grid += aux
    
    aux, edges = np.histogramdd(np.array([Z,Y,x_c]).T, weights=t_x*d_y*d_z, bins=(Ngrid,Ngrid,Ngrid), range=[[0,Ngrid],[0,Ngrid],[0,Ngrid]])
    grid += aux
    
    aux, edges = np.histogramdd(np.array([Z,y_c,X]).T, weights=d_x*t_y*d_z, bins=(Ngrid,Ngrid,Ngrid), range=[[0,Ngrid],[0,Ngrid],[0,Ngrid]])
    grid += aux
    
    aux, edges = np.histogramdd(np.array([Z,Y,X]).T, weights=d_x*d_y*d_z, bins=(Ngrid,Ngrid,Ngrid), range=[[0,Ngrid],[0,Ngrid],[0,Ngrid]])
    grid += aux
    
    return grid

def get_grid_id(x_in,y_in,z_in,boxsize,Ngrid):
    
    cell_len=np.float64(boxsize)/np.float64(Ngrid)
    
    x_dat=x_in/cell_len
    y_dat=y_in/cell_len
    z_dat=z_in/cell_len

    #Create a new grid which will contain the densities
    grid = np.zeros([Ngrid, Ngrid, Ngrid], dtype='float64')
    
    #Find cell center coordinates
    x_c = np.floor(x_dat).astype(int)
    y_c = np.floor(y_dat).astype(int)
    z_c = np.floor(z_dat).astype(int)
    
    return x_c,y_c,z_c

