# REFER TO NOTEBOOK FOR INFORMATION ON RATIONALE BEHIND CODE

import numpy as np
import h5py as hp
from scipy.fftpack import dct, idct
import dedalus.public as d3
from dedalus.core.evaluator import Evaluator
from pathlib import Path
import os

def cheb_poly(n, x):
    """
    Returns n-th Chebsyhev Polynomial evaluated at x
    """
    return np.cos(n * np.arccos(x))

def interpolate_z(u, Nx, Ny, Nz):
    """
    Extrema to roots grid interpolation for Gibson data along z-direction
    """
    u_new = np.zeros_like(u)
    GP_interpolated = np.cos(np.pi * (np.arange(0, Ny) - 1/2) / (Ny - 1))

    for i in range(Nx):
        for j in range(Nz):
            for k in range(3):
                z_slice = u[i, :, j, k] # slice at (x, y, z, n) = (i, y, j, k)
                coefficients = (2 / (Ny + 1)) * dct(z_slice, 2, norm = "backward") # obtaining coefficients via DCT

                coefficients[0] *= 0.5

                z_new = np.zeros_like(z_slice) # creating a test array for new slice

                for q in range(len(z_new)):
                    # n-th coefficient is multiplied by n-th Chebyshev polynomial evaluated at the q-th gridpoint
                    z_new[q] = sum(coefficients * cheb_poly(np.arange(0, Ny), GP_interpolated[q]))

                u_new[i, :, j, k] = z_new
                
    return u_new

def extract_gibson_data(filename):
    """
    Simple extraction scheme for Gibson data in .asc format
    """
    filepath = "/home/mabdulla/thesis/Couette_ECS/gibson_data/" + filename
    ascii_data = np.loadtxt(filepath)
    
    u = ascii_data.reshape(32, 35, 32, 3) # creating bold u
    
    u_new = interpolate_z(u, 32, 35, 32)
    
    return u_new

def gibson_to_dedalus(filename):
    """
    Saving invariant solution as checkpoint/restart file for forward
    integration in Dedalus
    """
    u_new = extract_gibson_data(filename)
    
    dtype = np.float64
    dealias = 3/2
    Nx = 32
    Ny = 35
    Nz = 32
    Lx = 5.511566058929462
    Ly = 2.513274122871834
    
    coords = d3.CartesianCoordinates('x', 'y', 'z')
    dist = d3.Distributor(coords, dtype = dtype)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias = dealias)
    ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias = dealias)
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds = (-1,1), dealias = dealias)
    x = xbasis.local_grid(1)
    y = xbasis.local_grid(1)
    z = zbasis.local_grid(1)
    
    ba = (xbasis,ybasis,zbasis)
    ba_p = (xbasis,ybasis)

    p = dist.Field(name='p', bases=ba)
    u = dist.VectorField(coords, name='u', bases=ba)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=ba_p)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=ba_p)
    tau_p = dist.Field(name='tau_p')
    
    for j in [0, 1, 2]:
        u['g'][j] = u_new[:,:,:,j]
    
    evaluator = Evaluator(dist,{'p':p, 'u':u})
    
    if "eq" in filename and int(filename[:-4][2:]) < 10:
        final_name = "eq0" + filename[:-4][2:]
    else:
        final_name = filename[:-4]
        
    check = evaluator.add_file_handler(Path("restart_files/" + final_name), iter=10, max_writes=1, virtual_file=True)
    check.add_tasks([u,p,tau_u1, tau_u2, tau_p])
    
    evaluator.evaluate_handlers([check])
    
def main():
    files = [file for file in os.listdir("gibson_data/") if ".asc" in file]
    for file in files:
        try:
            gibson_to_dedalus(file)
            print("{} saved as Dedalus file!".format(file[:-4]))
        except:
            # currently failing for TW2 - data has completely different shape & not sure why?
            print("Loop failed for {}".format(file[:-4]))
            continue
        
main()