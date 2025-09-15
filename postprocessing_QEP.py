#%%
from evs_class import *
import numpy as np
from scipy.io import savemat
import pickle

#%%
nuc = 6 # number of unit-cells in QEP_data folder
trunc = 28 # eigenspectrum truncation parameter. All the eigenvalues with imaginary part > trunc will be omitted

#%%
for i in range(nuc):

    # default
    a = 1 # length of the unit cell
    h = 1 # height of the unit cell

    file_o = 'QEP_data/uc_'+str(i)
    matfile_s = 'Matlab_data/uc_'+str(i)+'.mat'

    # Load the variables obtained by solving QEP
    with open(file_o, "rb") as picklefile:
        (ev, ef, ky, mesh, rh, mu) = pickle.load(picklefile)

    ########################## post-processing ##################################
    # Create an eigenspectrum object to handle filtering
    evs = eigenspectrum(ev, ef)
    # Select modes within the first Brillouin Zone
    evs.pick_first_BZ()
    # Select propagating and evanescent modes
    evs.seperate_ev()
    # Calculate gradients of the eigenfunctions
    evs.generate_gradient(mu)
    # truncate the eigenspectrum
    evs.generate_eigenspectrum(trunc, ky, a)
    # print the length of the eigenspectrum
    print(len(evs.krt))

    ########################### MATLAB readable form #############################
    # This section interpolates the NGSolve GridFunction solution onto a regular Cartesian grid
    inter_x = 101 # Number of interpolation points in x
    inter_y = 101 # Number of interpolation points in y

    x, y = np.meshgrid(np.linspace(0, a, inter_x), np.linspace(0, h, inter_y))

    kappa = np.zeros(len(evs.krt), dtype=complex)                   # eigenvalue
    phi = np.zeros((inter_y, inter_x, len(evs.krt)), dtype=complex) # eigenfunction
    psi = np.zeros((inter_y, inter_x, len(evs.krt)), dtype=complex) # gradient of the eigenfunction
    G = np.zeros((inter_y, inter_x))                                # shear modulus

    for k in range(len(evs.krt)):
        temp_phi = evs.phin[k]
        temp_psi = evs.psin[k]
        kappa[k] = evs.krt [k][0]
        for i in range(inter_y):
            for j in range(inter_x):
                phi[i, j, k] = temp_phi(mesh(x[i, j], y[i, j]))
                psi[i, j, k] = temp_psi(mesh(x[i, j], y[i, j]))
                G[i, j]  = mu(mesh(x[i, j], y[i, j]))

    # Saving variable to matlab file
    mdict = {'x' : x, 'y' : y, 'phi' : phi, 'psi' : psi, 'G':G, 'kappa' : kappa, 'ky' : ky}
    savemat(matfile_s, mdict)
# %%