#!/opt/local/bin/python

# Load the necessary packages: numpy, scipy, MDAnalysis, sklearn, etc.
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import pca, align
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0) #set the seed for the random number for debugging

path = "./vdac1h_pc/" # define the path to the set of files to load using the MDA Universe
file_name = "vdac1-dopc" # base file name to load
traj = [] # create a list of the trajectory files to be loaded
for i in range(0, 1001, 100):
    if i == 0:
        traj.append(path+file_name+'-'+str(i)+'to'+str(i+100)+'ns-100ps.dcd')
    else:
        traj.append(path+file_name+'-'+str(i+1)+'to'+str(i+100)+'ns-100ps.dcd')
# load the trajectory using the MDA Universe environment
u = mda.Universe(path + file_name + '.psf', traj)

# Create atom groups (this example uses the beta barrel of VDAC with the N terminus)
aligner = align.AlignTraj(u, u,in_memory=True).run()
BetaBarrel = u.select_atoms("backbone and not (resnum 1:25)")
# the center of mass of the barrel using the alpha carbons
CA = u.select_atoms("name CA")
CA_center = CA.center_of_mass()

# PCA using MDA
# use the select argument to select the atom group you would like to run PCA on
pca_vdac = pca.PCA(u, select="backbone and not (resnum 1:25)", align=True).run()
# the componenets are stored in pca_vdac.p_components with shape (natoms×3,natoms×3)

# Determine the number of components to keep: this selects those that account for 90% of the variance
n_pcs = np.where(pca_vdac.cumulated_variance > 0.9)[0][0]

# transform the atom group into the weights over each PC. The output has shape (n_frames, n_componenets)
pca_space = pca_vdac.transform(BetaBarrel, n_pcs)

# project the original traj onto each of the first component to visualize the motion. Can repeat for each component
projected = np.outer(pca_space[:, 0], pca_vdac.p_components[:, 0]) + pca_vdac.mean.flatten()
coordinates = projected.reshape(len(pca_space[:, 0]), -1, 3)

# Can create a new universe of these projected coordinates
proj1 = mda.Merge(BetaBarrel)
proj1.load_new(coordinates, order="fac")
# can write the traj to a new file (here I use XTC)
proj1.atoms.write(path + "comp1.pdb")
proj1.atoms.write(path + "comp1.xtc", frames='all')
