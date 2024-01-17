#!/opt/local/bin/python                                                                                                 

# Load the necessary packages: numpy, scipy, MDAnalysis, sklearn, etc.                                                  
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import pca, align
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)  # set the seed for the random number                                                                 


# # Load the trajectory using MDAnalysis                                                                                

path = "./vdac1h_pc/" # define the path to the set of files to load using the MDA Universe                              
file_name = "vdac1-dopc" # base file name to load                                                                       
traj = [] # create a list of the trajectory files to be loaded                                                          
for i in range(0, 1001, 100):
    if i == 0:
        traj.append(path+file_name+'-'+str(i)+'to'+str(i+100)+'ns-100ps.dcd')
    else:
        traj.append(path+file_name+'-'+str(i+1)+'to'+str(i+100)+'ns-100ps.dcd')
u = mda.Universe(path + file_name + '.psf', traj) # load the trajectory using the MDA Universe environment              


# Align the trajectory.                                                                                                 
# Create an AtomGroup containing the desired atoms from the simulation that we want to perform PCA on. (this example us\
es the beta barrel of VDAC without the N terminus and loops)                                                            

n_term = " and not (resid 1:25)"
loops = [34,38,48,52,64,68,76,79,88,94,104,108,120,122,132,135,146,148,158,163,175,177,185,188,197,201,212,215,228,230,\
238,241,252,254,265,273]
ex_ids = n_term
for i in range(0,len(loops),2):
    ex_ids += " and not (resid " + str(loops[i]) + ":" +str(loops[i+1]) + ")"
aligner = align.AlignTraj(u, u, select="backbone" + ex_ids, in_memory=True).run()
BetaBarrel = u.select_atoms("backbone" + ex_ids)
# store the AtomGroup information in an data frame [resnames, ids, atom names]                                          
Barrel_df = pd.DataFrame(np.array([BetaBarrel.resnames, BetaBarrel.resids, BetaBarrel.names]).T,columns=['resname', 're\
sid', 'atom'])

# # Run PCA on VDAC using MDAnalysis                                                                                    
# Use the select argument to select the desired atoms (idealy the same as the AtomGroup above)                          
# The components are stored in results.p_components with shape (natoms*3, natoms*3)                                     

pca_vdac = pca.PCA(u, select="backbone" + ex_ids, align=True).run()

# Determine the center of mass distance for each atom.

CA = u.select_atoms("name CA")
CA_center = np.zeros(3)
i = 0
for ts in u.trajectory:
    CA_center += CA.center_of_mass()
    i += 1
CA_center /= i
dr = CA_center - pca_vdac.mean

# Determine the number of components : here we keep the components that account for 90% of the variance                 

n_pcs = np.where(pca_vdac.cumulated_variance > 0.9)[0][0]

# Transform the AtomGroup into reduced space (the weights over each component). The output has shape (n_frames, n_compo\
nents)                                                                                                                  

pca_space = pca_vdac.transform(BetaBarrel, n_pcs)

# project the original traj onto each of the first component to visualize the motion. Can repeat for each component     

projected = np.outer(pca_space[:, 0], pca_vdac.p_components[:, 0]) + pca_vdac.mean.flatten()
coordinates = projected.reshape(len(pca_space[:, 0]), -1, 3)

# We can store information about each component as the beta value in the pdb. Here we use the raidial projection of the\
 component                                                                                                              

comp = pca_vdac.p_components[:,0].reshape(len(BetaBarrel),3)
beta = dr[:,0]*comp[:,0] + dr[:,1]*comp[:,1]
bfactor_norm = beta/(2*np.absolute(beta).max()) + 0.5

new_data = {'beta_1':beta, 'beta_norm_1':bfactor_norm}
Barrel_df.assign(**new_data)

# Can create a new universe of these projected coordinates                                                              

proj1 = mda.Merge(BetaBarrel)
proj1.load_new(coordinates, order="fac")
proj1.add_TopologyAttr('tempfactors')

proj1.atoms.tempfactors = bfactor_norm
proj1.atoms.write(path + "comp1.pdb")
proj1.atoms.write(path + "comp1.xtc", frames='all') # can write the traj to a new file (here we use XTC)                

# This last section can be repeated for as many compnents as desired
