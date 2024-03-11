# PCA

INTRODUCTION:
Run Principal Component Analysis on MD simulation data usin the MDAnalysis python package.

This contains a python script that has the base code necessary to run PCA on an MD simulation. The example in the code selects the 
backbone (minus the N terminus) of VDAC, which is embedded in a lipid bilayer and performs PCA on that selection. It then creates new 
coordinates for the protein projected into the reduced space and saves the new trajectories for the first few components into PDB and
XTC files. 

The script loads the psf file and all dcd files of an MD simulation. These are loaded using the Universe environment of the MDAnalysis package.

USAGE:
python PCA.py INPUT_PATH OUTPUT_PATH STRUCTURE_FILETYPE TRAJECTORY_FILETYPE VARIANCE_CUTOFF

REFERENCES:
 https://docs.mdanalysis.org/1.0.0/index.html 
 https://userguide.mdanalysis.org/stable/examples/analysis/reduced_dimensions/pca.html
