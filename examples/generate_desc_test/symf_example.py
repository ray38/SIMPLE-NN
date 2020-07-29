from ase import Atoms
from ase.build import add_adsorbate, fcc111
import numpy as np

from make_symf import make_simple_nn_fps
from make_symf import convert_simple_nn_fps

##############################################################################
# This is an example script to generate fingerprints (fps) from cffi scripts 
# in SIMPLE-NN package. 
# 
# It contains steps to do the following: 
#   1. Calculate the output directly using simple-NN via cffi interface. 
#   2. Reorganize into AMP's input format. 
# 
#                                                           N. Hu 07/13/2020
##############################################################################

# Necessary variable definitions
label = 'test' # for integration with AMPTorch module
forcetraining = True # for integration with AMPTorch module. 
# if forcetraining is set to False, it will drop the derivatives of fingerprints. 
cores = 1

# Define Gs variables for fps. 
Gs = {}
Gs["G2_etas"] = np.logspace(np.log10(0.05), np.log10(5.0), num=10)
Gs["G2_rs_s"] = [0] * 10
Gs["G4_etas"] = np.logspace(np.log10(0.005), np.log10(0.5), num=4)
Gs["G4_zetas"] = [1.0]
Gs["G4_gammas"] = [+1.0, -1.0]
Gs["cutoff"] = 6.5

# Generate ase.Atoms objects. 
slab = fcc111('Al', size=(2, 2, 4))
add_adsorbate(slab, 'H', 1.5, 'ontop')
slab.center(vacuum=10.0, axis=2)

# Define elements to calculate fps for
elements = ['Al', 'H']

# Step 1: obtain output from cffi as cffi_out
traj, calculated, cffi_out = make_simple_nn_fps(slab, Gs, elements=elements, label=label) 

# Step 2: reorganize into AMP input format
fps, fp_primes = convert_simple_nn_fps(traj, Gs, cffi_out, forcetraining, cores, save=False)

# Print the output if needed. 
print(fps)
print(fp_primes)