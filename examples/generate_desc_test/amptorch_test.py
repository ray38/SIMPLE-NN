# append the path for amptorch
import sys

import torch
import copy
from skorch import NeuralNetRegressor
from skorch.dataset import CVSplit
from skorch.callbacks import Checkpoint, EpochScoring
from skorch.callbacks.lr_scheduler import LRScheduler
import skorch.callbacks.base
from amptorch.gaussian import SNN_Gaussian
from amptorch.fp_simple_nn import make_amp_descriptors_simple_nn
from amptorch.model import FullNN, CustomMSELoss
from amptorch.data_preprocess import AtomsDataset, factorize_data, collate_amp, TestDataset
from amptorch.skorch_model import AMP
from amptorch.skorch_model.utils import target_extractor, energy_score, forces_score
from amptorch.analysis import parity_plot
from torch.utils.data import DataLoader
from torch.nn import init
from skorch.utils import to_numpy
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.visualize import view
from ase.io import read

# get training data
label = 'test'
images = read('data/water.extxyz', index=':')
forcetraining = True

# define symmetry functions to be used
Gs = {}
Gs["G2_etas"] = np.logspace(np.log10(0.05), np.log10(5.0), num=4)
Gs["G2_rs_s"] = [0] * 4
Gs["G4_etas"] = [0.005]
Gs["G4_zetas"] = [1.0]
Gs["G4_gammas"] = [+1.0, -1.0]
Gs["cutoff"] = 6.5

# make fingerprints # Can just be used for the first time to generate fps. 
make_amp_descriptors_simple_nn(images, Gs, ['H', 'O'], forcetraining=forcetraining, cores=1, label=label, save=True)

# data-preprocessing
training_data = AtomsDataset(
        images, 
        SNN_Gaussian, 
        Gs, 
        forcetraining=forcetraining,
        label=label, 
        cores=1, 
        delta_data=None)
unique_atoms = training_data.elements
fp_length = training_data.fp_length

# define device
device = "cpu"

net = NeuralNetRegressor(
    module=FullNN(unique_atoms, [fp_length, 3, 10], device, forcetraining=forcetraining),
    criterion=CustomMSELoss,
    criterion__force_coefficient=0.1,
    optimizer=torch.optim.LBFGS,
    optimizer__line_search_fn="strong_wolfe",
    lr=1e-3,
    batch_size=len(images),
    max_epochs=20,
    iterator_train__collate_fn=collate_amp,
    iterator_train__shuffle=True,
    iterator_valid__collate_fn=collate_amp,
    device=device,
    # train_split=0,
    verbose=1,
    callbacks=[
        EpochScoring(
            forces_score,
            on_train=True,
            use_caching=True,
            target_extractor=target_extractor,
        ),
        EpochScoring(
            energy_score,
            on_train=True,
            use_caching=True,
            target_extractor=target_extractor,
        ),
    ],
)


calc_amptorch = AMP(training_data, net, label=label)
calc_amptorch.train(overwrite=True)

# for evaluation in cpu
# net_load = copy.copy(net)
# calc_cpu = AMP(training_data, net_load, label)
# calc_cpu.load(filename='./results/trained_models/{}.pt'.format(label))

parity_plot(calc_amptorch, images, data="energy", label=label)
parity_plot(calc_amptorch, images, data="forces", label=label)