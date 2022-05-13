import os
import sys
import numpy as np
import torch

from weensembles.CombiningMethods import comb_picker
from weensembles.predictions_evaluation import compute_error_inconsistency
from utils import load_networks_outputs, print_memory_statistics
from weensembles.WeightedLinearEnsemble import WeightedLinearEnsemble

def test_mem(net_outputs=r'D:\skola\1\weighted_ensembles\tests\test_IM2012_2022\data\outputs', device="cuda"):
    print_memory_statistics()
    print("Loading networks")
    net_outputs = load_networks_outputs(nn_outputs_path=net_outputs, device=device, load_train_data=False)
    c, n, k = net_outputs["val_outputs"].shape
    print_memory_statistics()
    print("Creating ensemble")
    wle = WeightedLinearEnsemble(c=c, k=k, device=device)
    print_memory_statistics()
    print("Training ensemble")
    wle.fit(MP=net_outputs["val_outputs"], tar=net_outputs["val_labels"], combining_method="logreg_torch", verbose=4)