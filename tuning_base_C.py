import os
import argparse
import sys
from itertools import combinations
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch

from weensembles.CalibrationMethod import TemperatureScaling
from weensembles.predictions_evaluation import compute_error_inconsistency
from utils.utils import load_networks_outputs, evaluate_ens, evaluate_networks, linear_pw_ens_train_save, calibrating_ens_train_save, pairwise_accuracies_mat, average_variance
from ensemble_evaluation import ens_evaluation


def global_sweep_C():
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, help="Replication folder")
    parser.add_argument('-ens_sizes', nargs="+", default=[], help="Ensemble sizes to test")
    parser.add_argument('-ens_comb_file', type=str, default="", help="Path to file with listing of networks combinations to test. Can be used along with -ens_sizes.")
    parser.add_argument('-device', type=str, default="cpu", help="Device to use")
    parser.add_argument('-cifar', type=int, help="CIFAR type (10 or 100)")
    parser.add_argument('-verbose', default=0, type=int, help="Level of verbosity")
    parser.add_argument('-load_existing_models', type=str, choices=["no", "recalculate", "lazy"], default="no", help="Loading of present models. If no - all computations are performed again, \
                        if recalculate - existing models are loaded, but metrics are calculated again, if lazy - existing models are skipped.")
    parser.add_argument('-compute_pairwise_metrics', dest="compute_pwm", action="store_true", help="Whether to compute pairwise accuracies and calibration")
    parser.add_argument('-combining_methods', nargs='+', default=["logreg"], help="Logreg methods to use")
    parser.add_argument('-coupling_methods', nargs='+', default=['m2'], help="Coupling methods to use")
    parser.add_argument('-save_C', dest='save_sweep_C', action='store_true', help="Whether to save regularization coefficients C for logreg methods with sweep_C. Defaults to False.")
    parser.add_argument('-max_pow', type=float, default=1.0, help="Highest power to test. C will be in the range [1/10**max_pow, 10**max_pow]")
    parser.add_argument('-num_pows', type=int, default=11, help="Number of C values to test")
    parser.add_argument('-output_folder', type=str, default="exp_ensemble_evaluation", help="Folder name to save the outputs to.")
    parser.set_defaults(compute_pwm=False)
    parser.set_defaults(save_sweep_C=False)
    args = parser.parse_args()
    
    combs_to_test = []
    num_pows = args.num_pows
    num_pows += (1 - (num_pows % 2))
    C_vals = 10**np.linspace(start=-args.max_pow, stop=args.max_pow,
                        num=num_pows, endpoint=True)

    for cm in args.combining_methods:
           combs_to_test += ["{}{{base_C:{}}}".format(cm, c_val) for c_val in C_vals]
           
    args.combining_methods = combs_to_test
    ens_evaluation(args_dict=args)
    
 
if __name__ == "__main__":
     global_sweep_C()