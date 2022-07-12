import os
import argparse
import sys
from itertools import combinations
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import tensorflow as tf

from weensembles.CombiningMethods import comb_picker
from weensembles.predictions_evaluation import compute_error_inconsistency
from utils.utils import load_networks_outputs, evaluate_ens, evaluate_networks, linear_pw_ens_train_save, calibrating_ens_train_save, pairwise_accuracies_mat, average_variance
from utils.utils import prepare_computation_plan

def ens_evaluation(args_dict=None):
    if args_dict is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('-constituents_folder', type=str, help="Folder of combined classifiers outputs")
        parser.add_argument('-ens_sizes', nargs="+", default=[], type=int, help="Ensemble sizes to test")
        parser.add_argument('-ens_comb_file', type=str, default="",
                            help="Path to file with listing of networks combinations to test. Can be used along with -ens_sizes. Each combination should be specified on separate line by network names separated by commas.")
        parser.add_argument('-device', type=str, default="cpu", help="Device to use")
        parser.add_argument('-verbose', default=0, type=int, help="Level of verbosity")
        parser.add_argument('-load_existing_models', type=str, choices=["no", "recalculate", "lazy"], default="no", help="Loading of present models. If no - all computations are performed again, \
                            if recalculate - existing models are loaded, but metrics are calculated again, if lazy - existing models are skipped.")
        parser.add_argument('-combining_methods', nargs='+', default=["average"], help="Combining methods to use")
        parser.add_argument('-coupling_methods', nargs='+', default=['m2'], help="Coupling methods to use")
        parser.add_argument('-calibrating_methods', nargs='+', default=["TemperatureScaling"], help="Calibrating methods to test")
        parser.add_argument('-comp_precisions', nargs='+', default=["float"], help="Precisions which to test for the computations (float and/or double)")
        parser.add_argument('-output_folder', type=str, help="Path to a folder to save the outputs to.")
        parser.add_argument('-topl', nargs="+", default=[-1], type=int, help="Topl values to test")
        parser.add_argument('-process_ood', dest="process_ood", action="store_true",
                            help="Enables processing of ood samples. If specified, all the networks are expected to contain ood outputs.")
        parser.set_defaults(compute_pwm=False, save_sweep_C=False, process_ood=False)
        args = parser.parse_args()
    else:
        args = args_dict
        
    tf.config.experimental.set_visible_devices([], 'GPU')
    
    req_val_data = False
    for comb_m in args.combining_methods:
        comb_m_fun = comb_picker(comb_m, c=0, k=0)
        if comb_m_fun.req_val_:
            req_val_data = True
    
    net_outputs = load_networks_outputs(nn_outputs_path=args.constituents_folder,
                                        experiment_out_path=args.output_folder,
                                        device=args.device,
                                        dtype=torch.float64 if "double" in args.comp_precisions else torch.float32,
                                        load_train_data=req_val_data,
                                        load_ood_data=args.process_ood)
    
    comp_plan_pwc, comp_plan_cal = prepare_computation_plan(outputs_folder=args.output_folder,
                                                            networks_names=net_outputs["networks"],
                                                            combination_sizes=args.ens_sizes,
                                                            ens_comb_file=None if args.ens_comb_file == "" else args.ens_comb_file,
                                                            combining_methods=args.combining_methods,
                                                            coupling_methods=args.coupling_methods,
                                                            topl_values=args.topl,
                                                            calibrating_methods=args.calibrating_methods,
                                                            computational_precisions=args.comp_precisions,
                                                            loading_existing_models=args.load_existing_models,
                                                            device=args.device)
    
    pwc_metrics = comp_plan_pwc.execute_evaluate_save(net_outputs=net_outputs,
                                                      outputs_folder=args.output_folder,
                                                      req_comb_val_data=req_val_data,
                                                      verbose=args.verbose)
    cal_metrics = comp_plan_cal.execute_evaluate_save(net_outputs=net_outputs,
                                                      outputs_folder=args.output_folder,
                                                      req_comb_val_data=req_val_data,
                                                      verbose=args.verbose)
 
if __name__ == "__main__":
    ret = ens_evaluation()
    sys.exit(ret)