import os
import argparse
import numpy as np

from ensemble_evaluation import ens_evaluation


def global_sweep_C():
    parser = argparse.ArgumentParser()
    parser.add_argument('-constituents_folder', type=str, help="Replication folder")
    parser.add_argument('-ens_sizes', nargs="+", default=[], type=int, help="Ensemble sizes to test")
    parser.add_argument('-ens_comb_file', type=str, default="", help="Path to file with listing of networks combinations to test. Can be used along with -ens_sizes.")
    parser.add_argument('-device', type=str, default="cpu", help="Device to use")
    parser.add_argument('-verbose', default=0, type=int, help="Level of verbosity")
    parser.add_argument('-load_existing_models', type=str, choices=["no", "recalculate", "lazy"], default="no", help="Loading of present models. If no - all computations are performed again, \
                        if recalculate - existing models are loaded, but metrics are calculated again, if lazy - existing models are skipped.")
    parser.add_argument('-combining_methods', nargs='+', default=["logreg"], help="Logreg methods to use")
    parser.add_argument('-coupling_methods', nargs='+', default=['m2'], help="Coupling methods to use")
    parser.add_argument('-max_pow', type=float, default=1.0, help="Highest power to test. C will be in the range [1/10**max_pow, 10**max_pow]")
    parser.add_argument('-num_pows', type=int, default=11, help="Number of C values to test")
    parser.add_argument('-output_folder', type=str, default="exp_ensemble_evaluation", help="Folder name to save the outputs to.")
    parser.add_argument('-comp_precisions', nargs='+', default=["float"], help="Precisions which to test for the computations (float and/or double)")
    parser.add_argument('-calibrating_methods', nargs='+', default=["TemperatureScaling"], help="Calibrating methods to test")
    parser.add_argument('-test_set', type=str, default="test", help="Prefix of the constituent output files to compute testing metrics on. (Mainly for hyperparameter tuning.)")
    parser.add_argument('-topl', nargs="+", default=[-1], type=int, help="Topl values to test")
    parser.set_defaults(compute_pwm=False, save_sweep_C=False, process_ood=False)
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