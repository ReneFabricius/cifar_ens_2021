import argparse
import os
import copy
import pandas as pd

from ensemble_evaluation import ens_evaluation


def half_cif_evaluation(args_dict=None):
    if args_dict is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('-experiment_base', type=str, help="Folder containing replications outputs")
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
        parser.add_argument('-test_set', type=str, default="test", help="Prefix of the constituent output files to compute testing metrics on. (Mainly for hyperparameter tuning.)")
        parser.set_defaults(compute_pwm=False, save_sweep_C=False, process_ood=False)
        args = parser.parse_args()
    else:
        args = args_dict
        
    nets_df = pd.DataFrame()
    pwc_df = pd.DataFrame()
    cal_df = pd.DataFrame()

    repl = 0
    while os.path.exists(os.path.join(args.experiment_base, str(repl))):
        constit_folder = os.path.join(args.experiment_base, str(repl), "outputs")
        cur_outputs_folder = os.path.join(args.output_folder, str(repl))
        cur_args_dict = copy.copy(args)
        cur_args_dict.constituents_folder = constit_folder
        cur_args_dict.output_folder = cur_outputs_folder
        
        ens_evaluation(args_dict=cur_args_dict)
        
        cur_nets_df = pd.read_csv(os.path.join(cur_outputs_folder, "net_metrics.csv"))
        cur_pwc_df = pd.read_csv(os.path.join(cur_outputs_folder, "ens_pwc_metrics.csv"))
        cur_cal_df = pd.read_csv(os.path.join(cur_outputs_folder, "ens_cal_metrics.csv"))
        
        cur_nets_df["replication"] = repl
        cur_pwc_df["replication"] = repl
        cur_cal_df["replication"] = repl
        
        nets_df = pd.concat([nets_df, cur_nets_df], ignore_index=True)
        pwc_df = pd.concat([pwc_df, cur_pwc_df], ignore_index=True)
        cal_df = pd.concat([cal_df, cur_cal_df], ignore_index=True)
        
        repl += 1
        
    nets_df.to_csv(os.path.join(args.output_folder, "net_metrics.csv"), index=False)
    pwc_df.to_csv(os.path.join(args.output_folder, "ens_pwc_metrics.csv"), index=False)
    cal_df.to_csv(os.path.join(args.output_folder, "ens_cal_metrics.csv"), index=False)
    
    
if __name__ == "__main__":
    half_cif_evaluation()