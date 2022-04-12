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
from utils import load_networks_outputs, evaluate_ens, evaluate_networks, linear_pw_ens_train_save, calibrating_ens_train_save, pairwise_accuracies_mat, average_variance


def ens_evaluation(args_dict=None):
    if args_dict is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('-folder', type=str, help="Replication folder")
        parser.add_argument('-ens_sizes', nargs="+", default=[], help="Ensemble sizes to test")
        parser.add_argument('-ens_comb_file', type=str, default="", help="Path to file with listing of networks combinations to test. Can be used along with -ens_sizes.")
        parser.add_argument('-device', type=str, default="cpu", help="Device to use")
        parser.add_argument('-verbose', default=0, type=int, help="Level of verbosity")
        parser.add_argument('-load_existing_models', type=str, choices=["no", "recalculate", "lazy"], default="no", help="Loading of present models. If no - all computations are performed again, \
                            if recalculate - existing models are loaded, but metrics are calculated again, if lazy - existing models are skipped.")
        parser.add_argument('-compute_pairwise_metrics', dest="compute_pwm", action="store_true", help="Whether to compute pairwise accuracies and calibration")
        parser.add_argument('-combining_methods', nargs='+', default=["average"], help="Combining methods to use")
        parser.add_argument('-coupling_methods', nargs='+', default=['m2'], help="Coupling methods to use")
        parser.add_argument('-save_C', dest='save_sweep_C', action='store_true', help="Whether to save regularization coefficients C for logreg methods with sweep_C. Defaults to False.")
        parser.add_argument('-output_folder', type=str, default="exp_ensemble_evaluation", help="Folder name to save the outputs to.")
        parser.set_defaults(compute_pwm=False)
        parser.set_defaults(save_sweep_C=False)
        args = parser.parse_args()
    else:
        args = args_dict
    
    COMBINER_TRAINING_SAMPLES_PER_CLASS = 50
    
    dtp = torch.float32
    exper_output_folder = os.path.join(args.folder, args.output_folder)
    if not os.path.exists(exper_output_folder):
        os.mkdir(exper_output_folder)
        
    cal_metrics_file = os.path.join(exper_output_folder, "ens_cal_metrics.csv")
    pwc_metrics_file = os.path.join(exper_output_folder, "ens_pwc_metrics.csv")

    print("Loading networks outputs")
    net_outputs = load_networks_outputs(nn_outputs_path=os.path.join(args.folder, "outputs"), experiment_out_path=exper_output_folder,
                                        device=args.device, dtype=dtp)

    classes = torch.unique(net_outputs["val_labels"])
    n_classes = len(classes)
    val_size = len(net_outputs["val_labels"])
    
    if val_size > n_classes * COMBINER_TRAINING_SAMPLES_PER_CLASS:
        val_subset_size = n_classes * COMBINER_TRAINING_SAMPLES_PER_CLASS
        print("Warning: subsetting validation set to the size of {}".format(val_subset_size))
        _, val_ss_inds = train_test_split(np.arange(val_size), test_size=val_subset_size,
                                            random_state=42, stratify=net_outputs["val_labels"].cpu())
        net_outputs["val_labels"] = net_outputs["val_labels"][val_ss_inds]
        net_outputs["val_outputs"] = net_outputs["val_outputs"][:, val_ss_inds]
    
    print("Evaluating networks")
    df_net = evaluate_networks(net_outputs)
    df_net.to_csv(os.path.join(exper_output_folder, "net_metrics.csv"), index=False)
    
    net_pwa = pairwise_accuracies_mat(preds=net_outputs["test_outputs"], labs=net_outputs["test_labels"])
    
    networks = net_outputs["networks"]
    if os.path.exists(cal_metrics_file) and args.load_existing_models == "lazy":
        df_ens_cal = pd.read_csv(cal_metrics_file)
    else:
        df_ens_cal = pd.DataFrame()
    if os.path.exists(pwc_metrics_file) and args.load_existing_models == "lazy":
        df_ens_pwc = pd.read_csv(pwc_metrics_file)
    else:
        df_ens_pwc = pd.DataFrame()
    
    def get_combination_id(comb):
        mask = [net in comb for net in networks]
        if df_ens_cal.shape[0] > 0:
            cal_ids = list(df_ens_cal[(df_ens_cal[networks] == mask).prod(axis=1) == 1]["combination_id"])
            cal_id = cal_ids[0] if len(cal_ids) > 0 else None
        else:
            cal_id = None
        
        if df_ens_pwc.shape[0] > 0:
            pwc_ids = list(df_ens_pwc[(df_ens_pwc[networks] == mask).prod(axis=1) == 1]["combination_id"])
            pwc_id = pwc_ids[0] if len(pwc_ids) > 0 else None
        else:
            pwc_id = None
            
        if cal_id is not None and pwc_id is not None:
            if cal_id == pwc_id:
                return cal_id
            else:
                raise ValueError("Networks combination {} is in dataframes under two different ids".format(comb))
        
        if cal_id is not None:
            return cal_id
        if pwc_id is not None:
            return pwc_id
        
        max_cal = max([0] + list(df_ens_cal[df_ens_cal["combination_size"] == len(comb)]["combination_id"])) if df_ens_cal.shape[0] > 0 else 0
        max_pwc = max([0] + list(df_ens_pwc[df_ens_pwc["combination_size"] == len(comb)]["combination_id"])) if df_ens_pwc.shape[0] > 0 else 0
        return max(max_cal, max_pwc) + 1
    
    def process_combination(comb):
        nonlocal df_ens_cal, df_ens_pwc
        comb_size = len(comb)
        comb_id = get_combination_id(comb)
            
        mask = [net in comb for net in networks]
        nets_string = '+'.join(sorted(comb)) + "_"
        train_pred = net_outputs["train_outputs"][mask]
        val_pred = net_outputs["val_outputs"][mask]
        test_pred = net_outputs["test_outputs"][mask]
        train_lab = net_outputs["train_labels"]
        val_lab = net_outputs["val_labels"]
        test_lab = net_outputs["test_labels"]
        
        err_inc, all_cor = compute_error_inconsistency(preds=test_pred, tar=test_lab)
        mean_pwa_var = average_variance(inp=net_pwa[mask])
        
        _, lin_train_idx = train_test_split(np.arange(len(train_lab)), shuffle=True, stratify=train_lab.cpu(),
                                            test_size=n_classes * COMBINER_TRAINING_SAMPLES_PER_CLASS)
        combiner_val_pred = train_pred[:, lin_train_idx]
        combiner_val_lab = train_lab[lin_train_idx]
        
        cal_ens_outputs = calibrating_ens_train_save(predictors=val_pred, targets=val_lab, test_predictors=test_pred,
                                                        device=args.device, out_path=exper_output_folder, calibrating_methods=[TemperatureScaling],
                                                        prefix=nets_string, verbose=args.verbose, networks=comb, load_existing_models=args.load_existing_models,
                                                        computed_metrics=df_ens_cal, all_networks=networks)
        
        cal_ens_df, cal_net_df = evaluate_ens(ens_outputs=cal_ens_outputs, tar=test_lab)
        if cal_ens_df.shape[0] > 0:
            cal_ens_df[networks] = mask
            cal_ens_df[["combination_size", "combination_id", "err_incons", "all_cor", "mean_pwa_var"]] = [comb_size, comb_id, err_inc, all_cor, mean_pwa_var]
            df_ens_cal = pd.concat([df_ens_cal, cal_ens_df], ignore_index=True)

        lin_ens_outputs = linear_pw_ens_train_save(predictors=val_pred, targets=val_lab, test_predictors=test_pred,
                                                    device=args.device, out_path=exper_output_folder, networks=comb,
                                                    combining_methods=args.combining_methods,
                                                    coupling_methods=args.coupling_methods, prefix=nets_string,
                                                    verbose=args.verbose, val_predictors=combiner_val_pred,
                                                    val_targets=combiner_val_lab, load_existing_models=args.load_existing_models,
                                                    computed_metrics=df_ens_pwc, all_networks=networks,
                                                    save_sweep_C=args.save_sweep_C)
        
        lin_ens_df = evaluate_ens(ens_outputs=lin_ens_outputs, tar=test_lab)
        if lin_ens_df.shape[0] > 0:
            lin_ens_df[networks] = mask
            lin_ens_df[["combination_size", "combination_id", "err_incons", "all_cor", "mean_pwa_var"]] = [comb_size, comb_id, err_inc, all_cor, mean_pwa_var]
            df_ens_pwc = pd.concat([df_ens_pwc, lin_ens_df], ignore_index=True)
        
    for sss in [int(ens_sz) for ens_sz in args.ens_sizes]:
        print("Processing combinations of {} networks".format(sss))
        size_combs = list(combinations(networks, sss))
        for ss_i, ss in enumerate(size_combs):
            print("Processing combination {} out of {}.".format(ss_i + 1, len(size_combs)))
            process_combination(comb=ss)
        
            df_ens_pwc.to_csv(pwc_metrics_file, index=False)
            df_ens_cal.to_csv(cal_metrics_file, index=False)

        cal_metrics_up_to = os.path.join(exper_output_folder, "ens_cal_metrics_{}.csv".format(sss))
        pwc_metrics_up_to = os.path.join(exper_output_folder, "ens_pwc_metrics_{}.csv".format(sss))
        df_ens_pwc.to_csv(pwc_metrics_up_to, index=False)
        df_ens_cal.to_csv(cal_metrics_up_to, index=False)
       
    if args.ens_comb_file != "" and os.path.exists(args.ens_comb_file):
        print("Loading ensemble combinations from {}".format(args.ens_comb_file))
        ens_combs = pd.read_csv(args.ens_comb_file)
        for ri, row in ens_combs.iterrows():
            print("Processing combination {} out of {}.".format(ri + 1, len(ens_combs)))
            comb = list(ens_combs.columns[row])
            not_in = set(comb) - set(networks)
            if len(not_in) != 0:
                print("Warning, networks {} not present in provided networks outputs".format(not_in))
                continue
            process_combination(comb=comb)

            df_ens_pwc.to_csv(pwc_metrics_file, index=False)
            df_ens_cal.to_csv(cal_metrics_file, index=False)
    
    return 0
    
if __name__ == "__main__":
    ret = ens_evaluation()
    sys.exit(ret)