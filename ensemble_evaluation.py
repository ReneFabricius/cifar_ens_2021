import os
import argparse
from itertools import combinations
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from weensembles.CalibrationMethod import TemperatureScaling
from weensembles.predictions_evaluation import compute_error_inconsistency
from utils import load_networks_outputs, evaluate_ens, evaluate_networks, linear_pw_ens_train_save, calibrating_ens_train_save


def ens_evaluation():
    combining_methods = ["lda", "logreg", "logreg_no_interc", "logreg_sweep_C", "logreg_no_interc_sweep_C"]
    coupling_methods = ["m1", "m2", "bc"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, help="Replication folder")
    parser.add_argument('-ens_sizes', nargs="+", default=[2], help="Ensemble sizes to test")
    parser.add_argument('-device', type=str, default="cpu", help="Device to use")
    parser.add_argument('-cifar', type=int, help="CIFAR type (10 or 100)")
    parser.add_argument('-verbose', default=0, type=int, help="Level of verbosity")
    args = parser.parse_args()
    
    lin_ens_train_size = 50 * args.cifar
    exper_output_folder = os.path.join(args.folder, "exp_ensemble_evaluation")
    if not os.path.exists(exper_output_folder):
        os.mkdir(exper_output_folder)
        
    cal_metrics_file = os.path.join(exper_output_folder, "ens_cal_metrics.csv")
    pwc_metrics_file = os.path.join(exper_output_folder, "ens_pwc_metrics.csv")

    print("Loading networks outputs")
    net_outputs = load_networks_outputs(nn_outputs_path=os.path.join(args.folder, "outputs"), experiment_out_path=exper_output_folder, device=args.device)
    df_net = evaluate_networks(net_outputs)
    df_net.to_csv(os.path.join(exper_output_folder, "net_metrics.csv"), index=False)
    
    networks = net_outputs["networks"]

    df_ens_cal = pd.DataFrame(columns=(*networks,"combination_size", "combination_id", "calibrating_method", "accuracy", "nll", "ece", "err_incons"))
    df_ens_pwc = pd.DataFrame(columns=(*networks, "combination_size", "combination_id", "combining_method", "coupling_method", "accuracy", "nll", "ece", "err_incons"))
    
    for sss in [int(ens_sz) for ens_sz in args.ens_sizes]:
        print("Processing combinations of {} networks".format(sss))
        for ss_i, ss in enumerate(combinations(networks, sss)):
            mask = [net in ss for net in networks]
            nets_string = '+'.join(ss) + "_"
            train_pred = net_outputs["train_outputs"][mask]
            val_pred = net_outputs["val_outputs"][mask]
            test_pred = net_outputs["test_outputs"][mask]
            train_lab = net_outputs["train_labels"]
            val_lab = net_outputs["val_labels"]
            test_lab = net_outputs["test_labels"]
            
            err_inc = compute_error_inconsistency(preds=test_pred, tar=test_lab)
            
            _, lin_train_idx = train_test_split(np.arange(len(train_lab)), shuffle=True, stratify=train_lab.cpu(), train_size=lin_ens_train_size)
            lin_train_pred = train_pred[:, lin_train_idx]
            lin_train_lab = train_lab[lin_train_idx]
            
            cal_ens_outputs = calibrating_ens_train_save(predictors=val_pred, targets=val_lab, test_predictors=test_pred,
                                                         device=args.device, out_path=exper_output_folder, calibrating_methods=[TemperatureScaling],
                                                         prefix=nets_string, verbose=args.verbose, networks=ss)
            
            cal_ens_df, cal_net_df = evaluate_ens(ens_outputs=cal_ens_outputs, tar=test_lab)
            cal_ens_df[networks] = mask
            cal_ens_df[["combination_size", "combination_id", "err_incons"]] = [sss, ss_i, err_inc]
            df_ens_cal = pd.concat([df_ens_cal, cal_ens_df], ignore_index=True)

            lin_ens_outputs = linear_pw_ens_train_save(predictors=lin_train_pred, targets=lin_train_lab, test_predictors=test_pred,
                                                       device=args.device, out_path=exper_output_folder, combining_methods=combining_methods,
                                                       coupling_methods=coupling_methods, prefix=nets_string,
                                                       verbose=args.verbose, test_normality=False, val_predictors=val_pred,
                                                       val_targets=val_lab)
            
            lin_ens_df = evaluate_ens(ens_outputs=lin_ens_outputs, tar=test_lab)
            lin_ens_df[networks] = mask
            lin_ens_df[["combination_size", "combination_id", "err_incons"]] = [sss, ss_i, err_inc]
            df_ens_pwc = pd.concat([df_ens_pwc, lin_ens_df], ignore_index=True)
    
    # In case there is a bug in the following code
    df_ens_pwc.to_csv(os.path.join(exper_output_folder, "ens_pwc_metrics_new.csv"), index=False)
    df_ens_cal.to_csv(os.path.join(exper_output_folder, "ens_cal_metrics_new.csv"), index=False)

    old_df_ens_pwc = None
    if os.path.exists(pwc_metrics_file):
        old_df_ens_pwc = pd.read_csv(pwc_metrics_file)
    
    old_df_ens_cal = None
    if os.path.exists(cal_metrics_file):
        old_df_ens_cal = pd.read_csv(cal_metrics_file)    
    
    if old_df_ens_pwc is not None:
        for sss in set(old_df_ens_pwc["combination_size"]):
            if sss not in df_ens_pwc["combination_size"]:
                continue
            
            max_old_comb_id = max(old_df_ens_pwc[old_df_ens_pwc["combination_size"] == sss]["combination_id"])
            df_ens_pwc.loc[df_ens_pwc["combination_size"] == sss, "combination_id"] += max_old_comb_id
        
        df_ens_pwc = pd.concat([old_df_ens_pwc, df_ens_pwc], ignore_index=True)
        
    if old_df_ens_cal is not None:
        for sss in set(old_df_ens_cal["combination_size"]):
            if sss not in df_ens_cal["combination_size"]:
                continue
                
            max_old_comb_id = max(old_df_ens_cal[old_df_ens_cal["combination_size"] == sss]["combination_id"])
            df_ens_cal.loc[df_ens_cal["combination_size"] == sss, "combination_id"] += max_old_comb_id
            
        df_ens_cal = pd.concat([old_df_ens_cal, df_ens_cal], ignore_index=True)
    
    df_ens_pwc.to_csv(pwc_metrics_file, index=False)
    df_ens_cal.to_csv(cal_metrics_file, index=False)
    
if __name__ == "__main__":
    ens_evaluation()