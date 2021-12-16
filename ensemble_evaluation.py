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
    combining_methods = ["lda", "logreg", "logreg_no_interc", "logreg_sweep_C", "logreg_no_interc_sweep_C", "average", "cal_average", "prob_average", "cal_prob_average"]
    coupling_methods = ["m1", "m2", "bc", "sbt"]
    
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
    parser.set_defaults(compute_pwm=False)
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
        
        max_cal = max([0] + list(df_ens_cal[df_ens_cal["combination_size"] == len(comb)]["combination_id"]))
        max_pwc = max([0] + list(df_ens_pwc[df_ens_pwc["combination_size"] == len(comb)]["combination_id"]))
        return max(max_cal, max_pwc) + 1
    
    def process_combination(comb):
        nonlocal df_ens_cal, df_ens_pwc
        comb_size = len(comb)
        comb_id = get_combination_id(comb)
            
        mask = [net in comb for net in networks]
        nets_string = '+'.join(comb) + "_"
        train_pred = net_outputs["train_outputs"][mask]
        val_pred = net_outputs["val_outputs"][mask]
        test_pred = net_outputs["test_outputs"][mask]
        train_lab = net_outputs["train_labels"]
        val_lab = net_outputs["val_labels"]
        test_lab = net_outputs["test_labels"]
        
        err_inc, all_cor = compute_error_inconsistency(preds=test_pred, tar=test_lab)
        
        _, lin_train_idx = train_test_split(np.arange(len(train_lab)), shuffle=True, stratify=train_lab.cpu(), train_size=lin_ens_train_size)
        lin_train_pred = train_pred[:, lin_train_idx]
        lin_train_lab = train_lab[lin_train_idx]
        
        cal_ens_outputs = calibrating_ens_train_save(predictors=val_pred, targets=val_lab, test_predictors=test_pred,
                                                        device=args.device, out_path=exper_output_folder, calibrating_methods=[TemperatureScaling],
                                                        prefix=nets_string, verbose=args.verbose, networks=comb, load_existing_models=args.load_existing_models)
        
        cal_ens_df, cal_net_df = evaluate_ens(ens_outputs=cal_ens_outputs, tar=test_lab)
        if cal_ens_df.shape[0] > 0:
            cal_ens_df[networks] = mask
            cal_ens_df[["combination_size", "combination_id", "err_incons", "all_cor"]] = [comb_size, comb_id, err_inc, all_cor]
            df_ens_cal = pd.concat([df_ens_cal, cal_ens_df], ignore_index=True)

        lin_ens_outputs = linear_pw_ens_train_save(predictors=lin_train_pred, targets=lin_train_lab, test_predictors=test_pred,
                                                    device=args.device, out_path=exper_output_folder, combining_methods=combining_methods,
                                                    coupling_methods=coupling_methods, prefix=nets_string,
                                                    verbose=args.verbose, test_normality=False, val_predictors=val_pred,
                                                    val_targets=val_lab, load_existing_models=args.load_existing_models,
                                                    output_R_mats=args.compute_pwm)
        
        lin_ens_df = evaluate_ens(ens_outputs=lin_ens_outputs, tar=test_lab)
        if lin_ens_df.shape[0] > 0:
            lin_ens_df[networks] = mask
            lin_ens_df[["combination_size", "combination_id", "err_incons", "all_cor"]] = [comb_size, comb_id, err_inc, all_cor]
            df_ens_pwc = pd.concat([df_ens_pwc, lin_ens_df], ignore_index=True)
        
    for sss in [int(ens_sz) for ens_sz in args.ens_sizes]:
        print("Processing combinations of {} networks".format(sss))
        size_combs = list(combinations(networks, sss))
        for ss_i, ss in enumerate(size_combs):
            print("Progress {}%".format(100 * (ss_i + 1) // len(size_combs)), end="\r")
            process_combination(comb=ss)
        
        df_ens_pwc.to_csv(os.path.join(exper_output_folder, "ens_pwc_metrics_temp.csv"), index=False)
        df_ens_cal.to_csv(os.path.join(exper_output_folder, "ens_cal_metrics_temp.csv"), index=False)
       
    if args.ens_comb_file != "" and os.path.exists(args.ens_comb_file):
        print("Loading ensemble combinations from {}".format(args.ens_comb_file))
        ens_combs = pd.read_csv(args.ens_comb_file)
        for ri, row in ens_combs.iterrows():
            print("Progress {}%".format(100 * (ri + 1) // len(ens_combs)), end="\r")
            comb = list(ens_combs.columns[row])
            not_in = set(comb) - set(networks)
            if len(not_in) != 0:
                print("Warning, networks {} not present in provided networks outputs".format(not_in))
                continue
            process_combination(comb=comb)

    df_ens_pwc.to_csv(pwc_metrics_file, index=False)
    df_ens_cal.to_csv(cal_metrics_file, index=False)
    
if __name__ == "__main__":
    ens_evaluation()