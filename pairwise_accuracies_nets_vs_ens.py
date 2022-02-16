import torch
import argparse
import pandas as pd
import os
from utils import load_networks_outputs, compute_pairwise_accuracies, load_npy_arr, compute_pairwise_calibration, get_irrelevant_predictions


def pairwise_accuracies_nets_vs_ens():
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, required=True, help='Replication folder')
    parser.add_argument('-device', type=str, default='cpu', help='Device on which to execute the script')
    parser.add_argument('-combining_methods', nargs='+', default=["average"], help="Combining methods to use")
    parser.add_argument('-coupling_methods', nargs='+', default=['m2'], help="Coupling methods to use")

    args = parser.parse_args()

    experiment_folder = "exp_pairwise_acc_nets_vs_ens"
    exp_output_path = os.path.join(args.folder, experiment_folder)
    
    if not os.path.exists(exp_output_path):
        os.mkdir(exp_output_path)
        
    net_outputs_path = os.path.join(args.folder, "outputs")
    ens_outputs_path = os.path.join(args.folder, "exp_ensemble_evaluation")
    
    print("Processing networks outputs")
    net_outputs = load_networks_outputs(nn_outputs_path=net_outputs_path, experiment_out_path=exp_output_path, device=args.device)
    
    nets_df = pd.DataFrame()
    for net_i, net in enumerate(net_outputs["networks"]):
        pw_acc = compute_pairwise_accuracies(preds=net_outputs["test_outputs"][net_i], labs=net_outputs["test_labels"])
        pw_acc["network"] = net
        nets_df = pd.concat([nets_df, pw_acc], ignore_index=True)
    
    nets_df.to_csv(os.path.join(exp_output_path, "net_pairwise_acc.csv"), index=False)
    
    nets_string = "+".join(sorted(net_outputs["networks"]))
    
    print("Processing ens baseline")
    ens_baseline_name = nets_string + "_ens_test_outputs_cal_TemperatureScaling_prec_float.npy"
    ens_baseline_pred = load_npy_arr(file=os.path.join(ens_outputs_path, ens_baseline_name), device=args.device, dtype=torch.float32)
    ens_baseline_df = compute_pairwise_accuracies(preds=ens_baseline_pred, labs=net_outputs["test_labels"])
    ens_baseline_df.to_csv(os.path.join(exp_output_path, "ens_baseline_pairwise_acc.csv"), index=False)
    
    print("Processing ensembles")
    ens_df = pd.DataFrame()
    for co_m in args.combining_methods:
        for cp_m in args.coupling_methods:
            ens_name = "{}_ens_test_outputs_co_{}_cp_{}_prec_float.npy".format(nets_string, co_m, cp_m)
            ens_pred = load_npy_arr(file=os.path.join(ens_outputs_path, ens_name), device=args.device, dtype=torch.float32)
            ens_pw_acc = compute_pairwise_accuracies(preds=ens_pred, labs=net_outputs["test_labels"])
            ens_pw_acc[["combining_method", "coupling_method"]] = [co_m, cp_m]
            ens_df = pd.concat([ens_df, ens_pw_acc], ignore_index=True)
    ens_df.to_csv(os.path.join(exp_output_path, "ens_pairwise_acc.csv"), index=False)
    

if __name__ == "__main__":
    pairwise_accuracies_nets_vs_ens()
        