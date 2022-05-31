import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from weensembles.CalibrationMethod import TemperatureScaling
from weensembles.predictions_evaluation import compute_acc_topk, compute_nll, ECE_sweep

import torch

from utils.utils import load_networks_outputs, calibrating_ens_train_save, evaluate_networks, evaluate_ens

EXP_OUTPUTS_FOLDER = 'exp_subsets_sizes_calibration_outputs'


def ens_train_exp():
    calibration_methods = [TemperatureScaling]

    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, required=True, help='replication_folder')
    parser.add_argument('-max_fold_rep', type=int, default=30, help='max number of folds for each train size')
    parser.add_argument('-device', type=str, default='cpu', help='device on which to execute the script')
    parser.add_argument('-calibration_set', type=str, default='train',
                        help='Set from which to pick data for calibration. Can be train or val.')
    parser.add_argument('-verbosity', type=int, default='0',
                        help='Value greater than 0 enables detailed progress info. '
                             'May increase computational requirements')
    args = parser.parse_args()

    exper_outputs_path = os.path.join(args.folder, EXP_OUTPUTS_FOLDER)
    networks_outputs_folder = os.path.join(args.folder, 'outputs')

    if not os.path.exists(exper_outputs_path):
        os.mkdir(exper_outputs_path)

    print("Loading networks outputs")
    net_outputs = load_networks_outputs(networks_outputs_folder, exper_outputs_path, args.device)

    df_net = evaluate_networks(net_outputs)
    df_net.to_csv(os.path.join(exper_outputs_path, "net_metrics_" + args.calibration_set + ".csv"), index=False)

    df_ens = pd.DataFrame(columns=("calibrating_method", "train_size", "accuracy", "nll", "ece"))
    
    uncal_ens_output = torch.sum(torch.nn.Softmax(dim=2)(net_outputs["test_outputs"]), dim=0) / net_outputs["test_outputs"].shape[0]
    uncal_ens_acc = compute_acc_topk(tar=net_outputs["test_labels"], pred=uncal_ens_output, k=1)
    uncal_ens_nll = compute_nll(tar=net_outputs["test_labels"], pred=uncal_ens_output)
    uncal_ens_ece = ECE_sweep(pred=uncal_ens_output, tar=net_outputs["test_labels"])
    df_ens.loc[0] = ["NoCalibration", None, uncal_ens_acc, uncal_ens_nll, uncal_ens_ece]

    df_net_cal = pd.DataFrame(columns=("network", "calibrating_method", "train_size", "nll", "ece"))

    if args.calibration_set == "train":
        cal_labels = net_outputs["train_labels"]
        cal_outputs = net_outputs["train_outputs"]

    elif args.calibration_set == "val":
        cal_labels = net_outputs["val_labels"]
        cal_outputs = net_outputs["val_outputs"]

    n_samples = cal_labels.shape[0]

    min_t_size = 100
    max_t_size = 4950
    quot = 1.4
    cur_t_size = min_t_size
    while cur_t_size < max_t_size:
        print("Processing combiner train set size {}".format(cur_t_size))
        n_folds = n_samples // cur_t_size

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

        for fold_i, (_, comb_train_idxs) in enumerate(skf.split(np.zeros(n_samples),
                                                                cal_labels.detach().cpu().numpy())):
            if fold_i >= args.max_fold_rep:
                break

            real_t_size = len(comb_train_idxs)

            print("Processing fold {}".format(fold_i))
            print("Real train size {}".format(real_t_size))
            np.save(os.path.join(exper_outputs_path, "cal_set_{}_combiner_train_idx_size_{}_repl_{}.npy".format(args.calibration_set, cur_t_size, fold_i)),
                    comb_train_idxs)
            comb_train_idxs = torch.from_numpy(comb_train_idxs).to(device=torch.device(args.device), dtype=torch.long)
            comb_train_pred = cal_outputs[:, comb_train_idxs, :]
            comb_train_lab = cal_labels[comb_train_idxs]

            test_ens_results = calibrating_ens_train_save(predictors=comb_train_pred, targets=comb_train_lab,
                                                          test_predictors=net_outputs["test_outputs"],
                                                          device=torch.device(args.device), out_path=exper_outputs_path,
                                                          calibrating_methods=calibration_methods,
                                                          prefix="cal_set_{}_size_{}_repl_{}_".format(args.calibration_set, real_t_size, fold_i),
                                                          verbose=args.verbosity)

            ens_df_fold, net_cal_df_fold = evaluate_ens(test_ens_results)            
            ens_df_fold["train_size"] = real_t_size
            df_ens = pd.concat([df_ens, ens_df_fold], ignore_index=True)            
                        
            net_cal_df_fold["train_size"] = real_t_size           
            df_net_cal = pd.concat([df_net_cal, net_cal_df_fold], ignore_index=True)
                        
        cur_t_size = int(quot * cur_t_size)

    df_ens.to_csv(os.path.join(exper_outputs_path, 'ens_metrics_' + args.calibration_set + '.csv'), index=False)
    df_net_cal.to_csv(os.path.join(exper_outputs_path, 'net_cal_metrics_' + args.calibration_set + '.csv'), index=False)


if __name__ == '__main__':
    ens_train_exp()
