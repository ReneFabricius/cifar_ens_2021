import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from weighted_ensembles.predictions_evaluation import compute_acc_topk, compute_nll, ECE_sweep
from weighted_ensembles.CalibrationMethod import TemperatureScaling

import torch

from utils import load_networks_outputs, calibrating_ens_train_save

EXP_OUTPUTS_FOLDER = 'exp_subsets_sizes_calibration_outputs'


def ens_train_exp():
    calibration_methods = [TemperatureScaling]

    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, required=True, help='replication_folder')
    parser.add_argument('-max_fold_rep', type=int, default=30, help='max number of folds for each train size')
    parser.add_argument('-device', type=str, default='cpu', help='device on which to execute the script')
    args = parser.parse_args()

    exper_outputs_path = os.path.join(args.folder, EXP_OUTPUTS_FOLDER)
    networks_outputs_folder = os.path.join(args.folder, 'outputs')

    if not os.path.exists(exper_outputs_path):
        os.mkdir(exper_outputs_path)

    print("Loading networks outputs")
    net_outputs = load_networks_outputs(networks_outputs_folder, exper_outputs_path, args.device)

    df_net = pd.DataFrame(columns=("network", "accuracy", "nll", "ece"))
    for i, net in enumerate(net_outputs["networks"]):
        acc = compute_acc_topk(net_outputs["test_labels"], net_outputs["test_outputs"][i], 1)
        nll = compute_nll(net_outputs["test_labels"], net_outputs["test_outputs"][i], penultimate=True)
        ece = ECE_sweep(prob_pred=torch.nn.Softmax(dim=1)(net_outputs["test_outputs"][i]),
                        tar=net_outputs["test_labels"])
        df_net.loc[i] = [net, acc, nll, ece]

    df_net.to_csv(os.path.join(exper_outputs_path, "net_metrics.csv"), index=False)

    df_ens = pd.DataFrame(columns=("calibrating_method", "train_size", "accuracy", "nll", "ece"))
    df_ens_i = 0
    uncal_ens_output = torch.sum(torch.nn.Softmax(dim=2)(net_outputs["test_outputs"]), dim=0) / net_outputs["test_outputs"].shape[0]
    uncal_ens_acc = compute_acc_topk(y_cor=net_outputs["test_labels"], ps=uncal_ens_output, l=1)
    uncal_ens_nll = compute_nll(y_cor=net_outputs["test_labels"], ps=uncal_ens_output)
    uncal_ens_ece = ECE_sweep(prob_pred=uncal_ens_output, tar=net_outputs["test_labels"])
    df_ens.loc[df_ens_i] = ["NoCalibration", None, uncal_ens_acc, uncal_ens_nll, uncal_ens_ece]
    df_ens_i += 1

    df_net_cal = pd.DataFrame(columns=("network", "calibration_method", "train_size", "nll", "ece"))
    df_net_i = 0

    n_samples = net_outputs["train_labels"].shape[0]
    min_t_size = 80
    max_t_size = 4950
    quot = 1.4
    cur_t_size = min_t_size
    while cur_t_size < max_t_size:
        print("Processing combiner train set size {}".format(cur_t_size))
        n_folds = n_samples // cur_t_size

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

        for fold_i, (_, comb_train_idxs) in enumerate(skf.split(np.zeros(n_samples),
                                                        net_outputs["train_labels"].detach().cpu().clone().numpy())):
            if fold_i >= args.max_fold_rep:
                break

            real_t_size = len(comb_train_idxs)

            print("Processing fold {}".format(fold_i))
            print("Real train size {}".format(real_t_size))
            np.save(os.path.join(exper_outputs_path, "combiner_train_idx_size_{}_repl_{}.npy".format(cur_t_size, fold_i)),
                    comb_train_idxs)
            comb_train_idxs = torch.from_numpy(comb_train_idxs).to(device=torch.device(args.device), dtype=torch.long)
            comb_train_pred = net_outputs["train_outputs"][:, comb_train_idxs, :]
            comb_train_lab = net_outputs["train_labels"][comb_train_idxs]

            test_ens_results = calibrating_ens_train_save(predictors=comb_train_pred, targets=comb_train_lab,
                                                          test_predictors=net_outputs["test_outputs"],
                                                          device=torch.device(args.device), out_path=exper_outputs_path,
                                                          calibrating_methods=calibration_methods,
                                                          prefix="size_{}_repl_{}_".format(real_t_size, fold_i))

            for cal_m in calibration_methods:
                test_ens_res = test_ens_results.get(calibrating_method=cal_m.__name__)
                acc_ens = compute_acc_topk(net_outputs["test_labels"], test_ens_res, 1)
                nll_ens = compute_nll(net_outputs["test_labels"], test_ens_res)
                ece_ens = ECE_sweep(prob_pred=test_ens_res, tar=net_outputs["test_labels"])
                df_ens.loc[df_ens_i] = [cal_m.__name__, real_t_size, acc_ens, nll_ens, ece_ens]
                df_ens_i += 1

                test_net_res = test_ens_results.get_nets(calibrating_method=cal_m.__name__)
                for net_i, net in enumerate(net_outputs["networks"]):
                    cal_net_res = test_net_res[net_i]
                    nll_net = compute_nll(y_cor=net_outputs["test_labels"], ps=cal_net_res)
                    ece_net = ECE_sweep(prob_pred=cal_net_res, tar=net_outputs["test_labels"])
                    df_net_cal.loc[df_net_i] = [net, cal_m.__name__, real_t_size, nll_net, ece_net]
                    df_net_i += 1

        cur_t_size = int(quot * cur_t_size)

    df_ens.to_csv(os.path.join(exper_outputs_path, 'ens_metrics.csv'), index=False)
    df_net_cal.to_csv(os.path.join(exper_outputs_path, 'net_cal_metrics.csv'), index=False)


if __name__ == '__main__':
    ens_train_exp()
