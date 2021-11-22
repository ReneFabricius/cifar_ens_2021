import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from weensembles.predictions_evaluation import compute_acc_topk, compute_nll
from weensembles.SimplePWCombine import m1, m2, bc, m2_iter

import torch

from utils import load_networks_outputs, load_npy_arr, linear_pw_ens_train_save

TRAIN_OUTPUTS_FOLDER = 'exp_subsets_train_outputs'


def ens_train_exp():
    combining_methods = ["lda", "logreg", "logreg_no_interc"]
    coupling_methods = [m1, m2, m2_iter, bc]

    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, required=True, help='replication_folder')
    parser.add_argument('-train_size', type=int, default=500, help='size of lda training set')
    parser.add_argument('-cifar', default=100, type=int, help='cifar type (10 or 100)')
    parser.add_argument('-device', type=str, default='cpu', help='device on which to execute the script')
    args = parser.parse_args()

    exper_outputs_path = os.path.join(args.folder, TRAIN_OUTPUTS_FOLDER)
    networks_outputs_folder = os.path.join(args.folder, 'outputs')

    if not os.path.exists(exper_outputs_path):
        os.mkdir(exper_outputs_path)

    print("Loading networks outputs")
    net_outputs = load_networks_outputs(networks_outputs_folder, exper_outputs_path, args.device)

    df_net = pd.DataFrame(columns=("network", "accuracy", "nll"))
    for i, net in enumerate(net_outputs["networks"]):
        acc = compute_acc_topk(net_outputs["test_labels"], net_outputs["test_outputs"][i], 1)
        nll = compute_nll(net_outputs["test_labels"], net_outputs["test_outputs"][i], penultimate=True)
        df_net.loc[i] = [net, acc, nll]

    df_net.to_csv(os.path.join(exper_outputs_path, "net_accuracies.csv"), index=False)

    n_samples = net_outputs["train_labels"].shape[0]
    n_folds = n_samples // args.train_size
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    df = pd.DataFrame(columns=("combining_method", "coupling_method", "precision", "accuracy", "nll"))
    df_i = 0

    for fold_i, (_, lda_train_idxs) in enumerate(skf.split(np.zeros(n_samples),
                                                           net_outputs["train_labels"].detach().cpu().clone().numpy())):
        print("Processing fold ", fold_i)
        np.save(os.path.join(exper_outputs_path, str(fold_i) + "_lda_train_idx.npy"), lda_train_idxs)
        lda_train_idxs = torch.from_numpy(lda_train_idxs).to(device=torch.device(args.device), dtype=torch.long)
        lda_train_pred = net_outputs["train_outputs"][:, lda_train_idxs, :]
        lda_train_lab = net_outputs["train_labels"][lda_train_idxs]

        data_type = ["float", "double"]
        for dtype in data_type:
            test_ens_results = linear_pw_ens_train_save(predictors=lda_train_pred, targets=lda_train_lab,
                                                        test_predictors=net_outputs["test_outputs"],
                                                        device=torch.device(args.device), out_path=exper_outputs_path,
                                                        combining_methods=combining_methods,
                                                        coupling_methods=coupling_methods, prefix=(str(fold_i) + "_"),
                                                        double_accuracy=(dtype == "double"))

            for co_m in combining_methods:
                for cp_m in [cp.__name__ for cp in coupling_methods]:
                    test_ens_method_res = test_ens_results.get(co_m, cp_m)
                    acc_method = compute_acc_topk(net_outputs["test_labels"], test_ens_method_res, 1)
                    nll_method = compute_nll(net_outputs["test_labels"], test_ens_method_res)
                    df.loc[df_i] = [co_m, cp_m, dtype, acc_method, nll_method]
                    df_i += 1

    df.to_csv(os.path.join(exper_outputs_path, 'accuracies.csv'), index=False)


if __name__ == '__main__':
    ens_train_exp()
