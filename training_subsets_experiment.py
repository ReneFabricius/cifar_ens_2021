import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from weighted_ensembles.predictions_evaluation import compute_acc_topk, compute_nll
from weighted_ensembles.SimplePWCombine import m1, m2, bc, m2_iter

import torch

from utils import load_networks_outputs, load_npy_arr, ens_train_save

TRAIN_OUTPUTS_FOLDER = 'exp_subsets_train_outputs'


def ens_train_exp():
    pwc_methods = [m1, m2, m2_iter, bc]

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
    train_outputs, train_labels, val_outputs, val_labels, test_outputs, test_labels, networks = \
        load_networks_outputs(networks_outputs_folder, exper_outputs_path, args.device)

    df_net = pd.DataFrame(columns=("network", "accuracy", "nll"))
    for i, net in enumerate(networks):
        acc = compute_acc_topk(test_labels, test_outputs[i], 1)
        nll = compute_nll(test_labels, test_outputs[i], penultimate=True)
        df_net.loc[i] = [net, acc, nll]

    df_net.to_csv(os.path.join(exper_outputs_path, "net_accuracies.csv"), index=False)

    n_samples = train_labels.shape[0]
    n_folds = n_samples // args.train_size
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    df = pd.DataFrame(columns=("method", "accuracy", "nll"))
    df_i = 0

    for fold_i, (_, lda_train_idxs) in enumerate(skf.split(np.zeros(n_samples),
                                                           train_labels.detach().cpu().clone().numpy())):
        print("Processing fold ", fold_i)
        np.save(os.path.join(exper_outputs_path, str(fold_i) + "_lda_train_idx.npy"), lda_train_idxs)
        lda_train_idxs = torch.from_numpy(lda_train_idxs).to(device=torch.device(args.device), dtype=torch.long)
        lda_train_pred = train_outputs[:, lda_train_idxs, :]
        lda_train_lab = train_labels[lda_train_idxs]

        data_type = ["float", "double"]
        for dtype in data_type:
            test_ens_results = ens_train_save(lda_train_pred, lda_train_lab, test_outputs,
                                                torch.device(args.device), exper_outputs_path,
                                                pwc_methods, prefix=(str(fold_i) + "_"),
                                              double_accuracy=(dtype == "double"))

            for mi, test_ens_method_res in enumerate(test_ens_results):
                acc_method = compute_acc_topk(test_labels, test_ens_method_res, 1)
                nll_method = compute_nll(test_labels, test_ens_method_res)
                df.loc[df_i] = [pwc_methods[mi].__name__ + "_" + dtype, acc_method, nll_method]
                df_i += 1

    df.to_csv(os.path.join(exper_outputs_path, 'accuracies.csv'), index=False)


if __name__ == '__main__':
    ens_train_exp()
