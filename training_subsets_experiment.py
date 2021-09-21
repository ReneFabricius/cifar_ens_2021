import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.append('D:\\skola\\1\\weighted_ensembles')
from my_codes.weighted_ensembles.predictions_evaluation import compute_acc_topk

import torch

from utils import load_networks_outputs, load_npy_arr, ens_train_save

TRAIN_OUTPUTS_FOLDER = 'exp_subsets_train_outputs'


def ens_train_exp():
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

    df_net = pd.DataFrame(columns=("network", "accuracy"))
    for i, net in enumerate(networks):
        acc = compute_acc_topk(test_labels, test_outputs[i], 1)
        df_net.loc[i] = [net, acc]

    df_net.to_csv(os.path.join(exper_outputs_path, "net_accuracies.csv"), index=False)

    n_samples = train_labels.shape[0]
    n_folds = n_samples // args.train_size
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    df = pd.DataFrame(columns=('m1_acc', 'm2_acc', 'bc_acc'))

    for fold_i, (_, lda_train_idxs) in enumerate(skf.split(np.zeros(n_samples),
                                                           train_labels.detach().cpu().clone().numpy())):
        print("Processing fold ", fold_i)
        np.save(os.path.join(exper_outputs_path, str(fold_i) + "_lda_train_idx.npy"), lda_train_idxs)
        lda_train_idxs = torch.from_numpy(lda_train_idxs).to(device=torch.device(args.device), dtype=torch.long)
        lda_train_pred = train_outputs[:, lda_train_idxs, :]
        lda_train_lab = train_labels[lda_train_idxs]
        test_ens_m1, test_ens_m2, test_ens_bc = ens_train_save(lda_train_pred, lda_train_lab, test_outputs,
                                                               torch.device(args.device), exper_outputs_path,
                                                               str(fold_i) + "_")

        acc_m1 = compute_acc_topk(test_labels, test_ens_m1, 1)
        acc_m2 = compute_acc_topk(test_labels, test_ens_m2, 1)
        acc_bc = compute_acc_topk(test_labels, test_ens_bc, 1)

        df.loc[fold_i] = [acc_m1, acc_m2, acc_bc]

    df.to_csv(os.path.join(exper_outputs_path, 'accuracies.csv'), index=False)


if __name__ == '__main__':
    ens_train_exp()
