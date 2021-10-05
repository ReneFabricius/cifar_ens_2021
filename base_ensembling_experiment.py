import argparse
import os
import numpy as np
import sys
import shutil

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.append('D:\\skola\\1\\weighted_ensembles')
from my_codes.weighted_ensembles.predictions_evaluation import compute_acc_topk, compute_nll
from my_codes.weighted_ensembles.SimplePWCombine import m1, m2, bc, m2_iter

from utils import ens_train_save, load_networks_outputs

ENS_OUTPUTS_FOLDER = 'comb_outputs'
TRAIN_TRAIN = 'train_training'
VAL_TRAIN = 'val_training'


def ens_exp():
    pwc_methods = [m1, m2, m2_iter, bc]

    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, required=True, help='experiment root folder')
    parser.add_argument('-repl', type=int, default=1, help='number of replications')
    parser.add_argument('-cifar', default=100, type=int, help='cifar type (10 or 100)')
    parser.add_argument('-device', type=str, default='cpu', help='device on which to execute the script')
    args = parser.parse_args()

    df_ens = pd.DataFrame(columns=('repli', 'train_set', 'method', 'accuracy', 'nll'))
    df_ens_i = 0

    df_net = pd.DataFrame(columns=("repli", "network", "accuracy", "nll"))
    df_net_i = 0

    for repli in range(args.repl):
        repli_path = os.path.join(args.folder, str(repli))
        comb_out_path = os.path.join(repli_path, ENS_OUTPUTS_FOLDER)
        tt_out_path = os.path.join(comb_out_path, TRAIN_TRAIN)
        vt_out_path = os.path.join(comb_out_path, VAL_TRAIN)
        nn_outputs_path = os.path.join(repli_path, 'outputs')

        if os.path.exists(comb_out_path):
            shutil.rmtree(comb_out_path)
        os.mkdir(comb_out_path)
        if os.path.exists(tt_out_path):
            shutil.rmtree(tt_out_path)
        os.mkdir(tt_out_path)
        if os.path.exists(vt_out_path):
            shutil.rmtree(vt_out_path)
        os.mkdir(vt_out_path)

        train_outputs, train_labels, val_outputs, val_labels, test_outputs, test_labels, networks = \
            load_networks_outputs(nn_outputs_path, comb_out_path, args.device)

        for i, net in enumerate(networks):
            acc = compute_acc_topk(test_labels, test_outputs[i], 1)
            nll = compute_nll(test_labels, test_outputs[i], penultimate=True)
            df_net.loc[df_net_i] = [repli, net, acc, nll]
            df_net_i += 1

        _, lda_train_idx = train_test_split(np.arange(train_labels.shape[0]), test_size=val_labels.shape[0],
                                            shuffle=True, stratify=train_labels.cpu())

        np.save(os.path.join(tt_out_path, 'lda_train_idx.npy'), np.array(lda_train_idx))

        lda_train_outputs = train_outputs[:, lda_train_idx, :]
        lda_train_labels = train_labels[lda_train_idx]

        vt_test_ens_results = ens_train_save(val_outputs, val_labels, test_outputs,
                                             torch.device(args.device),
                                             vt_out_path, pwc_methods=pwc_methods)

        tt_test_ens_results = ens_train_save(lda_train_outputs, lda_train_labels, test_outputs,
                                             torch.device(args.device),
                                             tt_out_path, pwc_methods=pwc_methods)

        for mi, vt_ens_res in enumerate(vt_test_ens_results):
            acc_mi = compute_acc_topk(test_labels, vt_ens_res, 1)
            nll_mi = compute_nll(test_labels, vt_ens_res)
            df_ens.loc[df_ens_i] = [repli, 'vt', pwc_methods[mi].__name__, acc_mi, nll_mi]
            df_ens_i += 1

        for mi, tt_ens_res in enumerate(tt_test_ens_results):
            acc_mi = compute_acc_topk(test_labels, tt_ens_res, 1)
            nll_mi = compute_nll(test_labels, tt_ens_res)
            df_ens.loc[df_ens_i] = [repli, "tt", pwc_methods[mi].__name__, acc_mi, nll_mi]
            df_ens_i += 1

    df_ens.to_csv(os.path.join(args.folder, 'ensemble_accuracies.csv'), index=False)
    df_net.to_csv(os.path.join(args.folder, "net_accuracies.csv"), index=False)


if __name__ == '__main__':
    ens_exp()
