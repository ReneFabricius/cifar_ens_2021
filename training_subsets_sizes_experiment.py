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

EXP_OUTPUTS_FOLDER = 'exp_subsets_sizes_train_outputs'


def ens_train_exp():
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, required=True, help='replication_folder')
    parser.add_argument('-repl_num', type=int, default=30, help='max number of replications for each train size')
    parser.add_argument('-device', type=str, default='cpu', help='device on which to execute the script')
    args = parser.parse_args()

    exper_outputs_path = os.path.join(args.folder, EXP_OUTPUTS_FOLDER)
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

    df = pd.DataFrame(columns=("method", "train_size", "accuracy"))
    df_i = 0

    n_samples = train_labels.shape[0]
    min_t_size = 80
    max_t_size = 4950
    quot = 1.4
    cur_t_size = min_t_size
    while cur_t_size < max_t_size:
        print("Processing lda train set size {}".format(cur_t_size))
        n_folds = n_samples // cur_t_size

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

        for fold_i, (_, lda_train_idxs) in enumerate(skf.split(np.zeros(n_samples),
                                                               train_labels.detach().cpu().clone().numpy())):
            if fold_i >= args.repl_num:
                break

            real_t_size = len(lda_train_idxs)

            print("Processing fold {}".format(fold_i))
            print("Real train size {}".format(real_t_size))
            np.save(os.path.join(exper_outputs_path, "lda_train_idx_size_{}_repl_{}.npy".format(cur_t_size, fold_i)),
                    lda_train_idxs)
            lda_train_idxs = torch.from_numpy(lda_train_idxs).to(device=torch.device(args.device), dtype=torch.long)
            lda_train_pred = train_outputs[:, lda_train_idxs, :]
            lda_train_lab = train_labels[lda_train_idxs]
            test_ens_m1, test_ens_m2, test_ens_bc = ens_train_save(lda_train_pred, lda_train_lab, test_outputs,
                                                                   torch.device(args.device), exper_outputs_path,
                                                                   "size_{}_repl_{}_".format(real_t_size, fold_i))

            acc_m1 = compute_acc_topk(test_labels, test_ens_m1, 1)
            acc_m2 = compute_acc_topk(test_labels, test_ens_m2, 1)
            acc_bc = compute_acc_topk(test_labels, test_ens_bc, 1)

            df.loc[df_i] = ["m1", real_t_size, acc_m1]
            df_i += 1
            df.loc[df_i] = ["m2", real_t_size, acc_m2]
            df_i += 1
            df.loc[df_i] = ["bc", real_t_size, acc_bc]
            df_i += 1

        cur_t_size = int(quot * cur_t_size)

    df.to_csv(os.path.join(exper_outputs_path, 'accuracies.csv'), index=False)


if __name__ == '__main__':
    ens_train_exp()
