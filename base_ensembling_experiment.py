import argparse
import os
import numpy as np
import sys

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.append('D:\\skola\\1\\weighted_ensembles')
from my_codes.weighted_ensembles.predictions_evaluation import compute_acc_topk

from utils import ens_train_save, load_networks_outputs

ENS_OUTPUTS_FOLDER = 'comb_outputs'
TRAIN_TRAIN = 'train_training'
VAL_TRAIN = 'val_training'


def ens_exp():
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, required=True, help='experiment root folder')
    parser.add_argument('-repl', type=int, default=1, help='number of replications')
    parser.add_argument('-cifar', default=100, type=int, help='cifar type (10 or 100)')
    parser.add_argument('-device', type=str, default='cpu', help='device on which to execute the script')
    args = parser.parse_args()

    df = pd.DataFrame(columns=('vt_acc_m1', 'vt_acc_m2', 'vt_acc_bc', 'tt_acc_m1', 'tt_acc_m2', 'tt_acc_bc'))

    for repli in range(args.repl):
        repli_path = os.path.join(args.folder, str(repli))
        comb_out_path = os.path.join(repli_path, ENS_OUTPUTS_FOLDER)
        tt_out_path = os.path.join(comb_out_path, TRAIN_TRAIN)
        vt_out_path = os.path.join(comb_out_path, VAL_TRAIN)
        nn_outputs_path = os.path.join(repli_path, 'outputs')

        if not os.path.exists(comb_out_path):
            os.mkdir(comb_out_path)
        if not os.path.exists(tt_out_path):
            os.mkdir(tt_out_path)
        if not os.path.exists(vt_out_path):
            os.mkdir(vt_out_path)

        train_outputs, train_labels, val_outputs, val_labels, test_outputs, test_labels = \
            load_networks_outputs(nn_outputs_path, comb_out_path, args.device)

        _, lda_train_idx = train_test_split(np.arange(train_labels.shape[0]), test_size=val_labels.shape[0],
                                            shuffle=True, stratify=train_labels)

        np.save(os.path.join(tt_out_path, 'lda_train_idx.npy'), np.array(lda_train_idx))

        lda_train_outputs = train_outputs[:, lda_train_idx, :]
        lda_train_labels = train_labels[lda_train_idx]

        vt_test_ens_m1, vt_test_ens_m2, vt_test_ens_bc = ens_train_save(lda_train_outputs, lda_train_labels,
                                                                        test_outputs, torch.device(args.device),
                                                                        tt_out_path)

        tt_test_ens_m1, tt_test_ens_m2, tt_test_ens_bc = ens_train_save(val_outputs, val_labels, test_outputs,
                                                                        torch.device(args.device), vt_out_path)

        vt_acc_m2 = compute_acc_topk(test_labels, vt_test_ens_m2, 1)
        vt_acc_bc = compute_acc_topk(test_labels, vt_test_ens_bc, 1)
        vt_acc_m1 = compute_acc_topk(test_labels, vt_test_ens_m1, 1)

        tt_acc_m2 = compute_acc_topk(test_labels, tt_test_ens_m2, 1)
        tt_acc_bc = compute_acc_topk(test_labels, tt_test_ens_bc, 1)
        tt_acc_m1 = compute_acc_topk(test_labels, tt_test_ens_m1, 1)

        df.loc[repli] = [vt_acc_m1, vt_acc_m2, vt_acc_bc, tt_acc_m1, tt_acc_m2, tt_acc_bc]

    df.to_csv(os.path.join(args.folder, 'ensemble_accuracies.csv'), index=False)


if __name__ == '__main__':
    ens_exp()