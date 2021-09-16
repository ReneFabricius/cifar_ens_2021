import argparse
import os
import sys
import numpy as np

import torch
from sklearn.model_selection import train_test_split
sys.path.append('D:\\skola\\1\\weighted_ensembles\\my_codes\\weighted_ensembles')

from WeightedLDAEnsemble import WeightedLDAEnsemble
from SimplePWCombine import m1, m2, bc


ENS_OUTPUTS_FOLDER = 'comb_outputs'
TRAIN_TRAIN = 'train_training'
VAL_TRAIN = 'val_training'


def ens_exp():
    parser = argparse.ArgumentParser
    parser.add_argument('-folder', type=str, required=True, help='experiment root folder')
    parser.add_argument('-repl', type=int, default=1, help='number of replications')
    parser.add_argument('-cifar', default=100, type=int, help='cifar type (10 or 100)')
    parser.add_argument('-device', type=str, default='cpu', help='device on which to execute the script')
    args = parser.parse_args()

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

        networks = os.listdir(nn_outputs_path)

        networks_order = open(os.path.join(comb_out_path, 'networks_order.txt'))
        for net in networks:
            networks_order.write(net + "\n")
        networks_order.close()

        test_outputs = []
        for net in networks:
            test_outputs.append(load_npy_arr(os.path.join(nn_outputs_path, net, 'test_outputs.npy'), args.device).
                                unsqueeze(0))
        test_outputs = torch.cat(test_outputs, 0)
        test_labels = load_npy_arr(os.path.join(nn_outputs_path, networks[0], 'test_labels.npy'))

        train_outputs = []
        for net in networks:
            train_outputs.append(load_npy_arr(os.path.join(nn_outputs_path, net, 'train_outputs.npy'), args.device).
                                unsqueeze(0))
        train_outputs = torch.cat(train_outputs, 0)
        train_labels = load_npy_arr(os.path.join(nn_outputs_path, networks[0], 'train_labels.npy'))

        val_outputs = []
        for net in networks:
            val_outputs.append(load_npy_arr(os.path.join(nn_outputs_path, net, 'val_outputs.npy'), args.device).
                                 unsqueeze(0))
        val_outputs = torch.cat(val_outputs, 0)
        val_labels = load_npy_arr(os.path.join(nn_outputs_path, networks[0], 'val_labels.npy'))

        _, lda_train_idx = train_test_split(np.arange(train_labels.shape[0]), test_size=val_labels.shape[0],
                                            shuffle=True, stratify=train_labels)

        np.save(os.path.join(tt_out_path, 'lda_train_idx.npy'), np.array(lda_train_idx))

        lda_train_outputs = train_outputs[lda_train_idx]
        lda_train_labels = train_labels[lda_train_idx]

        ens_train_save(lda_train_outputs, lda_train_labels, test_outputs, torch.device(args.device), tt_out_path)

        ens_train_save(val_outputs, val_labels, test_outputs, torch.device(args.device), vt_out_path)


def load_npy_arr(file, device):
    return torch.from_numpy(np.load(file)).to(torch.device(device))


def ens_train_save(predictors, targets, test_predictors, device, out_path):
    ens = WeightedLDAEnsemble(predictors.shape[0], predictors.shape[1], device)
    ens.fit_penultimate(predictors, targets, verbose=True, test_normality=True)

    ens.save_coefs_csv(os.path.join(out_path, 'lda_coefs.csv'))
    ens.save_pvals(os.path.join(out_path, 'p_values.npy'))
    ens.save(os.path.join(out_path, 'model'))

    ens_test_out_m1 = ens.predict(test_predictors, m1)
    np.save(os.path.join(out_path, 'ens_test_outputs_m1.npy'), ens_test_out_m1.detach().cpu().clone().numpy())

    ens_test_out_m2 = ens.predict(test_predictors, m2)
    np.save(os.path.join(out_path, 'ens_test_outputs_m2.npy'), ens_test_out_m2.detach().cpu().clone().numpy())

    ens_test_out_bc = ens.predict(test_predictors, bc)
    np.save(os.path.join(out_path, 'ens_test_outputs_bc.npy'), ens_test_out_bc.detach().cpu().clone().numpy())


if __name__ == '__main__':
    ens_exp()