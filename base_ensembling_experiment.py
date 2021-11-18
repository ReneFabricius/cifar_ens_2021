import argparse
import os
import numpy as np
import shutil

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from weighted_ensembles.predictions_evaluation import compute_acc_topk, compute_nll
from weighted_ensembles.SimplePWCombine import m1, m2, bc, m2_iter

from utils import linear_pw_ens_train_save, load_networks_outputs

ENS_OUTPUTS_FOLDER = 'comb_outputs'
TRAIN_TRAIN = 'train_training'
VAL_TRAIN = 'val_training'


def ens_exp():
    coupling_methods = [m1, m2, m2_iter, bc]
    combining_methods = ["lda", "logreg", "logreg_no_interc"]

    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, required=True, help='experiment root folder')
    parser.add_argument('-repl', type=int, default=1, help='number of replications')
    parser.add_argument('-cifar', default=100, type=int, help='cifar type (10 or 100)')
    parser.add_argument('-device', type=str, default='cpu', help='device on which to execute the script')
    parser.add_argument('-save_R', dest='save_R', action='store_true',
                        help='Save R matrices entering into the coupling methods')
    parser.add_argument('-no_save_R', dest='save_R', action='store_false')
    parser.set_defaults(save_R=False)
    args = parser.parse_args()

    df_ens = pd.DataFrame(columns=('repli', 'train_set', 'combining_method', 'coupling_method', 'accuracy', 'nll'))
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

        net_outputs = load_networks_outputs(nn_outputs_path, comb_out_path, args.device)

        for i, net in enumerate(net_outputs["networks"]):
            acc = compute_acc_topk(net_outputs["test_labels"], net_outputs["test_outputs"][i], 1)
            nll = compute_nll(net_outputs["test_labels"], net_outputs["test_outputs"][i], penultimate=True)
            df_net.loc[df_net_i] = [repli, net, acc, nll]
            df_net_i += 1

        _, lda_train_idx = train_test_split(np.arange(net_outputs["train_labels"].shape[0]),
                                            test_size=net_outputs["val_labels"].shape[0],
                                            shuffle=True, stratify=net_outputs["train_labels"].cpu())

        np.save(os.path.join(tt_out_path, 'lda_train_idx.npy'), np.array(lda_train_idx))

        lda_train_outputs = net_outputs["train_outputs"][:, lda_train_idx, :]
        lda_train_labels = net_outputs["train_labels"][lda_train_idx]

        vt_test_ens_results = linear_pw_ens_train_save(net_outputs["val_outputs"], net_outputs["val_labels"],
                                                       net_outputs["test_outputs"],
                                                       torch.device(args.device),
                                                       vt_out_path, combining_methods=combining_methods,
                                                       coupling_methods=coupling_methods,
                                                       save_R_mats=args.save_R)

        tt_test_ens_results = linear_pw_ens_train_save(lda_train_outputs, lda_train_labels, net_outputs["test_outputs"],
                                                       torch.device(args.device),
                                                       tt_out_path, combining_methods=combining_methods,
                                                       coupling_methods=coupling_methods,
                                                       save_R_mats=args.save_R)

        for co_m in combining_methods:
            for cp_m in [cp.__name__ for cp in coupling_methods]:
                vt_ens_res = vt_test_ens_results.get(co_m, cp_m)
                acc_mi = compute_acc_topk(net_outputs["test_labels"], vt_ens_res, 1)
                nll_mi = compute_nll(net_outputs["test_labels"], vt_ens_res)
                df_ens.loc[df_ens_i] = [repli, 'vt', co_m, cp_m, acc_mi, nll_mi]
                df_ens_i += 1

        for co_m in combining_methods:
            for cp_m in [cp.__name__ for cp in coupling_methods]:
                tt_ens_res = tt_test_ens_results.get(co_m, cp_m)
                acc_mi = compute_acc_topk(net_outputs["test_labels"], tt_ens_res, 1)
                nll_mi = compute_nll(net_outputs["test_labels"], tt_ens_res)
                df_ens.loc[df_ens_i] = [repli, "tt", co_m, cp_m, acc_mi, nll_mi]
                df_ens_i += 1

    df_ens.to_csv(os.path.join(args.folder, 'ensemble_accuracies.csv'), index=False)
    df_net.to_csv(os.path.join(args.folder, "net_accuracies.csv"), index=False)


if __name__ == '__main__':
    ens_exp()
