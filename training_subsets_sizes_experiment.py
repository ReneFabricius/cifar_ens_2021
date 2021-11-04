import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from weighted_ensembles.predictions_evaluation import compute_acc_topk, compute_nll
from weighted_ensembles.SimplePWCombine import m1, m2, bc, m2_iter

import torch

from utils import load_networks_outputs, load_npy_arr, ens_train_save

EXP_OUTPUTS_FOLDER = 'exp_subsets_sizes_train_outputs'


def ens_train_exp():
    pwc_methods = [m1, m2, m2_iter, bc]

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

    df_net = pd.DataFrame(columns=("network", "accuracy", "nll"))
    for i, net in enumerate(net_outputs["networks"]):
        acc = compute_acc_topk(net_outputs["test_labels"], net_outputs["test_outputs"][i], 1)
        nll = compute_nll(net_outputs["test_labels"], net_outputs["test_outputs"][i], penultimate=True)
        df_net.loc[i] = [net, acc, nll]

    df_net.to_csv(os.path.join(exper_outputs_path, "net_accuracies.csv"), index=False)

    df = pd.DataFrame(columns=("method", "train_size", "accuracy", "nll"))
    df_i = 0

    n_samples = net_outputs["train_labels"].shape[0]
    min_t_size = 80
    max_t_size = 4950
    quot = 1.4
    cur_t_size = min_t_size
    while cur_t_size < max_t_size:
        print("Processing lda train set size {}".format(cur_t_size))
        n_folds = n_samples // cur_t_size

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

        for fold_i, (_, lda_train_idxs) in enumerate(skf.split(np.zeros(n_samples),
                                                        net_outputs["train_labels"].detach().cpu().clone().numpy())):
            if fold_i >= args.repl_num:
                break

            real_t_size = len(lda_train_idxs)

            print("Processing fold {}".format(fold_i))
            print("Real train size {}".format(real_t_size))
            np.save(os.path.join(exper_outputs_path, "lda_train_idx_size_{}_repl_{}.npy".format(cur_t_size, fold_i)),
                    lda_train_idxs)
            lda_train_idxs = torch.from_numpy(lda_train_idxs).to(device=torch.device(args.device), dtype=torch.long)
            lda_train_pred = net_outputs["train_outputs"][:, lda_train_idxs, :]
            lda_train_lab = net_outputs["train_labels"][lda_train_idxs]
            test_ens_results = ens_train_save(lda_train_pred, lda_train_lab, net_outputs["test_outputs"],
                                                torch.device(args.device), exper_outputs_path,
                                                pwc_methods, "size_{}_repl_{}_".format(real_t_size, fold_i))

            for mi, test_ens_res in enumerate(test_ens_results):
                acc_method = compute_acc_topk(net_outputs["test_labels"], test_ens_res, 1)
                nll_method = compute_nll(net_outputs["test_labels"], test_ens_res)
                df.loc[df_i] = [pwc_methods[mi].__name__, real_t_size, acc_method, nll_method]
                df_i += 1

        cur_t_size = int(quot * cur_t_size)

    df.to_csv(os.path.join(exper_outputs_path, 'accuracies.csv'), index=False)


if __name__ == '__main__':
    ens_train_exp()
