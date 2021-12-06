import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from weensembles.predictions_evaluation import compute_acc_topk, compute_nll

import torch

from utils import load_networks_outputs, load_npy_arr, linear_pw_ens_train_save

EXP_OUTPUTS_FOLDER = 'exp_subsets_sizes_train_outputs'


def ens_train_exp():
    combining_methods = ["lda", "logreg", "logreg_no_interc"]
    coupling_methods = ["m1", "m2", "m2_iter", "bc"]

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

    df = pd.DataFrame(columns=("combining_method", "coupling_method", "train_size", "accuracy", "nll"))
    df_i = 0

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

            test_ens_results = linear_pw_ens_train_save(predictors=comb_train_pred, targets=comb_train_lab,
                                                        test_predictors=net_outputs["test_outputs"],
                                                        device=torch.device(args.device), out_path=exper_outputs_path,
                                                        combining_methods=combining_methods,
                                                        coupling_methods=coupling_methods,
                                                        prefix="size_{}_repl_{}_".format(real_t_size, fold_i))

            for co_m in combining_methods:
                for cp_m in coupling_methods:
                    test_ens_res = test_ens_results.get(co_m, cp_m)
                    acc_method = compute_acc_topk(net_outputs["test_labels"], test_ens_res, 1)
                    nll_method = compute_nll(net_outputs["test_labels"], test_ens_res)
                    df.loc[df_i] = [co_m, cp_m, real_t_size, acc_method, nll_method]
                    df_i += 1

        cur_t_size = int(quot * cur_t_size)

    df.to_csv(os.path.join(exper_outputs_path, 'accuracies.csv'), index=False)


if __name__ == '__main__':
    ens_train_exp()
