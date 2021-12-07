import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import torch

from utils import load_networks_outputs, evaluate_networks, linear_pw_ens_train_save, evaluate_ens

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

    df_net = evaluate_networks(net_outputs)
    df_net.to_csv(os.path.join(exper_outputs_path, "net_metrics.csv"), index=False)

    df = pd.DataFrame(columns=("combining_method", "coupling_method", "train_size", "accuracy", "nll", "ece"))

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

            df_ens_ts = evaluate_ens(ens_outputs=test_ens_results, tar=net_outputs["test_labels"])
            df_ens_ts["train_size"] = real_t_size
            df = pd.concat([df, df_ens_ts], ignore_index=True)
            
        cur_t_size = int(quot * cur_t_size)

    df.to_csv(os.path.join(exper_outputs_path, 'accuracies.csv'), index=False)


if __name__ == '__main__':
    ens_train_exp()
