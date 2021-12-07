import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import torch

from utils import load_networks_outputs, linear_pw_ens_train_save, evaluate_ens, evaluate_networks

TRAIN_OUTPUTS_FOLDER = 'exp_subsets_train_outputs'


def ens_train_exp():
    coupling_methods = ["m1", "m2", "m2_iter", "bc"]
    combining_methods = ["lda", "logreg", "logreg_no_interc"]

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
    net_outputs = load_networks_outputs(networks_outputs_folder, exper_outputs_path, args.device)

    df_net = evaluate_networks(net_outputs) 
    df_net.to_csv(os.path.join(exper_outputs_path, "net_metrics.csv"), index=False)

    n_samples = net_outputs["train_labels"].shape[0]
    n_folds = n_samples // args.train_size
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    df = pd.DataFrame(columns=("combining_method", "coupling_method", "precision", "accuracy", "nll", "ece"))

    for fold_i, (_, lda_train_idxs) in enumerate(skf.split(np.zeros(n_samples),
                                                           net_outputs["train_labels"].detach().cpu().clone().numpy())):
        print("Processing fold ", fold_i)
        np.save(os.path.join(exper_outputs_path, str(fold_i) + "_lda_train_idx.npy"), lda_train_idxs)
        lda_train_idxs = torch.from_numpy(lda_train_idxs).to(device=torch.device(args.device), dtype=torch.long)
        lda_train_pred = net_outputs["train_outputs"][:, lda_train_idxs, :]
        lda_train_lab = net_outputs["train_labels"][lda_train_idxs]

        data_type = ["float", "double"]
        for dtype in data_type:
            test_ens_results = linear_pw_ens_train_save(predictors=lda_train_pred, targets=lda_train_lab,
                                                        test_predictors=net_outputs["test_outputs"],
                                                        device=torch.device(args.device), out_path=exper_outputs_path,
                                                        combining_methods=combining_methods,
                                                        coupling_methods=coupling_methods, prefix=(str(fold_i) + "_"),
                                                        double_accuracy=(dtype == "double"))

            ens_df_dtp = evaluate_ens(ens_outputs=test_ens_results, tar=net_outputs["test_labels"])
            ens_df_dtp["precision"] = dtype
            df = pd.concat([df, ens_df_dtp], ignore_index=True)
            
    df.to_csv(os.path.join(exper_outputs_path, 'accuracies.csv'), index=False)


if __name__ == '__main__':
    ens_train_exp()
