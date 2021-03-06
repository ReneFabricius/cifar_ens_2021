import argparse
import os
import numpy as np
import shutil

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from utils.utils import linear_pw_ens_train_save, load_networks_outputs, evaluate_networks, evaluate_ens

ENS_OUTPUTS_FOLDER = 'comb_outputs'
TRAIN_TRAIN = 'train_training'
VAL_TRAIN = 'val_training'


def ens_exp():
    coupling_methods = ["m1", "m2", "m2_iter", "bc"]
    combining_methods = ["lda", "logreg", "logreg_no_interc"]

    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, required=True, help='experiment root folder')
    parser.add_argument('-repl', type=int, default=1, help='number of replications')
    parser.add_argument('-cifar', default=100, type=int, help='cifar type (10 or 100)')
    parser.add_argument('-device', type=str, default='cpu', help='device on which to execute the script')
    args = parser.parse_args()

    df_ens = pd.DataFrame(columns=('repli', 'train_set', 'combining_method', 'coupling_method', 'accuracy', 'nll', 'ece'))
    df_net = pd.DataFrame(columns=("repli", "network", "accuracy", "nll", "ece"))

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

        repli_df_net = evaluate_networks(net_outputs)
        repli_df_net["repli"] = repli
        df_net = pd.concat([df_net, repli_df_net], ignore_index=True)
        
        _, co_m_train_idx = train_test_split(np.arange(net_outputs["train_labels"].shape[0]),
                                            test_size=net_outputs["val_labels"].shape[0],
                                            shuffle=True, stratify=net_outputs["train_labels"].cpu())

        np.save(os.path.join(tt_out_path, 'lda_train_idx.npy'), np.array(co_m_train_idx))

        co_m_train_outputs = net_outputs["train_outputs"][:, co_m_train_idx, :]
        co_m_train_labels = net_outputs["train_labels"][co_m_train_idx]

        vt_test_ens_results = linear_pw_ens_train_save(predictors=net_outputs["val_outputs"], targets=net_outputs["val_labels"],
                                                       test_predictors=net_outputs["test_outputs"],
                                                       device=torch.device(args.device),
                                                       out_path=vt_out_path, combining_methods=combining_methods,
                                                       coupling_methods=coupling_methods)

        tt_test_ens_results = linear_pw_ens_train_save(predictors=co_m_train_outputs, targets=co_m_train_labels, 
                                                       test_predictors=net_outputs["test_outputs"],
                                                       device=torch.device(args.device),
                                                       out_path=tt_out_path, combining_methods=combining_methods,
                                                       coupling_methods=coupling_methods)

        df_ens_repli_vt = evaluate_ens(ens_outputs=vt_test_ens_results, tar=net_outputs["test_labels"])
        df_ens_repli_tt = evaluate_ens(ens_outputs=tt_test_ens_results, tar=net_outputs["test_labels"])
        df_ens_repli_vt["repli"] = repli
        df_ens_repli_vt["train_set"] = "vt"
        df_ens_repli_tt["repli"] = repli
        df_ens_repli_tt["train_set"] = "tt"
        df_ens = pd.concat([df_ens, df_ens_repli_vt, df_ens_repli_tt], ignore_index=True)

    df_ens.to_csv(os.path.join(args.folder, 'ensemble_metrics.csv'), index=False)
    df_net.to_csv(os.path.join(args.folder, "net_metrics.csv"), index=False)


if __name__ == '__main__':
    ens_exp()
