import argparse
import os
import numpy as np
import shutil

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from weighted_ensembles.predictions_evaluation import compute_acc_topk, compute_nll
from weighted_ensembles.SimplePWCombine import m1, m2, bc, m2_iter

from utils import ens_train_save, load_networks_outputs, print_memory_statistics

ENS_OUTPUTS_FOLDER = 'comb_outputs'
TRAIN_TRAIN = 'train_training'
VAL_TRAIN = 'val_training'


def ens_exp():
    pwc_methods = [m1, m2, m2_iter, bc]

    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, required=True, help='experiment root folder')
    parser.add_argument('-fold_size', type=int, default=500, required=True, help='LDA training set size')
    parser.add_argument('-repl', type=int, default=1, help='number of replications')
    parser.add_argument('-cifar', default=100, type=int, help='cifar type (10 or 100)')
    parser.add_argument('-device', type=str, default='cpu', help='device on which to execute the script')
    parser.add_argument('-save_R', dest='save_R', action='store_true',
                        help='Save R matrices entering into the coupling methods')
    parser.add_argument('-no_save_R', dest='save_R', action='store_false')
    parser.set_defaults(save_R=False)
    args = parser.parse_args()

    torch_dev = torch.device(args.device)
    torch_dtp = torch.float32
    load_device = args.device if args.cifar == 10 else "cpu"
    torch_dev_load = torch.device(load_device)

    df_ens = pd.DataFrame(columns=('repli', 'fold',  'train_set', 'method', 'accuracy', 'nll'))
    df_ens_i = 0

    df_net = pd.DataFrame(columns=("repli", "network", "accuracy", "nll"))
    df_net_i = 0

    print_memory_statistics()

    for repli in range(args.repl):
        print("Processing replication {}".format(repli))
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

        print("Loading networks outputs")
        train_outputs, train_labels, val_outputs, val_labels, test_outputs, test_labels, networks = \
            load_networks_outputs(nn_outputs_path, comb_out_path, load_device)

        print("Evaluating networks")
        for i, net in enumerate(networks):
            acc = compute_acc_topk(test_labels, test_outputs[i], 1)
            nll = compute_nll(test_labels, test_outputs[i], penultimate=True)
            df_net.loc[df_net_i] = [repli, net, acc, nll]
            df_net_i += 1

        df_net.to_csv(os.path.join(args.folder, "net_accuracies.csv"), index=False)

        test_labels = test_labels.to(device=torch_dev)

        train_set_size = len(train_labels)
        val_set_size = len(val_labels)

        n_folds_train = train_set_size // args.fold_size
        n_folds_val = val_set_size // args.fold_size

        skf_train = StratifiedKFold(n_splits=n_folds_train, shuffle=True)
        skf_val = StratifiedKFold(n_splits=n_folds_val, shuffle=True)

        param_sets = [{"skf": skf_train, "train_set": "tt", "train_preds": train_outputs, "train_labs": train_labels,
                       "out_fold": tt_out_path},
                      {"skf": skf_val, "train_set": "vt", "train_preds": val_outputs, "train_labs": val_labels,
                       "out_fold": vt_out_path}]

        for par in param_sets:
            print("Processing train set {}".format(par["train_set"]))
            for fold_i, (_, fold_idxs) in enumerate(par["skf"].split(np.zeros(len(par["train_labs"])),
                                                               par["train_labs"].detach().cpu().clone().numpy())):

                np.save(os.path.join(par["out_fold"], '{}_lda_{}_idx.npy'.format(fold_i, par["train_set"])),
                        np.array(fold_idxs))
                fold_idxs = torch.from_numpy(fold_idxs).to(device=torch_dev_load, dtype=torch.long)
                fold_pred = par["train_preds"][:, fold_idxs, :].to(device=torch_dev, dtype=torch_dtp)
                fold_lab = par["train_labs"][fold_idxs].to(device=torch_dev, dtype=torch_dtp)

                print_memory_statistics()

                fold_ens_results = ens_train_save(predictors=fold_pred, targets=fold_lab,
                                                  test_predictors=test_outputs, device=torch_dev,
                                                  out_path=par["out_fold"],
                                                  pwc_methods=pwc_methods, prefix="fold_{}_".format(fold_i),
                                                  verbose=False, test_normality=False,
                                                  save_R_mats=args.save_R)

                for mi, ens_res in enumerate(fold_ens_results):
                    acc_mi = compute_acc_topk(test_labels, ens_res, 1)
                    nll_mi = compute_nll(test_labels, ens_res)
                    df_ens.loc[df_ens_i] = [repli, fold_i, par["train_set"], pwc_methods[mi].__name__, acc_mi, nll_mi]
                    df_ens_i += 1

                print_memory_statistics()

    df_ens.to_csv(os.path.join(args.folder, 'ensemble_accuracies.csv'), index=False)


if __name__ == '__main__':
    with torch.no_grad():
        ens_exp()