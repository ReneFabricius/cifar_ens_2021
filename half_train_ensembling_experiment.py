import argparse
import os
import numpy as np
import shutil

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from weensembles.predictions_evaluation import compute_acc_topk, compute_nll
from weensembles.CouplingMethods import m1, m2, bc, m2_iter
from weensembles.CombiningMethods import lda, logreg, logreg_no_interc, logreg_sweep_C, logreg_no_interc_sweep_C

from utils import linear_pw_ens_train_save, load_networks_outputs, print_memory_statistics

ENS_OUTPUTS_FOLDER = 'comb_outputs'
TRAIN_TRAIN = 'train_training'
VAL_TRAIN = 'val_training'


def ens_exp():
    coupling_methods = [m1, m2, m2_iter, bc]
    combining_methods = [lda, logreg, logreg_no_interc]

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

    df_ens = pd.DataFrame(columns=('repli', 'fold',  'train_set', 'combining_method', 'coupling_method',
                                   'accuracy', 'nll'))
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
        net_outputs = load_networks_outputs(nn_outputs_path, comb_out_path, load_device)

        print("Evaluating networks")
        for i, net in enumerate(net_outputs["networks"]):
            acc = compute_acc_topk(net_outputs["test_labels"], net_outputs["test_outputs"][i], 1)
            nll = compute_nll(net_outputs["test_labels"], net_outputs["test_outputs"][i], penultimate=True)
            df_net.loc[df_net_i] = [repli, net, acc, nll]
            df_net_i += 1

        df_net.to_csv(os.path.join(args.folder, "net_accuracies.csv"), index=False)

        test_labels = net_outputs["test_labels"].to(device=torch_dev)

        train_set_size = len(net_outputs["train_labels"])
        val_set_size = len(net_outputs["val_labels"])

        n_folds_train = train_set_size // args.fold_size
        n_folds_val = val_set_size // args.fold_size

        skf_train = StratifiedKFold(n_splits=n_folds_train, shuffle=True)
        skf_val = StratifiedKFold(n_splits=n_folds_val, shuffle=True)

        param_sets = [{"skf": skf_train, "train_set": "tt", "train_preds": net_outputs["train_outputs"],
                       "train_labs": net_outputs["train_labels"],
                       "out_fold": tt_out_path},
                      {"skf": skf_val, "train_set": "vt", "train_preds": net_outputs["val_outputs"],
                       "train_labs": net_outputs["val_labels"],
                       "out_fold": vt_out_path}]

        for par in param_sets:
            print("Processing train set {}".format(par["train_set"]))
            for fold_i, (_, fold_idxs) in enumerate(par["skf"].split(np.zeros(len(par["train_labs"])),
                                                               par["train_labs"].detach().cpu().clone().numpy())):

                print("Processing fold {}".format(fold_i))
                print_memory_statistics(list_tensors=True)

                np.save(os.path.join(par["out_fold"], '{}_lin_comb_{}_idx.npy'.format(fold_i, par["train_set"])),
                        np.array(fold_idxs))
                fold_idxs = torch.from_numpy(fold_idxs).to(device=torch_dev_load, dtype=torch.long)
                fold_pred = par["train_preds"][:, fold_idxs, :].to(device=torch_dev, dtype=torch_dtp)
                fold_lab = par["train_labs"][fold_idxs].to(device=torch_dev, dtype=torch_dtp)

                print("Memory before ensembling")
                print_memory_statistics()

                fold_ens_results = linear_pw_ens_train_save(predictors=fold_pred, targets=fold_lab,
                                                            test_predictors=net_outputs["test_outputs"], device=torch_dev,
                                                            out_path=par["out_fold"],
                                                            combining_methods=combining_methods,
                                                            coupling_methods=coupling_methods, prefix="fold_{}_".format(fold_i),
                                                            verbose=False, test_normality=False,
                                                            save_R_mats=args.save_R)

                for co_m in [co.__name__ for co in combining_methods]:
                    for cp_m in [cp.__name__ for cp in coupling_methods]:
                        ens_res = fold_ens_results.get(co_m, cp_m)
                        acc_mi = compute_acc_topk(test_labels, ens_res, 1)
                        nll_mi = compute_nll(test_labels, ens_res)
                        df_ens.loc[df_ens_i] = [repli, fold_i, par["train_set"], co_m, cp_m, acc_mi, nll_mi]
                        df_ens_i += 1

                del fold_ens_results
                print("Memory after saving results")
                print_memory_statistics()

    df_ens.to_csv(os.path.join(args.folder, 'ensemble_accuracies.csv'), index=False)


if __name__ == '__main__':
    with torch.no_grad():
        ens_exp()