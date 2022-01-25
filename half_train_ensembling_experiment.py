import argparse
from cgi import test
import os
import numpy as np
import shutil

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from utils import evaluate_ens, evaluate_networks, linear_pw_ens_train_save, load_networks_outputs, print_memory_statistics

ENS_OUTPUTS_FOLDER = 'comb_outputs'
TRAIN_TRAIN = 'train_training'
VAL_TRAIN = 'val_training'


def ens_exp():
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, required=True, help='experiment root folder')
    parser.add_argument('-fold_size', type=int, default=500, required=True, help='Combiner training set size')
    parser.add_argument('-repl', type=int, default=1, help='number of replications')
    parser.add_argument('-cifar', default=100, type=int, help='cifar type (10 or 100)')
    parser.add_argument('-device', type=str, default='cpu', help='device on which to execute the script')
    parser.add_argument('-coupling_methods', nargs='+', default=["m2"], help='Coupling methods to use')
    parser.add_argument('-load_existing_models', type=str, choices=["no", "recalculate"], default="no", help="Loading of present models. If no - all computations are performed again, \
                        if recalculate - existing models are loaded, but metrics are calculated again.")
    parser.add_argument('-combining_methods', nargs='+', default=["average"], help="Combining methods to use")
    parser.add_argument('-verbose', type=int, default=0, help="Verbosity level.")
    args = parser.parse_args()

    torch_dev = torch.device(args.device)
    torch_dtp = torch.float32
    load_device = args.device if args.cifar == 10 else "cpu"
    torch_dev_load = torch.device(load_device)

    df_ens = pd.DataFrame(columns=('repli', 'fold',  'train_set', 'combining_method', 'coupling_method',
                                   'accuracy', 'nll', 'ece'))

    df_net = pd.DataFrame(columns=("repli", "network", "accuracy", "nll", "ece"))

    print_memory_statistics()

    for repli in range(args.repl):
        print("Processing replication {}".format(repli))
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

        print("Loading networks outputs")
        net_outputs = load_networks_outputs(nn_outputs_path, comb_out_path, load_device)

        print("Evaluating networks")
        net_df_repli = evaluate_networks(net_outputs)
        net_df_repli["repli"] = repli
        df_net = pd.concat([df_net, net_df_repli], ignore_index=True)
        df_net.to_csv(os.path.join(args.folder, "net_metrics.csv"), index=False)

        test_outputs = net_outputs["test_outputs"].to(device=torch_dev)
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
                                                               par["train_labs"].detach().cpu().numpy())):
                print("Processing fold {}".format(fold_i))
                np.save(os.path.join(par["out_fold"], '{}_lin_comb_{}_idx.npy'.format(fold_i, par["train_set"])),
                        np.array(fold_idxs))
                fold_idxs = torch.from_numpy(fold_idxs).to(device=torch_dev_load, dtype=torch.long)
                fold_pred = par["train_preds"][:, fold_idxs, :].to(device=torch_dev, dtype=torch_dtp)
                fold_lab = par["train_labs"][fold_idxs].to(device=torch_dev)

                fold_ens_results = linear_pw_ens_train_save(predictors=fold_pred, targets=fold_lab,
                                                            test_predictors=test_outputs, device=torch_dev,
                                                            out_path=par["out_fold"],
                                                            combining_methods=args.combining_methods,
                                                            coupling_methods=args.coupling_methods, prefix="fold_{}_".format(fold_i),
                                                            verbose=args.verbose,
                                                            load_existing_models=args.load_existing_models)

                ens_df_fold = evaluate_ens(ens_outputs=fold_ens_results, tar=test_labels)
                ens_df_fold["repli"] = repli
                ens_df_fold["fold"] = fold_i
                ens_df_fold["train_set"] = par["train_set"]
                df_ens = pd.concat([df_ens, ens_df_fold], ignore_index=True)
                
                del fold_ens_results

            df_ens.to_csv(os.path.join(args.folder, 'ensemble_metrics.csv'), index=False)


if __name__ == '__main__':
    ens_exp()
    