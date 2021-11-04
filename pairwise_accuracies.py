import torch
import argparse
import pandas as pd
import os
import re
from utils import load_networks_outputs, compute_pairwise_accuracies, load_npy_arr


def pairwise_accuracies():
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, required=True, help='experiment root folder')
    parser.add_argument('-repl', type=int, default=1, help='number of replications')
    parser.add_argument('-folds', type=int, default=1, help='number of folds')
    parser.add_argument('-cifar', default=100, type=int, help='cifar type (10 or 100)')
    parser.add_argument('-device', type=str, default='cpu', help='device on which to execute the script')
    args = parser.parse_args()

    load_device = args.device if args.cifar == 10 else "cpu"
    train_types = ["train_training", "val_training"]

    outputs_match = "ens_test_R_"
    if args.folds == 1:
        pattern = "^" + outputs_match + "(.*).npy$"
    else:
        pattern = "^fold_\\d+_" + outputs_match + "(.*).npy$"

    df_ens = pd.DataFrame(columns=('repli', 'fold', 'train_set', 'precision', 'class1', 'class2', 'accuracy'))

    df_net = pd.DataFrame(columns=("repli", "network", 'class1', 'class2', "accuracy"))

    for repli in range(args.repl):
        print("Processing repli {}".format(repli))

        repli_path = os.path.join(args.folder, str(repli))
        nn_outputs_path = os.path.join(repli_path, "outputs")
        ens_outputs_path = os.path.join(repli_path, "comb_outputs")

        net_outputs = load_networks_outputs(nn_outputs_path, device="cuda")
        labs = net_outputs["test_labels"]
        for ni, net in enumerate(net_outputs["networks"]):
            cur_n_out = net_outputs["test_outputs"][ni, :, :]
            net_df = compute_pairwise_accuracies(cur_n_out, labs)
            net_df["repli"] = repli
            net_df["network"] = net
            df_net = pd.concat([df_net, net_df])

        files = os.listdir(os.path.join(ens_outputs_path, train_types[0]))
        ptrn = re.compile(pattern)
        precisions = list(set([re.search(ptrn, f).group(1) for f in files if re.search(ptrn, f) is not None]))

        for tr_tp in train_types:
            print("Processing train type {}".format(tr_tp))

            for prec in precisions:
                if args.folds == 1:
                    file_name = outputs_match + prec + ".npy"
                    file_path = os.path.join(ens_outputs_path, tr_tp, file_name)
                    R_mat = load_npy_arr(file_path, args.device)
                    df = compute_pairwise_accuracies(R_mat, labs)
                    df["repli"] = repli
                    df["fold"] = 0
                    df["train_set"] = tr_tp
                    df["precision"] = prec
                    df_ens = pd.concat([df_ens, df])

                else:
                    for foldi in range(args.folds):
                        print("Processing fold {}".format(foldi))
                        file_name = "fold_" + str(foldi) + "_" + outputs_match + prec + ".npy"
                        file_path = os.path.join(ens_outputs_path, tr_tp, file_name)
                        R_mat = load_npy_arr(file_path, args.device)
                        df = compute_pairwise_accuracies(R_mat, labs)
                        df["repli"] = repli
                        df["fold"] = foldi
                        df["train_set"] = tr_tp
                        df["precision"] = prec
                        df_ens = pd.concat([df_ens, df])

    df_ens.to_csv(os.path.join(args.folder, 'ensemble_pw_accuracies.csv'), index=False)
    df_net.to_csv(os.path.join(args.folder, "net_pw_accuracies.csv"), index=False)


if __name__ == '__main__':
    with torch.no_grad():
        pairwise_accuracies()