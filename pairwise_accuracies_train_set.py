import torch
import argparse
import pandas as pd
import os
import re
from utils.utils import load_networks_outputs, compute_pairwise_accuracies, load_npy_arr, compute_pairwise_calibration, get_irrelevant_predictions


def pairwise_accuracies():
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, required=True, help='experiment root folder')
    parser.add_argument('-repl', type=int, default=1, help='number of replications')
    parser.add_argument('-folds', type=int, default=1, help='number of folds')
    parser.add_argument('-cifar', default=100, type=int, help='cifar type (10 or 100)')
    parser.add_argument('-device', type=str, default='cpu', help='device on which to execute the script')
    parser.add_argument('-comp_irrel', dest='comp_irrel', action='store_true',
                        help='Output LDA predictions for irrelevant classes')
    parser.add_argument('-no_comp_irrel', dest='comp_irrel', action='store_false')
    parser.set_defaults(comp_irrel=False)
    args = parser.parse_args()

    train_types = ["train_training", "val_training"]

    outputs_match = "ens_test_R_"
    if args.folds == 1:
        pattern = "^" + outputs_match + "co_(.*?)_prec_(.*?).npy$"
    else:
        pattern = "^fold_\\d+_" + outputs_match + "co_(.*?)_prec_(.*?).npy$"

    df_ens = pd.DataFrame(columns=('repli', 'fold', 'train_set', 'precision',
                                   'combining_method', 'class1', 'class2', 'accuracy'))

    df_net = pd.DataFrame(columns=("repli", "network", 'class1', 'class2', "accuracy"))

    df_ens_cal = pd.DataFrame(columns=('repli', 'fold', 'train_set', 'precision',
                                       'combining_method', 'class1', 'class2',
                                       'conf_min', 'conf_max', 'bin_accuracy', 'bin_count'))

    df_ens_irrel = pd.DataFrame(columns=('repli', 'fold', 'train_set', 'precision',
                                         'combining_method', 'class1', 'class2',
                                         'pred1', 'pred2'))

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
            df_net = pd.concat([df_net, net_df], ignore_index=True)

        files = os.listdir(os.path.join(ens_outputs_path, train_types[0]))
        ptrn = re.compile(pattern)
        combining_methods = list(set([re.search(ptrn, f).group(1) for f in files if re.search(ptrn, f) is not None]))
        precisions = list(set([re.search(ptrn, f).group(2) for f in files if re.search(ptrn, f) is not None]))

        for tr_tp in train_types:
            print("Processing train type {}".format(tr_tp))

            for co_m in combining_methods:
                for prec in precisions:
                    if args.folds == 1:
                        file_name = "{}co_{}_prec_{}.npy".format(outputs_match, co_m, prec)
                        file_path = os.path.join(ens_outputs_path, tr_tp, file_name)
                        R_mat = load_npy_arr(file_path, args.device)
                        df = compute_pairwise_accuracies(R_mat, labs)
                        df["repli"] = repli
                        df["fold"] = 0
                        df["train_set"] = tr_tp
                        df["precision"] = prec
                        df["combining_method"] = co_m
                        df_ens = pd.concat([df_ens, df], ignore_index=True)

                        df_cal = compute_pairwise_calibration(R_mat, labs)
                        df_cal["repli"] = repli
                        df_cal["fold"] = 0
                        df_cal["train_set"] = tr_tp
                        df_cal["precision"] = prec
                        df_cal["combining_method"] = co_m
                        df_ens_cal = pd.concat([df_ens_cal, df_cal], ignore_index=True)

                        if args.comp_irrel:
                            df_irrel = get_irrelevant_predictions(R_mat, labs)
                            df_irrel["repli"] = repli
                            df_irrel["fold"] = 0
                            df_irrel["train_set"] = tr_tp
                            df_irrel["precision"] = prec
                            df_irrel["combining_method"] = co_m
                            df_ens_irrel = pd.concat([df_ens_irrel, df_irrel], ignore_index=True)

                    else:
                        for foldi in range(args.folds):
                            print("Processing fold {}".format(foldi))
                            file_name = "fold_{}_{}co_{}_prec_{}.npy".format(foldi, outputs_match, co_m, prec)
                            file_path = os.path.join(ens_outputs_path, tr_tp, file_name)
                            R_mat = load_npy_arr(file_path, args.device)
                            df = compute_pairwise_accuracies(R_mat, labs)
                            df["repli"] = repli
                            df["fold"] = foldi
                            df["train_set"] = tr_tp
                            df["precision"] = prec
                            df["combining_method"] = co_m
                            df_ens = pd.concat([df_ens, df], ignore_index=True)

                            df_cal = compute_pairwise_calibration(R_mat, labs)
                            df_cal["repli"] = repli
                            df_cal["fold"] = foldi
                            df_cal["train_set"] = tr_tp
                            df_cal["precision"] = prec
                            df_cal["combining_method"] = co_m
                            df_ens_cal = pd.concat([df_ens_cal, df_cal], ignore_index=True)

                            if args.comp_irrel:
                                df_irrel = get_irrelevant_predictions(R_mat, labs)
                                df_irrel["repli"] = repli
                                df_irrel["fold"] = foldi
                                df_irrel["train_set"] = tr_tp
                                df_irrel["precision"] = prec
                                df_irrel["combining_method"] = co_m
                                df_ens_irrel = pd.concat([df_ens_irrel, df_irrel], ignore_index=True)

    df_ens.to_csv(os.path.join(args.folder, 'ensemble_pw_accuracies.csv'), index=False)
    df_net.to_csv(os.path.join(args.folder, "net_pw_accuracies.csv"), index=False)
    df_ens_cal.to_csv(os.path.join(args.folder, "ensemble_pw_calibration.csv"), index=False)
    if args.comp_irrel:
        df_ens_irrel.to_csv(os.path.join(args.folder, "ensemble_pw_irrelevant.csv"), index=False)


if __name__ == '__main__':
    with torch.no_grad():
        pairwise_accuracies()