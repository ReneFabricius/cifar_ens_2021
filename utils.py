import os
import numpy as np
import pandas as pd
import torch
import regex as re

from weighted_ensembles.WeightedLDAEnsemble import WeightedLDAEnsemble


def load_npy_arr(file, device):
    return torch.from_numpy(np.load(file)).to(torch.device(device))


def load_networks_outputs(nn_outputs_path, experiment_out_path=None, device='cpu'):
    """
    Loads network outputs for single replication. Dimensions in the output tensors are network, sample, class.
    :param nn_outputs_path: replication outputs path.
    :param experiment_out_path: if not None a path to folder where to store networks_order file
    containing the order of the networks
    :param device: device to use
    :return: dictionary with network outputs and labels
    """
    networks = os.listdir(nn_outputs_path)

    if experiment_out_path is not None:
        networks_order = open(os.path.join(experiment_out_path, 'networks_order.txt'), 'w')
        for net in networks:
            networks_order.write(net + "\n")
        networks_order.close()

    test_outputs = []
    for net in networks:
        test_outputs.append(load_npy_arr(os.path.join(nn_outputs_path, net, 'test_outputs.npy'), device).
                            unsqueeze(0))
    test_outputs = torch.cat(test_outputs, 0)
    test_labels = load_npy_arr(os.path.join(nn_outputs_path, networks[0], 'test_labels.npy'), device)

    train_outputs = []
    for net in networks:
        train_outputs.append(load_npy_arr(os.path.join(nn_outputs_path, net, 'train_outputs.npy'), device).
                             unsqueeze(0))
    train_outputs = torch.cat(train_outputs, 0)
    train_labels = load_npy_arr(os.path.join(nn_outputs_path, networks[0], 'train_labels.npy'), device)

    val_outputs = []
    for net in networks:
        val_outputs.append(load_npy_arr(os.path.join(nn_outputs_path, net, 'val_outputs.npy'), device).
                           unsqueeze(0))
    val_outputs = torch.cat(val_outputs, 0)
    val_labels = load_npy_arr(os.path.join(nn_outputs_path, networks[0], 'val_labels.npy'), device)

    return {"train_outputs": train_outputs, "train_labels": train_labels, "val_outputs": val_outputs,
            "val_labels": val_labels, "test_outputs": test_outputs, "test_labels": test_labels,
            "networks": networks}


def ens_train_save(predictors, targets, test_predictors, device, out_path, pwc_methods,
                   double_accuracy=False, prefix='', verbose=True, test_normality=True,
                   save_R_mats=False):
    dtp = torch.float64 if double_accuracy else torch.float32
    ens = WeightedLDAEnsemble(predictors.shape[0], predictors.shape[2], device, dtp=dtp)
    ens.fit_penultimate(predictors, targets, verbose=verbose, test_normality=test_normality)

    ens.save_coefs_csv(os.path.join(out_path, prefix + 'lda_coefs_{}.csv'.format("double" if double_accuracy else "float")))
    ens.save_pvals(os.path.join(out_path, prefix + 'p_values_{}.npy'.format("double" if double_accuracy else "float")))
    ens.save(os.path.join(out_path, prefix + 'model_{}'.format("double" if double_accuracy else "float")))

    ens_test_results = []
    for m_i, pwc_method in enumerate(pwc_methods):
        ens_test_out_method = ens.predict_proba(test_predictors, pwc_method, output_R=save_R_mats)
        if save_R_mats:
            ens_test_out_method, ens_test_R = ens_test_out_method
            if m_i == 0:
                np.save(os.path.join(out_path, "{}ens_test_R_{}.npy".format(prefix,
                                                                   ("double" if double_accuracy else "float"))),
                        ens_test_R.detach().cpu().numpy())

        ens_test_results.append(ens_test_out_method)
        np.save(os.path.join(out_path,
                             "{}ens_test_outputs_{}_{}.npy".format(prefix, pwc_method.__name__,
                                                                   ("double" if double_accuracy else "float"))),
                ens_test_out_method.detach().cpu().numpy())

    return ens_test_results


def print_memory_statistics():
    allocated = torch.cuda.memory_allocated()
    max_allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_reserved = torch.cuda.max_memory_reserved()

    print("Allocated current: {:.3f}GB, max {:.3f}GB".format(allocated / 2**30, max_allocated / 2**30))
    print("Reserved current: {:.3f}GB, max {:.3f}GB".format(reserved / 2**30, max_reserved / 2**30))


def average_Rs(outputs_path, replications, folds=None, device="cuda"):
    train_types = ["train_training", "val_training"]
    outputs_folder = "comb_outputs"
    net_outputs_folder = "outputs"
    outputs_match = "ens_test_R_"
    if folds is None:
        pattern = "^" + outputs_match + "(.*).npy$"
    else:
        pattern = "^fold_\\d+_" + outputs_match + "(.*).npy$"

    files = os.listdir(os.path.join(outputs_path, "0", outputs_folder, train_types[0]))
    ptrn = re.compile(pattern)
    precisions = list(set([re.search(ptrn, f).group(1) for f in files if re.search(ptrn, f) is not None]))
    net_outputs = load_networks_outputs(os.path.join(outputs_path, "0", net_outputs_folder), None, device)
    test_labels = net_outputs["test_labels"]

    classes = len(torch.unique(test_labels))
    labels_mask = [(test_labels == ci).unsqueeze(0) for ci in range(classes)]
    labels_mask = torch.cat(labels_mask, 0)

    for tr_tp in train_types:
        print("Processing train type {}".format(tr_tp))
        repli_list = []
        for repli in range(replications):
            print("Processing repli {}".format(repli))
            prec_list = []
            for prec in precisions:
                if folds is None:
                    file_name = outputs_match + prec + ".npy"
                    file_path = os.path.join(outputs_path, str(repli), outputs_folder, tr_tp, file_name)
                    R_mat = load_npy_arr(file_path, device)
                    n, k, k = R_mat.shape

                    class_mats = []
                    for ci in range(k):
                        class_mats.append(R_mat[labels_mask[ci]].unsqueeze(0))

                    class_sep_R = torch.cat(class_mats, 0)

                    mean_dim = [1]

                else:
                    fold_mats = []
                    for foldi in range(folds):
                        print("Processing fold {}".format(foldi))
                        file_name = "fold_" + str(foldi) + "_" + outputs_match + prec + ".npy"
                        file_path = os.path.join(outputs_path, str(repli), outputs_folder, tr_tp, file_name)
                        R_mat = load_npy_arr(file_path, device)
                        fold_mats.append(R_mat.unsqueeze(0))

                    R_mat = torch.cat(fold_mats, 0)
                    f, n, k, k = R_mat.shape
                    class_mats = []
                    for ci in range(k):
                        class_mats.append(R_mat[:, labels_mask[ci]].unsqueeze(1))

                    class_sep_R = torch.cat(class_mats, 1)

                    mean_dim = [0, 2]

                aggr_R = torch.mean(class_sep_R, mean_dim).unsqueeze(0)
                prec_list.append(aggr_R)

            prec_mat = torch.cat(prec_list, 0).unsqueeze(0)
            repli_list.append(prec_mat)

        repli_mat = torch.cat(repli_list, 0)
        repli_aggr = torch.mean(repli_mat, 0)

        np.save(os.path.join(outputs_path, tr_tp + "_class_aggr_R.npy"), repli_aggr.cpu().numpy())


def compute_pairwise_accuracies(preds, labs):
    """
    Computes pairwise accuracies of the provided predictions according to provided labels.
    :param preds: 2D tensor of probabilistic predictions of the size samples×classes,
    or 3D tensor of pairwise probabilities of the size samples×classes×classes
    :param labs: correct labels
    :return: DataFrame containing pairwise accuracies
    """
    class_n = len(torch.unique(labs))
    df = pd.DataFrame(columns=("class1", "class2", "accuracy"))
    df_i = 0
    for c1 in range(class_n):
        for c2 in range(c1 + 1, class_n):
            sample_mask = (labs == c1) + (labs == c2)

            if len(preds.shape) == 2:
                relev_preds = preds[sample_mask, :][:, [c1, c2]]
            elif len(preds.shape) == 3:
                c1_relev_preds = preds[sample_mask, :, :][:, c1, c2]
                c2_relev_preds = preds[sample_mask, :, :][:, c2, c1]
                relev_preds = torch.cat([c1_relev_preds.unsqueeze(1), c2_relev_preds.unsqueeze(1)], dim=1)

            top_v, top_i = torch.topk(relev_preds, 1, dim=1)
            ti = top_i.squeeze(dim=1)
            relev_labs = labs[sample_mask]
            paired_labs = torch.zeros_like(relev_labs)
            paired_labs[relev_labs == c2] = 1
            acc = torch.sum(ti == paired_labs) / float(len(paired_labs))
            df.loc[df_i] = [c1, c2, acc.item()]
            df_i += 1

    return df


def compute_pairwise_calibration(R_mat, labs):
    """
    Computes calibration plot data (confidence vs accuracy) for each class pair.
    :param R_mat: Matrices of outputs from LDA models.
    :param labs: Correct labels
    :return: Dataframe containing class pair, start and end value of confidence interval (conf_min; conf_max],
    accuracy of predictions falling to the bin and number of predictions falling to the bin
    """
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    class_n = len(torch.unique(labs))
    df = pd.DataFrame(columns=("class1", "class2", "conf_min", "conf_max", "bin_accuracy", "bin_count"))
    df_i = 0
    for c1 in range(class_n):
        for c2 in range(c1 + 1, class_n):
            sample_mask = (labs == c1) + (labs == c2)
            c1_relev_preds = R_mat[sample_mask, :, :][:, c1, c2]
            c2_relev_preds = R_mat[sample_mask, :, :][:, c2, c1]
            for bei in range(1, len(bins)):
                bs = bins[bei - 1]
                be = bins[bei]
                if bei == 1:
                    c1_bin_mask = c1_relev_preds <= be
                    c2_bin_mask = c2_relev_preds <= be
                else:
                    c1_bin_mask = (c1_relev_preds > bs) & (c1_relev_preds <= be)
                    c2_bin_mask = (c2_relev_preds > bs) & (c2_relev_preds <= be)

                relev_labs = labs[sample_mask]
                c1_cor_count = torch.sum(relev_labs[c1_bin_mask] == c1).item()
                c2_cor_count = torch.sum(relev_labs[c2_bin_mask] == c2).item()
                bin_samples_count = torch.sum(c1_bin_mask).item() + torch.sum(c2_bin_mask).item()
                bin_correct = c1_cor_count + c2_cor_count
                bin_acc = bin_correct / bin_samples_count if bin_samples_count != 0 else pd.NA

                df.loc[df_i] = [c1, c2, bs, be, bin_acc, bin_samples_count]
                df_i += 1

    return df


def get_irrelevant_predictions(R_mat, labs):
    """
    Computes irrelevant outputs of LDA models. Irrelevant outputs are predictions for all classes for which the LDA
    wasn't trained.
    :param R_mat: Matrices of outputs from LDA models.
    :param labs: Correct labels.
    :return: Dataframe containing information about class pair (for which the LDA was trained) and predictions
    for class 1 and class 2 of the LDA model.
    """
    class_n = len(torch.unique(labs))
    df = pd.DataFrame(columns=("class1", "class2", "pred1", "pred2"))
    for c1 in range(class_n):
        for c2 in range(c1 + 1, class_n):
            sample_mask = (labs == c1) + (labs == c2)
            sample_mask_inv = (sample_mask != True)
            irrel_count = torch.sum(sample_mask_inv)
            c1_irrelev_preds = R_mat[sample_mask_inv, :, :][:, c1, c2]
            c2_irrelev_preds = R_mat[sample_mask_inv, :, :][:, c2, c1]
            cur_df = pd.DataFrame(data={"class1": [c1]*irrel_count, "class2": [c2]*irrel_count,
                                        "pred1": c1_irrelev_preds.tolist(),
                                        "pred2": c2_irrelev_preds.tolist()})
            df = pd.concat([df, cur_df])

    return df



