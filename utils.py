import os
import numpy as np
import torch

from weighted_ensembles.WeightedLDAEnsemble import WeightedLDAEnsemble


def load_npy_arr(file, device):
    return torch.from_numpy(np.load(file)).to(torch.device(device))


def load_networks_outputs(nn_outputs_path, experiment_out_path, device):
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

    return train_outputs, train_labels, val_outputs, val_labels, test_outputs, test_labels, networks


def ens_train_save(predictors, targets, test_predictors, device, out_path, pwc_methods,
                   double_accuracy=False, prefix='', verbose=True, test_normality=True):
    dtp = torch.float64 if double_accuracy else torch.float32
    ens = WeightedLDAEnsemble(predictors.shape[0], predictors.shape[2], device, dtp=dtp)
    ens.fit_penultimate(predictors, targets, verbose=verbose, test_normality=test_normality)

    ens.save_coefs_csv(os.path.join(out_path, prefix + 'lda_coefs_{}.csv'.format("double" if double_accuracy else "float")))
    ens.save_pvals(os.path.join(out_path, prefix + 'p_values_{}.npy'.format("double" if double_accuracy else "float")))
    ens.save(os.path.join(out_path, prefix + 'model_{}'.format("double" if double_accuracy else "float")))

    ens_test_results = []
    for pwc_method in pwc_methods:
        ens_test_out_method = ens.predict_proba(test_predictors, pwc_method)
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
