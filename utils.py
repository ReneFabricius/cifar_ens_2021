import os
from pickle import load
import numpy as np
import pandas as pd
import torch
import regex as re
import gc
from weensembles.CouplingMethods import coup_picker

from weensembles.WeightedLinearEnsemble import WeightedLinearEnsemble
from weensembles.predictions_evaluation import ECE_sweep, compute_acc_topk, compute_nll
from weensembles.CalibrationEnsemble import CalibrationEnsemble
from weensembles.CombiningMethods import comb_picker
from weensembles.utils import cuda_mem_try


def load_npy_arr(file, device, dtype):
    arr = torch.from_numpy(np.load(file)).to(device=torch.device(device), dtype=dtype)
    arr.requires_grad_(False)
    return arr


def load_networks_outputs(nn_outputs_path, experiment_out_path=None, device='cpu', dtype=torch.float, load_train_data=True):
    """
    Loads network outputs for single replication. Dimensions in the output tensors are network, sample, class.
    :param nn_outputs_path: replication outputs path.
    :param experiment_out_path: if not None a path to folder where to store networks_order file
    containing the order of the networks
    :param device: device to use
    :return: dictionary with network outputs and labels
    """
    networks = [fold for fold in os.listdir(nn_outputs_path) if os.path.isdir(os.path.join(nn_outputs_path, fold))]

    if experiment_out_path is not None:
        networks_order = open(os.path.join(experiment_out_path, 'networks_order.txt'), 'w')
        for net in networks:
            networks_order.write(net + "\n")
        networks_order.close()

    
    test_outputs = []
    for net in networks:
        test_outputs.append(load_npy_arr(os.path.join(nn_outputs_path, net, 'test_outputs.npy'), device=device, dtype=dtype).
                            unsqueeze(0))
    test_outputs = torch.cat(test_outputs, 0)
    test_labels = load_npy_arr(os.path.join(nn_outputs_path, networks[0], 'test_labels.npy'), device=device, dtype=torch.long)

    if load_train_data:
        train_outputs = []
        for net in networks:
            train_outputs.append(load_npy_arr(os.path.join(nn_outputs_path, net, 'train_outputs.npy'), device=device, dtype=dtype).
                                unsqueeze(0))
        train_outputs = torch.cat(train_outputs, 0)
        train_labels = load_npy_arr(os.path.join(nn_outputs_path, networks[0], 'train_labels.npy'), device=device, dtype=torch.long)

    val_outputs = []
    for net in networks:
        val_outputs.append(load_npy_arr(os.path.join(nn_outputs_path, net, 'val_outputs.npy'), device=device, dtype=dtype).
                           unsqueeze(0))
    val_outputs = torch.cat(val_outputs, 0)
    val_labels = load_npy_arr(os.path.join(nn_outputs_path, networks[0], 'val_labels.npy'), device=device, dtype=torch.long)

    ret = {"val_outputs": val_outputs,
            "val_labels": val_labels, "test_outputs": test_outputs, "test_labels": test_labels,
            "networks": networks}
    if load_train_data:
        ret["train_outputs"] = train_outputs
        ret["train_labels"] = train_labels
    
    return ret

class LinearPWEnsOutputs:
    def __init__(self, combining_methods, coupling_methods, store_R=False):
        self.comb_m_ = combining_methods
        self.coup_m_ = coupling_methods

        self.outputs_ = [[None for cp_m in coupling_methods] for co_m in combining_methods]
        
        self.has_R_ = store_R
        if store_R:
            self.R_mats_ = [None for co_m in combining_methods]

    def store(self, combining_method, coupling_method, output, R_mat=None):
        co_i = self.comb_m_.index(combining_method)
        cp_i = self.coup_m_.index(coupling_method)
        self.outputs_[co_i][cp_i] = output
        if self.has_R_ and R_mat is not None:
            self.R_mats_[co_i] = R_mat

    def get(self, combining_method, coupling_method):
        co_i = self.comb_m_.index(combining_method)
        cp_i = self.coup_m_.index(coupling_method)
        return self.outputs_[co_i][cp_i]
    
    def get_R(self, combining_method):
        if not self.has_R_:
            return None
        co_i = self.comb_m_.index(combining_method)
        return self.R_mats_[co_i]
        
    def get_combining_methods(self):
        return self.comb_m_
    
    def get_coupling_methods(self):
        return self.coup_m_


class CalibratingEnsOutputs:
    def __init__(self, calibrating_methods, networks):
        self.cal_m_ = calibrating_methods
        self.nets_ = networks
        self.ens_outputs_ = [None for cal_m in calibrating_methods]
        self.net_outputs_ = [None for cal_m in calibrating_methods]

    def store(self, calibrating_method, ens_output, nets_outputs):
        cal_i = self.cal_m_.index(calibrating_method)
        self.ens_outputs_[cal_i] = ens_output
        self.net_outputs_[cal_i] = nets_outputs

    def get(self, calibrating_method):
        cal_i = self.cal_m_.index(calibrating_method)
        return self.ens_outputs_[cal_i]

    def get_nets_outputs(self, calibrating_method):
        cal_i = self.cal_m_.index(calibrating_method)
        return self.net_outputs_[cal_i]
    
    def get_networks(self):
        return self.nets_
    
    def get_calibrating_methods(self):
        return self.cal_m_


def linear_pw_ens_train_save(predictors, targets, test_predictors, device, out_path, combining_methods,
                             coupling_methods, networks,
                             double_accuracy=False, prefix='', verbose=0,
                             val_predictors=None, val_targets=None,
                             load_existing_models="no", computed_metrics=None, all_networks=None,
                             save_sweep_C=False):
    """
    Trains LinearWeightedEnsemble using all possible combinations of provided combining_methods and coupling_methods.
    Combines outputs given in test_predictors, saves them and returns them in an instance of LinearPWEnsOutputs.

    Args:
        predictors (torch tensor): Training predictors. Tensor of shape c×n×k.
        targets (torch tensor): Training targets.
        test_predictors (torch tensor): Testing predictors. Tensor of shape c×n_t×k.
        device (string): Device to use.
        out_path (string): Folder to save the outputs to.
        combining_methods (list): List of combining methods to use.
        coupling_methods (list): List of coupling methods to use.
        networks (list): List of network names present in the current ensemble.
        double_accuracy (bool, optional): Whether to use double accuracy. Defaults to False.
        prefix (str, optional): Prefix to prepend to file names with outputs. Defaults to ''.
        verbose (int, optional): Verbosity level. Defaults to 0.
        test_normality (bool, optional): Whether to test normality of predictors. Defaults to True.
        save_R_mats (bool, optional): Whether to save resulting R matrices. Defaults to False.
        val_predictors (torch tensor, optional): Validation predictors. Tensor of shape c×n_v×k. Required if sweep_C is True. Defaults to None.
        val_targets (torch tensor, optional): Validation targets. Required if sweep_C is True. Defaults to None.
        load_existing_models (string): Existing models loading strategy. Possible values are no, recalculate and lazy.
            If no is chosen, no loading is performed and all models are trained.
            If recalculate is chosen, models are loaded if corresponding model file exists and metrics are recalculated.
            If lazy is chosem, models for which both model file and metrics are present are skipped. Defaults to no.
        computed_metrics (pandas.Dataframe): Already computed metrics. Required if load_existing_models is lazy.
        all_networks (list): List of all networks names in the experiment. Required if load_existing_models is lazy.
        save_sweep_C (bool, optional): Whether to save C coefficients of logreg combining methods using sweep_C. Defaults to False.
    Raises:
        rerr: [description]

    Returns:
        LinearPWEnsOutputs: Obtained test predictions.
    """
    dtp = torch.float64 if double_accuracy else torch.float32
    ens_test_results = LinearPWEnsOutputs(combining_methods, coupling_methods, store_R=False)
    if load_existing_models == "lazy":
        net_mask = [net in networks for net in all_networks]
   
    for co_m in combining_methods:
        comb_m = comb_picker(co_m=co_m, c=0, k=0)
        if comb_m is None:
            raise ValueError("Unknown combining method: {}".format(co_m))
    
    for cp_m in coupling_methods:
        coup_m = coup_picker(cp_m=cp_m)
        if coup_m is None:
            raise ValueError("Unknown coupling method: {}".format(cp_m))
    
    for co_mi, co_m in enumerate(combining_methods):
        if verbose > 0:
            print("Processing combining method {}".format(co_m))
        model_file = os.path.join(out_path, prefix + co_m + '_model_{}'.format("double" if double_accuracy else "float"))
        model_loaded = False
        co_m_fun = comb_picker(co_m, c=0, k=0, device=device, dtype=dtp)
        if co_m_fun.req_val_ and (val_predictors is None or val_targets is None):
            raise ValueError("Combining method {} requires validation data, but val_predictors or val_targets are None".format(co_m))
        
        save_C = save_sweep_C and hasattr(co_m_fun, "sweep_C_")
            
        model_exists = os.path.exists(model_file)

        metrics_exist = False
        if load_existing_models == "lazy" and computed_metrics.shape[0] > 0:
            comb_metrics = computed_metrics[(computed_metrics[all_networks] == net_mask).prod(axis=1) == 1]
            co_comb_metrics = comb_metrics[comb_metrics["combining_method"] == co_m]
            if co_comb_metrics.shape[0] == len(coupling_methods):
                metrics_exist = True
            else:
                computed_metrics.drop(co_comb_metrics.index, inplace=True)

        if load_existing_models == "lazy" and metrics_exist:
            continue
        
        ens = WeightedLinearEnsemble(c=predictors.shape[0], k=predictors.shape[2], device=device, dtp=dtp)
        if load_existing_models == "no" or not model_exists:
            if co_m_fun.req_val_:
                ens.fit(MP=predictors, tar=targets, verbose=verbose, combining_method=co_m,
                        MP_val=val_predictors, tar_val=val_targets, save_C=save_C)
            else:
                ens.fit(MP=predictors, tar=targets, verbose=verbose, combining_method=co_m, save_C=save_C)
        else:
            ens.load(model_file)
            model_loaded = True

        if not model_loaded:
            ens.save(model_file)
            ens.save_coefs_csv(
                os.path.join(out_path, prefix + co_m + '_coefs_{}.csv'.format("double" if double_accuracy else "float")))
            if save_C:
                ens.save_C_coefs(
                    os.path.join(out_path, prefix + co_m + '_coefs_C_{}.csv'.format("double" if double_accuracy else "float"))
                )

        for cp_mi, cp_m in enumerate(coupling_methods):
            ens_test_out_method = cuda_mem_try(
                fun=lambda bsz: ens.predict_proba(MP=test_predictors, coupling_method=cp_m, batch_size=bsz, verbose=verbose),
                start_bsz=test_predictors.shape[1],
                verbose=verbose,
                device=device)

            ens_test_results.store(co_m, cp_m, ens_test_out_method, R_mat=None)
            np.save(os.path.join(out_path,
                                 "{}ens_test_outputs_co_{}_cp_{}_prec_{}.npy".format(prefix, co_m, cp_m,
                                                                                     ("double" if double_accuracy else "float"))),
                    ens_test_out_method.detach().cpu().numpy())

    return ens_test_results


def calibrating_ens_train_save(predictors, targets, test_predictors, device, out_path, calibrating_methods,
                               networks, double_accuracy=False, prefix='', verbose=0,
                               load_existing_models="no", computed_metrics=None, all_networks=None):
    """
    Trains CalibrationdEnsemble using each provided calibrating_method.
    Combines outputs given in test_predictors, saves them and returns them in an instance of CalibratingEnsOutput.

    Args:
        predictors (torch tensor): Training predictors. Tensor of shape c×n×k.
        targets (torch tensor): Training targets.
        test_predictors (torch tensor): Testing predictors. Tensor of shape c×n_t×k.
        device (string): Torch device to use.
        out_path (string): Path to folder for saving outputs.
        calibrating_methods (list): Calibrating methods to use.
        double_accuracy (bool): Whether to use double accuracy. Defaults to False.
        prefix (string): Prefix for file names of saved outputs. Defaults to ''.
        verbose (int): Level of verbosity. Defaults to 0.
        load_existing_models (string): Existing models loading strategy. Possible values are no, recalculate and lazy.
            If no is chosen, no loading is performed and all models are trained.
            If recalculate is chosen, models are loaded if corresponding model file exists and metrics are recalculated.
            If lazy is chosem, models for which both model file and metrics are present are skipped. Defaults to no.
        computed_metrics (pandas.Dataframe): Already computed metrics. Required if load_existing_models is lazy. Defaults to None.
        all_networks (list): List of all networks names in the experiment. Required if load_existing_models is lazy. Defaults to None.

    Returns:
        CalibratingEnsOutput: instance with ensemble and calibrated networks outputs.
    """
    dtp = torch.float64 if double_accuracy else torch.float32
    ens_test_results = CalibratingEnsOutputs(calibrating_methods=[cal_m.__name__ for cal_m in calibrating_methods],
                                             networks=networks)
    if load_existing_models == "lazy":
        net_mask = [net in networks for net in all_networks]
    
    for cal_mi, cal_m in enumerate(calibrating_methods):
        if verbose > 0:
            print("Processing calibrating method {}".format(cal_m.__name__))
        model_file = os.path.join(out_path,
                              prefix + cal_m.__name__ + '_model_{}'.format("double" if double_accuracy else "float"))
        model_loaded = False
        model_exists = os.path.exists(model_file)

        metrics_exist = False
        if load_existing_models == "lazy" and computed_metrics.shape[0] > 0:
            comb_metrics = computed_metrics[(computed_metrics[all_networks] == net_mask).prod(axis=1) == 1]
            cal_comb_metrics = comb_metrics[comb_metrics["calibrating_method"] == cal_m.__name__]
            if cal_comb_metrics.shape[0] > 0:
                metrics_exist = True
                
        if load_existing_models == "lazy" and metrics_exist:
            continue
        
        ens = CalibrationEnsemble(c=predictors.shape[0], k=predictors.shape[2], device=device, dtp=dtp)
        if load_existing_models == "no" or not model_exists:
            ens.fit(MP=predictors, tar=targets, calibration_method=cal_m, verbose=verbose)
        else:
            ens.load(model_file)
            model_loaded = True

        ens.save_coefs_csv(
            os.path.join(out_path,
                         prefix + cal_m.__name__ + '_coefs_{}.csv'.format("double" if double_accuracy else "float")))
        
        if not model_loaded:
            ens.save(model_file)

        ens_m_out, net_m_out = ens.predict_proba(MP=test_predictors, output_net_preds=True)

        ens_test_results.store(calibrating_method=cal_m.__name__, ens_output=ens_m_out, nets_outputs=net_m_out)
        np.save(os.path.join(out_path,
                             "{}ens_test_outputs_cal_{}_prec_{}.npy".format(prefix, cal_m.__name__,
                                                                            "double" if double_accuracy else "float")),
                ens_m_out.detach().cpu().numpy())
        np.save(os.path.join(out_path,
                             "{}nets_cal_test_outputs_cal_{}_prec_{}.npy".format(prefix, cal_m.__name__,
                                                                                 "double" if double_accuracy else "float")),
                net_m_out.detach().cpu().numpy())

    return ens_test_results


def print_memory_statistics(list_tensors=False):
    allocated = torch.cuda.memory_allocated()
    max_allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_reserved = torch.cuda.max_memory_reserved()

    print("Allocated current: {:.3f}GB, max {:.3f}GB".format(allocated / 2 ** 30, max_allocated / 2 ** 30))
    print("Reserved current: {:.3f}GB, max {:.3f}GB".format(reserved / 2 ** 30, max_reserved / 2 ** 30))

    if list_tensors:
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size(), obj.device)
            except:
                pass


def average_Rs(outputs_path, replications, folds=None, device="cuda"):
    train_types = ["train_training", "val_training"]
    outputs_folder = "comb_outputs"
    net_outputs_folder = "outputs"
    outputs_match = "ens_test_R_"
    if folds is None:
        pattern = "^" + outputs_match + "co_(.*?)_prec_(.*?).npy$"
    else:
        pattern = "^fold_\\d+_" + outputs_match + "co_(.*?)_prec_(.*?).npy$"

    files = os.listdir(os.path.join(outputs_path, "0", outputs_folder, train_types[0]))
    ptrn = re.compile(pattern)
    precisions = list(set([re.search(ptrn, f).group(2) for f in files if re.search(ptrn, f) is not None]))
    combining_methods = list(set([re.search(ptrn, f).group(1) for f in files if re.search(ptrn, f) is not None]))
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
            co_m_list = []
            for co_m in combining_methods:
                print("Processing combining method {}".format(co_m))
                prec_list = []
                for prec in precisions:
                    if folds is None:
                        file_name = outputs_match + "co_" + co_m + "_prec_" + prec + ".npy"
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
                            file_name = "fold_" + str(
                                foldi) + "_" + outputs_match + "co_" + co_m + "_prec_" + prec + ".npy"
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
                co_m_list.append(prec_mat)

            co_m_mat = torch.cat(co_m_list, 0).unsqueeze(0)
            repli_list.append(co_m_mat)

        repli_mat = torch.cat(repli_list, 0)
        repli_aggr = torch.mean(repli_mat, 0)

        np.save(os.path.join(outputs_path, tr_tp + "_class_aggr_R.npy"), repli_aggr.cpu().numpy())
        combining_methods_pd = pd.DataFrame(combining_methods)
        combining_methods_pd.to_csv(os.path.join(outputs_path, "R_mat_co_m_names.csv"), index=False, header=False)


def pairwise_accuracies_mat(preds, labs):
    """Computes matrices of pairwise accuracies for provided predictions according to provided labels

    Args:
        preds (torch.tensor): Tensor of predictions, shape: predictors x samples x classes
        labs (torch.tensor): Correct labels. Tensor of shape samples.
    """
    dev = preds.device
    dtp = preds.dtype
    c, n, k = preds.shape
    PWA = torch.zeros(c, k, k, device=dev, dtype=dtp)
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            mask = (labs == c1) + (labs == c2)
            cur_n = torch.sum(mask)
            cur_preds = preds[:, mask][:, :, [c1, c2]]
            cur_labs = labs[mask]
            c1m = cur_labs == c1
            c2m = cur_labs == c2
            cur_labs[c1m] = 0
            cur_labs[c2m] = 1
            _, cur_inds = torch.topk(cur_preds, k=1, dim=-1)
            cur_accs = torch.sum(cur_inds.squeeze() == cur_labs, dim=-1) / cur_n
            PWA[:, c1, c2] = cur_accs
            PWA[:, c2, c1] = cur_accs
    
    return PWA
            
            
def average_variance(inp, var_dim=0):
    """Computes variance over the specified dimension of the inpuit tensor and then averages it over all remaining dimensions.

    Args:
        inp (torch.tensor): Input tensor
        var_dim (int, optional): Dimension over which to compute variance. Defaults to 0.
    """
    vars = torch.var(inp, dim=var_dim, unbiased=False)
    return torch.mean(vars).item()
            
            
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
            cur_df = pd.DataFrame(data={"class1": [c1] * irrel_count, "class2": [c2] * irrel_count,
                                        "pred1": c1_irrelev_preds.tolist(),
                                        "pred2": c2_irrelev_preds.tolist()})
            df = pd.concat([df, cur_df], ignore_index=True)

    return df


def test_model(model_path, test_inputs_path):
    device = "cuda"
    cp_methods = ["m1", "m2", "m2_iter", "bc"]
    networks = os.listdir(test_inputs_path)

    test_outputs = []
    for net in networks:
        test_outputs.append(load_npy_arr(os.path.join(test_inputs_path, net, 'test_outputs.npy'), device).
                            unsqueeze(0))
    test_outputs = torch.cat(test_outputs, 0)
    test_labels = load_npy_arr(os.path.join(test_inputs_path, networks[0], 'test_labels.npy'), device)

    ens = WeightedLinearEnsemble(test_outputs.shape[0], test_outputs.shape[2], device=device)
    ens.load(model_path)

    for cp_m in cp_methods:
        cp_m_pred = ens.predict_proba(test_outputs, cp_m)
        acc = compute_acc_topk(tar=test_labels, pred=cp_m_pred, k=1)
        nll = compute_nll(tar=test_labels, pred=cp_m_pred)
        print("Method {}, accuracy: {}, nll: {}".format(cp_m, acc, nll))
        

def compute_calibration_plot(prob_pred, labs, bin_n=10, softmax=False):
    """
    Computes calibration plot for the predictions. Plot will have specified number of bins.
    Args:
        preds (torch tensor): Predictions of the classifier. Torch tensor of the shape samples × classes.
        labs (torch tensor): Correct labels for the samples. Torch tensor of the shape samples.
        bins (int, optional): Number of bins in the plot. Defaults to 10.
        softmax (bool, optional): Whether to apply softmax on the predictions before processing.
    """
    dtp = prob_pred.dtype
    
    if softmax:
        prob_pred = torch.nn.Softmax(dim=1)(prob_pred)
        
    top_probs, top_inds = torch.topk(input=prob_pred, k=1, dim=1)
    top_probs = top_probs.squeeze()
    top_inds = top_inds.squeeze()
    cor_pred = top_inds == labs

    df = pd.DataFrame(columns=("bin_start", "bin_end", "sample_num", "mean_conf", "mean_acc"))
    df_i = 0
    
    step = 1 / bin_n
    for st in np.linspace(0.0, 1.0, bin_n, endpoint=False):
        if st == 0:
            cur = top_probs <= step
        elif st + step >= 1.0:
            cur = top_probs > st
        else:
            cur = (top_probs > st) & (top_probs <= st + step)
        if any(cur):
            fxm = torch.mean(top_probs[cur])
            ym = torch.mean(cor_pred[cur].to(dtype=dtp))
            bin_sam_n = torch.sum(cur)
            df.loc[df_i] = [st, st + step, bin_sam_n.item(), fxm.item(), ym.item()]
            df_i += 1
                
    return df

    
def evaluate_networks(net_outputs):
    """
    Computes accuracy, negative log likelihood and estimated calibration error for provided test outputs of penultimate layer of networks.
    Args:
        net_outputs (dict): Dictionary of network outputs as generated by function load_networks_outputs.

    Returns:
        pandas.DataFrame: Data frame containing metrics of networks
    """
    df_net = pd.DataFrame(columns=("network", "accuracy", "nll", "ece"))
    for i, net in enumerate(net_outputs["networks"]):
        acc = compute_acc_topk(tar=net_outputs["test_labels"], pred=net_outputs["test_outputs"][i], k=1)
        nll = compute_nll(tar=net_outputs["test_labels"], pred=net_outputs["test_outputs"][i], penultimate=True)
        ece = ECE_sweep(pred=net_outputs["test_outputs"][i], tar=net_outputs["test_labels"], penultimate=True)
        df_net.loc[i] = [net, acc, nll, ece]
    
    return df_net


def evaluate_ens(ens_outputs, tar):
    """
    Evaluates metrics of ensemble outputs.
    Args:
        ens_outputs (object): Instance of LinearPWEnsOutputs or CalibratingEnsOutputs.
        tar (tensor): Correct labels.
    Returns:
        pandas.DataFrame: Data frame cintaining metrics of ensembles. In case of CalibratingEnsOutputs also returns second data frame containg metrics of calibrated networks.
    """
    if isinstance(ens_outputs, LinearPWEnsOutputs):
        df_ens = pd.DataFrame(columns=("combining_method", "coupling_method", "accuracy", "nll", "ece"))
        df_i = 0
        pw_dfs = []
        
        combining_methods = ens_outputs.get_combining_methods()
        coupling_methods = ens_outputs.get_coupling_methods()
        for co_m in combining_methods:
            for cp_m in coupling_methods:
                pred = ens_outputs.get(co_m, cp_m)
                if pred is None:
                    continue
                
                acc = compute_acc_topk(pred=pred, tar=tar, k=1)
                nll = compute_nll(pred=pred, tar=tar)
                ece = ECE_sweep(pred=pred, tar=tar)
                df_ens.loc[df_i] = [co_m, cp_m, acc, nll, ece]
                df_i += 1

            if ens_outputs.has_R_:
                pw_metrics = compute_pw_metrics(R=ens_outputs.get_R(combining_method=co_m), tar=tar)
                pw_metrics["combining_method"] = co_m
                pw_dfs.append(pw_metrics)

        if ens_outputs.has_R_:
            df_ens_pw = pd.concat(pw_dfs, ignore_index=True) 
            df_ens = df_ens.join(df_ens_pw.set_index("combining_method", on="combining_method"))
        
        return df_ens

    elif isinstance(ens_outputs, CalibratingEnsOutputs):
        df_ens = pd.DataFrame(columns=("calibrating_method", "accuracy", "nll", "ece"))
        df_net = pd.DataFrame(columns=("network", "calibrating_method", "nll", "ece"))
        df_net_i = 0
        calibration_methods = ens_outputs.get_calibrating_methods()
        networks = ens_outputs.get_networks()
        

        for mi, cal_m in enumerate(calibration_methods):
            pred = ens_outputs.get(calibrating_method=cal_m)
            if pred is None:
                continue
            
            acc = compute_acc_topk(pred=pred, tar=tar, k=1)
            nll = compute_nll(pred=pred, tar=tar)
            ece = ECE_sweep(pred=pred, tar=tar)
            df_ens.loc[mi] = [cal_m, acc, nll, ece]

            net_pred = ens_outputs.get_nets_outputs(cal_m)
            for net_i, net in enumerate(networks):
                pred = net_pred[net_i]
                nll_net = compute_nll(pred=pred, tar=tar)
                ece_net = ECE_sweep(pred=pred, tar=tar)
                df_net.loc[df_net_i] = [net, cal_m, nll_net, ece_net]
                df_net_i += 1

        return df_ens, df_net

    else:
        print("Unsupported ensemble output format")
        return None


def compute_pw_metrics(R, tar):
    """Computes pairwise accuracies and calibration for R matrix.

    Args:
        R (torch.tensor): Matrices of pairwise probabilities. Shape n × k × k, where n is number of samples and k in number of classes.
        tar (torch.tensor): Correct class labels. Shape n.
    """
    n, k, k = R.shape
    eces =[]
    accs = []
    for fc in range(k):
        for sc in range(fc + 1, k):
            sample_mask = (tar == fc) + (tar == sc)
            pw_pred = R[sample_mask][:, [fc, sc], [sc, fc]]
            pw_tar = tar[sample_mask]
            mask_fc = pw_tar == fc
            mask_sc = pw_tar == sc
            pw_tar[mask_fc] = 0
            pw_tar[mask_sc] = 1
            acc = compute_acc_topk(pred=pw_pred, tar=pw_tar, k=1)
            ece = ECE_sweep(pred=pw_pred, tar=pw_tar)
            eces.append(ece)
            accs.append(acc)
    
    eces = torch.tensor(eces)
    accs = torch.tensor(accs)
    ece_mean = torch.mean(eces).item()
    ece_var = torch.var(eces).item()
    ece_min = torch.min(eces).item()
    ece_max = torch.max(eces).item()
    acc_mean = torch.mean(accs).item()
    acc_var = torch.var(accs).item()
    acc_min = torch.min(accs).item()
    acc_max = torch.max(accs).item()
    
    df = pd.DataFrame({"pw_acc_mean": [acc_mean], "pw_acc_var": [acc_var], "pw_acc_min": [acc_min], "pw_acc_max": [acc_max],
                       "pw_ece_mean": [ece_mean], "pw_ece_var": [ece_var], "pw_ece_min": [ece_min], "pw_ece_max": [ece_max]})
    
    return df


def bc_mapping(k):
    rws = int(k * (k - 1) / 2)
    # Mapping h is used such that elements of {1, ..., k(k-1)/2}
    # are placed into upper triangle of k x k matrix row by row from left to right.
    M = torch.zeros(rws, k - 1)
    for c in range(k - 1):
        rs = int((c + 1) * k - (c + 1) * (c + 2) / 2)
        re = int((c + 2) * k - (c + 2) * (c + 3) / 2)
        M[rs:re, c] = -1
        oi = c
        cs = 0
        while oi >= 0:
            M[cs + oi, c] = 1
            oi -= 1
            cs += k - (c - oi)
            
    # More effective (hopefully)
    triu_ind = torch.triu_indices(k, k, offset=1)
    ones = triu_ind[1].unsqueeze(0).expand(k - 1, rws)
    rang = torch.arange(1, k).unsqueeze(1)
    M_ef = torch.zeros(k - 1, rws)
    M_ef[ones == rang] = 1
    min_ones = triu_ind[0].unsqueeze(0).expand(k - 1, rws)
    M_ef[min_ones == rang] = -1

    return M, M_ef