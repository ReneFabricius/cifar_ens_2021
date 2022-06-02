from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch

from weensembles.predictions_evaluation import compute_error_inconsistency, compute_acc_topk, compute_nll, ECE_sweep
from weensembles.Ensemble import Ensemble
import utils.utils as ens_utils
from weensembles.CalibrationEnsemble import CalibrationEnsemble
from weensembles.WeightedLinearEnsemble import WeightedLinearEnsemble
from weensembles.CombiningMethods import comb_picker
from weensembles.utils import cuda_mem_try


class ComputationPlan(ABC):
    def __init__(self, plan: pd.DataFrame, metrics: pd.DataFrame, metrics_name: str, device: str,
                 model_file_format: str, model_coefs_file_format: str,
                 model_pred_file_format: str) -> None:
        self.plan_ = plan
        self.metrics_ = metrics
        self.metrics_name_ = metrics_name
        self.dev_ = device
        self.mod_f_form_ = model_file_format
        self.mod_csv_f_form_ = model_coefs_file_format
        self.mod_pred_f_form_ = model_pred_file_format
    
    def save_metrics(self, outputs_folder: str):
        if self.metrics_ is not None and self.metrics_.shape[0] > 0:
            file = os.path.join(outputs_folder, self.metrics_name_)
            self.metrics_.to_csv(file, index=False)
        
    def save_model(self, outputs_folder: str, model: Ensemble, nets: str, method: str, comp_prec: str, verbose: int=0):
        model_name = self.mod_f_form_.format(nets, method, comp_prec)
        model.save(file=os.path.join(outputs_folder, model_name), verbose=verbose)
        model_coefs_name = self.mod_csv_f_form_.format(nets, method, comp_prec)
        model.save_coefs_csv(file=os.path.join(outputs_folder, model_coefs_name))
        
    def compute_metrics(self, predictions, labels):
        acc1 = compute_acc_topk(pred=predictions, tar=labels, k=1)
        acc5 = compute_acc_topk(pred=predictions, tar=labels, k=5)
        nll = compute_nll(pred=predictions, tar=labels, penultimate=False)
        ece = ECE_sweep(pred=predictions, tar=labels, penultimate=False)
        
        return acc1, acc5, nll, ece
        
    @abstractmethod
    def _process_model(self, comb_id: int, comb_mask: List[bool], comp_precision: str,
                            val_pred: torch.tensor, val_labels: torch.tensor,
                            test_pred: torch.tensor, test_labels: torch.tensor,
                            err_inc: float, all_cor: float, mean_pwa_var: float,
                            outputs_folder:str, nets_string: str, networks: List[str],
                            combiner_val_pred: torch.tensor=None, combiner_val_labels: torch.tensor=None,
                            verbose: int=0):
        pass 
                

    
    def execute_evaluate_save(self, net_outputs: Dict[str, np.array], outputs_folder: str, req_comb_val_data: bool, verbose: int=0) -> pd.DataFrame:
        COMBINER_SAMPLES_PER_CLASS = 50
        self.save_metrics(outputs_folder=outputs_folder)
        networks = net_outputs["networks"]
        
        print("Evaluating networks")
        df_net = ens_utils.evaluate_networks(net_outputs)
        df_net.to_csv(os.path.join(outputs_folder, "net_metrics.csv"), index=False)
        
        net_pwa = ens_utils.pairwise_accuracies_mat(preds=net_outputs["test_outputs"], labs=net_outputs["test_labels"])

        print("Processing ensembles")
        for comb_id in self.plan_["combination_id"].drop_duplicates():
            comb_mask = self.plan_[self.plan_["combination_id"] == comb_id][networks].drop_duplicates().iloc[0].tolist()
            comb_nets = np.array(networks)[np.array(comb_mask)]
            if verbose > 0:
                print("Processing combination of networks: {}".format(comb_nets))
            comb_nets_string = '+'.join(comb_nets)
            val_pred = net_outputs["val_outputs"][comb_mask]
            test_pred = net_outputs["test_outputs"][comb_mask]
            val_lab = net_outputs["val_labels"]
            test_lab = net_outputs["test_labels"]
            classes = torch.unique(val_lab)
            n_classes = len(classes)
            
            err_inc, all_cor = compute_error_inconsistency(preds=test_pred, tar=test_lab)
            mean_pwa_var = ens_utils.average_variance(inp=net_pwa[comb_mask])
        
            if req_comb_val_data: 
                train_pred = net_outputs["train_outputs"][comb_mask]
                train_lab = net_outputs["train_labels"]
                _, lin_train_idx = train_test_split(np.arange(len(train_lab)), shuffle=True, stratify=train_lab.cpu(),
                                                    test_size=n_classes * COMBINER_SAMPLES_PER_CLASS)
                combiner_val_pred = train_pred[:, lin_train_idx]
                combiner_val_lab = train_lab[lin_train_idx]
            else:
                combiner_val_pred = None
                combiner_val_lab = None
            
            comp_precisions = self.plan_[self.plan_["combination_id"] == comb_id]["computational_precision"].drop_duplicates()
            for comp_prec in comp_precisions:
                if verbose > 0:
                    print("Processing computational precision {}".format(comp_prec))
                prec_dtype = torch.float32 if comp_prec == "float" else torch.float64
                self._process_model(comb_id=comb_id, comb_mask=comb_mask, comp_precision=comp_prec,
                                    nets_string=comb_nets_string, networks=networks,
                                    val_pred=val_pred.to(dtype=prec_dtype) if val_pred is not None else None,
                                    val_labels=val_lab,
                                    test_pred=test_pred.to(dtype=prec_dtype) if test_pred is not None else None,
                                    test_labels=test_lab,
                                    err_inc=err_inc, all_cor=all_cor, mean_pwa_var=mean_pwa_var,
                                    combiner_val_pred=combiner_val_pred.to(dtype=prec_dtype) if combiner_val_pred is not None else None,
                                    combiner_val_labels=combiner_val_lab, verbose=verbose,
                                    outputs_folder=outputs_folder)
  

class ComputationPlanPWC(ComputationPlan):
    def __init__(self, plan: pd.DataFrame, metrics: pd.DataFrame, device: str="cpu") -> None:
        super().__init__(plan=plan, metrics=metrics, device=device,
                         metrics_name="ens_pwc_metrics.csv",
                         model_file_format="{}_model_co_{}_prec_{}",
                         model_coefs_file_format="{}_csv_coefs_co_{}_prec_{}.csv",
                         model_pred_file_format="{}_ens_test_outputs_co_{}_cp_{}_prec_{}_topl_{}.npy") 
     
    def _process_model(self, comb_id: int, comb_mask: List[bool], comp_precision: str,
                       val_pred: torch.tensor, val_labels: torch.tensor,
                       test_pred: torch.tensor, test_labels: torch.tensor,
                       err_inc: float, all_cor: float, mean_pwa_var: float,
                       outputs_folder: str, nets_string: str, networks: List[str],
                       combiner_val_pred: torch.tensor=None, combiner_val_labels: torch.tensor=None,
                       verbose: int=0):
        
        comb_m_names = self.plan_[(self.plan_["combination_id"] == comb_id) &
                                  (self.plan_["computational_precision"] == comp_precision)]["combining_method"].drop_duplicates()
        c, n, k = val_pred.shape if val_pred is not None else test_pred.shape
        comp_dtype = torch.float32 if comp_precision == "float" else torch.float64
        for comb_m in comb_m_names:
            if verbose > 0:
                print("Processing combining method {}".format(comb_m))
            model_file = self.plan_[(self.plan_["combination_id"] == comb_id) &
                                    (self.plan_["computational_precision"] == comp_precision) &
                                    (self.plan_["combining_method"] == comb_m)]["model_file"].iloc[0]
            wle = WeightedLinearEnsemble(c=c, k=k, device=self.dev_, dtp=comp_dtype)
            if pd.isna(model_file):
                wle.fit(preds=val_pred, labels=val_labels, combining_method=comb_m,
                        verbose=verbose, val_preds=combiner_val_pred, val_labels=combiner_val_labels)
                self.save_model(outputs_folder=outputs_folder, model=wle, nets=nets_string,
                                method=comb_m, comp_prec=comp_precision, verbose=verbose)
            else:
               wle.load(os.path.join(outputs_folder, model_file))
            
            coup_methods = self.plan_[(self.plan_["combination_id"] == comb_id) &
                                      (self.plan_["computational_precision"] == comp_precision) &
                                      (self.plan_["combining_method"] == comb_m)]["coupling_method"].drop_duplicates()
            for coup_m in coup_methods:
                if verbose > 0:
                    print("Processing coupling method {}".format(coup_m))
                topl_vals = self.plan_[(self.plan_["combination_id"] == comb_id) &
                                      (self.plan_["computational_precision"] == comp_precision) &
                                      (self.plan_["combining_method"] == comb_m) &
                                      (self.plan_["coupling_method"] == coup_m)]["topl"].drop_duplicates()
                for topl in topl_vals:
                    if verbose > 0:
                        print("Processing topl value {}".format(topl)) 
                    ens_test_pred = cuda_mem_try(fun=lambda bsz: wle.predict_proba(preds=test_pred,
                                                                                   coupling_method=coup_m,
                                                                                   verbose=verbose, l=topl if topl > 0 else k,
                                                                                   batch_size=bsz),
                                                 start_bsz=test_pred.shape[1],
                                                 device=self.dev_,
                                                 dec_coef=0.8, verbose=verbose)
                    pred_name = self.mod_pred_f_form_.format(nets_string, comb_m, coup_m, comp_precision, topl)
                    np.save(os.path.join(outputs_folder, pred_name), arr=ens_test_pred.detach().cpu().numpy())
                    acc1, acc5, nll, ece = self.compute_metrics(predictions=ens_test_pred, labels=test_labels)
                    row_data = [comb_mask + [comb_id, sum(comb_mask), err_inc, all_cor, mean_pwa_var,
                                            comb_m, coup_m, topl, comp_precision,
                                            acc1, acc5, nll, ece]]
                    row_cols = networks + ["combination_id", "combination_size", "err_incons", "all_cor", "mean_pwa_var",
                                           "combining_method", "coupling_method", "topl", "computational_precision",
                                           "accuracy1", "accuracy5", "nll", "ece"]
                    row_df = pd.DataFrame(data=row_data, columns=row_cols)
                    self.metrics_ = pd.concat([self.metrics_, row_df])
                    self.save_metrics(outputs_folder=outputs_folder)
                                 

class ComputationPlanCAL(ComputationPlan):
    def __init__(self, plan: pd.DataFrame, metrics: pd.DataFrame, device: str="cpu") -> None:
        super().__init__(plan=plan, metrics=metrics, device=device,
                         metrics_name="ens_cal_metrics.csv",
                         model_file_format="{}_model_cal_{}_prec_{}",
                         model_coefs_file_format="{}_csv_coefs_cal_{}_prec_{}.csv",
                         model_pred_file_format="{}_ens_test_outputs_cal_{}_prec_{}.npy") 
 
    def _process_model(self, comb_id: int, comb_mask: List[bool], comp_precision: str,
                       val_pred: torch.tensor, val_labels: torch.tensor,
                       test_pred: torch.tensor, test_labels: torch.tensor,
                       err_inc: float, all_cor: float, mean_pwa_var: float,
                       outputs_folder: str, nets_string: str, networks: List[str],
                       combiner_val_pred: torch.tensor=None, combiner_val_labels: torch.tensor=None,
                       verbose: int=0):
        
        cal_m_names = self.plan_[(self.plan_["combination_id"] == comb_id) &
                                  (self.plan_["computational_precision"] == comp_precision)]["calibrating_method"].drop_duplicates()
        c, n, k = val_pred.shape if val_pred is not None else test_pred.shape
        comp_dtype = torch.float32 if comp_precision == "float" else torch.float64
        for cal_m in cal_m_names:
            if verbose > 0:
                print("Processing calibrating method {}".format(cal_m))
            model_file = self.plan_[(self.plan_["combination_id"] == comb_id) &
                                    (self.plan_["computational_precision"] == comp_precision) &
                                    (self.plan_["calibrating_method"] == cal_m)]["model_file"].iloc[0]
            cale = CalibrationEnsemble(c=c, k=k, device=self.dev_, dtp=comp_dtype)
            if pd.isna(model_file):
                cale.fit(preds=val_pred, labels=val_labels, calibration_method=cal_m,
                        verbose=verbose, val_preds=combiner_val_pred, val_labels=combiner_val_labels)
                self.save_model(outputs_folder=outputs_folder, model=cale, nets=nets_string,
                                method=cal_m, comp_prec=comp_precision, verbose=verbose)
            else:
               cale.load(os.path.join(outputs_folder, model_file))
            
            ens_test_pred = cale.predict_proba(preds=test_pred)
            
            pred_name = self.mod_pred_f_form_.format(nets_string, cal_m, comp_precision)
            np.save(os.path.join(outputs_folder, pred_name), arr=ens_test_pred.detach().cpu().numpy())
            acc1, acc5, nll, ece = self.compute_metrics(predictions=ens_test_pred, labels=test_labels)
            row_data = [comb_mask + [comb_id, sum(comb_mask), err_inc, all_cor, mean_pwa_var,
                                    cal_m, comp_precision,
                                    acc1, acc5, nll, ece]]
            row_cols = networks + ["combination_id", "combination_size", "err_incons", "all_cor", "mean_pwa_var",
                                    "calibrating_method", "computational_precision",
                                    "accuracy1", "accuracy5", "nll", "ece"]
            row_df = pd.DataFrame(data=row_data, columns=row_cols)
            self.metrics_ = pd.concat([self.metrics_, row_df])
            self.save_metrics(outputs_folder=outputs_folder)
    