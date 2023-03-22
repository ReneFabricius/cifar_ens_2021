import argparse
import torch
import numpy as np
import regex as re
import os
from typing import List
import pandas as pd

from weensembles.WeightedLinearEnsemble import WeightedLinearEnsemble as WLE
from weensembles.CalibrationEnsemble import CalibrationEnsemble as CAL
from weensembles.predictions_evaluation import compute_acc_topk
from weensembles.utils import cuda_mem_try
from utils.utils import load_npy_arr, depth_walk


def compute_error(folder, match, labels):
    outputs = np.load(os.path.join(folder, match.string))
    if not hasattr(outputs, "files"):
        acc_top1 = compute_acc_topk(pred=torch.from_numpy(outputs).to(labels.device), tar=labels, k=1)
        return 1.0 - acc_top1
    
    inds = torch.from_numpy(outputs["indices"]).to(labels.device)
    top_inds = inds[:, 0].squeeze()
    acc = torch.sum(top_inds == labels) / labels.shape[0]
    return 1.0 - acc.item()
    

def compute_absolute_CE(args_dict=None):
    if args_dict is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('-device', type=str, default="cpu", help="Device to use")
        parser.add_argument('-verbose', default=0, type=int, help="Level of verbosity")
        parser.add_argument('-testing_root', type=str, help="Path to a root of testing folder tree.")
        parser.add_argument('-outputs_folder', type=str, help="Path to a folder for storing outputs")
        args = parser.parse_args()
    else:
        args = args_dict

    CORRUPT_FOLDERS = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
        "speckle_noise",
        "gaussian_blur",
        "spatter",
        "saturate"]
    CORRUPT_LEVELS = 5
    nets_df_name = "nets_CE_detailed.csv"
    pwc_df_name = "pwc_CE_detailed.csv"
    cal_df_name = "cal_CE_detailed.csv"
    
    net_df = pd.DataFrame()
    pwc_df = pd.DataFrame()
    cal_df = pd.DataFrame()
   
    ex_net = re.compile(r"^(?P<net>.*?_IM2012)_outputs.npy$") 
    ex_pwc = re.compile(r"^(?P<nets>.*?)_ens_test_outputs_short_co_(?P<comb_m>.*?)_cp_(?P<coup_m>.*?)_prec_(?P<prec>.*?)_topl_(?P<topl>\d*).npz$")
    ex_cal = re.compile(r"^(?P<nets>.*?)_ens_test_outputs_short_cal_(?P<cal_m>.*?)_prec_(?P<prec>.*?).npz$")
    
    for cor_fold in CORRUPT_FOLDERS:
        if args.verbose > 0:
            print(f"Processing folder {cor_fold}")
        for cor_lev in range(1, CORRUPT_LEVELS + 1):
            if args.verbose > 0:
                print(f"Processing corruption level {cor_lev}")
            cur_fold = os.path.join(args.testing_root, cor_fold, str(cor_lev))
            cur_f_list = os.listdir(cur_fold) if os.path.exists(cur_fold) else []
            labels = load_npy_arr(os.path.join(cur_fold, "labels.npy"), device=args.device, dtype=torch.int32)
            
            net_output_fs = [mtch for mtch in map(ex_net.match, cur_f_list) if mtch is not None]
            if args.verbose > 1:
                print(f"Networks found {net_output_fs}")
            for net_out_f in net_output_fs:
                error = compute_error(folder=cur_fold, match=net_out_f, labels=labels)
                row = pd.DataFrame(
                    {
                        "network": net_out_f['net'],
                        "corruption_type": cor_fold,
                        "corruption_level": cor_lev,
                        "corruption_error": error
                    },
                    index=[0]
                )
                net_df = pd.concat([net_df, row], ignore_index=True)
                
            pwc_output_fs = [mtch for mtch in map(ex_pwc.match, cur_f_list) if mtch is not None]
            if args.verbose > 1:
                print(f"Ensembles pwc found {pwc_output_fs}")
            for pwc_out_f in pwc_output_fs:
                error = compute_error(folder=cur_fold, match=pwc_out_f, labels=labels)
                sorted_nets = '+'.join(sorted(pwc_out_f['nets'].split('+')))
                row = pd.DataFrame(
                    {
                        "nets": sorted_nets,
                        "comb_size": len(pwc_out_f['nets'].split('+')),
                        "combining_method": pwc_out_f['comb_m'],
                        "coupling_method": pwc_out_f['coup_m'],
                        "computational_precision": pwc_out_f['prec'],
                        "topl": pwc_out_f['topl'],
                        "corruption_type": cor_fold,
                        "corruption_level": cor_lev,
                        "corruption_error": error
                    },
                    index=[0]
                )
                pwc_df = pd.concat([pwc_df, row], ignore_index=True)
                
            cal_output_fs = [mtch for mtch in map(ex_cal.match, cur_f_list) if mtch is not None]
            if args.verbose > 1:
                print(f"Ensembles cal found {cal_output_fs}")
            for cal_out_f in cal_output_fs:
                error = compute_error(folder=cur_fold, match=cal_out_f, labels=labels)
                sorted_nets = '+'.join(sorted(cal_out_f['nets'].split('+')))
                row = pd.DataFrame(
                    {
                        "nets": sorted_nets,
                        "comb_size": len(cal_out_f['nets'].split('+')),
                        "calibrating_method": cal_out_f['cal_m'],
                        "computational_precision": cal_out_f['prec'],
                        "corruption_type": cor_fold,
                        "corruption_level": cor_lev,
                        "corruption_error": error
                    },
                    index=[0]
                )
                cal_df = pd.concat([cal_df, row], ignore_index=True)
            
    net_df.to_csv(os.path.join(args.outputs_folder, nets_df_name), index=False)                
    pwc_df.to_csv(os.path.join(args.outputs_folder, pwc_df_name), index=False)
    cal_df.to_csv(os.path.join(args.outputs_folder, cal_df_name), index=False)

    return {"net_df": net_df, "pwc_df": pwc_df, "cal_df": cal_df}

if __name__ == "__main__":
    compute_absolute_CE()