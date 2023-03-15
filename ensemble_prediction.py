import argparse
import torch
import numpy as np
import regex as re
import os
from typing import List
import pandas as pd

from weensembles.WeightedLinearEnsemble import WeightedLinearEnsemble as WLE
from weensembles.CalibrationEnsemble import CalibrationEnsemble as CAL
from weensembles.utils import cuda_mem_try
from utils.utils import load_npy_arr, depth_walk


def load_net_outputs(folder:str, networks:List[str], device:str="cpu", dtype:torch.dtype=torch.float):
    outputs = []
    for net in networks:
        output = load_npy_arr(file=os.path.join(folder, f"{net}_outputs.npy"), device=device, dtype=dtype)
        outputs.append(output.unsqueeze(0))
    
    return torch.cat(outputs, dim=0)


def ens_prediction(args_dict=None):
    if args_dict is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('-models_folder', type=str, help="Folder of trained ensemble models")
        parser.add_argument('-ens_sizes', nargs="+", default=[], type=int, help="Ensemble sizes to test")
        parser.add_argument('-ens_comb_file', type=str, default="",
                            help="Path to file with listing of networks combinations to test. Can be used along with -ens_sizes. Each combination should be specified on separate line by network names separated by commas.")
        parser.add_argument('-configurations_file', type=str, default="",
                            help="Path to a file listing configurations to test. Csv file with header and columns combining_method, coupling_method, topl.")
        parser.add_argument('-device', type=str, default="cpu", help="Device to use")
        parser.add_argument('-verbose', default=0, type=int, help="Level of verbosity")
        parser.add_argument('-calibrating_methods', nargs='+', default=["TemperatureScaling"], help="Calibrating methods to test")
        parser.add_argument('-comp_precisions', nargs='+', default=["float"], help="Precisions which to test for the computations (float and/or double)")
        parser.add_argument('-testing_root', type=str, help="Path to a root of testing folder tree.")
        parser.add_argument('-save_shortened', dest="save_shortened", action="store_true")
        args = parser.parse_args()
    else:
        args = args_dict
        
    print("Running with arguments:")
    print(args)
    K = 1000        # number of classes
    save_topk = 5
    
    if args.save_shortened:
        file_content_name = "ens_test_outputs_short"
        file_extension = ".npz"
    else:
        file_content_name = "ens_test_outputs"
        file_extension = ".npy"

    subfolders = depth_walk(root=args.testing_root, exact_depth=2)
    print(f"Subfolders found: {subfolders}")
    out_f_listdir = os.listdir(args.models_folder) if os.path.exists(args.models_folder) else [] 
    comp_precision = args.comp_precisions[0]
    
    configs = pd.read_csv(args.configurations_file)
    configs.reset_index()
    
    for index, row in configs.iterrows():
        print(f"Processing config {row}")
        pwc_pattern = f"^(?P<nets>.*?)_model_co_{row['combining_method']}_prec_{comp_precision}$"
        pwc_comp_patt = re.compile(pwc_pattern)
        
        pwc_models = [mtch for mtch in map(pwc_comp_patt.match, out_f_listdir) if mtch is not None]
        for pwc_model in pwc_models:
            print(f"Processing model {pwc_model.string}")
            networks_sorted = pwc_model["nets"].split("+")
            
            wle = WLE(c=len(networks_sorted), k=K, device=args.device)
            wle.load(os.path.join(args.models_folder, pwc_model.string), verbose=args.verbose)
            
            for subf in subfolders:
                out_f_name = f"{pwc_model['nets']}_{file_content_name}_co_{row['combining_method']}_cp_{row['coupling_method']}_prec_{args.comp_precisions[0]}_topl_{row['topl']}{file_extension}"
                out_f = os.path.join(args.testing_root, subf, out_f_name)
                if os.path.exists(out_f):
                    if args.verbose > 0:
                        print(f"Skipping existing file {out_f}")
                    continue
                
                net_out = load_net_outputs(folder=os.path.join(args.testing_root, subf), networks=wle.constituent_names_, device=args.device)
                wle_pred = cuda_mem_try(
                    fun = lambda bsz: wle.predict_proba(
                        preds=net_out, coupling_method=row['coupling_method'],
                        verbose=args.verbose, l=row["topl"], batch_size=bsz),
                    start_bsz=40 if row["topl"] > 50 else 1000,
                    device=args.device,
                    dec_coef=0.5,
                    verbose=args.verbose)
                
                if args.save_shortened:
                    vals, inds = torch.topk(wle_pred, k=save_topk, dim=-1)                    
                    sm_denoms = torch.sum(torch.exp(wle_pred), dim=-1)
                    np.savez(out_f, values=vals.cpu(), indices=inds.cpu(), sm_denominators=sm_denoms.cpu())
                else:
                    np.save(out_f, wle_pred.cpu())
   
    cal_method = args.calibrating_methods[0]
    cal_pattern = f"^(?P<nets>.*?)_model_cal_{cal_method}_prec_{comp_precision}$"
    cal_comp_patt = re.compile(cal_pattern)
    
    cal_models = [mtch for mtch in map(cal_comp_patt.match, out_f_listdir) if mtch is not None]

    for cal_model in cal_models:
        print(f"Processing model {cal_model.string}")
        networks_sorted = cal_model["nets"].split("+")
        
        cal = CAL(c=len(networks_sorted), k=K, device=args.dev)
        cal.load(cal_model.string, verbose=args.verbose)
        
        for subf in subfolders:
            net_out = load_net_outputs(folder=os.path.join(args.testing_root, subf), networks=cal.constituent_names_, device=args.device)
            cal_pred = cuda_mem_try(
                fun = lambda bsz: cal.predict_proba(
                    preds=net_out, verbose=args.verbose, batch_size=bsz),
                start_bsz=500,
                device=args.device,
                dec_coef=0.5,
                verbose=args.verbose)
            out_f_name = f"{cal_model['nets']}_{file_content_name}_cal_{cal_method}_prec_{comp_precision}{file_extension}"
            out_f = os.path.join(args.testing_root, subf, out_f_name)

            if args.save_shortened:
                vals, inds = torch.topk(cal_pred, k=save_topk, dim=-1)
                sm_denoms = torch.sum(torch.exp(cal_pred), dim=-1)
                np.savez(out_f, values=vals.cpu(), indices=inds.cpu(), sm_denominators=sm_denoms.cpu())
            else:                
                np.save(out_f, cal_pred.cpu())


if __name__ == "__main__":
    ens_prediction() 


 