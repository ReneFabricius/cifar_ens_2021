from numpy.lib.npyio import load
import torch
import re
import argparse
import os
import pandas as pd

from utils import compute_calibration_plot, load_networks_outputs, load_npy_arr

def comp_plot_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, required=True, help='Folder to load classification outputs from')
    parser.add_argument('-labels_folder', type=str, required=True, help='Folder with correct labels')
    parser.add_argument('-outputs_type', type=str, default='net', help='Type of classifier (for file naming and softmax). Possible values are: net, cal_net, cal_ens, pw_ens')
    parser.add_argument('-device', type=str, default='cpu', help='device on which to execute the script')

    args = parser.parse_args()

    if args.outputs_type == 'cal_net':
        prim_ptrn = re.compile("^cal_set_(?P<calibration_set>.+?)_size_(?P<size>\d+?)_repl_(?P<repl>\d+?)_nets_cal_test_outputs_cal_(?P<cal_method>.+?)_prec_(?P<precision>.+?).npy$")
        alt_ptrn = None
    elif args.outputs_type == 'cal_ens':
        prim_ptrn = re.compile("^cal_set_(?P<calibration_set>.+?)_size_(?P<size>\d+?)_repl_(?P<repl>\d+?)_ens_test_outputs_cal_(?P<calibration_method>.+?)_prec_(?P<precision>.+?).npy$")
        alt_ptrn = re.compile("^cal_set_(?P<calibration_set>.+?)_size_(?P<size>\d+?)_ens_test_outputs_cal_(?P<calibration_method>.+?)_prec_(?P<precision>.+?).npy$")
    elif args.outputs_type == 'pw_ens':
        prim_ptrn = re.compile("^fold_(?P<fold>\d+?)_ens_test_outputs_co_(?P<combining_method>.+?)_cp_(?P<coupling_method>.+?)_prec_(?P<precision>.+?).npy$")
        alt_ptrn = re.compile("^ens_test_outputs_co_(?P<combining_method>.+?)_cp_(?P<coupling_method>.+?)_prec_(?P<precision>.+?).npy$")

    print("Loading networks outputs")
    nets_outputs = load_networks_outputs(nn_outputs_path=args.labels_folder, device=args.device)    
    labs = nets_outputs["test_labels"]
     
    dfs_list = [] 
    if args.outputs_type == 'net':
        print("Processing net outputs")
        for ni, net in enumerate(nets_outputs["networks"]):
            net_df = compute_calibration_plot(prob_pred=nets_outputs["test_outputs"][ni], labs=labs, softmax=True)
            net_df["network"] = net
            dfs_list.append(net_df)
    else:
        print("Processing outputs")
        files = [f for f in os.listdir(args.folder) if os.path.isfile(os.path.join(args.folder, f))]
        valid_files = list(filter(prim_ptrn.match, files))
        if len(valid_files) == 0 and alt_ptrn is not None:
            valid_files = list(filter(alt_ptrn.match, files))
            prim_ptrn = alt_ptrn
        
        print("{} outputs files found".format(len(valid_files)))
        
        if args.outputs_type == "cal_net":
            print("Reading networks order file")
            with open(os.path.join(args.folder, "networks_order.txt")) as f:
                cont = f.read()
                cal_networks = list(filter(None, cont.split("\n")))
        
        for fi, pred_file in enumerate(valid_files):
            print("Processing file {}".format(fi))
            m = re.match(prim_ptrn, pred_file)
            predictions = load_npy_arr(file=os.path.join(args.folder, pred_file), device=args.device)
            if args.outputs_type == "cal_net":
                net_df = []
                for cal_ni, cal_n in enumerate(cal_networks):
                    net_cal_df = compute_calibration_plot(prob_pred=predictions[cal_ni], labs=labs)
                    net_cal_df["network"] = cal_n
                    net_df.append(net_cal_df)
                cal_df = pd.concat(net_df, ignore_index=True)
            else:    
                cal_df = compute_calibration_plot(prob_pred=predictions, labs=labs)
            cal_df = cal_df.assign(**m.groupdict())
            dfs_list.append(cal_df)
        
    if len(dfs_list) == 0:
        return 0
    
    res_df = pd.concat(dfs_list, ignore_index=True)
    res_df.to_csv(os.path.join(args.folder, "cal_plots_{}.csv".format(args.outputs_type)), index=False)    
        
    
if __name__ == '__main__':
    comp_plot_data()