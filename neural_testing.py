import torch
import argparse
import os
from itertools import product
import pandas as pd
import numpy as np
from timeit import default_timer as timer

from weensembles.WeightedLinearEnsemble import WeightedLinearEnsemble
from weensembles.predictions_evaluation import ECE_sweep, compute_acc_topk, compute_nll
from weensembles.utils import cuda_mem_try

from utils import load_networks_outputs

def test_neural():
    parser = argparse.ArgumentParser()
    parser.add_argument("-repl_folder", type=str, help="Training replication folder")
    parser.add_argument("-learning_rates", nargs="+", default=[0.1], help="Learning rates to test")
    parser.add_argument("-momentums", nargs="+", default=[0.9], help="Momentums to test")
    parser.add_argument("-batch_sizes", nargs="+", default=[500], help="Batch sizes to test")
    parser.add_argument("-epochs", nargs="+", default=[10], help="Numbers of epochs to test")
    parser.add_argument("-verbose", type=int, help="Level of verbosity")
    parser.add_argument("-coupling_methods", nargs="+", default=["m2"], help="Coupling methods to test")
    parser.add_argument("-device", type=str, default="cpu", help="device on which to perform computations")
    args = parser.parse_args()
    
    nets_outputs = os.path.join(args.repl_folder, "outputs")
    exp_outputs = os.path.join(args.repl_folder, "exp_neural_test")
    exp_df = os.path.join(exp_outputs, "neural_testing.csv")
    if not os.path.exists(exp_outputs):
        os.mkdir(exp_outputs)
    
    print("Loading networks outputs")
    net_outputs = load_networks_outputs(nn_outputs_path=nets_outputs, experiment_out_path=exp_outputs, device=args.device)
    c, n, k = net_outputs["train_outputs"].shape
    
    lrs = [float(lr) for lr in args.learning_rates]
    mmts = [float(mmt) for mmt in args.momentums]
    bszs = [int(bsz) for bsz in args.batch_sizes]
    epochs = [int(e) for e in args.epochs]
    
    df = pd.DataFrame(columns=("coupling_method_train", "coupling_method_test", 
                               "learning_rate", "momentum", "batch_size", "epochs",
                               "train_acc", "train_nll", "train_ece",
                               "test_acc", "test_nll", "test_ece",
                               "train_time"))
    df_i = 0
    for pars in product(lrs, mmts, bszs, epochs, args.coupling_methods):
        start = timer()
        ens = WeightedLinearEnsemble(c=c, k=k, device=args.device)
        ens.fit(MP=None, tar=None, MP_val=net_outputs["val_outputs"], tar_val=net_outputs["val_labels"], verbose=args.verbose,
                combining_method="neural_" + pars[4], batch_size=pars[2], epochs=pars[3], lr=pars[0], momentum=pars[1],
                test_period=10)
        end = timer()
        train_time = end - start
        
        train_string = "lr_{}_mmt_{}_bsz_{}_e_{}_cptr_{}".format(*pars)
        print("Saving coefficients")
        ens.save(os.path.join(exp_outputs, "model_{}".format(train_string)))
        for cp_m_test in args.coupling_methods:
            print("Testing coupling method {}".format(cp_m_test))
            train_pred = cuda_mem_try(
                    fun=lambda bsz: ens.predict_proba(MP=net_outputs["val_outputs"], l=k, coupling_method=cp_m_test,
                                                                verbose=max(args.verbose - 2, 0), batch_size=bsz),
                    start_bsz=net_outputs["val_outputs"].shape[1], verbose=args.verbose, device=args.device)
            test_pred = cuda_mem_try(
                    fun=lambda bsz: ens.predict_proba(MP=net_outputs["test_outputs"], l=k, coupling_method=cp_m_test,
                                                                verbose=max(args.verbose - 2, 0), batch_size=bsz),
                    start_bsz=net_outputs["test_outputs"].shape[1], verbose=args.verbose, device=args.device)
            
            print("Computing metrics")
            train_acc = compute_acc_topk(pred=train_pred, tar=net_outputs["val_labels"], k=1) 
            train_nll = compute_nll(pred=train_pred, tar=net_outputs["val_labels"])
            train_ece = ECE_sweep(pred=train_pred, tar=net_outputs["val_labels"])
            test_acc = compute_acc_topk(pred=test_pred, tar=net_outputs["test_labels"], k=1)
            test_nll = compute_nll(pred=test_pred, tar=net_outputs["test_labels"])
            test_ece = ECE_sweep(pred=test_pred, tar=net_outputs["test_labels"])
            df.loc[df_i] = [pars[4], cp_m_test,
                            pars[0], pars[1], pars[2], pars[3],
                            train_acc, train_nll, train_ece,
                            test_acc, test_nll, test_ece,
                            train_time]
            df_i += 1
            
            test_string = train_string + "_cpte_{}".format(cp_m_test)
            np.save(os.path.join(exp_outputs, "test_outputs_{}.npy".format(test_string)), test_pred.cpu().numpy())
            
        df.to_csv(exp_df, index=False)
        
        
if __name__ == "__main__":
    test_neural()
