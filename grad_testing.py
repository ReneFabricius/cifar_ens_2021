import torch
import argparse
import os
from itertools import product
import pandas as pd
import numpy as np
from timeit import default_timer as timer

from weensembles.WeightedLinearEnsemble import WeightedLinearEnsemble
from weensembles.CombiningMethods import grad_comb
from weensembles.predictions_evaluation import ECE_sweep, compute_acc_topk, compute_nll
from weensembles.utils import cuda_mem_try

from utils.utils import load_networks_outputs

def test_grad():
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
    exp_outputs = os.path.join(args.repl_folder, "exp_grad_test")
    exp_df = os.path.join(exp_outputs, "grad_testing.csv")
    if not os.path.exists(exp_outputs):
        os.mkdir(exp_outputs)
    
    print("Loading networks outputs")
    net_outputs = load_networks_outputs(nn_outputs_path=nets_outputs, experiment_out_path=exp_outputs, device=args.device)
    c, n, k = net_outputs["train_outputs"].shape
    
    lrs = [float(lr) for lr in args.learning_rates]
    mmts = [float(mmt) for mmt in args.momentums]
    bszs = [int(bsz) for bsz in args.batch_sizes]
    epochs = [int(e) for e in args.epochs]
    
    wle = WeightedLinearEnsemble(c=c, k=k, device=args.device)
    df = pd.DataFrame(columns=("coupling_method_train", "coupling_method_test", 
                               "learning_rate", "momentum", "batch_size", "epochs",
                               "train_acc", "train_nll", "train_ece",
                               "test_acc", "test_nll", "test_ece",
                               "train_time"))
    df_i = 0
    for pars in product(lrs, mmts, bszs, epochs, args.coupling_methods):
        start = timer()
        coefs = grad_comb(X=net_outputs["val_outputs"], y=net_outputs["val_labels"], wle=wle,
                          coupling_method=pars[4], verbose=args.verbose, lr=pars[0], momentum=pars[1],
                          batch_sz=pars[2], epochs=pars[3], test_period=10,
                          return_coefs=True)
        end = timer()
        train_time = end - start
        
        train_string = "lr_{}_mmt_{}_bsz_{}_e_{}_cptr_{}".format(*pars)
        print("Saving coefficients")
        np.save(os.path.join(exp_outputs, "coefs_{}.npy".format(train_string)), coefs.cpu().numpy())
        for cp_m_test in args.coupling_methods:
            print("Testing coupling method {}".format(cp_m_test))
            train_pred = cuda_mem_try(
                    fun=lambda bsz: wle.predict_proba(preds=net_outputs["val_outputs"], l=k, coupling_method=cp_m_test, coefs=coefs,
                                                                verbose=max(args.verbose - 2, 0), batch_size=bsz),
                    start_bsz=net_outputs["val_outputs"].shape[1], verbose=args.verbose, device=args.device)
            test_pred = cuda_mem_try(
                    fun=lambda bsz: wle.predict_proba(preds=net_outputs["test_outputs"], l=k, coupling_method=cp_m_test, coefs=coefs,
                                                                verbose=max(args.verbose - 2, 0), batch_size=bsz),
                    start_bsz=net_outputs["test_outputs"].shape[1], verbose=args.verbose, device=args.device)
            
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
    test_grad()
