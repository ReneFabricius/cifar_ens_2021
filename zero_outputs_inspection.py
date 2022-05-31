import numpy as np
import torch
import os

from utils.utils import load_networks_outputs, load_npy_arr

from weensembles.WeightedLinearEnsemble import WeightedLinearEnsemble

def zero_outputs_inspection():
    method = "m1"
    repl = 0
    train_set = "val"
    precision = "float"
    dev = "cuda"
    print_num = 10

    torch.set_printoptions(linewidth=160)
    np.set_printoptions(precision=4, linewidth=160)
    torch_dev = torch.device(dev)
    torch_dtp = torch.float32 if precision == "float" else torch.float64
    base_dir = "D:\\skola\\1\\weighted_ensembles\\tests\\test_cifar_2021\\data\\data_train_val_c10"
    net_outputs_path = os.path.join(base_dir, str(repl), "outputs")
    ens_output = os.path.join(base_dir, str(repl), "comb_outputs",
                              train_set + "_training", "ens_test_outputs_" + method + "_" + precision + ".npy")
    model_file = os.path.join(base_dir, str(repl), "comb_outputs",
                              train_set + "_training", "model_" + precision)

    net_outputs = load_networks_outputs(net_outputs_path, None, dev)

    ens_output = load_npy_arr(ens_output, dev)

    correct_probabilities = ens_output.gather(dim=1, index=net_outputs["test_labels"].unsqueeze(1))

    num_zeros = torch.sum(correct_probabilities == 0).item()

    print("Number of zero probability correct outputs {}".format(num_zeros))

    zero_inds = (correct_probabilities == 0).nonzero(as_tuple=True)[0]

    ens = WeightedLinearEnsemble(device=torch_dev, dtp=torch_dtp)
    ens.load(model_file)

    for i, zero_ind in enumerate(zero_inds[:print_num]):
        nets_out = net_outputs["test_outputs"][:, [zero_ind], :]
        print("Correct label: {}".format(net_outputs["test_labels"][zero_ind].item()))
        for ni, net in enumerate(net_outputs["networks"]):
            print("Network {} prediction:\n{}".format(net, net_outputs["test_outputs"][ni, zero_ind].cpu().numpy()))

        ens_output = ens.predict_proba(nets_out, method, debug_pwcm=True)













