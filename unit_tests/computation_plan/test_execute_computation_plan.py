from utils.utils import prepare_computation_plan
import unittest
import pandas as pd
import numpy as np
import torch
import shutil
import os

from utils.utils import load_networks_outputs

class Test_ExecCompPlan(unittest.TestCase):
    def test_empty_folder_cal_plan(self):
        net_outputs = load_networks_outputs(nn_outputs_path="./unit_tests/computation_plan/networks_folder",
                                            experiment_out_path="./unit_tests/computation_plan/outputs_folder_4",
                                            device="cpu",
                                            dtype=torch.float32,
                                            load_train_data=False)

        
        pwc_plan, cal_plan = prepare_computation_plan(
            outputs_folder="./unit_tests/computation_plan/outputs_folder_4",
            networks_names=net_outputs["networks"],
            combination_sizes= [3],
            ens_comb_file=None,
            combining_methods=["logreg"],
            coupling_methods=["bc", "m1"],
            topl_values=[5, -1],
            calibrating_methods=["TemperatureScaling"],
            computational_precisions=["float"],
            loading_existing_models="lazy"
        )
        
        cal_plan.execute_evaluate_save(net_outputs=net_outputs, outputs_folder="./unit_tests/computation_plan/outputs_folder_4",
                                       req_comb_val_data=False)
        
        pwc_plan_new, cal_plan_new = prepare_computation_plan(
            outputs_folder="./unit_tests/computation_plan/outputs_folder_4",
            networks_names=net_outputs["networks"],
            combination_sizes= [3],
            ens_comb_file=None,
            combining_methods=["logreg"],
            coupling_methods=["bc", "m1"],
            topl_values=[5, -1],
            calibrating_methods=["TemperatureScaling"],
            computational_precisions=["float"],
            loading_existing_models="lazy"
        )

        assert(cal_plan_new.plan_.shape[0] == 0)
        shutil.rmtree("./unit_tests/computation_plan/outputs_folder_4")
        os.mkdir("./unit_tests/computation_plan/outputs_folder_4")

    def test_empty_folder_pwc_plan(self):
        net_outputs = load_networks_outputs(nn_outputs_path="./unit_tests/computation_plan/networks_folder",
                                            experiment_out_path="./unit_tests/computation_plan/outputs_folder_4",
                                            device="cpu",
                                            dtype=torch.float32,
                                            load_train_data=False)

        
        pwc_plan, cal_plan = prepare_computation_plan(
            outputs_folder="./unit_tests/computation_plan/outputs_folder_4",
            networks_names=net_outputs["networks"],
            combination_sizes= [3],
            ens_comb_file=None,
            combining_methods=["logreg"],
            coupling_methods=["bc", "m1"],
            topl_values=[5, -1],
            calibrating_methods=["TemperatureScaling"],
            computational_precisions=["float"],
            loading_existing_models="lazy"
        )
        
        pwc_plan.execute_evaluate_save(net_outputs=net_outputs, outputs_folder="./unit_tests/computation_plan/outputs_folder_4",
                                       req_comb_val_data=False)
        
        pwc_plan_new, cal_plan_new = prepare_computation_plan(
            outputs_folder="./unit_tests/computation_plan/outputs_folder_4",
            networks_names=net_outputs["networks"],
            combination_sizes= [3],
            ens_comb_file=None,
            combining_methods=["logreg"],
            coupling_methods=["bc", "m1"],
            topl_values=[5, -1],
            calibrating_methods=["TemperatureScaling"],
            computational_precisions=["float"],
            loading_existing_models="lazy"
        )

        assert(pwc_plan_new.plan_.shape[0] == 0)
        shutil.rmtree("./unit_tests/computation_plan/outputs_folder_4")
        os.mkdir("./unit_tests/computation_plan/outputs_folder_4")
