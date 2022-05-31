from utils.utils import prepare_computation_plan
import unittest
import pandas as pd
import numpy as np

class Test_PrepCompPlan(unittest.TestCase):
    def test_empty_folder_pwc_plan(self):
        pwc_plan, cal_plan, pwc_metric, cal_metric = prepare_computation_plan(
            outputs_folder="./unit_tests/computation_plan/outputs_folder_2",
            networks_names=["clip_ViT-B-16_LP", "densenet201", "resnet16"],
            combination_sizes= [2, 3],
            ens_comb_file=None,
            combining_methods=["logreg"],
            coupling_methods=["bc", "m1"],
            topl_values=[5, 10, -1],
            calibrating_methods=["TemperatureScaling"],
            computational_precisions=["float"],
            loading_existing_models="lazy"
        )
        
        pwc_data = [[True, True, True, 1, 'logreg', 'bc', 5, 'float', np.nan],
                    [True, True, True, 1, 'logreg', 'bc', 10, 'float', np.nan],
                    [True, True, True, 1, 'logreg', 'bc', -1, 'float', np.nan],
                    [True, True, True, 1, 'logreg', 'm1', 5, 'float', np.nan],
                    [True, True, True, 1, 'logreg', 'm1', 10, 'float', np.nan],
                    [True, True, True, 1, 'logreg', 'm1', -1, 'float', np.nan],
                    [True, False, True, 2, 'logreg', 'bc', 5, 'float', np.nan],
                    [True, False, True, 2, 'logreg', 'bc', 10, 'float', np.nan],
                    [True, False, True, 2, 'logreg', 'bc', -1, 'float', np.nan],
                    [True, False, True, 2, 'logreg', 'm1', 5, 'float', np.nan],
                    [True, False, True, 2, 'logreg', 'm1', 10, 'float', np.nan],
                    [True, False, True, 2, 'logreg', 'm1', -1, 'float', np.nan],
                    [True, True, False, 3, 'logreg', 'bc', 5, 'float', np.nan],
                    [True, True, False, 3, 'logreg', 'bc', 10, 'float', np.nan],
                    [True, True, False, 3, 'logreg', 'bc', -1, 'float', np.nan],
                    [True, True, False, 3, 'logreg', 'm1', 5, 'float', np.nan],
                    [True, True, False, 3, 'logreg', 'm1', 10, 'float', np.nan],
                    [True, True, False, 3, 'logreg', 'm1', -1, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'bc', 5, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'bc', 10, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'bc', -1, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'm1', 5, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'm1', 10, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'm1', -1, 'float', np.nan]]
        pwc_columns = ['clip_ViT-B-16_LP', 'densenet201', 'resnet16', 'combination_id',
                       'combining_method', 'coupling_method', 'topl',
                       'computational_precision', 'model_file']
        
        exp_pwc_plan = pd.DataFrame(data=pwc_data, columns=pwc_columns)
        ret_pwc_plan = pwc_plan.plan_
        
        compared_columns = list(exp_pwc_plan.columns)
        compared_columns.remove("combination_id")
        
        ret_pwc_plan.sort_values(by=compared_columns, ignore_index=True, inplace=True)
        exp_pwc_plan.sort_values(by=compared_columns, ignore_index=True, inplace=True)
        pwc_comparison = exp_pwc_plan[compared_columns].compare(ret_pwc_plan[compared_columns])
        
        assert(pwc_comparison.shape == (0, 0))
    
    def test_empty_folder_cal_plan(self):
        pwc_plan, cal_plan, pwc_metric, cal_metric = prepare_computation_plan(
            outputs_folder="./unit_tests/computation_plan/outputs_folder_2",
            networks_names=["clip_ViT-B-16_LP", "densenet201", "resnet16"],
            combination_sizes= [2, 3],
            ens_comb_file=None,
            combining_methods=["logreg"],
            coupling_methods=["bc", "m1"],
            topl_values=[5, 10, -1],
            calibrating_methods=["TemperatureScaling"],
            computational_precisions=["float"],
            loading_existing_models="lazy"
        )
        
        cal_data = [[True, True, True, 1, 'TemperatureScaling', 'float', np.nan],
                    [True, False, True, 2, 'TemperatureScaling', 'float', np.nan],
                    [True, True, False, 3, 'TemperatureScaling', 'float', np.nan],
                    [False, True, True, 4, 'TemperatureScaling', 'float', np.nan]]
        cal_columns = ['clip_ViT-B-16_LP', 'densenet201', 'resnet16', 'combination_id',
                       'calibrating_method', 'computational_precision', 'model_file']
        
        exp_cal_plan = pd.DataFrame(data=cal_data, columns=cal_columns)
        ret_cal_plan = cal_plan.plan_
        
        compared_columns = list(exp_cal_plan.columns)
        compared_columns.remove("combination_id")
        
        ret_cal_plan.sort_values(by=compared_columns, ignore_index=True, inplace=True)
        exp_cal_plan.sort_values(by=compared_columns, ignore_index=True, inplace=True)
        cal_comparison = exp_cal_plan[compared_columns].compare(ret_cal_plan[compared_columns])
        
        assert(cal_comparison.shape == (0, 0))

    def test_existing_models_folder_pwc_plan(self):
        pwc_plan, cal_plan, pwc_metric, cal_metric = prepare_computation_plan(
            outputs_folder="./unit_tests/computation_plan/outputs_folder_1",
            networks_names=["clip_ViT-B-16_LP", "densenet201", "resnet16"],
            combination_sizes= [2, 3],
            ens_comb_file=None,
            combining_methods=["logreg"],
            coupling_methods=["bc", "m1"],
            topl_values=[5, 10, -1],
            calibrating_methods=["TemperatureScaling"],
            computational_precisions=["float"],
            loading_existing_models="lazy"
        )
        
        pwc_data = [[True, True, True, 1, 'logreg', 'bc', 5, 'float', "resnet16+clip_ViT-B-16_LP+densenet201_model_co_m_logreg_prec_float"],
                    [True, True, True, 1, 'logreg', 'bc', 10, 'float', "resnet16+clip_ViT-B-16_LP+densenet201_model_co_m_logreg_prec_float"],
                    [True, True, True, 1, 'logreg', 'bc', -1, 'float', "resnet16+clip_ViT-B-16_LP+densenet201_model_co_m_logreg_prec_float"],
                    [True, True, True, 1, 'logreg', 'm1', 5, 'float', "resnet16+clip_ViT-B-16_LP+densenet201_model_co_m_logreg_prec_float"],
                    [True, True, True, 1, 'logreg', 'm1', 10, 'float', "resnet16+clip_ViT-B-16_LP+densenet201_model_co_m_logreg_prec_float"],
                    [True, True, True, 1, 'logreg', 'm1', -1, 'float', "resnet16+clip_ViT-B-16_LP+densenet201_model_co_m_logreg_prec_float"],
                    [True, False, True, 2, 'logreg', 'bc', 5, 'float', "resnet16+clip_ViT-B-16_LP_model_co_m_logreg_prec_float"],
                    [True, False, True, 2, 'logreg', 'bc', 10, 'float', "resnet16+clip_ViT-B-16_LP_model_co_m_logreg_prec_float"],
                    [True, False, True, 2, 'logreg', 'bc', -1, 'float', "resnet16+clip_ViT-B-16_LP_model_co_m_logreg_prec_float"],
                    [True, False, True, 2, 'logreg', 'm1', 5, 'float', "resnet16+clip_ViT-B-16_LP_model_co_m_logreg_prec_float"],
                    [True, False, True, 2, 'logreg', 'm1', 10, 'float', "resnet16+clip_ViT-B-16_LP_model_co_m_logreg_prec_float"],
                    [True, False, True, 2, 'logreg', 'm1', -1, 'float', "resnet16+clip_ViT-B-16_LP_model_co_m_logreg_prec_float"],
                    [True, True, False, 3, 'logreg', 'bc', 5, 'float', np.nan],
                    [True, True, False, 3, 'logreg', 'bc', 10, 'float', np.nan],
                    [True, True, False, 3, 'logreg', 'bc', -1, 'float', np.nan],
                    [True, True, False, 3, 'logreg', 'm1', 5, 'float', np.nan],
                    [True, True, False, 3, 'logreg', 'm1', 10, 'float', np.nan],
                    [True, True, False, 3, 'logreg', 'm1', -1, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'bc', 5, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'bc', 10, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'bc', -1, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'm1', 5, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'm1', 10, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'm1', -1, 'float', np.nan]]
        pwc_columns = ['clip_ViT-B-16_LP', 'densenet201', 'resnet16', 'combination_id',
                       'combining_method', 'coupling_method', 'topl',
                       'computational_precision', 'model_file']
        
        exp_pwc_plan = pd.DataFrame(data=pwc_data, columns=pwc_columns)
        ret_pwc_plan = pwc_plan.plan_
        
        compared_columns = list(exp_pwc_plan.columns)
        compared_columns.remove("combination_id")
        
        ret_pwc_plan.sort_values(by=compared_columns, ignore_index=True, inplace=True)
        exp_pwc_plan.sort_values(by=compared_columns, ignore_index=True, inplace=True)
        pwc_comparison = exp_pwc_plan[compared_columns].compare(ret_pwc_plan[compared_columns])
        
        assert(pwc_comparison.shape == (0, 0))
    
    def test_existing_models_folder_cal_plan(self):
        pwc_plan, cal_plan, pwc_metric, cal_metric = prepare_computation_plan(
            outputs_folder="./unit_tests/computation_plan/outputs_folder_1",
            networks_names=["clip_ViT-B-16_LP", "densenet201", "resnet16"],
            combination_sizes= [2, 3],
            ens_comb_file=None,
            combining_methods=["logreg"],
            coupling_methods=["bc", "m1"],
            topl_values=[5, 10, -1],
            calibrating_methods=["TemperatureScaling"],
            computational_precisions=["float"],
            loading_existing_models="lazy"
        )
        
        cal_data = [[True, True, True, 1, 'TemperatureScaling', 'float', np.nan],
                    [True, False, True, 2, 'TemperatureScaling', 'float', np.nan],
                    [True, True, False, 3, 'TemperatureScaling', 'float', np.nan],
                    [False, True, True, 4, 'TemperatureScaling', 'float', "densenet201+resnet16_model_cal_m_TemperatureScaling_prec_float"]]
        cal_columns = ['clip_ViT-B-16_LP', 'densenet201', 'resnet16', 'combination_id',
                       'calibrating_method', 'computational_precision', 'model_file']
        
        exp_cal_plan = pd.DataFrame(data=cal_data, columns=cal_columns)
        ret_cal_plan = cal_plan.plan_
        
        compared_columns = list(exp_cal_plan.columns)
        compared_columns.remove("combination_id")
        
        ret_cal_plan.sort_values(by=compared_columns, ignore_index=True, inplace=True)
        exp_cal_plan.sort_values(by=compared_columns, ignore_index=True, inplace=True)
        cal_comparison = exp_cal_plan[compared_columns].compare(ret_cal_plan[compared_columns])
        
        assert(cal_comparison.shape == (0, 0))
 
    def test_existing_models_existing_metrics_folder_pwc_plan(self):
        pwc_plan, cal_plan, pwc_metric, cal_metric = prepare_computation_plan(
            outputs_folder="./unit_tests/computation_plan/outputs_folder_3",
            networks_names=["clip_ViT-B-16_LP", "densenet201", "resnet16"],
            combination_sizes= [2, 3],
            ens_comb_file=None,
            combining_methods=["logreg"],
            coupling_methods=["bc", "m1"],
            topl_values=[5, 10, -1],
            calibrating_methods=["TemperatureScaling"],
            computational_precisions=["float"],
            loading_existing_models="lazy"
        )
        
        pwc_data = [[True, True, True, 1, 'logreg', 'bc', 5, 'float', "resnet16+clip_ViT-B-16_LP+densenet201_model_co_m_logreg_prec_float"],
                    [True, True, True, 1, 'logreg', 'bc', 10, 'float', "resnet16+clip_ViT-B-16_LP+densenet201_model_co_m_logreg_prec_float"],
                    [True, True, True, 1, 'logreg', 'bc', -1, 'float', "resnet16+clip_ViT-B-16_LP+densenet201_model_co_m_logreg_prec_float"],
                    [True, True, True, 1, 'logreg', 'm1', 5, 'float', "resnet16+clip_ViT-B-16_LP+densenet201_model_co_m_logreg_prec_float"],
                    [True, True, True, 1, 'logreg', 'm1', 10, 'float', "resnet16+clip_ViT-B-16_LP+densenet201_model_co_m_logreg_prec_float"],
                    [True, True, True, 1, 'logreg', 'm1', -1, 'float', "resnet16+clip_ViT-B-16_LP+densenet201_model_co_m_logreg_prec_float"],
                    [True, False, True, 2, 'logreg', 'bc', 5, 'float', "resnet16+clip_ViT-B-16_LP_model_co_m_logreg_prec_float"],
                    [True, False, True, 2, 'logreg', 'bc', 10, 'float', "resnet16+clip_ViT-B-16_LP_model_co_m_logreg_prec_float"],
                    [True, False, True, 2, 'logreg', 'bc', -1, 'float', "resnet16+clip_ViT-B-16_LP_model_co_m_logreg_prec_float"],
                    [True, False, True, 2, 'logreg', 'm1', 5, 'float', "resnet16+clip_ViT-B-16_LP_model_co_m_logreg_prec_float"],
                    [True, False, True, 2, 'logreg', 'm1', 10, 'float', "resnet16+clip_ViT-B-16_LP_model_co_m_logreg_prec_float"],
                    [True, False, True, 2, 'logreg', 'm1', -1, 'float', "resnet16+clip_ViT-B-16_LP_model_co_m_logreg_prec_float"],
                    [True, True, False, 3, 'logreg', 'bc', 5, 'float', "densenet201+clip_ViT-B-16_LP_model_co_m_logreg_prec_float"],
                    [True, True, False, 3, 'logreg', 'bc', 10, 'float', "densenet201+clip_ViT-B-16_LP_model_co_m_logreg_prec_float"],
                    [True, True, False, 3, 'logreg', 'm1', 5, 'float', "densenet201+clip_ViT-B-16_LP_model_co_m_logreg_prec_float"],
                    [True, True, False, 3, 'logreg', 'm1', 10, 'float', "densenet201+clip_ViT-B-16_LP_model_co_m_logreg_prec_float"],
                    [False, True, True, 4, 'logreg', 'bc', 5, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'bc', 10, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'bc', -1, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'm1', 5, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'm1', 10, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'm1', -1, 'float', np.nan]]
        pwc_columns = ['clip_ViT-B-16_LP', 'densenet201', 'resnet16', 'combination_id',
                       'combining_method', 'coupling_method', 'topl',
                       'computational_precision', 'model_file']
        
        exp_pwc_plan = pd.DataFrame(data=pwc_data, columns=pwc_columns)
        ret_pwc_plan = pwc_plan.plan_
        
        compared_columns = list(exp_pwc_plan.columns)
        compared_columns.remove("combination_id")
        
        ret_pwc_plan.sort_values(by=compared_columns, ignore_index=True, inplace=True)
        exp_pwc_plan.sort_values(by=compared_columns, ignore_index=True, inplace=True)
        pwc_comparison = exp_pwc_plan[compared_columns].compare(ret_pwc_plan[compared_columns])
        
        assert(pwc_comparison.shape == (0, 0))
    
    def test_existing_models_existing_metrics_folder_cal_plan(self):
        pwc_plan, cal_plan, pwc_metric, cal_metric = prepare_computation_plan(
            outputs_folder="./unit_tests/computation_plan/outputs_folder_3",
            networks_names=["clip_ViT-B-16_LP", "densenet201", "resnet16"],
            combination_sizes= [2, 3],
            ens_comb_file=None,
            combining_methods=["logreg"],
            coupling_methods=["bc", "m1"],
            topl_values=[5, 10, -1],
            calibrating_methods=["TemperatureScaling"],
            computational_precisions=["float"],
            loading_existing_models="lazy"
        )
        
        cal_data = [[True, True, True, 1, 'TemperatureScaling', 'float', np.nan],
                    [True, False, True, 2, 'TemperatureScaling', 'float', np.nan],
                    [False, True, True, 4, 'TemperatureScaling', 'float', "densenet201+resnet16_model_cal_m_TemperatureScaling_prec_float"]]
        cal_columns = ['clip_ViT-B-16_LP', 'densenet201', 'resnet16', 'combination_id',
                       'calibrating_method', 'computational_precision', 'model_file']
        
        exp_cal_plan = pd.DataFrame(data=cal_data, columns=cal_columns)
        ret_cal_plan = cal_plan.plan_
        
        compared_columns = list(exp_cal_plan.columns)
        compared_columns.remove("combination_id")
        
        ret_cal_plan.sort_values(by=compared_columns, ignore_index=True, inplace=True)
        exp_cal_plan.sort_values(by=compared_columns, ignore_index=True, inplace=True)
        cal_comparison = exp_cal_plan[compared_columns].compare(ret_cal_plan[compared_columns])
        
        assert(cal_comparison.shape == (0, 0))
 
    def test_existing_models_existing_metrics_folder_specified_combinations_pwc_plan(self):
        pwc_plan, cal_plan, pwc_metric, cal_metric = prepare_computation_plan(
            outputs_folder="./unit_tests/computation_plan/outputs_folder_3",
            networks_names=["clip_ViT-B-16_LP", "densenet201", "resnet16"],
            combination_sizes= [3],
            ens_comb_file="./unit_tests/computation_plan/combinations.txt",
            combining_methods=["logreg"],
            coupling_methods=["bc", "m1"],
            topl_values=[5, 10, -1],
            calibrating_methods=["TemperatureScaling"],
            computational_precisions=["float"],
            loading_existing_models="lazy"
        )
        
        pwc_data = [[True, True, True, 1, 'logreg', 'bc', 5, 'float', "resnet16+clip_ViT-B-16_LP+densenet201_model_co_m_logreg_prec_float"],
                    [True, True, True, 1, 'logreg', 'bc', 10, 'float', "resnet16+clip_ViT-B-16_LP+densenet201_model_co_m_logreg_prec_float"],
                    [True, True, True, 1, 'logreg', 'bc', -1, 'float', "resnet16+clip_ViT-B-16_LP+densenet201_model_co_m_logreg_prec_float"],
                    [True, True, True, 1, 'logreg', 'm1', 5, 'float', "resnet16+clip_ViT-B-16_LP+densenet201_model_co_m_logreg_prec_float"],
                    [True, True, True, 1, 'logreg', 'm1', 10, 'float', "resnet16+clip_ViT-B-16_LP+densenet201_model_co_m_logreg_prec_float"],
                    [True, True, True, 1, 'logreg', 'm1', -1, 'float', "resnet16+clip_ViT-B-16_LP+densenet201_model_co_m_logreg_prec_float"],
                    [False, True, True, 4, 'logreg', 'bc', 5, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'bc', 10, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'bc', -1, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'm1', 5, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'm1', 10, 'float', np.nan],
                    [False, True, True, 4, 'logreg', 'm1', -1, 'float', np.nan]]
        pwc_columns = ['clip_ViT-B-16_LP', 'densenet201', 'resnet16', 'combination_id',
                       'combining_method', 'coupling_method', 'topl',
                       'computational_precision', 'model_file']
        
        exp_pwc_plan = pd.DataFrame(data=pwc_data, columns=pwc_columns)
        ret_pwc_plan = pwc_plan.plan_
        
        compared_columns = list(exp_pwc_plan.columns)
        compared_columns.remove("combination_id")
        
        ret_pwc_plan.sort_values(by=compared_columns, ignore_index=True, inplace=True)
        exp_pwc_plan.sort_values(by=compared_columns, ignore_index=True, inplace=True)
        pwc_comparison = exp_pwc_plan[compared_columns].compare(ret_pwc_plan[compared_columns])
        
        assert(pwc_comparison.shape == (0, 0))
    
    def test_existing_models_existing_metrics_folder_specified_combinations_cal_plan(self):
        pwc_plan, cal_plan, pwc_metric, cal_metric = prepare_computation_plan(
            outputs_folder="./unit_tests/computation_plan/outputs_folder_3",
            networks_names=["clip_ViT-B-16_LP", "densenet201", "resnet16"],
            combination_sizes= [3],
            ens_comb_file="./unit_tests/computation_plan/combinations.txt",
            combining_methods=["logreg"],
            coupling_methods=["bc", "m1"],
            topl_values=[5, 10, -1],
            calibrating_methods=["TemperatureScaling"],
            computational_precisions=["float"],
            loading_existing_models="lazy"
        )
        
        cal_data = [[True, True, True, 1, 'TemperatureScaling', 'float', np.nan],
                    [False, True, True, 4, 'TemperatureScaling', 'float', "densenet201+resnet16_model_cal_m_TemperatureScaling_prec_float"]]
        cal_columns = ['clip_ViT-B-16_LP', 'densenet201', 'resnet16', 'combination_id',
                       'calibrating_method', 'computational_precision', 'model_file']
        
        exp_cal_plan = pd.DataFrame(data=cal_data, columns=cal_columns)
        ret_cal_plan = cal_plan.plan_
        
        compared_columns = list(exp_cal_plan.columns)
        compared_columns.remove("combination_id")
        
        ret_cal_plan.sort_values(by=compared_columns, ignore_index=True, inplace=True)
        exp_cal_plan.sort_values(by=compared_columns, ignore_index=True, inplace=True)
        cal_comparison = exp_cal_plan[compared_columns].compare(ret_cal_plan[compared_columns])
        
        assert(cal_comparison.shape == (0, 0))

    def test_existing_models_existing_metrics_folder_pwc_metrics(self):
        pwc_plan, cal_plan, pwc_metric, cal_metric = prepare_computation_plan(
            outputs_folder="./unit_tests/computation_plan/outputs_folder_3",
            networks_names=["clip_ViT-B-16_LP", "densenet201", "resnet16"],
            combination_sizes= [2, 3],
            ens_comb_file=None,
            combining_methods=["logreg"],
            coupling_methods=["bc", "m1"],
            topl_values=[5, 10, -1],
            calibrating_methods=["TemperatureScaling"],
            computational_precisions=["float"],
            loading_existing_models="lazy"
        )
        
        pwc_data = [["logreg_torch",	"m1",0.78716,1.2019379138946533,0.30336296558380127,True,True,	False,2,1,0.18435999751091003,0.663,0.0002651164832059294],
                    ["logreg_torch",	"m2",0.80172,1.0294718742370605,0.20988452434539795,True,True,	False,2,1,0.18435999751091003,0.663,0.0002651164832059294],
                    ["logreg_torch",	"bc",0.74712,1.4554779529571533,0.3728276491165161,True,True,	False,2,1,0.18435999751091003,0.663,0.0002651164832059294],
                    ["logreg",		"m1",0.7872,1.2047868967056274,0.30450013279914856,True,True,	False,2,1,0.18435999751091003,0.663,0.0002651164832059294],
                    ["logreg",		"m2",0.80214,1.0277280807495117,0.20963628590106964,True,True,	False,2,1,0.18435999751091003,0.663,0.0002651164832059294],
                    ["logreg",		"bc",0.74384,1.4622012376785278,0.37030452489852905,True,True,	False,2,1,0.18435999751091003,0.663,0.0002651164832059294],
                    ["average",		"m1",0.80474,0.7484594583511353,0.034744199365377426,True,True,	False,2,1,0.18435999751091003,0.663,0.0002651164832059294],
                    ["average",		"m2",0.80474,0.7483724355697632,0.034555718302726746,True,True,	False,2,1,0.18435999751091003,0.663,0.0002651164832059294],
                    ["average",		"bc",0.8047,1.467559576034546,0.4871610999107361,True,True,		False,2,1,0.18435999751091003,0.663,0.0002651164832059294],
                    ["cal_average",	"m1",0.80304,0.8969166278839111,0.11197781562805176,True,True,	False,2,1,0.18435999751091003,0.663,0.0002651164832059294],
                    ["cal_average",	"m2",0.80304,0.8956654071807861,0.11254053562879562,True,True,	False,2,1,0.18435999751091003,0.663,0.0002651164832059294],
                    ["cal_average",	"bc",0.80322,2.6419050693511963,0.7175244688987732,True,True,	False,2,1,0.18435999751091003,0.663,0.0002651164832059294],
                    ["grad_m2",		"m1",0.80566,0.7480303645133972,0.026364009827375412,True,True,	False,2,1,0.18435999751091003,0.663,0.0002651164832059294],
                    ["grad_m2",		"m2",0.80566,0.7455103993415833,0.027345161885023117,True,True,	False,2,1,0.18435999751091003,0.663,0.0002651164832059294],
                    ["grad_m2",		"bc",0.80492,1.4517050981521606,0.4821453094482422,True,True,	False,2,1,0.18435999751091003,0.663,0.0002651164832059294]]
        pwc_columns = ["combining_method", "coupling_method", "accuracy", "nll",
                       "ece", "densenet201", "clip_ViT-B-16_LP", "resnet16",
                       "combination_size", "combination_id", "err_incons",
                       "all_cor", "mean_pwa_var"]
        
        non_float_columns = ["combining_method", "coupling_method", "densenet201",
                             "clip_ViT-B-16_LP", "resnet16",
                             "combination_size"]
        
        exp_pwc_metrics = pd.DataFrame(data=pwc_data, columns=pwc_columns)
        ret_pwc_metrics = pwc_metric
        
        compared_columns = list(exp_pwc_metrics.columns)
        compared_columns.remove("combination_id")
        ret_pwc_metrics.sort_values(by=compared_columns, ignore_index=True, inplace=True)
        exp_pwc_metrics.sort_values(by=compared_columns, ignore_index=True, inplace=True)

        compared_float_columns = [col for col in compared_columns if col not in non_float_columns]
        
        pwc_float_comparison = np.isclose(ret_pwc_metrics[compared_float_columns], exp_pwc_metrics[compared_float_columns])
        pwc_float_comp_differences = np.size(pwc_float_comparison) - np.count_nonzero(pwc_float_comparison)
        
        pwc_comparison = exp_pwc_metrics[non_float_columns].compare(ret_pwc_metrics[non_float_columns])
        pwc_comp_differences = np.size(pwc_comparison)
        
        assert(pwc_comp_differences + pwc_float_comp_differences == 0)
    
    def test_existing_models_existing_metrics_folder_cal_metrics(self):
        pwc_plan, cal_plan, pwc_metric, cal_metric = prepare_computation_plan(
            outputs_folder="./unit_tests/computation_plan/outputs_folder_3",
            networks_names=["clip_ViT-B-16_LP", "densenet201", "resnet16"],
            combination_sizes= [2, 3],
            ens_comb_file=None,
            combining_methods=["logreg"],
            coupling_methods=["bc", "m1"],
            topl_values=[5, 10, -1],
            calibrating_methods=["TemperatureScaling"],
            computational_precisions=["float"],
            loading_existing_models="lazy"
        )
        
        cal_data = [["TemperatureScaling",0.79772,0.7975190877914429,0.12244042009115219,True,True,False,2,1,0.18435999751091003,0.663,0.0002651164832059294]]
        cal_columns = ["calibrating_method",
                       "accuracy", "nll", "ece",
                       "densenet201", "clip_ViT-B-16_LP", "resnet16",
                       "combination_size", "combination_id", 
                       "err_incons", "all_cor", "mean_pwa_var"]
        
        non_float_columns = ["calibrating_method", "densenet201",
                             "clip_ViT-B-16_LP", "resnet16",
                             "combination_size"]
        
        exp_cal_metrics = pd.DataFrame(data=cal_data, columns=cal_columns)
        ret_cal_metrics = cal_metric
        
        compared_columns = list(exp_cal_metrics.columns)
        compared_columns.remove("combination_id")
        ret_cal_metrics.sort_values(by=compared_columns, ignore_index=True, inplace=True)
        exp_cal_metrics.sort_values(by=compared_columns, ignore_index=True, inplace=True)

        compared_float_columns = [col for col in compared_columns if col not in non_float_columns]
        
        cal_float_comparison = np.isclose(ret_cal_metrics[compared_float_columns], exp_cal_metrics[compared_float_columns])
        cal_float_comp_differences = np.size(cal_float_comparison) - np.count_nonzero(cal_float_comparison)
        
        cal_comparison = exp_cal_metrics[non_float_columns].compare(ret_cal_metrics[non_float_columns])
        cal_comp_differences = np.size(cal_comparison)
        
        assert(cal_comp_differences + cal_float_comp_differences == 0)

    def test_existing_models_existing_metrics_folder_empty_plans(self):
            pwc_plan, cal_plan, pwc_metric, cal_metric = prepare_computation_plan(
                outputs_folder="./unit_tests/computation_plan/outputs_folder_3",
                networks_names=["clip_ViT-B-16_LP", "densenet201", "resnet16"],
                combination_sizes= [2, 3],
                ens_comb_file=None,
                combining_methods=[],
                coupling_methods=["bc", "m1"],
                topl_values=[5, 10, -1],
                calibrating_methods=[],
                computational_precisions=["float"],
                loading_existing_models="lazy"
            )
            
            assert(pwc_plan.plan_.size == 0)
            assert(cal_plan.plan_.size == 0)
        

if __name__ == "__main__":
    unittest.main()