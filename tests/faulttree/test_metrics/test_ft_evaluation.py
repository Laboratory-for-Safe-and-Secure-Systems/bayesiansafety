import math

from unittest.mock import patch
import pytest

from bayesiansafety.faulttree import Evaluation
#fixtures provided via conftest


def test_cutsets_probabilities_elevator(fixture_ft_elevator_model):

    # (set, val, tolerance)
    correct_probabilities =[({"IHF_LOG_FA"}, 1.199e-3, 1e-5), ({"IHF_LS_FP", "IHF_SIG_BS_FP"}, 7.957e-6, 1e-8), ({"EMI_LS_FP", "IHF_BS_FP"}, 5.554e-10, 1e-12), ({"IHF_LS_FP", "EMI_BS_FP"}, 2.223e-10, 1e-12)]

    model, _ = fixture_ft_elevator_model
    evaluator = Evaluation(model)
    candidate_cutsets = evaluator.get_cutsets_with_probabilities(at_time=0)

    for correct_pair in correct_probabilities:
        for candidate_pair in candidate_cutsets.items():
            if len(correct_pair[0] ^ candidate_pair[0]) == 0:

                assert math.isclose(candidate_pair[1], correct_pair[1], abs_tol=correct_pair[2])


def test_birnbaum_importance_elevator(fixture_ft_elevator_model):
    #correct_importance (name, value, tolerance)
    correct_importances =[("IHF_LOG_FA", 1.0, 0.0), ("EMI_LOG_FA", 1.0, 0.0), ("SYS_LOG_FA", 1.0, 0.0), ("IHF_LS_FP", 1.195e-2, 2e-6) , ("IHF_SIG_LS_FP", 1.195e-2, 2e-6),
                            ("EMI_SIG_LS_FP", 1.195e-2, 2e-6), ("EMI_LS_FP", 1.195e-2, 2e-6), ("IHF_SIG_BS_FP", 8.102e-4, 2e-6), ("IHF_BS_FP", 8.102e-4, 2e-6), ("EMI_BS_FP", 8.102e-4, 2e-6),
                            ("EMI_SIG_BS_FP", 8.102e-4, 2e-6), ("IHF_SIG_LO_FP", 1.028e-5, 2e-8), ("IHF_SIG_HI_FP", 1.028e-5, 2e-8), ("EMI_SIG_LO_FP", 1.028e-5, 2e-8), ("EMI_SIG_HI_FP", 1.028e-5, 2e-8)]

    model, _ = fixture_ft_elevator_model
    evaluator = Evaluation(model)
    candidate_importances = evaluator.get_importances(at_time=0, importance_type="birnbaum")

    for correct_pair in correct_importances:
        for candidate_pair in candidate_importances:
            if correct_pair[0] == candidate_pair[0]:
                assert math.isclose(candidate_pair[1], correct_pair[1], abs_tol=correct_pair[2])


def test_fussell_vesely_importance_elevator(fixture_ft_elevator_model):
    #correct_importance (name, value, tolerance)
    correct_importances =[("IHF_LOG_FA", 9.918e-1, 3e-4), ("IHF_LS_FP", 7.902e-3, 3e-6), ("IHF_SIG_BS_FP", 6.667e-3, 3e-6), ("IHF_BS_FP", 1.339e-3, 3e-6), ("EMI_LOG_FA", 2.299e-4, 3e-7),
                        ("IHF_SIG_LS_FP", 9.881e-5, 3e-8), ("EMI_SIG_LS_FP", 2.747e-6, 3e-9), ("EMI_LS_FP", 2.747e-6, 3e-9), ("EMI_BS_FP", 1.863e-7, 3e-10), ("EMI_SIG_BS_FP", 1.863e-7, 3e-10),
                        ("IHF_SIG_LO_FP", 8.499e-8, 3e-11), ("IHF_SIG_HI_FP", 8.499e-8, 3e-11), ("EMI_SIG_LO_FP", 2.363e-9, 3e-12), ("EMI_SIG_HI_FP", 2.363e-9, 3e-12), ("SYS_LOG_FA", 0.0, 1e-12) ]

    model, _ = fixture_ft_elevator_model
    evaluator = Evaluation(model)
    candidate_importances = evaluator.get_importances(at_time=0, importance_type="fussel_vesely")

    for correct_pair in correct_importances:
        for candidate_pair in candidate_importances:
            if correct_pair[0] == candidate_pair[0]:
                assert math.isclose(candidate_pair[1], correct_pair[1], abs_tol=correct_pair[2])


def test_risk_reduction_worth_elevator(fixture_ft_elevator_model):
    # (name, value, tolerance)
    correct_worths =[("IHF_LOG_FA", 121.417017497818, 2e-1), ("IHF_LS_FP", 1.00796472414712, 1e-3), ("IHF_SIG_BS_FP", 1.0067117323905, 1e-3), ("IHF_BS_FP", 1.00134053060509, 1e-3),
                    ("EMI_LOG_FA", 1.0002299493893, 1e-4), ("IHF_SIG_LS_FP", 1.00009882114207, 1e-5), ("EMI_SIG_LS_FP", 1.000002746977577, 1e-6), ("EMI_LS_FP", 1.00000274697757, 1e-6),
                    ("EMI_BS_FP", 1.00000018627048, 1e-7), ("EMI_SIG_BS_FP", 1.00000018627048, 1e-7), ("IHF_SIG_LO_FP", 1.00000008499473, 1e-8), ("IHF_SIG_HI_FP", 1.00000008499473, 1e-8),
                    ("EMI_SIG_LO_FP", 1.00000000236287, 1e-9), ("EMI_SIG_HI_FP", 1.00000000236287, 1e-9), ("SYS_LOG_FA", 1.0, 0.0)]

    model, _ = fixture_ft_elevator_model
    evaluator = Evaluation(model)
    candidate_worths = evaluator.get_risk_worths(method='rrw', at_time=0, scoped_event=None)

    for correct_pair in correct_worths:
        for candidate_pair in candidate_worths:
            if correct_pair[0] == candidate_pair[0]:
                assert math.isclose(candidate_pair[1], correct_pair[1], abs_tol=correct_pair[2])


def test_risk_achievement_worth_elevator(fixture_ft_elevator_model):
    # (name, value, tolerance)
    correct_worths =[("EMI_LOG_FA", 828.0, 1.0), ("SYS_LOG_FA", 828.0, 1.0), ("IHF_LOG_FA", 827.0, 1.0), ("EMI_SIG_LS_FP", 10.88, 1e-1), ("EMI_LS_FP", 10.88, 1e-1), ("IHF_SIG_LS_FP", 10.88, 1e-1),
                      ("IHF_LS_FP", 10.88, 1e-1), ("EMI_BS_FP", 1.670, 1e-2), ("EMI_SIG_BS_FP", 1.670, 1e-2), ("IHF_BS_FP", 1.669, 1e-2), ("IHF_SIG_BS_FP", 1.663, 1e-2), ("EMI_SIG_LO_FP", 1.008, 1e-3),
                      ("EMI_SIG_HI_FP", 1.008, 1e-3), ("IHF_SIG_HI_FP", 1.008, 1e-3), ("IHF_SIG_LO_FP", 1.008, 1e-3) ]

    model, _ = fixture_ft_elevator_model
    evaluator = Evaluation(model)
    candidate_worths = evaluator.get_risk_worths(method='raw', at_time=0, scoped_event=None)

    for correct_pair in correct_worths:
        for candidate_pair in candidate_worths:
            if correct_pair[0] == candidate_pair[0]:
                assert math.isclose(candidate_pair[1], correct_pair[1], abs_tol=correct_pair[2])



def test_get_risk_worths_wrong_method_raises_exception(fixture_ft_and_only_model):
    bad_method = "bad"
    expected_exc_substring = "Invalid risk analysis specified"
    model, _ = fixture_ft_and_only_model
    evaluator = Evaluation(model)

    with pytest.raises(Exception) as e:
        assert evaluator.get_risk_worths(method=bad_method, at_time=0, scoped_event=None)

    assert expected_exc_substring in str(e.value)


def test_get_importances_wrong_metric_raises_exception(fixture_ft_and_only_model):
    bad_method = "bad"
    expected_exc_substring = "Invalid importance metric specified"
    model, _ = fixture_ft_and_only_model
    evaluator = Evaluation(model)

    with pytest.raises(Exception) as e:
        assert evaluator.get_importances(at_time=0, importance_type=bad_method)

    assert expected_exc_substring in str(e.value)


@patch("builtins.print")
@patch("matplotlib.pyplot.savefig")
@patch("os.makedirs")
@pytest.mark.parametrize("plot_dir", [None, "Something"])
def test_evaluate_fault_tree(mock_print, mock_savefig, mock_makedirs, fixture_ft_and_only_model, plot_dir):
    model, _ = fixture_ft_and_only_model
    evaluator = Evaluation(model)

    evaluator.evaluate_fault_tree(start_time = 0, stop_time = 2, simulation_steps=2, plot_dir=plot_dir, include_risk_worths=True)

    if plot_dir is not None:
        mock_savefig.assert_called()
