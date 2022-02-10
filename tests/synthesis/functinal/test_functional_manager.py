import pytest
import numpy as np

#fixtures provided via conftest

from bayesiansafety.faulttree import FaultTreeProbNode
from bayesiansafety.synthesis.functional import FunctionalManager
from bayesiansafety.synthesis.functional import Behaviour, FunctionalConfiguration

@pytest.mark.parametrize("configs",
[    ## FunctionalConfiguration(node_instance, environmental_factors, thresholds, weights=None, time_func=None, func_params=None, behaviour=Behaviour.REPLACEMENT))
    # one config set with one functional config
    {"ID_1":[ FunctionalConfiguration(    FaultTreeProbNode("IHF_BS_FP", 1.7e-3, True), {"BN_A":[("NODE_A", "STATE_NODE_A_No")]}, {"BN_A":[("NODE_A", 0)]}, None, None, None, Behaviour.REPLACEMENT) ]},

    # one config set with multiple functional configs
    {"ID_2":[ FunctionalConfiguration(    FaultTreeProbNode("IHF_BS_FP", 1.7e-3, True), {"BN_A":[("NODE_A", "STATE_NODE_A_No")]}, {"BN_A":[("NODE_A", 0)]}, None, None, None, Behaviour.REPLACEMENT),
              FunctionalConfiguration(    FaultTreeProbNode("EMI_SIG_HI_FP", 1.7e-3, True), {"BN_B":[("NODE_B", "STATE_NODE_B_No")]}, {"BN_B":[("NODE_B", 0.123)]}, None, None, None, Behaviour.OVERLAY) ]},

    # two config sets with multiple different nodes each
    {"ID_3_a":[FunctionalConfiguration(    FaultTreeProbNode("IHF_BS_FP", 1.7e-3, True), {"BN_A":[("NODE_A", "STATE_NODE_A_No")]}, {"BN_A":[("NODE_A", 0.456)]}, None, None, None, Behaviour.REPLACEMENT),
              FunctionalConfiguration(    FaultTreeProbNode("EMI_SIG_HI_FP", 1.7e-3, True), {"BN_B":[("NODE_B", "STATE_NODE_B_No")]}, {"BN_B":[("NODE_B", 0.123)]}, None, None, None, Behaviour.OVERLAY) ],
     "ID_3_b":[FunctionalConfiguration(    FaultTreeProbNode("IHF_BS_FP", 1.7e-3, True), {"BN_A":[("NODE_C", "STATE_NODE_C_No")]}, {"BN_A":[("NODE_C", 0.147)]}, None, None, None, Behaviour.REPLACEMENT),
              FunctionalConfiguration(    FaultTreeProbNode("EMI_SIG_HI_FP", 1.7e-3, True), {"BN_B":[("NODE_A", "STATE_NODE_A_No")]}, {"BN_B":[("NODE_A", 0.987)]}, None, None, None, Behaviour.OVERLAY) ] },
])
def test_instantiate_success(configs, fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn_a, _ = fixture_bn_confounder_param("BN_A")
    bn_b, _ = fixture_bn_confounder_param("BN_B")

    baynets = {"BN_A":bn_a, "BN_B":bn_b}


    functional_manager = FunctionalManager(ft_inst=tree, bayesian_nets=baynets, configurations=configs)

    assert functional_manager.ft_inst == tree
    assert functional_manager.bayesian_nets == baynets
    assert functional_manager.configurations == configs

    for config_id in configs.keys():
        assert functional_manager._FunctionalManager__builder[config_id]


@pytest.mark.parametrize("req_id, observations",
[    ("ID_1", None),
    ("ID_1", {"BN_A":[("NODE_C", "STATE_NODE_C_Yes")]} ),
    ("ID_2", {"BN_A":[("NODE_C", "STATE_NODE_C_No")]} ),
    ("ID_3", {"BN_A":[("NODE_C", "STATE_NODE_C_Yes")], "BN_B":[("NODE_C", "STATE_NODE_C_No")] } ),
])
def test_build_functional_fault_tree_successful(req_id, observations, fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn_a, _ = fixture_bn_confounder_param("BN_A")
    bn_b, _ = fixture_bn_confounder_param("BN_B")
    baynets = {"BN_A":bn_a, "BN_B":bn_b}

    conf_1 = FunctionalConfiguration(node_instance=tree.get_elem_by_name("IHF_BS_FP"),
                                    environmental_factors={ "BN_A": [("NODE_A", "STATE_NODE_A_No")]},
                                    thresholds={"BN_A": [("NODE_A", 0.123)]},
                                    weights={"BN_A": [("NODE_A", 0.5)] },
                                    time_func=None,
                                    func_params=None,
                                    behaviour=Behaviour.ADDITION)

    conf_2 = FunctionalConfiguration(node_instance=tree.get_elem_by_name("EMI_LS_FP"),
                                    environmental_factors={ "BN_A": [("NODE_A", "STATE_NODE_A_Yes")],  "BN_B" :[("NODE_B", "STATE_NODE_B_No")] } ,
                                    thresholds= {"BN_A": [("NODE_A", 0.1)], "BN_B": [("NODE_B", 0.2)]},
                                    weights={"BN_A": [("NODE_A", 1)], "BN_B": [("NODE_B", 1)]},
                                    time_func=None,
                                    func_params=None,
                                    behaviour=Behaviour.OVERLAY)

    conf_3 = FunctionalConfiguration(node_instance=tree.get_elem_by_name("IHF_SIG_LO_FP"),
                                    environmental_factors={ "BN_A": [("NODE_A", "STATE_NODE_A_Yes")]},
                                    thresholds={"BN_A": [("NODE_A", 0.123)]},
                                    weights=None, # due to replacement weight for node will be 0
                                    time_func=np.exp,
                                    func_params={"x":123},
                                    behaviour=Behaviour.FUNCTIONAL)

    configs = {    "ID_1":[conf_1], "ID_2":[conf_2], "ID_3":[conf_3]}

    functional_manager = FunctionalManager(ft_inst=tree, bayesian_nets=baynets, configurations=configs)

    functional_ft = functional_manager.build_functional_fault_tree(config_id=req_id, bn_observations=observations)

    assert isinstance(functional_ft, type(tree))



def test_build_functional_fault_tree_invalid_id_raises_excpetion( fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn_a, _ = fixture_bn_confounder_param("BN_A")
    baynets = {"BN_A":bn_a}

    conf_1 = FunctionalConfiguration(node_instance=tree.get_elem_by_name("IHF_BS_FP"),
                                    environmental_factors={ "BN_A": [("NODE_A", "STATE_NODE_A_No")]},
                                    thresholds={"BN_A": [("NODE_A", 0.123)]},
                                    weights={"BN_A": [("NODE_A", 0.5)] },
                                    time_func=None,
                                    func_params=None,
                                    behaviour=Behaviour.ADDITION)

    configs = {    "ID_1":[conf_1]}
    functional_manager = FunctionalManager(ft_inst=tree, bayesian_nets=baynets, configurations=configs)

    expected_exc_substring = "There is no managed functional configuration for given config id"

    with pytest.raises(ValueError) as e:
        assert functional_manager.build_functional_fault_tree(config_id="BAD_ID", bn_observations=False)

    assert expected_exc_substring in str(e.value)
