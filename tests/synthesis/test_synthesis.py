import pytest
from unittest.mock import patch
#fixtures provided via conftest

from bayesiansafety.core import BayesianNetwork
from bayesiansafety.faulttree import BayesianFaultTree
from bayesiansafety.faulttree import FaultTreeLogicNode
from bayesiansafety.faulttree import FaultTreeProbNode
from bayesiansafety.synthesis import Synthesis
from bayesiansafety.synthesis.functional import Behaviour

def test_instantiate_success(fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn_a, _ = fixture_bn_confounder_param("BN_A")
    bn_b, _ = fixture_bn_confounder_param("BN_B")

    trees = {"elevator_model":tree}
    baynets = {"BN_A":bn_a, "BN_B":bn_b}

    bay_safety = Synthesis(fault_trees=trees, bayesian_nets=baynets)

    assert bay_safety.fault_trees == trees
    assert bay_safety.bayesian_nets == baynets
    assert bay_safety._Synthesis__models.keys() == set( list(trees.keys()) + list(baynets.keys()) )


@pytest.mark.parametrize("bad_trees, bad_bns",
[ ( None    , None    ),
  ( None    , {}    ),
  ( {}        , None    ),
  ( dict()    , dict())
])
def test_non_initialized_ctor_params_raise_exception(bad_trees, bad_bns):
    expected_exc_substring = "Given parameters can't be default or none"

    with pytest.raises(TypeError) as e:
        assert Synthesis(fault_trees=bad_trees, bayesian_nets=bad_bns)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("bad_trees",
[     {"BadTree":"BadTree"},
    {"BadTree":None},

    {"BadTree":BayesianNetwork("test_FT", [("A", "B")])},

    {    "GoodTree":BayesianFaultTree("test_FT", [FaultTreeProbNode('A', 2.78e-6), FaultTreeProbNode('B', 4.12e-4)] , [FaultTreeLogicNode('AND', ['A', 'B'], "AND" )]),
        "BadTree":BayesianNetwork("test_BN", [("A", "B")])
    }
])
def test_invalid_types_in_ctor_trees_raise_exception(bad_trees, fixture_bn_confounder_param):
    bn_a, _ = fixture_bn_confounder_param("BN_A")
    baynets = {"BN_A":bn_a}

    expected_exc_substring = "All given Fault Trees must be an instantiation of"

    with pytest.raises(TypeError) as e:
        assert Synthesis(fault_trees=bad_trees, bayesian_nets=baynets)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("bad_bns",
[     {"BadBayNet":"BadBayNet"},
    {"BadBayNet": None},

    {"BadBayNet":BayesianFaultTree("test_FT", [FaultTreeProbNode('A', 2.78e-6), FaultTreeProbNode('B', 4.12e-4)] , [FaultTreeLogicNode('AND', ['A', 'B'], "AND" )])},

    {    "GoodBayNet":BayesianNetwork("test_BN", [("A", "B")]),
        "BadBayNet":BayesianFaultTree("test_FT", [FaultTreeProbNode('A', 2.78e-6), FaultTreeProbNode('B', 4.12e-4)] , [FaultTreeLogicNode('AND', ['A', 'B'], "AND" )])
    }
])
def test_invalid_types_in_ctor_bns_raise_exception(bad_bns, fixture_ft_elevator_model):
    tree, _ = fixture_ft_elevator_model
    trees = {"elevator_model":tree}

    expected_exc_substring = "All given Bayesian Networks must be an instantiation of"

    with pytest.raises(TypeError) as e:
        assert Synthesis(fault_trees=trees, bayesian_nets=bad_bns)

    assert expected_exc_substring in str(e.value)


def test_set_hybrid_configurations_success(fixture_hybrid_config, fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree_1, _ = fixture_ft_elevator_model
    tree_2 = tree_1.copy()
    bn_a, _ = fixture_bn_confounder_param("BN_A")

    trees = {tree_1.name:tree_1, tree_2.name:tree_2}
    baynets = {"BN_A":bn_a}

    bay_safety = Synthesis(fault_trees=trees, bayesian_nets=baynets)

    ft_configs = {
                    tree_1.name: {"ID_1":fixture_hybrid_config("ID_1"), "ID_2":fixture_hybrid_config("ID_2")},
                    tree_2.name: {"ID_3":fixture_hybrid_config("ID_3")}
                }

    assert bay_safety._Synthesis__hybrid_managers is None

    bay_safety.set_hybrid_configurations(ft_configurations = ft_configs)

    assert bay_safety._Synthesis__hybrid_managers is not None


def test_set_functional_configurations_success(fixture_functional_config, fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree_1, _ = fixture_ft_elevator_model
    tree_2 = tree_1.copy()
    bn_a, _ = fixture_bn_confounder_param("BN_A")

    trees = {tree_1.name:tree_1, tree_2.name:tree_2}
    baynets = {"BN_A":bn_a}

    bay_safety = Synthesis(fault_trees=trees, bayesian_nets=baynets)

    conf_1, _ = fixture_functional_config(Behaviour.REPLACEMENT, name="IHF_LS_FP")
    conf_2, _ = fixture_functional_config(Behaviour.ADDITION, name="EMI_SIG_HI_FP")
    conf_3, _ = fixture_functional_config(Behaviour.FUNCTIONAL, name="EMI_BS_FP")

    ft_configs = {
                    tree_1.name: {"ID_1":[conf_1], "ID_2":[conf_1, conf_2]},
                    tree_2.name: {"ID_3":[conf_3]}
                }

    assert bay_safety._Synthesis__functional_managers is None

    bay_safety.set_functional_configurations(ft_configurations = ft_configs)

    assert bay_safety._Synthesis__functional_managers is not None


def test_requesting_hybrid_service_before_setting_configs_raises_exception(fixture_hybrid_config, fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree_1, _ = fixture_ft_elevator_model
    bn_a, _ = fixture_bn_confounder_param("BN_A")

    trees = {tree_1.name:tree_1}
    baynets = {"BN_A":bn_a}
    expected_exc_substring = "No hybrid managers are instantiated. Make sure to set the hybrid configurations before requesting associated services"

    bay_safety = Synthesis(fault_trees=trees, bayesian_nets=baynets)

    with pytest.raises(Exception) as e:
        assert bay_safety.get_hybrid_networks(ft_name=tree_1.name, config_id="ID_1")

    assert expected_exc_substring in str(e.value)


def test_requesting_functional_service_before_setting_configs_raises_exception(fixture_functional_config, fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree_1, _ = fixture_ft_elevator_model
    bn_a, _ = fixture_bn_confounder_param("BN_A")

    trees = {tree_1.name:tree_1}
    baynets = {"BN_A":bn_a}
    expected_exc_substring = "No functional managers are instantiated. Make sure to set the functional configurations before requesting associated services"

    bay_safety = Synthesis(fault_trees=trees, bayesian_nets=baynets)
    conf_1, _ = fixture_functional_config(Behaviour.REPLACEMENT, name="IHF_LS_FP")

    with pytest.raises(Exception) as e:
        assert bay_safety.evaluate_functional_fault_trees(ft_name=tree_1.name, bn_observations=None, ft_time_scales=None)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("bad_name", [ "BadName", 5, None ])
def test_get_model_by_name_invalid_name_raises_exception(bad_name, fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn_a, _ = fixture_bn_confounder_param("BN_A")

    trees = {tree.name:tree}
    baynets = {"BN_A":bn_a}
    expected_exc_substring = "could not be found in given models"

    bay_safety = Synthesis(fault_trees=trees, bayesian_nets=baynets)

    with pytest.raises(ValueError) as e:
        assert bay_safety.get_model_by_name(bad_name)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("scoped_name", [ "BN_A", "elevator_model"])
def test_get_model_by_name_returns_correct(scoped_name, fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn_a, _ = fixture_bn_confounder_param("BN_A")

    trees = {tree.name:tree}
    baynets = {"BN_A":bn_a}

    bay_safety = Synthesis(fault_trees=trees, bayesian_nets=baynets)

    model_elem =  bay_safety.get_model_by_name(scoped_name)

    assert model_elem.name == scoped_name
    assert isinstance(model_elem, (BayesianFaultTree, BayesianNetwork))



def test_get_extended_fault_tree_successful(fixture_hybrid_config, fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn_a, _ = fixture_bn_confounder_param("BN_A")

    trees = {tree.name:tree}
    baynets = {"BN_A":bn_a}

    ft_configs = { tree.name: {"ID_1":fixture_hybrid_config("ID_1")} }

    bay_safety = Synthesis(fault_trees=trees, bayesian_nets=baynets)
    bay_safety.set_hybrid_configurations(ft_configurations = ft_configs)

    ext_ft =  bay_safety.get_extended_fault_tree(ft_name=tree.name, config_id="ID_1", bn_observations=None)

    assert isinstance(ext_ft, BayesianFaultTree)


def test_get_hybrid_networks_successful(fixture_hybrid_config, fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn_a, _ = fixture_bn_confounder_param("BN_A")

    trees = {tree.name:tree}
    baynets = {"BN_A":bn_a}

    ft_configs = { tree.name: {"ID_1":fixture_hybrid_config("ID_1")} }

    bay_safety = Synthesis(fault_trees=trees, bayesian_nets=baynets)
    bay_safety.set_hybrid_configurations(ft_configurations = ft_configs)

    hybrid_networks =  bay_safety.get_hybrid_networks(ft_name=tree.name, config_id="ID_1", at_time=0, fix_other_bns=True)

    assert all( isinstance(hybrid_bn, BayesianNetwork) for hybrid_bn in hybrid_networks.values() )


def test_get_functional_fault_tree_successful(fixture_functional_config, fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn_a, _ = fixture_bn_confounder_param("BN_A")

    trees = {tree.name:tree}
    baynets = {"BN_A":bn_a}

    conf, _ = fixture_functional_config(Behaviour.REPLACEMENT, name="IHF_LS_FP")
    ft_configs = { tree.name: {"ID_1":[conf]} }

    bay_safety = Synthesis(fault_trees=trees, bayesian_nets=baynets)
    bay_safety.set_functional_configurations(ft_configurations = ft_configs)

    func_ft =  bay_safety.get_functional_fault_tree(ft_name=tree.name, config_id="ID_1", bn_observations=None)

    assert isinstance(func_ft, BayesianFaultTree)


@patch("builtins.print")
@patch("matplotlib.pyplot.savefig")
@patch("os.makedirs")
@pytest.mark.parametrize("time_scales", [None, {"elevator_model": (0 , 1e5, 2, "TEST")}, {"invalid_ft": (0 , 1e5, 2, "TEST")}])
def test_evaluate_functional_fault_trees_successful(mock_print, mock_savefig, mock_makedirs, time_scales, fixture_functional_config, fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn_a, _ = fixture_bn_confounder_param("BN_A")

    trees = {tree.name:tree}
    baynets = {"BN_A":bn_a}

    conf, _ = fixture_functional_config(Behaviour.REPLACEMENT, name="IHF_LS_FP")
    ft_configs = { tree.name: {"ID_1":[conf]} }

    bay_safety = Synthesis(fault_trees=trees, bayesian_nets=baynets)
    bay_safety.set_functional_configurations(ft_configurations = ft_configs)

    bay_safety.evaluate_functional_fault_trees(ft_name=tree.name, bn_observations=None, ft_time_scales=time_scales)

    if time_scales and tree.name in time_scales:
        mock_savefig.assert_called()


@patch("builtins.print")
@patch("matplotlib.pyplot.savefig")
@patch("os.makedirs")
@pytest.mark.parametrize("time_scales", [None, {"elevator_model": (0 , 1e5, 2, "TEST")}, {"invalid_ft": (0 , 1e5, 2, "TEST")}])
def test_evaluate_hybrid_fault_trees_successful(mock_print, mock_savefig, mock_makedirs, time_scales, fixture_hybrid_config, fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn_a, _ = fixture_bn_confounder_param("BN_A")

    trees = {tree.name:tree}
    baynets = {"BN_A":bn_a}

    ft_configs = { tree.name: {"ID_1":fixture_hybrid_config("ID_1")} }

    bay_safety = Synthesis(fault_trees=trees, bayesian_nets=baynets)
    bay_safety.set_hybrid_configurations(ft_configurations = ft_configs)

    bay_safety.evaluate_hybrid_fault_trees(ft_name=tree.name, bn_observations=None, ft_time_scales=time_scales)

    if time_scales and tree.name in time_scales:
        mock_savefig.assert_called()
