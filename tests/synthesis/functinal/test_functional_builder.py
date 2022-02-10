import pytest
import numpy as np

from bayesiansafety.faulttree import FaultTreeProbNode
from bayesiansafety.faulttree import FaultTreeLogicNode
from bayesiansafety.synthesis.functional import FunctionalBuilder
from bayesiansafety.synthesis.functional import Behaviour, FunctionalConfiguration


def test_instantiate_success():
    node_instance = FaultTreeProbNode( "test", 0.15, True)
    env_factors = { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")]}
    threshs = {"BN_A": [("NODE_A", 0.05)]}
    behaviour = Behaviour.ADDITION


    config = FunctionalConfiguration(node_instance=node_instance, environmental_factors=env_factors, thresholds=threshs, weights=None, time_func=None, func_params=None, behaviour=behaviour)
    builder = FunctionalBuilder(ft_node_inst=node_instance, configuration=config)

    assert builder.ft_node_inst.name == node_instance.name
    assert builder.configuration == config
    assert builder.associated_probabilities is None



@pytest.mark.parametrize("invalid_node_inst",
[    None,
    "BAD",
    FaultTreeLogicNode("logic_1", ["A", "B"]),
    FunctionalConfiguration(node_instance=FaultTreeProbNode( "test", 0.15, True), environmental_factors={ "BN_A": [("NODE_A", "NODE_STATE_A_Yes")]}, thresholds={"BN_A": [("NODE_A", 0.05)]}, behaviour=Behaviour.ADDITION),
])
def test_invalid_ft_node_type_raises_exception(invalid_node_inst):
    valid_node_instance = FaultTreeProbNode( "test", 0.15, True)
    env_factors = { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")]}
    threshs = {"BN_A": [("NODE_A", 0.05)]}
    behaviour = Behaviour.ADDITION

    valid_config = FunctionalConfiguration(node_instance=valid_node_instance, environmental_factors=env_factors, thresholds=threshs, weights=None, time_func=None, func_params=None, behaviour=behaviour)

    expected_exc_substring = "Fault tree node instance must be of type"

    with pytest.raises(TypeError) as e:
        assert FunctionalBuilder(ft_node_inst=invalid_node_inst, configuration=valid_config)

    assert expected_exc_substring in str(e.value)


#associated_probabilities (dict<str, list<<tuple<str, float>>>): Dictionary of relevant nodes and their marginal probabilities to build the funcional.
#                Dictionary key = bn name, value = list of tuples of bn node name and marginal probability for the it's designated state.


@pytest.mark.parametrize("at_time, expected_val",
[ (0, 0.39535),
  (1e4, 0.8944999793),
])
def test_create_functional_for_behaviour_addition_successful(at_time, expected_val, fixture_functional_config):
    config, node_instance = fixture_functional_config(Behaviour.ADDITION)
    asso_probs = {"BN_A": [("NODE_A", 0.789)]}
    expected_params = ["at_time"]

    builder = FunctionalBuilder(ft_node_inst=node_instance, configuration=config)

    func, params = builder.create_functional(associated_probabilities=asso_probs , new_configuration=None)

    assert func is not None
    assert params is not None
    for exp_param in expected_params:
        assert exp_param in params

    assert callable(func)
    assert np.isclose( func(at_time), expected_val, atol=1e-9)


@pytest.mark.parametrize("at_time, expected_val",
[ (0, 0.789),
  (1e4, 0.789),
])
def test_create_functional_for_behaviour_repalacement_successful(at_time, expected_val, fixture_functional_config):
    config, node_instance = fixture_functional_config(Behaviour.REPLACEMENT)
    asso_probs = {"BN_A": [("NODE_A", 0.789)]}

    builder = FunctionalBuilder(ft_node_inst=node_instance, configuration=config)

    func, params = builder.create_functional(associated_probabilities=asso_probs , new_configuration=None)

    assert func is not None
    assert callable(func)
    assert not params

    assert np.isclose( func(at_time), expected_val, atol=1e-9)


@pytest.mark.parametrize("at_time, expected_val",
[ (0, 4.70192715e-06),
  (1e4, 2.765839385e-3),
])
def test_create_functional_for_behaviour_overlay_successful(at_time, expected_val, fixture_functional_config):
    config, node_instance = fixture_functional_config(Behaviour.OVERLAY)
    asso_probs = {"BN_A": [("NODE_A", 0.123)], "BN_B": [("NODE_B", 0.456)], "BN_C": [("NODE_C", 0.789)] }

    expected_params = ["at_time"]

    builder = FunctionalBuilder(ft_node_inst=node_instance, configuration=config)

    func, params = builder.create_functional(associated_probabilities=asso_probs , new_configuration=None)

    assert func is not None
    assert params is not None
    for exp_param in expected_params:
        assert exp_param in params

    assert callable(func)
    assert np.isclose( func(at_time), expected_val, atol=1e-9)



@pytest.mark.parametrize("at_time, expected_val",
[ (0, 1),   # Note: Specified behaviour in conftest.py is np.exp with param x
  (123, 2.619517319e53),
  (-5, 6.737946999e-3),
])
def test_create_functional_for_behaviour_functional_successful(at_time, expected_val, fixture_functional_config):
    config, node_instance = fixture_functional_config(Behaviour.FUNCTIONAL)
    asso_probs = {}

    expected_params = ["x"]

    builder = FunctionalBuilder(ft_node_inst=node_instance, configuration=config)

    func, params = builder.create_functional(associated_probabilities=asso_probs , new_configuration=None)

    assert func is not None
    assert params is not None
    for exp_param in expected_params:
        assert exp_param in params

    assert callable(func)
    assert np.isclose( func(at_time), expected_val, atol=1e-9)



@pytest.mark.parametrize("at_time, mod_rate, expected_val",
[ (0, 5e-5, 5e-5),
  (1e4, 5e-5, 0.3934693403),
  (5e5, 1e-4, 1.0),
])
def test_create_functional_for_behaviour_rate_successful(at_time, mod_rate, expected_val, fixture_functional_config):
    config, node_instance = fixture_functional_config(Behaviour.RATE)
    asso_probs = {}

    expected_params = ["at_time", "frate"]

    builder = FunctionalBuilder(ft_node_inst=node_instance, configuration=config)

    func, params = builder.create_functional(associated_probabilities=asso_probs , new_configuration=None)

    assert func is not None
    assert params is not None
    for exp_param in expected_params:
        assert exp_param in params

    assert callable(func)
    assert np.isclose( func(at_time=at_time, frate=mod_rate), expected_val, atol=1e-9)


def test_create_functional_for_behaviour_parameter_raises_excpetion(fixture_functional_config):
    config, node_instance = fixture_functional_config(Behaviour.PARAMETER)
    asso_probs = {}

    expected_exc_substring = "is currently not supported"

    builder = FunctionalBuilder(ft_node_inst=node_instance, configuration=config)

    with pytest.raises(NotImplementedError) as e:
        assert builder.create_functional(associated_probabilities=asso_probs , new_configuration=None)

    assert expected_exc_substring in str(e.value)



def test_create_functional_with_other_config_successful(fixture_functional_config):
    config, node_instance = fixture_functional_config(Behaviour.ADDITION)
    other_config, other_node_instance = fixture_functional_config(Behaviour.REPLACEMENT)
    asso_probs = {"BN_A": [("NODE_A", 0.789)]}
    comparision_time = 123

    builder = FunctionalBuilder(ft_node_inst=node_instance, configuration=config)

    ref_func, ref_params = builder.create_functional(associated_probabilities=asso_probs , new_configuration=None)
    other_func, other_params = builder.create_functional(associated_probabilities=asso_probs, new_configuration=other_config)

    assert ref_func != other_func
    assert callable(ref_func)
    assert callable(other_func)
    assert ref_func(at_time=comparision_time) != other_func(at_time=comparision_time)
