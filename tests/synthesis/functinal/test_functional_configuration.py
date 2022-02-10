import pytest
import numpy as np

from bayesiansafety.faulttree import FaultTreeProbNode
from bayesiansafety.synthesis.functional import Behaviour, FunctionalConfiguration

@pytest.mark.parametrize("node_instance, env_factors, threshs, weights, func, params, behaviour, expected_node_weights, expected_w0",
[    ( FaultTreeProbNode( "node_1", 0.11, True),  { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")] },  {"BN_A": [("NODE_A", 0.01)] },  None, None, None, Behaviour.REPLACEMENT,   {"BN_A": [("NODE_A", 1.0)] }, 0.0) ,
    ( FaultTreeProbNode( "node_2", 0.12, True),  { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")] },  {"BN_A": [("NODE_A", 0.02)] },  None, None, None, Behaviour.ADDITION,  {"BN_A": [("NODE_A", 0.5)] }, 0.5) ,
    ( FaultTreeProbNode( "node_3", 0.13, True),  { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")] },  {"BN_A": [("NODE_A", 0.03)] },  {"BN_A": [("NODE_A", 3)] }, None, None, Behaviour.REPLACEMENT,   {"BN_A": [("NODE_A", 1.0)] }, 0.0) ,  ## Note: weights are always normalized to 0...1
    ( FaultTreeProbNode( "node_4", 0.14, True),  { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")] },  {"BN_A": [("NODE_A", 0.04)] },  {"BN_A": [("NODE_A", 4)] }, None, None, Behaviour.ADDITION,  {"BN_A": [("NODE_A", 0.8)] }, 0.2) ,      ## Note: weights are normalized - therfore NODE_A is 4x as important as w_0

    ( FaultTreeProbNode( "node_5", 0.15, True),  { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")], "BN_B": [("NODE_B", "STATE_NODE_B_No")] },  {"BN_A": [("NODE_A", 0.05)], "BN_B": [("NODE_B", 0.50)] }, None, None, None, Behaviour.OVERLAY,   {"BN_A": [("NODE_A", 1/3)], "BN_B": [("NODE_B", 1/3)] }, 1/3) ,
    ( FaultTreeProbNode( "node_6", 0.16, True),  { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")], "BN_B": [("NODE_B", "STATE_NODE_B_No")] },  {"BN_A": [("NODE_A", 0.06)], "BN_B": [("NODE_B", 0.60)] }, {"BN_A": [("NODE_A", 0.123)], "BN_B": [("NODE_B", 0.456)] }, None, None, Behaviour.OVERLAY,  {"BN_A": [("NODE_A", 0.123)], "BN_B": [("NODE_B",  0.456)] }, 0.943912) ,
    ( FaultTreeProbNode( "node_7", 0.17, True),  { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")], "BN_B": [("NODE_B", "STATE_NODE_B_No")] },  {"BN_A": [("NODE_A", 0.07)], "BN_B": [("NODE_B", 0.70)] }, {"BN_A": [("NODE_A", 7)], "BN_B": [("NODE_B", 70)] }, None, None, Behaviour.OVERLAY,  {"BN_A": [("NODE_A", 7/491)], "BN_B": [("NODE_B",  70/491)] }, 1/491) ,

    ( FaultTreeProbNode( "node_8", 0.18, True),  { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")] },  {"BN_A": [("NODE_A", 0.08)] },  None, np.exp, {"x": 8}, Behaviour.RATE,  {}, 1.0) ,
    ( FaultTreeProbNode( "node_9", 0.19, True),  { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")] },  {"BN_A": [("NODE_A", 0.09)] },  None, np.sin, {"x": 9}, Behaviour.FUNCTIONAL,  {}, 1.0) ,
    ( FaultTreeProbNode("node_10", 0.20, True),  { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")] },  {"BN_A": [("NODE_A", 0.10)] },  None, np.cos, {"x":10}, Behaviour.RATE,  {}, 1.0) ,
    ( FaultTreeProbNode("node_11", 0.21, True),  { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")] },  {"BN_A": [("NODE_A", 0.11)] },  None, np.tan, {"x":11}, Behaviour.PARAMETER,  {}, 1.0) ,

])
def test_instantiate_success(node_instance, env_factors, threshs, weights, func, params, behaviour, expected_node_weights, expected_w0):
    config = FunctionalConfiguration(node_instance=node_instance, environmental_factors=env_factors, thresholds=threshs, weights=weights, time_func=func, func_params=params, behaviour=behaviour)

    assert config.node_instance == node_instance
    assert config.behaviour == behaviour
    assert config.environmental_factors == env_factors
    assert config.thresholds == threshs
    assert config.time_func == func
    assert config.func_params == params
    assert np.isclose(config.get_weight_orig_fn(), expected_w0)

    for bn_name in expected_node_weights.keys():
        cur_exp_weights = dict(expected_node_weights[bn_name])
        cur_real_weights = dict(config.weights[bn_name])

        for node_name, exp_val in cur_exp_weights.items():
            real_val = cur_real_weights[node_name]
            assert np.isclose(exp_val, real_val)


def test_invalid_nr_of_bns_for_behaviour_replacement_raises_excpetion():
    node_instance = FaultTreeProbNode( "test", 0.15, True)
    env_factors = { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")], "BAD_BN": [("NODE_B", "STATE_NODE_B_No")] }
    threshs = {"BN_A": [("NODE_A", 0.05)], "BAD_BN": [("NODE_B", 0.50)] }
    behaviour = Behaviour.REPLACEMENT

    expected_exc_substring = "only one fixed BN is allowed"

    with pytest.raises(ValueError) as e:
        assert FunctionalConfiguration(node_instance=node_instance, environmental_factors=env_factors, thresholds=threshs, weights=None, time_func=None, func_params=None, behaviour=behaviour)

    assert expected_exc_substring in str(e.value)


def test_invalid_nr_of_bns_for_behaviour_addition_raises_excpetion():
    node_instance = FaultTreeProbNode( "test", 0.15, True)
    env_factors = { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")], "BAD_BN": [("NODE_B", "STATE_NODE_B_No")] }
    threshs = {"BN_A": [("NODE_A", 0.05)], "BAD_BN": [("NODE_B", 0.50)] }
    behaviour = Behaviour.ADDITION

    expected_exc_substring = "only one fixed BN is allowed"

    with pytest.raises(ValueError) as e:
        assert FunctionalConfiguration(node_instance=node_instance, environmental_factors=env_factors, thresholds=threshs, weights=None, time_func=None, func_params=None, behaviour=behaviour)

    assert expected_exc_substring in str(e.value)


def test_invalid_nr_of_bn_nodes_for_behaviour_replacement_raises_excpetion():
    node_instance = FaultTreeProbNode( "test", 0.15, True)
    env_factors = { "BN_A": [("NODE_A", "NODE_STATE_A_Yes"), ("BAD_NODE", "STATE_NODE_B_No")]}
    threshs = {"BN_A": [("NODE_A", 0.05), ("BAD_NODE", 0.123)]}
    behaviour = Behaviour.REPLACEMENT

    expected_exc_substring = "only one fixed node is allowed"

    with pytest.raises(ValueError) as e:
        assert FunctionalConfiguration(node_instance=node_instance, environmental_factors=env_factors, thresholds=threshs, weights=None, time_func=None, func_params=None, behaviour=behaviour)

    assert expected_exc_substring in str(e.value)


def test_invalid_nr_of_bn_nodes_for_behaviour_addition_raises_excpetion():
    node_instance = FaultTreeProbNode( "test", 0.15, True)
    env_factors = { "BN_A": [("NODE_A", "NODE_STATE_A_Yes"), ("BAD_NODE", "STATE_NODE_B_No")]}
    threshs = {"BN_A": [("NODE_A", 0.05), ("BAD_NODE", 0.123)]}
    behaviour = Behaviour.ADDITION

    expected_exc_substring = "only one fixed node is allowed"

    with pytest.raises(ValueError) as e:
        assert FunctionalConfiguration(node_instance=node_instance, environmental_factors=env_factors, thresholds=threshs, weights=None, time_func=None, func_params=None, behaviour=behaviour)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("behaviour, env_factors, thresholds",
[    ( Behaviour.ADDITION    , { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")]}    ,  {} ),

                                                                                  # Note: BN_A missing
    ( Behaviour.REPLACEMENT    , { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")]}    ,  {"BAD_BN": [("NODE_B", 0.05)]} ),

    ( Behaviour.OVERLAY        , { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")],        # Note: BN_B missing
                                "BN_B" :[("NODE_B", "NODE_STATE_B_No")]}    ,  {"BN_A": [("NODE_A", 0.05)]} ),

    ( Behaviour.OVERLAY        , { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")],
                                "BN_B" :[("NODE_B", "NODE_STATE_B_No")],       # Note: BN_B missing
                                "BN_C" :[("NODE_C", "NODE_STATE_C_Yes")] }    ,  {"BN_A": [("NODE_A", 0.05)], "BN_C": [("NODE_C", 0.05)]} ),

    ( Behaviour.FUNCTIONAL    , { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")],
                                "BN_B" :[("NODE_B", "NODE_STATE_B_No")],       # Note: BN_A missing
                                "BN_C" :[("NODE_C", "NODE_STATE_C_Yes")] }    ,  {"BN_B": [("NODE_B", 0.05)], "BN_C": [("NODE_C", 0.05)]} ),

    ( Behaviour.RATE        , { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")],
                                "BN_B" :[("NODE_B", "NODE_STATE_B_No")],       # Note: BN_C missing
                                "BN_C" :[("NODE_C", "NODE_STATE_C_Yes")] }    ,  {"BN_B": [("NODE_B", 0.05)], "BN_A": [("NODE_A", 0.05)]} ),
])
def test_unspecified_thresholds_bn_level_raises_excpetion(behaviour, env_factors, thresholds):
    node_instance = FaultTreeProbNode( "test", 0.15, True)
    # params are needed to test variation 'functional' and 'rate' in the last two parameterized cases
    dummy_func = np.exp
    dummy_params = {"x":123}

    expected_exc_substring = "For each given contributing BN"

    with pytest.raises(ValueError) as e:
        assert FunctionalConfiguration(node_instance=node_instance, environmental_factors=env_factors, thresholds=thresholds, weights=None, time_func=dummy_func, func_params=dummy_params, behaviour=behaviour)

    assert expected_exc_substring in str(e.value)



@pytest.mark.parametrize("behaviour, env_factors, thresholds",
[                                                                                # Note: NODE_A missing
    ( Behaviour.ADDITION    , { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")]}    ,  {"BN_A": [("BAD_NODE", 0.05)]} ),
    ( Behaviour.REPLACEMENT    , { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")]}    ,  {"BN_A": [("BAD_NODE", 0.05)]} ),

    ( Behaviour.OVERLAY        , { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")],        # Note: NODE_B missing
                                "BN_B" :[("NODE_B", "NODE_STATE_B_No")]}    ,  {"BN_A": [("NODE_A", 0.05)], "BN_B": [("NODE_A", 0.05)]}, ),

    ( Behaviour.OVERLAY        , { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")],
                                "BN_B" :[("NODE_B", "NODE_STATE_B_No")],       # Note: BN_B NODE_B missing
                                "BN_C" :[("NODE_C", "NODE_STATE_C_Yes")] }    ,  {"BN_A": [("NODE_A", 0.05), ("NODE_B", 0.05)], "BN_B": [("NODE_A", 0.05)], "BN_C": [("NODE_C", 0.05)] } ),

    ( Behaviour.FUNCTIONAL    , { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")],
                                "BN_B" :[("NODE_B", "NODE_STATE_B_No")],       # Note: BN_A NODE_A missing
                                "BN_C" :[("NODE_C", "NODE_STATE_C_Yes")] }    ,  {"BN_A": [("NODE_B", 0.05)], "BN_B": [("NODE_B", 0.05)], "BN_C": [("NODE_C", 0.05)]} ),

    ( Behaviour.RATE        , { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")],
                                "BN_B" :[("NODE_B", "NODE_STATE_B_No")],       # Note: BN_C NODE_C missing
                                "BN_C" :[("NODE_C", "NODE_STATE_C_Yes")] }    ,  {"BN_A": [("NODE_A", 0.05), ("NODE_B", 0.05), ("NODE_C", 0.05)], "BN_B": [("NODE_B", 0.05)], "BN_C": [("NODE_A", 0.05), ("NODE_B", 0.05)]} ),
])
def test_unspecified_thresholds_node_level_raises_excpetion(behaviour, env_factors, thresholds):
    node_instance = FaultTreeProbNode( "test", 0.15, True)
    # params are needed to test variation 'functional' and 'rate' in the last two parameterized cases
    dummy_func = np.exp
    dummy_params = {"x":123}

    expected_exc_substring = "For each contributing node of each BN a probability threshold needs to be defined."

    with pytest.raises(ValueError) as e:
        assert FunctionalConfiguration(node_instance=node_instance, environmental_factors=env_factors, thresholds=thresholds, weights=None, time_func=dummy_func, func_params=dummy_params, behaviour=behaviour)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("behaviour, env_factors, thresholds",
[
    ( Behaviour.ADDITION    , { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")]}    ,  {"BN_A": [("NODE_A", -1)]} ),
    ( Behaviour.REPLACEMENT    , { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")]}    ,  {"BN_A": [("NODE_A", 15), ("NODE_B", 0.05)]} ),

    ( Behaviour.OVERLAY        , { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")],
                                "BN_B" :[("NODE_B", "NODE_STATE_B_No")]}    ,  {"BN_A": [("NODE_A", 0.05)], "BN_B": [("NODE_B", 0x1F)]}, ),

    ( Behaviour.OVERLAY        , { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")],
                                "BN_B" :[("NODE_B", "NODE_STATE_B_No")],
                                "BN_C" :[("NODE_C", "NODE_STATE_C_Yes")] }    ,  {"BN_A": [("NODE_A", 0.05), ("NODE_B", 0.05)], "BN_B": [("NODE_B", 0b00110)], "BN_C": [("NODE_C", 0.05)] } ),

    ( Behaviour.FUNCTIONAL    , { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")],
                                "BN_B" :[("NODE_B", "NODE_STATE_B_No")],
                                "BN_C" :[("NODE_C", "NODE_STATE_C_Yes")] }    ,  {"BN_A": [("NODE_A", 1.01), ("NODE_B", 0.05)], "BN_B": [("NODE_B", 0.05)], "BN_C": [("NODE_C", 0.05)]} ),

    ( Behaviour.RATE        , { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")],
                                "BN_B" :[("NODE_B", "NODE_STATE_B_No")],
                                "BN_C" :[("NODE_C", "NODE_STATE_C_Yes")] }    ,  {"BN_A": [("NODE_A", 0.05), ("NODE_B", 0.05), ("NODE_C", -1.23)], "BN_B": [("NODE_B", 0.05)], "BN_C": [("NODE_A", 0.05), ("NODE_B", 0.05), ("NODE_C", 0.05)]} ),
])
def test_outofbounds_threshold_raises_excpetion(behaviour, env_factors, thresholds):
    node_instance = FaultTreeProbNode( "test", 0.15, True)
    # params are needed to test variation 'functional' and 'rate' in the last two parameterized cases
    dummy_func = np.exp
    dummy_params = {"x":123}

    expected_exc_substring = "out of bounds (0...1.0) with value"

    with pytest.raises(ValueError) as e:
        assert FunctionalConfiguration(node_instance=node_instance, environmental_factors=env_factors, thresholds=thresholds, weights=None, time_func=dummy_func, func_params=dummy_params, behaviour=behaviour)

    assert expected_exc_substring in str(e.value)



@pytest.mark.parametrize("behaviour, bad_func, bad_params",
[    (Behaviour.FUNCTIONAL, None, None),
    (Behaviour.FUNCTIONAL, np.exp, None),
    (Behaviour.FUNCTIONAL, None, {"x":123}),
    (Behaviour.RATE, "Bad", {"x":123}),
    (Behaviour.PARAMETER, lambda x: x +1, {}),
])
def test_invalid_functional_tuple_raises_excpetion(behaviour, bad_func, bad_params):
    node_instance = FaultTreeProbNode( "test", 0.15, True)
    env_factors = { "BN_A": [("NODE_A", "NODE_STATE_A_Yes"), ("NODE_A", "STATE_NODE_A_No")]}
    threshs = {"BN_A": [("NODE_A", 0.05)]}

    expected_exc_substring = "a callable function and associated parameters need to be provided"

    with pytest.raises(ValueError) as e:
        assert FunctionalConfiguration(node_instance=node_instance, environmental_factors=env_factors, thresholds=threshs, weights=None, time_func=bad_func, func_params=bad_params, behaviour=behaviour)

    assert expected_exc_substring in str(e.value)



# basically redundant due to the test inside 'test_instantiate_success' -> maybe refactor
@pytest.mark.parametrize("config, expected_w0",
[    ( FunctionalConfiguration( FaultTreeProbNode( "node_1", 0.11, True),  { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")] },  {"BN_A": [("NODE_A", 0.01)] },  None, None, None, Behaviour.REPLACEMENT),  0.0) ,
    ( FunctionalConfiguration( FaultTreeProbNode( "node_2", 0.12, True),  { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")] },  {"BN_A": [("NODE_A", 0.02)] },  None, None, None, Behaviour.ADDITION),  0.5) ,
    ( FunctionalConfiguration( FaultTreeProbNode( "node_3", 0.13, True),  { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")] },  {"BN_A": [("NODE_A", 0.03)] },  {"BN_A": [("NODE_A", 3)] }, None, None, Behaviour.REPLACEMENT), 0.0) ,  ## Note: weights are always normalized to 0...1
    ( FunctionalConfiguration( FaultTreeProbNode( "node_4", 0.14, True),  { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")] },  {"BN_A": [("NODE_A", 0.04)] },  {"BN_A": [("NODE_A", 4)] }, None, None, Behaviour.ADDITION), 0.2) ,     ## Note: weights are normalized - therfore NODE_A is 4x as important as w_0
    ( FunctionalConfiguration( FaultTreeProbNode( "node_5", 0.15, True),  { "BN_A": [("NODE_A", "NODE_STATE_A_Yes")] },  {"BN_A": [("NODE_A", 0.08)] },  None, np.exp, {"x": 8}, Behaviour.FUNCTIONAL), 1.0) ,                     ## Note: Only w_0 matters and therefore is 1.0
])
def test_get_weight_orig_fn_returns_correct(config, expected_w0):
    assert np.isclose(config.get_weight_orig_fn(), expected_w0)
