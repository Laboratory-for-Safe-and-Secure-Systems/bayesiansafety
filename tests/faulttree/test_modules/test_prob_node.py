import math

import numpy as np
import pytest

from bayesiansafety.faulttree import FaultTreeProbNode

#fixtures provided via conftest

@pytest.mark.parametrize("name, probability_of_failure, is_time_dependent, values", [("A", 0.12, True, [0.88, 0.12]), ("B", 0.87, False, [0.13, 0.87]),
                                                                             ("C", 0.0, False, [1.0, 0.0]), ("D", 1.0, True, [0.0, 1.0])])
def test_instantiate_node_success(name, probability_of_failure, is_time_dependent, values):
    node = FaultTreeProbNode(name=name, probability_of_failure=probability_of_failure, is_time_dependent=is_time_dependent)

    assert isinstance(node, FaultTreeProbNode)
    assert node.is_time_dependent == is_time_dependent
    assert node.probability_of_failure == probability_of_failure
    assert np.isclose(node.probability_of_no_failure, 1-probability_of_failure, atol=0.0)
    assert np.allclose(np.array(node.values), values, atol=0.0)


def test_instantiate_node_creates_correct_cpt():
    name = "test"
    is_time_dependent = True
    prob_no_failure = 0.8
    prob_failure = 0.2

    tolerance = 1e-32
    expected = np.array([prob_no_failure, prob_failure], np.float64)

    node = FaultTreeProbNode(name=name, probability_of_failure=prob_failure, is_time_dependent=is_time_dependent)

    assert np.allclose(node.values, expected, atol=tolerance)


def test_get_node_type_returns_correct():
    name = "test"
    prob = 0.2
    is_time_dependent = True

    expected_node_type = "PROB"

    node = FaultTreeProbNode(name=name, probability_of_failure=prob, is_time_dependent=is_time_dependent)

    assert node.get_node_type() == expected_node_type

def test_get_time_behaviour_returns_correct():
    name = "test"
    prob = 0.2
    is_time_dependent = True

    node = FaultTreeProbNode(name=name, probability_of_failure=prob, is_time_dependent=is_time_dependent)
    func, params = node.get_time_behaviour()

    assert func == node._FaultTreeProbNode__default_time_func
    assert params == {}


@pytest.mark.parametrize("original_prob, updated_prob", [(1.0, 0.0), (0.0, 1.0), (0.1, 0.8), (0.99,  0.001)])
def test_change_frate_sets_correct(original_prob, updated_prob):
    name = "test"
    is_time_dependent = False
    node = FaultTreeProbNode(name=name, probability_of_failure=original_prob, is_time_dependent=is_time_dependent)

    assert node.probability_of_failure == original_prob
    assert np.isclose([node.probability_of_no_failure], [1 - original_prob], atol=0.0)

    node.change_frate(probability_of_failure=updated_prob)

    assert node.probability_of_failure == updated_prob
    assert np.isclose([node.probability_of_no_failure], [1 - updated_prob], atol=0.0)


@pytest.mark.parametrize("bad_frate, is_time_dependent", [("0", True), (hex(12), False), (oct(12), True)])
def test_change_frate_invalid_type_raises_exception(bad_frate, is_time_dependent):
    name = "test"
    original_prob = 0.123
    expected_exc_substring = "Probability of failure must be a number"

    node = FaultTreeProbNode(name=name, probability_of_failure=original_prob, is_time_dependent=is_time_dependent)

    with pytest.raises(TypeError) as e:
        assert node.change_frate(probability_of_failure=bad_frate)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("wrong_prob, is_time_dependent",
                        [(-0.1, False), (1.1,  True),
                         (-1e-32, True), (5e32, False),
                         (int(123), True), (float(-456), False)])
def test_instantiate_node_wrong_probability_raises_exception(wrong_prob, is_time_dependent):
    name = "test"
    expected_exc_substring = "Invalid probability of failure"

    with pytest.raises(ValueError) as e:
        assert FaultTreeProbNode(name=name, probability_of_failure=wrong_prob, is_time_dependent=is_time_dependent)

    assert expected_exc_substring in str(e.value)


def test_get_cpt_at_time_wrong_time_raises_exception():
    name = "test"
    is_time_dependent = True
    prob_failure = 0.2

    bad_time = -123
    expected_exc_substring = "Invalid time:"

    node = FaultTreeProbNode(name=name, probability_of_failure=prob_failure, is_time_dependent=is_time_dependent)

    with pytest.raises(Exception) as e:
        assert node.get_cpt_at_time(at_time=bad_time)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("probability_of_failure, is_time_dependent, time, expected",
                        [(0.1, False, 0, np.array([[0.9], [0.1]])),
                         (0.54, False, 9876, np.array([[0.46], [0.54]])),
                         (0.1, True, 0, np.array([[0.9], [0.1]])),
                         (1e-5, True, 1e5, np.array([[0.3678794412], [0.6321205588]])) ])
def test_get_cpt_at_time_default_behaviour_returns_correct(probability_of_failure, is_time_dependent, time, expected):
    name = "test"
    tolerance = 1e-10
    node = FaultTreeProbNode(name=name, probability_of_failure=probability_of_failure, is_time_dependent=is_time_dependent)

    assert np.allclose(node.get_cpt_at_time(at_time=time).get_probabilities(), expected, atol=tolerance)



def test_change_time_behaviour_func_set_correct():
    new_func = math.exp

    name = "test"
    prob = 0.2
    is_time_dependent = True

    params = {}

    node = FaultTreeProbNode(name=name, probability_of_failure=prob, is_time_dependent=is_time_dependent)

    node.change_time_behaviour(fn_behaviour=new_func, params=params)
    assert type(node._FaultTreeProbNode__time_func) == type(new_func)
    assert node._FaultTreeProbNode__time_func == new_func


def test_change_time_behaviour_params_set_correct():
    new_func = math.exp

    name = "test"
    prob = 0.2
    is_time_dependent = True

    keys = ["A", "B", "C"]
    val = 1.0
    params = dict.fromkeys(keys, val)
    node = FaultTreeProbNode(name=name, probability_of_failure=prob, is_time_dependent=is_time_dependent)

    node.change_time_behaviour(fn_behaviour=new_func, params=params)
    assert node._FaultTreeProbNode__time_func_params == params


@pytest.mark.parametrize("bad_func",  [("BAD"), (5), []])
def test_change_time_behaviour_wrong_func_type_raises_exception(bad_func):
    name = "test"
    prob = 0.2
    is_time_dependent = True
    params = dict.fromkeys(["first_param"])

    expected_exc_substring = "Invalid object of type"

    node = FaultTreeProbNode(name=name, probability_of_failure=prob, is_time_dependent=is_time_dependent)

    with pytest.raises(Exception) as e:
        assert node.change_time_behaviour(fn_behaviour=bad_func, params=params)

    assert expected_exc_substring in str(e.value)


def test_change_time_behaviour_wrong_params_type_raises_exception():
    new_func = math.exp

    name = "test"
    prob = 0.2
    is_time_dependent = True

    bad_params = "BAD PARAMS"

    expected_exc_substring = "Invalid params of type"

    node = FaultTreeProbNode(name=name, probability_of_failure=prob, is_time_dependent=is_time_dependent)

    with pytest.raises(Exception) as e:
        assert node.change_time_behaviour(fn_behaviour=new_func, params=bad_params)

    assert expected_exc_substring in str(e.value)


def test_change_time_behaviour_sets_correct_pbf_time_at_zero():
    expected_pbf = 1.23e-4
    time_dependent_pbf = 9.87e-6

    def dummy_pbf_func(at_time):
        if at_time == 0:
            return expected_pbf

        return time_dependent_pbf

    name = "test"
    orig_prob = 0.2
    is_time_dependent = True
    new_func = dummy_pbf_func
    params = {}

    node = FaultTreeProbNode(name=name, probability_of_failure=orig_prob, is_time_dependent=is_time_dependent)

    node.change_time_behaviour(fn_behaviour=new_func, params=params)
    assert node.probability_of_failure == expected_pbf


def test_change_time_behaviour_changes_time_dependent_to_true():
    new_func = math.exp

    name = "test"
    orig_prob = 0.2
    is_time_dependent = False
    params = {}

    node = FaultTreeProbNode(name=name, probability_of_failure=orig_prob, is_time_dependent=is_time_dependent)

    node.change_time_behaviour(fn_behaviour=new_func, params=params)
    assert node.is_time_dependent != is_time_dependent


def test_reset_time_behaviour_correct_initial_states():
    expected_pbf = 1.23e-4
    time_dependent_pbf = 9.87e-6

    def dummy_pbf_func(at_time):
        if at_time == 0:
            return expected_pbf

        return time_dependent_pbf

    name = "test"
    orig_prob = 0.2
    is_time_dependent = False
    new_func = dummy_pbf_func
    params = {}

    node = FaultTreeProbNode(name=name, probability_of_failure=orig_prob, is_time_dependent=is_time_dependent)
    default_cpt_vals = node.values

    node.change_time_behaviour(fn_behaviour=new_func, params=params)
    node.reset_time_behaviour()

    assert node.probability_of_failure == orig_prob
    assert node.is_time_dependent == is_time_dependent
    assert np.allclose(node.values, default_cpt_vals, atol=0.0)


@pytest.mark.parametrize("time, expected", [(0.0, 1.23e-4), (10,  10),(9876, 9876)])
def test_evaluate_time_func_no_args_correct_pbf_differnt_times(time, expected):
    zero_pbf = 1.23e-4
    def dummy_pbf_func(at_time):
        if at_time == 0:
            return zero_pbf

        return time

    name = "test"
    orig_prob = 0.2
    is_time_dependent = True
    new_func = dummy_pbf_func
    params = {}

    node = FaultTreeProbNode(name=name, probability_of_failure=orig_prob, is_time_dependent=is_time_dependent)

    node.change_time_behaviour(fn_behaviour=new_func, params=params)
    assert node.get_cpt_at_time(at_time=time).get_probabilities()[1] == expected


@pytest.mark.parametrize("time, argument, expected", [(0.0, 1, 1.23e-4), (10,  2, 20),(9876, 3, 29628)])
def test_evaluate_time_func_args_correct_pbf_differnt_times(time, argument, expected):
    zero_pbf = 1.23e-4
    def dummy_pbf_func(at_time, passed_argument):
        if at_time == 0:
            return zero_pbf

        return time * passed_argument

    name = "test"
    orig_prob = 0.2
    is_time_dependent = True
    new_func = dummy_pbf_func
    params = {}
    params["passed_argument"] = argument

    node = FaultTreeProbNode(name=name, probability_of_failure=orig_prob, is_time_dependent=is_time_dependent)

    node.change_time_behaviour(fn_behaviour=new_func, params=params)
    assert node.get_cpt_at_time(at_time=time).get_probabilities()[1] == expected


@pytest.mark.parametrize("time, argument, kwarg, expected", [(0.0, 1, None, 0), (0.0, 1, 10, 1.23e-3), (10,  2, 20, 400),(9876, 3, 0.1, 2962.8)])
def test_evaluate_time_func_kwargs_correct_pbf_differnt_times(time, argument, kwarg, expected):
    zero_pbf = 1.23e-4
    def dummy_pbf_func(at_time, passed_argument, kwarg_a = 0):
        if at_time == 0:
            return zero_pbf * kwarg_a

        return time * passed_argument * kwarg_a

    name = "test"
    orig_prob = 0.2
    rtol = 1e-4
    is_time_dependent = True
    new_func = dummy_pbf_func
    params = {}
    params["passed_argument"] = argument
    if kwarg is not None:
        params["kwarg_a"] = kwarg

    node = FaultTreeProbNode(name=name, probability_of_failure=orig_prob, is_time_dependent=is_time_dependent)

    node.change_time_behaviour(fn_behaviour=new_func, params=params)
    assert np.isclose(node.get_cpt_at_time(at_time=time).get_probabilities()[1], expected, rtol=rtol)
