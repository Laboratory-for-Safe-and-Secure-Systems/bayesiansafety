import pytest
import numpy as np

from bayesiansafety.core import ConditionalProbabilityTable
# fixtures provided via conftest


@pytest.mark.parametrize("name, var_card, vals                      , evi               , evi_card , states",
                         [("",       2,  [[0.1], [0.9]], None, None, {"": ["True", "False"]}),
                          ("test",  3,  [[0.12], [0.34], [0.54]], None, None, {"test": ["A", "B", "C"]}),
                          ("grade",  3,  [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                          [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                          [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]], ['diff', 'intel'], [2, 3],   {"grade": ["good", "bad", "catastrophic"]})])
def test_instantiate_success(name, var_card, vals, evi, evi_card, states):
    CPT = ConditionalProbabilityTable(name=name, variable_card=var_card,
                                      values=vals, evidence=evi, evidence_card=evi_card, state_names=states)

    assert CPT.name == name
    assert CPT.variable_card == var_card
    assert CPT.values == vals
    assert CPT.evidence == evi
    assert CPT.evidence_card == evi_card
    assert CPT.state_names == states


def test_copy_succes():
    CPT = ConditionalProbabilityTable(name="test", variable_card=3, values=[[0.12], [0.34], [0.54]],
                                      evidence=None, evidence_card=None, state_names={"test": ["A", "B", "C"]})

    copied_CPT = CPT.copy()

    assert CPT.name == copied_CPT.name
    assert CPT.variable_card == copied_CPT.variable_card
    assert np.allclose(CPT.values, copied_CPT.values, atol=1e-32)
    assert CPT.evidence == copied_CPT.evidence
    assert CPT.evidence_card == copied_CPT.evidence_card
    assert CPT.state_names == copied_CPT.state_names


@pytest.mark.parametrize("combination, expected",
                         [({"A": "s_a0", "B": "s_b0", "C": "s_c0"}, 0.0123),
                          ({"A": "s_a0", "B": "s_b0", "C": "s_c1"}, 0.0456),
                             ({"A": "s_a0", "B": "s_b1", "C": "s_c0"}, 0.789),
                             ({"A": "s_a0", "B": "s_b1", "C": "s_c1"}, 0.1531),
                             ({"A": "s_a1", "B": "s_b0", "C": "s_c0"}, 0.147),
                             ({"A": "s_a1", "B": "s_b0", "C": "s_c1"}, 0.258),
                             ({"A": "s_a1", "B": "s_b1", "C": "s_c0"}, 0.369),
                             ({"A": "s_a1", "B": "s_b1", "C": "s_c1"}, 0.226)])
def test_get_value_returns_correct(combination, expected):
    #      s_b0         sb_1
    # s_c0     s_c1   s_c0   s_1
    CPT = ConditionalProbabilityTable(name="A", variable_card=2, values=[[0.0123, 0.0456, 0.789, 0.1531],  # s_a0
                                                                         [0.147, 0.258, 0.369, 0.226]],  # s_a1
                                      evidence=["B", "C"], evidence_card=[2, 2], state_names={"A": ["s_a0", "s_a1"],
                                                                                              "B": ["s_b0", "s_b1"],
                                                                                              "C": ["s_c0", "s_c1"]})

    assert np.isclose(CPT.get_value(combination), expected, atol=0.0)


@pytest.mark.parametrize("bad_combination", [{"A": "s_a0", "B": "s_b1"},  {"A": "s_a0", "B": "s_b1", "C": "s_c0", "D": "s_d1"}])
def test_get_value_wrong_number_of_query_states_raises_exception(bad_combination):
    expected_exc_substring = "do not match with the CPTs named vars"
    #      s_b0         sb_1
    # s_c0     s_c1   s_c0   s_1
    CPT = ConditionalProbabilityTable(name="A", variable_card=2, values=[[0.0123, 0.0456, 0.789, 0.1531],  # s_a0
                                                                         [0.147, 0.258, 0.369, 0.226]],  # s_a1
                                      evidence=["B", "C"], evidence_card=[2, 2], state_names={"A": ["s_a0", "s_a1"],
                                                                                              "B": ["s_b0", "s_b1"],
                                                                                              "C": ["s_c0", "s_c1"]})
    with pytest.raises(ValueError) as e:
        assert CPT.get_value(bad_combination)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("bad_combination", [{"A": "s_a2", "B": "s_b1", "C": "s_c0"}, {"A": "s_a0", "B": "s_b1", "C": 0}])
def test_get_value_invalid_state_name_raises_exception(bad_combination):
    expected_exc_substring = "are not part of the listed state names"
    #      s_b0         sb_1
    # s_c0     s_c1   s_c0   s_1
    CPT = ConditionalProbabilityTable(name="A", variable_card=2, values=[[0.0123, 0.0456, 0.789, 0.1531],  # s_a0
                                                                         [0.147, 0.258, 0.369, 0.226]],  # s_a1
                                      evidence=["B", "C"], evidence_card=[2, 2], state_names={"A": ["s_a0", "s_a1"],
                                                                                              "B": ["s_b0", "s_b1"],
                                                                                              "C": ["s_c0", "s_c1"]})
    with pytest.raises(ValueError) as e:
        assert CPT.get_value(bad_combination)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("state_name, expected", [("A", 0), ("B", 1),  ("C", 2), ])
def test_get_index_of_state_returns_correct(state_name, expected):
    CPT = ConditionalProbabilityTable(name="test", variable_card=3, values=[[0.12], [0.34], [0.54]],
                                      evidence=None, evidence_card=None, state_names={"test": ["A", "B", "C"]})

    assert CPT.get_index_of_state(state_name) == expected


def test_get_index_of_state_empty_state_names_raises_exception():
    expected_exc_substring = "Conditional probability table does not contain state names"
    var_name = "test"
    queried_state_name = "A"
    empty_state_names = {}

    CPT = ConditionalProbabilityTable(name=var_name, variable_card=3, values=[[0.12], [0.34], [0.54]],
                                      evidence=None, evidence_card=None, state_names=empty_state_names)

    with pytest.raises(ValueError) as e:
        assert CPT.get_index_of_state(queried_state_name)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("bad_query", [0, None, {"test": "A"}])
def test_get_index_of_state_non_string_query_raises_exception(bad_query):
    expected_exc_substring = "is not a string"
    var_name = "test"
    state_names = {var_name: ["A", "B", "C"]}

    CPT = ConditionalProbabilityTable(name=var_name, variable_card=3, values=[[0.12], [0.34], [0.54]],
                                      evidence=None, evidence_card=None, state_names=state_names)

    with pytest.raises(TypeError) as e:
        assert CPT.get_index_of_state(bad_query)

    assert expected_exc_substring in str(e.value)


def test_get_index_of_state_invalid_state_name_raises_exception():
    expected_exc_substring = "is not specified for this variable"
    var_name = "test"
    invalid_state = "bad_name"
    state_names = {var_name: ["A", "B", "C"]}

    CPT = ConditionalProbabilityTable(name=var_name, variable_card=3, values=[[0.12], [0.34], [0.54]],
                                      evidence=None, evidence_card=None, state_names=state_names)

    with pytest.raises(ValueError) as e:
        assert CPT.get_index_of_state(invalid_state)

    assert expected_exc_substring in str(e.value)
