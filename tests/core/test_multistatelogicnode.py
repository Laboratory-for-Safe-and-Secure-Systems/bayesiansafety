import numpy as np
import pytest

from bayesiansafety.core import MultistateLogicNode
# fixtures provided via conftest


@pytest.mark.parametrize("name, inputs, state_names, cards, bad_states, logic_type, expected_cards, expected_bstates",
                         [("test_1", ["A", "B"], {"A": ["a", "aa"], "B":["b", "bb"]}, None, None, "AND", [2, 2], {"A": 1, "B": 1}),
                          ("test_2", ["A", "B", "C"], {"A": ["a", "aa"], "B":["b", "bb", "bbb"], "C":["c", "cc", "ccc", "cccc"]}, [2, 3, 4], None, "OR", [2, 3, 4], {"A": 1, "B": 1, "C": 1}),
                          ("test_3", ["A", "B"], {"A": ["a"*i for i in range(1, 11)], "B":["b", "bb"]}, [10, 2], {"A": 9, "B": 1}, "AND", [10, 2], {"A": 9, "B": 1})])
def test_instantiate_node_success(name, inputs, state_names, cards, bad_states, logic_type, expected_cards, expected_bstates):
    msate_node = MultistateLogicNode(name=name, input_nodes=inputs, input_state_names=state_names,
                                     cardinalities=cards, bad_states=bad_states, logic_type=logic_type)

    assert isinstance(msate_node, MultistateLogicNode)
    assert msate_node.name == name
    assert np.allclose(msate_node.cardinalities, expected_cards, atol=0.0)
    assert msate_node.bad_states == expected_bstates
    assert msate_node.cpt is not None
    assert state_names.items() <= msate_node.input_state_names.items()
    assert state_names.items() <= msate_node.cpt.state_names.items()


@pytest.mark.parametrize("logic_type", ["AND", "OR"])
def test_get_node_type_returns_correct(logic_type):
    name = "test"
    inputs = ["A", "B"]
    msate_node = MultistateLogicNode(name=name, input_nodes=inputs, input_state_names={}, cardinalities=None, bad_states=None, logic_type=logic_type)

    assert msate_node.get_node_type() == logic_type


@pytest.mark.parametrize("inputs,", [[], ["A"]])
def test_instantiate_invalid_nr_of_inputs_raises_exception(inputs):
    expected_exc_substring = "You need to specify at least two nodes"
    name = "test"

    with pytest.raises(ValueError) as e:
        assert MultistateLogicNode(name=name, input_nodes=inputs, input_state_names={})

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("logic_type,", ["TEST", 5, None])
def test_instantiate_invalid_logic_type_raises_exception(logic_type):
    expected_exc_substring = "Valid types are 'OR' and 'AND'"
    name = "test"
    inputs = ["A", "B"]

    with pytest.raises(ValueError) as e:
        assert MultistateLogicNode(name=name, input_nodes=inputs, input_state_names={}, logic_type=logic_type)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("inputs            , cards     , bad_states            , logic_type, expected_cpt",
                         [(["A", "B"], [2, 2], None, "OR",  [[1, 0, 0, 0], [0, 1, 1, 1]]),
                          (["A", "B"], [2, 2], None, "AND", [[1, 1, 1, 0], [0, 0, 0, 1]]),
                          (["A", "B", "C"], [2, 2, 3], {"A": 1, "B": 1, "C": 1}, "OR", [[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
                          (["A", "B", "C"], [2, 2, 3], {"A": 1, "B": 1, "C": 1}, "AND",[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]),
                          (["A", "B", "C"], [2, 2, 3], {"A": 0, "B": 1, "C": 2}, "OR", [[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]]),
                          (["A", "B", "C"], [2, 2, 3], {"A": 0, "B": 1, "C": 2}, "AND",[[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]),
                          ])
def test_instantiate_creates_valid_probability_table(inputs, cards, bad_states, logic_type, expected_cpt):
    name = "test"
    msate_node = MultistateLogicNode(name=name, input_nodes=inputs, input_state_names={}, cardinalities=cards, bad_states=bad_states, logic_type=logic_type)

    reshaped_values = msate_node.cpt.get_probabilities().reshape(msate_node.cardinalities[0], -1)
    assert np.allclose(reshaped_values, expected_cpt, atol=0.0)
