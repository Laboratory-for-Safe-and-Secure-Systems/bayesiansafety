import numpy as np
import pytest

from bayesiansafety.faulttree import FaultTreeLogicNode
#fixtures provided via conftest

@pytest.mark.parametrize("name, inputs, logic_type",
                        [("A", ["B", "C"], "AND"), ("A", ["B", "C", "D"], "AND"),
                        ("A", ["B", "C"], "OR"), ("A", ["B", "C", "D"], "OR") ])
def test_instantiate_node_success(name, inputs, logic_type):
    node = FaultTreeLogicNode(name=name, input_nodes=inputs, logic_type=logic_type)
    assert isinstance(node, FaultTreeLogicNode)

@pytest.mark.parametrize("name, bad_inputs",
                        [("A", []), ("A", ["B"])])
def test_instantiate_node_wrong_inputs_raises_exception(name, bad_inputs):
    expected_exc_substring = "Invalid input nodes:"

    with pytest.raises(Exception) as e:
        assert FaultTreeLogicNode(name=name, input_nodes=bad_inputs)

    assert expected_exc_substring in str(e.value)


def test_instantiate_node_wrong_logic_type_raises_exception():
    name = "test"
    inputs = ["A", "B"]
    bad_logic_type = "test"
    expected_exc_substring = "Unsupported logic type:"

    with pytest.raises(Exception) as e:
        assert FaultTreeLogicNode(name=name, input_nodes=inputs, logic_type=bad_logic_type)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("node_type, expected", [("AND", "AND"),("OR", "OR",)])
def test_get_node_type_returns_correct(node_type, expected):
    name = "test"
    inputs = ["A", "B"]
    node = FaultTreeLogicNode(name=name, input_nodes=inputs, logic_type=node_type)

    assert node.get_node_type() == expected


@pytest.mark.parametrize("node_type, inputs, expected",
                         [("OR", ["A", "B"], np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]], np.float64)),
                         ("AND", ["A", "B"], np.array([[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], np.float64)) ])
def test_instantiate_node_creates_correct_cpt(node_type, inputs, expected):
    name = "test"
    tolerance = 1e-32
    node = FaultTreeLogicNode(name=name, input_nodes=inputs, logic_type=node_type)

    assert np.allclose(node.cpt.get_probabilities(), expected, atol=tolerance)
