from unittest.mock import patch
import pytest
import numpy as np
# fixtures provided via conftest

from bayesiansafety.core import BayesianNetwork
from bayesiansafety.core import DiscreteFactor
from bayesiansafety.core import ConditionalProbabilityTable


def test_instantiate_skeleton_success():
    model_name = 'test'
    correct_node_connections = [('A', "CENTRAL"), ('B', "CENTRAL"), ('C', "CENTRAL")]
    correct_node_names = set(["A", "B", "C", "CENTRAL"])

    model = BayesianNetwork(name=model_name, node_connections=correct_node_connections)

    assert model.name == model_name
    assert len(set(model.node_connections) ^ set(correct_node_connections)) == 0
    assert len(set(model.model_elements.keys()) ^ correct_node_names) == 0


def test_instantiate_skeleton_with_cycles_raises_exception():
    expected_exc_substring = "Cycles are not allowed in a DAG"
    model_name = 'test'
    cyclic_node_connections = [('A', "B"), ('B', "A")]

    with pytest.raises(ValueError) as e:
        assert BayesianNetwork(
            name=model_name, node_connections=cyclic_node_connections)

    assert expected_exc_substring in str(e.value)


def test_instantiate_add_cpt_success():
    model_name = 'test'
    correct_node_connections = [('NODE_A', "NODE_C"), ('NODE_B', "NODE_C")]

    NODE_A = ConditionalProbabilityTable(name="NODE_A", variable_card=2, values=[[0.1], [0.9]], evidence=None, evidence_card=None, state_names={"NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"]})
    NODE_B = ConditionalProbabilityTable(name="NODE_B", variable_card=2, values=[[0.12, 0.34], [0.88, 0.66]], evidence=["NODE_A"], evidence_card=[2], state_names={"NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"], "NODE_B": ["STATE_NODE_B_Yes", "STATE_NODE_B_No"]})
    NODE_C = ConditionalProbabilityTable(name="NODE_C", variable_card=2, values=[[0.98, 0.76], [0.02, 0.24]], evidence=["NODE_A"], evidence_card=[2], state_names={"NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"], "NODE_C": ["STATE_NODE_C_Yes", "STATE_NODE_C_No"]})

    model = BayesianNetwork(name=model_name, node_connections=correct_node_connections)
    model.add_cpts(NODE_A, NODE_B, NODE_C)

    assert model.model_elements["NODE_A"] == NODE_A
    assert model.model_elements["NODE_B"] == NODE_B
    assert model.model_elements["NODE_C"] == NODE_C


@pytest.mark.parametrize("fixture, expected_nodes", [("fixture_bn_confounder_param", ["NODE_A"]), ("fixture_bn_collider_param", ["NODE_A", "NODE_B"])])
def test_get_root_nodes_successful(fixture, expected_nodes, request):
    model_name = 'test'
    model, marginals = request.getfixturevalue(fixture)(model_name)

    queried_roots = model.get_root_node_names()

    assert len(queried_roots) == len(expected_nodes)
    assert len(set(queried_roots) ^ set(expected_nodes)) == 0


@pytest.mark.parametrize("invalid_cpt_instance", [None, {"NODE_C": [0.1, 0.9]}, DiscreteFactor("NODE_C", ["A", "B"], [2, 2], np.ones(4))])
def test_add_cpts_invalid_instance_raises_exception(invalid_cpt_instance):
    expected_exc_substring = "CPTs to add must be instantiations of"
    model_name = 'test'
    node_connections = [('NODE_A', "NODE_C"), ('NODE_B', "NODE_C")]

    NODE_A = ConditionalProbabilityTable(name="NODE_A", variable_card=2, values=[[0.1], [0.9]], evidence=None, evidence_card=None, state_names={"NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"]})
    NODE_B = ConditionalProbabilityTable(name="NODE_B", variable_card=2, values=[[0.12, 0.34], [0.88, 0.66]], evidence=["NODE_A"], evidence_card=[2], state_names={"NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"], "NODE_B": ["STATE_NODE_B_Yes", "STATE_NODE_B_No"]})

    model = BayesianNetwork(name=model_name, node_connections=node_connections)

    with pytest.raises(TypeError) as e:
        assert model.add_cpts(NODE_A, NODE_B, invalid_cpt_instance)

    assert expected_exc_substring in str(e.value)


def test_copy_success(fixture_bn_confounder_param):
    model_name = 'test'
    model, _ = fixture_bn_confounder_param(model_name)

    copied_model = model.copy()

    assert model.name == copied_model.name
    assert len(set(model.node_connections) ^ set(copied_model.node_connections)) == 0
    assert len(set(model.model_elements.keys()) ^ set(copied_model.model_elements.keys())) == 0
    assert type(model) == type(copied_model)
    assert isinstance(copied_model, BayesianNetwork)


@pytest.mark.parametrize("fixture, expected", [("fixture_bn_independent_nodes_only_param", set(["NODE_A", "NODE_B", "NODE_C"])),
                                               ("fixture_bn_confounder_param", set()),
                                               ("fixture_bn_collider_param", set())])
def test_get_independent_nodes_returns_correct(fixture, expected, request):
    model_name = 'test'
    model, _ = request.getfixturevalue(fixture)(model_name)

    assert model.get_independent_nodes() == expected


@patch("matplotlib.pyplot.show")
@pytest.mark.parametrize("title, edge_labels, options", [("Test_1", None, None),
                                                         ("Test_2", {("NODE_A", "NODE_B"): "Test_2"}, None),
                                                         ("Test_3", {("NODE_A", "NODE_B"): "Test_3"}, {"font_size": 10, "node_size": 2000})])
def test_plot_graph_executes_successfully(mock_show, title, edge_labels, options, fixture_bn_confounder_param):
    model, _ = fixture_bn_confounder_param(title)
    model.plot_graph(title=title, edge_labels=edge_labels, options=options)
    mock_show.assert_called()


@pytest.mark.parametrize("fixture, prefix, expected", [("fixture_bn_confounder_param", "", set(["NODE_A", "NODE_B", "NODE_C"])),
                                                       ("fixture_bn_collider_param", "test_1_", set(["test_1_NODE_A", "test_1_NODE_B", "test_1_NODE_C"])),
                                                       ("fixture_bn_independent_nodes_only_param", "test_2_", set(["test_2_NODE_A", "test_2_NODE_B", "test_2_NODE_C"]))])
def test_get_prefixed_copy_returns_correct(fixture, prefix, expected, request):
    model_name = 'test'
    model, _ = request.getfixturevalue(fixture)(model_name)

    prefixed_copy = BayesianNetwork.get_prefixed_copy(model, prefix)

    assert isinstance(prefixed_copy, BayesianNetwork)
    assert len(set(prefixed_copy.model_elements.keys()) ^ expected) == 0


@pytest.mark.parametrize("invalid_bn_inst", [None, 5, "test", DiscreteFactor("NODE_C", ["A", "B"], [2, 2], np.ones(4))])
def test_get_prefixed_copy_invalid_bn_instance_type_raises_excpetion(invalid_bn_inst):
    expected_exc_substring = "Passed BN instance must be an instantiation of the class"
    prefix = "test"

    with pytest.raises(TypeError) as e:
        assert BayesianNetwork.get_prefixed_copy(invalid_bn_inst, prefix)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("invalid_prefix", [None, 5, {"NODE_A": "pref_A", "NODE_B": "pref_B"}, DiscreteFactor("NODE_C", ["A", "B"], [2, 2], np.ones(4))])
def test_get_prefixed_copy_invalid_prefix_string_type_raises_excpetion(fixture_bn_confounder_param, invalid_prefix):
    expected_exc_substring = "Passed prefix string must be string object and not an instantiation of"
    model_name = 'test'
    model, _ = fixture_bn_confounder_param(model_name)

    with pytest.raises(TypeError) as e:
        assert BayesianNetwork.get_prefixed_copy(model, invalid_prefix)

    assert expected_exc_substring in str(e.value)
