import pytest
import numpy as np

from bayesiansafety.core.inference import TwinNetwork
PREFIX = TwinNetwork.TWIN_NET_PREFIX

# fixtures provided via conftest


@pytest.mark.parametrize("fixture, expected_new_node_connections, expected_nr_new_nodes",
                         [ ("fixture_bn_collider_param", [("NODE_A", PREFIX + "NODE_C"), ("NODE_B", PREFIX + "NODE_C")], 1),
                           ("fixture_bn_confounder_param", [
                           ("NODE_A", PREFIX + "NODE_B"), ("NODE_A", PREFIX + "NODE_C")], 2),
                           ("fixture_bn_independent_nodes_only_param", [], 0), ])
def test_build_twin_network_success(fixture, expected_new_node_connections, expected_nr_new_nodes, request):
    model_name = 'test'
    model, _ = request.getfixturevalue(fixture)(model_name)

    twin_model = TwinNetwork.build_twin_network(model)

    assert len(twin_model.model_elements.keys()) == (
        len(model.model_elements.keys()) + expected_nr_new_nodes)

    assert len(twin_model.node_connections) == len(
        model.node_connections) + len(expected_new_node_connections)

    for tup in expected_new_node_connections:
        assert tup in twin_model.node_connections


def test_build_twin_network_isolated_nodes_success(fixture_bn_independent_nodes_only_param):
    model_name = 'test'
    model, _ = fixture_bn_independent_nodes_only_param(model_name)

    twin_model = TwinNetwork.build_twin_network(model)

    # in this case only exogenous vars
    assert set(twin_model.model_elements.keys()) == set(
        model.model_elements.keys())


def test_create_twin_node_success(fixture_bn_causal_queries_param):
    model_name = 'test'
    model = fixture_bn_causal_queries_param(model_name)

    root_vars = model.get_root_node_names()

    for name, elem in model.model_elements.items():
        twin_node = TwinNetwork.create_twin_node(elem, root_vars)

        assert isinstance(twin_node, type(elem))
        assert elem.name in twin_node.name and TwinNetwork.TWIN_NET_PREFIX == twin_node.name[:len(TwinNetwork.TWIN_NET_PREFIX)]
        assert len(elem.evidence) == len(twin_node.evidence) if elem.evidence is not None else not twin_node.evidence
        
        assert np.allclose(np.array(twin_node.evidence_card), np.array(elem.evidence_card)) if elem.evidence_card is not None else not twin_node.evidence_card
        assert np.allclose(np.array(twin_node.variable_card), np.array(elem.variable_card))
        assert np.allclose(np.array(twin_node.values), np.array(elem.values))

        assert len(twin_node.state_names.keys()) == len(elem.state_names.keys()) if elem.state_names is not None else not twin_node.state_names
        if elem.state_names:
            elem_states = []
            twin_states = []
            [elem_states.extend(cur_states)
             for cur_states in elem.state_names.values()]
            [twin_states.extend(cur_states)
             for cur_states in twin_node.state_names.values()]

            assert len(elem_states) == len(twin_states) and sorted(
                elem_states) == sorted(twin_states)
