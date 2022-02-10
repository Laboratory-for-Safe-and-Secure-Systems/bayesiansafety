import pytest
#fixtures provided via conftest

from bayesiansafety.synthesis.hybrid import HybridConfiguration


@pytest.mark.parametrize("name, shareds, couplings, states",
[    # arugments partially missing
    ( "config_1", {"BN_A":["NODE_A"]}, {} , {}),
    ( "config_2", {"BN_A":["NODE_A"]}, {} , {"BN_A":[("NODE_A", "STATE_NODE_A_No")]}),
    ( "config_3", {"BN_A":["NODE_A"]}, {"BN_A":[("NODE_A", "OR_SIG_HI")]} , {"BN_A":[("NODE_A", "STATE_NODE_A_No")]}),
    ( "config_4", {"BN_A":["NODE_A"], "BN_B":["NODE_B", "NODE_C"]}, {"BN_A":[("NODE_A", "OR_SIG_HI")]} , {"BN_A":[("NODE_A", "STATE_NODE_A_No")]}),
    ( "config_5", {"BN_A":["NODE_A", "NODE_B", "NODE_C"]}, {"BN_A":[("NODE_A", "OR_SIG_HI")]} , {"BN_A":[("NODE_A", "STATE_NODE_A_No")]}),
    ( "config_6", {"BN_A":["NODE_A"], "BN_B":["NODE_B", "NODE_C"]}, {"BN_A":[("NODE_A", "OR_SIG_HI")]} , {"BN_A":[("NODE_A", "STATE_NODE_A_No")]}),
    ( "config_7", {"BN_A":["NODE_A"], "BN_B":["NODE_B", "NODE_C"]},     {"BN_A":[("NODE_A", "OR_SIG_HI")], "BN_B":[("NODE_B", "OR_SIG_LO"), ("NODE_C", "AND_TOP")]}, {"BN_A":[("NODE_A", "STATE_NODE_A_No")], "BN_B":[("NODE_B", "STATE_NODE_B_Yes"), ("NODE_C", "STATE_NODE_C_Yes")]}),
])
def test_instantiate_success(name, shareds, couplings, states):
    config = HybridConfiguration(name=name, shared_nodes=shareds, ft_coupling_points=couplings, pbf_states=states)

    assert config.name == name
    assert config.ft_coupling_points == couplings
    assert config.pbf_states == states
    assert config.shared_nodes == shareds



def test_invalid_mounting_point_not_in_shared_raises_exception():
    name = "test"
    shared_nodes = {"BN_A":["NODE_A"]}
    couplings = {"BN_A":[("BAD_NODE", "AND_TOP")]}
    pbf_states = {"BN_A":[("NODE_A", "STATE_NODE_A_No")]}

    expected_exc_substring = "that are not part of the shared nodes for this BN."

    with pytest.raises(ValueError) as e:
        assert HybridConfiguration(name=name, shared_nodes=shared_nodes, ft_coupling_points=couplings, pbf_states=pbf_states)

    assert expected_exc_substring in str(e.value)


def test_invalid_pbf_node_not_in_shared_raises_exception():
    name = "test"
    shared_nodes = {"BN_A":["NODE_A"]}
    couplings = {"BN_A":[("NODE_A", "AND_TOP")]}
    pbf_states = {"BN_A":[("BAD_NODE", "STATE_NODE_A_No")]}

    expected_exc_substring = "Not all nodes are part of the shared nodes"

    with pytest.raises(ValueError) as e:
        assert HybridConfiguration(name=name, shared_nodes=shared_nodes, ft_coupling_points=couplings, pbf_states=pbf_states)

    assert expected_exc_substring in str(e.value)
