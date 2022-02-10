import pytest
#fixtures provided via conftest

from bayesiansafety.core import BayesianNetwork
from bayesiansafety.faulttree import BayesianFaultTree
from bayesiansafety.synthesis.hybrid import HybridConfiguration
from bayesiansafety.synthesis.hybrid import HybridBuilder


def test_instantiate_success(fixture_bn_confounder_param, fixture_ft_elevator_model):
    bn_inst, _ = fixture_bn_confounder_param("BN_A")
    ft_inst, _ = fixture_ft_elevator_model

    name = "test"
    shared_nodes = {"BN_A":["NODE_A"]}
    couplings = {"BN_A":[("NODE_A", "AND_TOP")]}
    pbf_states = {"BN_A":[("NODE_A", "STATE_NODE_A_No")]}

    config = HybridConfiguration(name=name, shared_nodes=shared_nodes, ft_coupling_points=couplings, pbf_states=pbf_states)

    asso_bns = {"BN_A":bn_inst}
    builder = HybridBuilder(ft_inst=ft_inst, configuration=config, associated_bns=asso_bns)


    assert builder.ft_inst == ft_inst
    assert builder.configuration  == config
    assert builder.associated_bns  == asso_bns


@pytest.mark.parametrize("observations, expected_edges",
[ (None                                        , [("BN_A_NODE_A", "AND_TOP"), ("BN_A_NODE_B", "OR_SIG_HI")] ),
  ({"BN_A":[("NODE_C", "STATE_NODE_C_Yes")]}, [("BN_A_NODE_A", "AND_TOP"), ("BN_A_NODE_B", "OR_SIG_HI")] ),
])
def test_get_extended_fault_tree_successful(observations, expected_edges, fixture_bn_confounder_param, fixture_ft_elevator_model,):
    bn_inst, _ = fixture_bn_confounder_param("BN_A")
    ft_inst, _ = fixture_ft_elevator_model

    name = "test"
    shared_nodes = {"BN_A":["NODE_A", "NODE_B"]}
    couplings = {"BN_A":[("NODE_A", "AND_TOP"), ("NODE_B", "OR_SIG_HI")]}
    pbf_states = {"BN_A":[("NODE_A", "STATE_NODE_A_No"), ("NODE_B", "STATE_NODE_B_Yes")]}

    config = HybridConfiguration(name=name, shared_nodes=shared_nodes, ft_coupling_points=couplings, pbf_states=pbf_states)

    asso_bns = {"BN_A":bn_inst}
    builder = HybridBuilder(ft_inst=ft_inst, configuration=config, associated_bns=asso_bns)

    ext_ft_inst = builder.get_extended_fault_tree(bn_observations=observations)

    assert isinstance(ext_ft_inst, BayesianFaultTree)
    for src, dest in expected_edges:
        assert ext_ft_inst.model.has_edge(src, dest)



@pytest.mark.parametrize("at_time, fix, shareds, couplings, states,  expected_edges",
[    (    0,
        True,
        {"BN_A":["NODE_A", "NODE_B"]},
        {"BN_A":[("NODE_A", "AND_TOP"), ("NODE_B", "OR_SIG_HI")]},
        {"BN_A":[("NODE_A", "STATE_NODE_A_No"), ("NODE_B", "STATE_NODE_B_Yes")]},
        {"BN_A": [("BN_A_NODE_A", "AND_TOP"), ("BN_A_NODE_B", "OR_SIG_HI"),
            # check also linked BN nodes
            ("BN_A_NODE_A", "BN_A_NODE_B"), ("BN_A_NODE_A", "BN_A_NODE_C")]}
    ),

    (    123,
        True,
        {"BN_A":["NODE_A", "NODE_B"]},
        {"BN_A":[("NODE_A", "AND_TOP"), ("NODE_B", "OR_SIG_HI")]},
        {"BN_A":[("NODE_A", "STATE_NODE_A_No"), ("NODE_B", "STATE_NODE_B_Yes")]},
        {"BN_A": [("BN_A_NODE_A", "AND_TOP"), ("BN_A_NODE_B", "OR_SIG_HI"),
            # check also linked BN nodes
            ("BN_A_NODE_A", "BN_A_NODE_B"), ("BN_A_NODE_A", "BN_A_NODE_C")]}
    ),

    (    456,
        True,
        {"BN_A":["NODE_A", "NODE_B"], "BN_B":["NODE_B"]},
        {"BN_A":[("NODE_A", "AND_TOP"), ("NODE_B", "OR_SIG_HI")], "BN_B":[("NODE_B", "OR_SIG_LO")]},
        {"BN_A":[("NODE_A", "STATE_NODE_A_No"), ("NODE_B", "STATE_NODE_B_Yes")], "BN_B":[("NODE_B", "STATE_NODE_B_Yes")]},
        {"BN_A": [("BN_A_NODE_A", "AND_TOP"), ("BN_A_NODE_B", "OR_SIG_HI"),
            # check also linked BN nodes
            ("BN_A_NODE_A", "BN_A_NODE_B"), ("BN_A_NODE_A", "BN_A_NODE_C")],
         "BN_B":[("BN_B_NODE_B", "OR_SIG_LO"),
            # check also linked BN nodes
            ("BN_B_NODE_A", "BN_B_NODE_B"), ("BN_B_NODE_A", "BN_B_NODE_C")]}
    ),

    (    789,
        False,
        {"BN_A":["NODE_A", "NODE_B"], "BN_B":["NODE_B"]},
        {"BN_A":[("NODE_A", "AND_TOP"), ("NODE_B", "OR_SIG_HI")], "BN_B":[("NODE_B", "OR_SIG_LO")]},
        {"BN_A":[("NODE_A", "STATE_NODE_A_No"), ("NODE_B", "STATE_NODE_B_Yes")], "BN_B":[("NODE_B", "STATE_NODE_B_Yes")]},
        [("BN_A_NODE_A", "AND_TOP"), ("BN_A_NODE_B", "OR_SIG_HI"), ("BN_B_NODE_B", "OR_SIG_LO"),
            # check also linked BN nodes
            ("BN_A_NODE_A", "BN_A_NODE_B"), ("BN_A_NODE_A", "BN_A_NODE_C"),
            ("BN_B_NODE_A", "BN_B_NODE_B"), ("BN_B_NODE_A", "BN_B_NODE_C")]
    )
])
def test_get_hybrid_networks_returns_correct(at_time, fix, shareds, couplings, states,  expected_edges, fixture_bn_confounder_param, fixture_ft_elevator_model,):
    ft_inst, _ = fixture_ft_elevator_model
    config = HybridConfiguration(name=ft_inst.name, shared_nodes=shareds, ft_coupling_points=couplings, pbf_states=states)

    asso_bns = {}
    for bn_name in shareds.keys():
        bn_inst, _ = fixture_bn_confounder_param(bn_name)
        asso_bns[bn_name] = bn_inst

    builder = HybridBuilder(ft_inst=ft_inst, configuration=config, associated_bns=asso_bns)

    hybrid_networks = builder.get_hybrid_networks(at_time=at_time, fix_other_bns=fix)

    assert all(isinstance(bn, BayesianNetwork) for bn in hybrid_networks.values())
    assert all(str(at_time) in bn_name for bn_name in hybrid_networks.keys())

    if fix is True:
        for bn_name, exp_edges in expected_edges.items():
            for hybrid_name, bn_inst in hybrid_networks.items():
                if bn_name in hybrid_name:
                    for src, dest in exp_edges:
                        assert bn_inst.has_edge(src, dest)

    else:
        assert len(hybrid_networks) == 1
        first_bn_name = next(iter(hybrid_networks))

        for src, dest in expected_edges:
            assert hybrid_networks[first_bn_name].has_edge(src, dest)


def test_get_hybrid_networks_negative_time_raises_excpetion(fixture_hybrid_config, fixture_bn_confounder_param, fixture_ft_elevator_model,):
    invalid_time = -123
    bn_inst, _ = fixture_bn_confounder_param("BN_A")
    ft_inst, _ = fixture_ft_elevator_model
    config = fixture_hybrid_config("test")

    asso_bns = {"BN_A":bn_inst}
    builder = HybridBuilder(ft_inst=ft_inst, configuration=config, associated_bns=asso_bns)

    expected_exc_substring = "A time stamp for evaluation can not be negative but was requested at time"

    with pytest.raises(ValueError) as e:
        assert builder.get_hybrid_networks(at_time=invalid_time, fix_other_bns=True)

    assert expected_exc_substring in str(e.value)
