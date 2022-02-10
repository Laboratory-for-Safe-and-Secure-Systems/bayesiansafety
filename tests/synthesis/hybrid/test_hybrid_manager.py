import pytest
#fixtures provided via conftest

from bayesiansafety.synthesis.hybrid import HybridManager
from bayesiansafety.synthesis.hybrid import HybridConfiguration

@pytest.mark.parametrize("configs, all_shared_nodes, all_couplings, all_pbf_states",
[    # arugments partially missing
    ( {"ID_1":HybridConfiguration("elevator_model", {"BN_A":["NODE_A"]}, {}, {}) },
             {"BN_A":{"NODE_A"}}    , {}            , {}       ),
    ( {"ID_2":HybridConfiguration("elevator_model", {"BN_A":["NODE_A"]}, {} , {"BN_A":[("NODE_A", "STATE_NODE_A_No")]}) },
             {"BN_A":{"NODE_A"}}    ,  {}    , {"BN_A":{("NODE_A", "STATE_NODE_A_No")}} ),
    ( {"ID_3":HybridConfiguration("elevator_model", {"BN_A":["NODE_A"]}, {"BN_A":[("NODE_A", "OR_SIG_HI")]} , {"BN_A":[("NODE_A", "STATE_NODE_A_No")]}) },
         {"BN_A":{"NODE_A"}}    ,  {"BN_A":{("NODE_A", "OR_SIG_HI")}}    , {"BN_A":{("NODE_A", "STATE_NODE_A_No")}} ),

    # all arguments satisfied
    ( {"ID_4":HybridConfiguration("elevator_model", {"BN_A":["NODE_A"], "BN_B":["NODE_B", "NODE_C"]}, {"BN_A":[("NODE_A", "OR_SIG_HI")]} , {"BN_A":[("NODE_A", "STATE_NODE_A_No")]}) },
         {"BN_A":{"NODE_A"}, "BN_B":{"NODE_B", "NODE_C"}}    ,  {"BN_A":{("NODE_A", "OR_SIG_HI")}}    , {"BN_A":{("NODE_A", "STATE_NODE_A_No")}} ),

    # more shared nodes than used
    ( {"ID_4":HybridConfiguration("elevator_model", {"BN_A":["NODE_A", "NODE_B", "NODE_C"]}, {"BN_A":[("NODE_A", "OR_SIG_HI")]} , {"BN_A":[("NODE_A", "STATE_NODE_A_No")]}) },
     {"BN_A":{"NODE_A", "NODE_B", "NODE_C"}}    ,  {"BN_A":{("NODE_A", "OR_SIG_HI")}}    , {"BN_A":{("NODE_A", "STATE_NODE_A_No")}} ),

    # multiple shared nodes from different networks
    ( {"ID_5":HybridConfiguration("elevator_model", {"BN_A":["NODE_A"], "BN_B":["NODE_B", "NODE_C"]}, {"BN_A":[("NODE_A", "OR_SIG_HI")]} , {"BN_A":[("NODE_A", "STATE_NODE_A_No")]}) },
     {"BN_A":{"NODE_A"}, "BN_B":{"NODE_B", "NODE_C"}}    ,  {"BN_A":{("NODE_A", "OR_SIG_HI")}}    , {"BN_A":{("NODE_A", "STATE_NODE_A_No")}} ),

    # multiple bayesian networks in reasonable usage
    ( {"ID_6":HybridConfiguration("elevator_model",     {"BN_A":["NODE_A"], "BN_B":["NODE_B", "NODE_C"]},     {"BN_A":[("NODE_A", "OR_SIG_HI")], "BN_B":[("NODE_B", "OR_SIG_LO"), ("NODE_C", "AND_TOP")]},     {"BN_A":[("NODE_A", "STATE_NODE_A_No")], "BN_B":[("NODE_B", "STATE_NODE_B_Yes"), ("NODE_C", "STATE_NODE_C_Yes")]}) },
     {"BN_A":{"NODE_A"}, "BN_B":{"NODE_B", "NODE_C"}}    ,  {"BN_A":{("NODE_A", "OR_SIG_HI")}, "BN_B":{("NODE_B", "OR_SIG_LO"), ("NODE_C", "AND_TOP")}}    , {"BN_A":{("NODE_A", "STATE_NODE_A_No")}, "BN_B":{("NODE_B", "STATE_NODE_B_Yes"), ("NODE_C", "STATE_NODE_C_Yes")}} ),

    # multiple hybrid configurations
    ( {"ID_7_a":HybridConfiguration("elevator_model", {"BN_A":["NODE_A"]}, {"BN_A":[("NODE_A", "OR_SIG_HI")]} , {"BN_A":[("NODE_A", "STATE_NODE_A_No")]}) ,
       "ID_7_b":HybridConfiguration("elevator_model", {"BN_B":["NODE_B", "NODE_C"]}, {"BN_B":[("NODE_B", "OR_SIG_HI"), ("NODE_C", "OR_SIG_LO")]} , {"BN_B":[("NODE_B", "STATE_NODE_B_No"), ("NODE_C", "STATE_NODE_C_Yes")]})},
     {"BN_A":{"NODE_A"}, "BN_B":{"NODE_B", "NODE_C"}}    ,  {"BN_A":{("NODE_A", "OR_SIG_HI")}, "BN_B":{("NODE_B", "OR_SIG_HI"), ("NODE_C", "OR_SIG_LO")}}    , {"BN_A":{("NODE_A", "STATE_NODE_A_No")}, "BN_B":{("NODE_B", "STATE_NODE_B_No"), ("NODE_C", "STATE_NODE_C_Yes")}} )
])
def test_instantiate_success(configs, all_shared_nodes, all_couplings, all_pbf_states, fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model

    baynets = {bn_name:(fixture_bn_confounder_param(bn_name))[0] for bn_name in all_shared_nodes.keys()}

    hybrid_manager = HybridManager(ft_inst=tree, bayesian_nets=baynets, configurations=configs)

    assert hybrid_manager.ft_inst == tree
    assert hybrid_manager.bayesian_nets == baynets
    assert hybrid_manager.configurations == configs
    assert hybrid_manager._HybridManager__all_shared_nodes == all_shared_nodes
    assert hybrid_manager._HybridManager__all_pbf_states == all_pbf_states
    assert hybrid_manager._HybridManager__all_ft_coupling_points == all_couplings

    for config_id in configs.keys():
        assert hybrid_manager._HybridManager__builder[config_id]


def test_shared_node_not_in_bn_raises_exception(fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn, _ = fixture_bn_confounder_param("BN_A")
    baynets = {"BN_A":bn}

    shared_nodes = {"BN_A":["NODE_A", "BAD_NODE"]}
    couplings = {"BN_A":[("NODE_A", "AND_TOP")]}
    pbf_states = {"BN_A":[("NODE_A", "STATE_NODE_A_No"), ("BAD_NODE", "STATE_BAD_NODE_Yes")]}
    config = HybridConfiguration(name=tree.name, shared_nodes=shared_nodes, ft_coupling_points=couplings, pbf_states=pbf_states)

    configs = {"test":config}

    expected_exc_substring = "that are not part of the BN"

    with pytest.raises(ValueError) as e:
        assert HybridManager(ft_inst=tree, bayesian_nets=baynets, configurations=configs)

    assert expected_exc_substring in str(e.value)


def test_coupling_node_not_in_ft_raises_exception(fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn, _ = fixture_bn_confounder_param("BN_A")
    baynets = {"BN_A":bn}

    shared_nodes = {"BN_A":["NODE_A", "NODE_B"]}
    couplings = {"BN_A":[("NODE_A", "AND_TOP"), ("NODE_B", "BAD_FT_NODE")]}
    pbf_states = {"BN_A":[("NODE_A", "STATE_NODE_A_No"), ("NODE_B", "STATE_NODE_B_Yes")]}
    config = HybridConfiguration(name=tree.name, shared_nodes=shared_nodes, ft_coupling_points=couplings, pbf_states=pbf_states)

    configs = {"ID_0":config}

    expected_exc_substring = "Mounting nodes for Faul Tree"

    with pytest.raises(ValueError) as e:
        assert HybridManager(ft_inst=tree, bayesian_nets=baynets, configurations=configs)

    assert expected_exc_substring in str(e.value)


def test_coupling_node_not_logical_raises_exception(fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn, _ = fixture_bn_confounder_param("BN_A")
    baynets = {"BN_A":bn}

    shared_nodes = {"BN_A":["NODE_A", "NODE_B"]}
    couplings = {"BN_A":[("NODE_A", "AND_TOP"), ("NODE_B", "IHF_SIG_BS_FP")]}
    pbf_states = {"BN_A":[("NODE_A", "STATE_NODE_A_No"), ("NODE_B", "STATE_NODE_B_Yes")]}
    config = HybridConfiguration(name=tree.name, shared_nodes=shared_nodes, ft_coupling_points=couplings, pbf_states=pbf_states)

    configs = {"ID_0":config}

    expected_exc_substring = "must be an instance of FaultTreeLogicNode"

    with pytest.raises(TypeError) as e:
        assert HybridManager(ft_inst=tree, bayesian_nets=baynets, configurations=configs)

    assert expected_exc_substring in str(e.value)


def test_invalid_pbf_state_raises_exception(fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn, _ = fixture_bn_confounder_param("BN_A")
    baynets = {"BN_A":bn}

    shared_nodes = {"BN_A":["NODE_A", "NODE_B"]}
    couplings = {"BN_A":[("NODE_A", "AND_TOP"), ("NODE_B", "OR_SIG_LO")]}
    pbf_states = {"BN_A":[("NODE_A", "STATE_NODE_A_No"), ("NODE_B", "BAD_STATE")]}
    config = HybridConfiguration(name=tree.name, shared_nodes=shared_nodes, ft_coupling_points=couplings, pbf_states=pbf_states)

    configs = {"ID_0":config}

    expected_exc_substring = "Invalid PBF state"

    with pytest.raises(ValueError) as e:
        assert HybridManager(ft_inst=tree, bayesian_nets=baynets, configurations=configs)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("req_id, observations",
[    ( "ID_0", None),
    ( "ID_1", {"BN_A":[("NODE_C", "STATE_NODE_C_Yes")]}),
])
def test_build_extended_ft_success(req_id, observations, fixture_hybrid_config, fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn, _ = fixture_bn_confounder_param("BN_A")
    baynets = {"BN_A":bn}
    configs = {"ID_0":fixture_hybrid_config(tree.name), "ID_1":fixture_hybrid_config("test")}

    hybrid_manager = HybridManager(ft_inst=tree, bayesian_nets=baynets, configurations=configs)

    ext_ft = hybrid_manager.build_extended_ft(config_id=req_id, bn_observations=observations)
    assert isinstance(ext_ft, type(tree))


def test_build_extended_ft_invalid_id_raises_excpetion(fixture_hybrid_config, fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn, _ = fixture_bn_confounder_param("BN_A")
    baynets = {"BN_A":bn}
    configs = {"ID_0":fixture_hybrid_config(tree.name), "ID_1":fixture_hybrid_config("test")}

    hybrid_manager = HybridManager(ft_inst=tree, bayesian_nets=baynets, configurations=configs)

    expected_exc_substring = "There is no managed hybrid configuration for given config id"

    with pytest.raises(ValueError) as e:
        assert hybrid_manager.build_extended_ft(config_id="BAD_ID", bn_observations=None)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("req_id, at_time, fix",
[    ( "ID_0", 0, True),
    ( "ID_0", 123, True),
    ( "ID_1", 456, False),
])
def test_build_hybrid_networks_success(req_id, at_time, fix, fixture_hybrid_config, fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn, _ = fixture_bn_confounder_param("BN_A")
    baynets = {"BN_A":bn}
    configs = {"ID_0":fixture_hybrid_config(tree.name), "ID_1":fixture_hybrid_config("test")}

    hybrid_manager = HybridManager(ft_inst=tree, bayesian_nets=baynets, configurations=configs)

    hybrid_networks = hybrid_manager.build_hybrid_networks(config_id=req_id, at_time=at_time, fix_other_bns=fix)

    assert all(isinstance(hybrid_bn, type(bn)) for hybrid_bn in hybrid_networks.values())


def test_build_hybrid_networks_invalid_id_raises_excpetion(fixture_hybrid_config, fixture_ft_elevator_model, fixture_bn_confounder_param):
    tree, _ = fixture_ft_elevator_model
    bn, _ = fixture_bn_confounder_param("BN_A")
    baynets = {"BN_A":bn}
    configs = {"ID_0":fixture_hybrid_config(tree.name), "ID_1":fixture_hybrid_config("test")}

    hybrid_manager = HybridManager(ft_inst=tree, bayesian_nets=baynets, configurations=configs)

    expected_exc_substring = "There is no managed hybrid configuration for given config id"

    with pytest.raises(ValueError) as e:
        assert hybrid_manager.build_hybrid_networks(config_id="BAD_ID", at_time=0, fix_other_bns=False)

    assert expected_exc_substring in str(e.value)
