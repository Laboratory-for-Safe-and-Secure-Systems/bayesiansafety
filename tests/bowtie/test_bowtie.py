import pytest
import networkx as nx
import numpy as np

from bayesiansafety.core import BayesianNetwork
from bayesiansafety.bowtie import BayesianBowTie


@pytest.mark.parametrize("et_fixture,             ft_fixture,             model_name,   pivot_elem, trigg_state, add_affected_nodes",
    [ ("fixture_et_causal_arc_param"      , "fixture_ft_fatram_paper_model", "direct"    , None,     None        ,     None),
      ("fixture_et_consequence_arc_param" , "fixture_ft_or_only_model" , "explicit"        , "OR",     None        ,     None),
      ("fixture_et_dont_care_param"          , "fixture_ft_and_only_model", "trigger"        , None,   "working"        ,      None),
      ("fixture_et_train_derailment_param", "fixture_ft_mocus_book_model", "affected"    , None,        None        , {"Falls":"no", "Hits":"no"}),
      ("fixture_et_dont_care_param"          , "fixture_ft_elevator_model", "explict_trigger", "OR_LS", "working"    , None),
      ("fixture_et_consequence_arc_param" , "fixture_ft_elevator_model", "full"            ,"EMI_LOG_FA", "failing", {"FE1":"e12"}),
    ])
def test_bayesianbowtie_instantiate_success(et_fixture, ft_fixture, model_name, pivot_elem, trigg_state, add_affected_nodes, request):
    et_model, consequence_probabilities = request.getfixturevalue(et_fixture)("EventTree")
    ft_model, correct_cutsets            = request.getfixturevalue(ft_fixture)

    expected_pivot = pivot_elem if pivot_elem else ft_model.get_top_level_event_name()

    bt_model = BayesianBowTie(name=model_name, bay_ft=ft_model, bay_et=et_model,
                                pivot_node=pivot_elem, triggering_state=trigg_state, causal_arc_et_nodes=add_affected_nodes)


    assert isinstance(bt_model, BayesianBowTie)
    assert isinstance(bt_model.model, BayesianNetwork)
    assert bt_model.name == model_name
    assert bt_model.et_model == et_model
    assert bt_model.ft_model == ft_model
    assert bt_model.pivot_node == expected_pivot
    assert bt_model.triggering_state == trigg_state


def test_bayesianbowtie_invalid_pivot_node_raises_excpetion(fixture_et_train_derailment_param, fixture_ft_elevator_model):
    expected_exc_substring = "Bow-Tie can not be built. Pivot node"
    et_model, consequence_probabilities = fixture_et_train_derailment_param("EventTree")
    ft_model, correct_cutsets            = fixture_ft_elevator_model
    invalid_pivot_node = "INVALID"

    with pytest.raises(ValueError) as e:
        assert BayesianBowTie(name="TEST", bay_ft=ft_model, bay_et=et_model, pivot_node=invalid_pivot_node)

    assert expected_exc_substring in str(e.value)


def test_bayesianbowtie_get_elem_by_name_success(fixture_et_train_derailment_param, fixture_ft_elevator_model):
    et_model, consequence_probabilities = fixture_et_train_derailment_param("EventTree")
    ft_model, correct_cutsets            = fixture_ft_elevator_model
    tle_name = 'AND_TOP'

    bt_model = BayesianBowTie(name="TEST", bay_ft=ft_model, bay_et=et_model)

    tle_element = ft_model.model_elements[tle_name]
    requested_elem = bt_model.get_elem_by_name(tle_name)

    assert tle_element.name == requested_elem.name
    assert isinstance(tle_element, type(requested_elem))


def test_bayesianbowtie_get_element_invalid_name_raises_excpetion(fixture_et_train_derailment_param, fixture_ft_elevator_model):
    expected_exc_substring = "Scoped element:"
    et_model, consequence_probabilities = fixture_et_train_derailment_param("EventTree")
    ft_model, correct_cutsets            = fixture_ft_elevator_model
    invalid_element = "INVALID"

    bt_model = BayesianBowTie(name="TEST", bay_ft=ft_model, bay_et=et_model)

    with pytest.raises(ValueError) as e:
        assert bt_model.get_elem_by_name(invalid_element)

    assert expected_exc_substring in str(e.value)


def test_bayesianbowtie_copy_success(fixture_et_train_derailment_param, fixture_ft_elevator_model):
    et_model, consequence_probabilities = fixture_et_train_derailment_param("EventTree")
    ft_model, correct_cutsets            = fixture_ft_elevator_model

    bt_model = BayesianBowTie(name="TEST", bay_ft=ft_model, bay_et=et_model)
    copy_model = bt_model.copy()

    assert isinstance(copy_model, BayesianBowTie)
    assert isinstance(copy_model.model, BayesianNetwork)
    assert bt_model.name == copy_model.name
    assert nx.is_isomorphic(bt_model.et_model.model, copy_model.et_model.model)
    assert nx.is_isomorphic(bt_model.ft_model.model, copy_model.ft_model.model)
    assert bt_model.pivot_node == copy_model.pivot_node
    assert bt_model.triggering_state == copy_model.triggering_state



@pytest.mark.parametrize("ft_fixture, et_fixture, model_name, outcome_probabilities",
    [ ("fixture_ft_and_only_model", "fixture_et_causal_arc_param", "causal", {"c1":0.1, "c2":0.63, "c3":0.27, "safe":1 - 1.00906216e-13}),
      ("fixture_ft_and_only_model", "fixture_et_consequence_arc_param", "consequence", {"c1":0.037, "c2":0.963, "safe":1 - 1.00906216e-13}),
      ("fixture_ft_and_only_model", "fixture_et_dont_care_param", "dontcare", {"c1":0.061, "c2":0.939, "safe":1 - 1.00906216e-13})
     ])
def test_bayesianbowtie_get_consequence_likelihoods_success(ft_fixture, et_fixture, model_name, outcome_probabilities, request):
    et_model, consequence_probabilities = request.getfixturevalue(et_fixture)("EventTree")
    ft_model, correct_cutsets            = request.getfixturevalue(ft_fixture)

    tle_prob = 1.00906216e-13 # at time t = 0
    bt_model = BayesianBowTie(name="TEST", bay_ft=ft_model, bay_et=et_model)

    result = bt_model.get_consequence_likelihoods()

    for outcome, expected_prob in outcome_probabilities.items():
        if outcome != "safe":
            expected_prob = expected_prob * tle_prob
        else:
            expected_prob = 1 - 1.00906216e-13

        res_val = result.get_value({et_model.get_consequence_node_name():outcome})
        assert np.allclose(res_val, expected_prob, atol=1e-17)
