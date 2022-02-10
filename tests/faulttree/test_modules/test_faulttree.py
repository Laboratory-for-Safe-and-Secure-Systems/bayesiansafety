from unittest.mock import patch
import pytest

from bayesiansafety.faulttree import BayesianFaultTree
from bayesiansafety.faulttree import FaultTreeLogicNode
from bayesiansafety.faulttree import FaultTreeProbNode
#fixtures provided via conftest

def test_instantiate_empty_model():
    model = BayesianFaultTree(name=None, probability_nodes=[], logic_nodes=[])
    assert True

def test_instantiate_success():
    A = FaultTreeProbNode(name='A'   , probability_of_failure=2.78e-6)
    B = FaultTreeProbNode(name='B'   , probability_of_failure=4.12e-4)
    C = FaultTreeProbNode(name='C'   , probability_of_failure=8.81e-5)

    AND   = FaultTreeLogicNode(name='AND', input_nodes=['A', 'B', 'C'], logic_type="AND" )


    given_probability_nodes = [A, B, C]
    given_logic_nodes       = [AND]

    model_name = "test"
    elem_names = set([elem.name for elem in given_probability_nodes + given_logic_nodes])
    correct_node_connections = set([('A', "AND"), ('B', "AND"), ('C', "AND")])

    model = BayesianFaultTree(model_name, given_probability_nodes, given_logic_nodes)

    assert model.name == model_name
    assert model.probability_nodes == given_probability_nodes
    assert model.logic_nodes == given_logic_nodes
    assert len(set(model.model_elements.keys()) ^ elem_names) == 0
    assert len(set(model.node_connections) ^ correct_node_connections) == 0


###### public methods
@pytest.mark.parametrize("queried_name, expected", [("EMI_BS_FP", FaultTreeProbNode),("AND_TOP", FaultTreeLogicNode)])
def test_get_elem_by_name_returns_correct(queried_name, expected, fixture_ft_elevator_model):
    model, _ = fixture_ft_elevator_model
    queried_element = model.get_elem_by_name(node_name=queried_name)
    assert isinstance(queried_element, expected)


def test_get_elem_by_name_wrong_name_raises_exception(fixture_ft_and_only_model):
    bad_node = "BAD_NODE"
    expected_exc_substring = "Scoped element:"
    model, _ = fixture_ft_and_only_model

    with pytest.raises(Exception) as e:
        assert model.get_elem_by_name(node_name=bad_node)

    assert expected_exc_substring in str(e.value)


def test_get_top_level_event_name_multiple_tle_raises_exception():
    expected_exc_substring = "Tree contains more than one leaf"
    dummy_name = "test"

    A = FaultTreeProbNode(name='A'   , probability_of_failure=0.5)
    B = FaultTreeProbNode(name='B'   , probability_of_failure=0.1)
    TLE_1   = FaultTreeLogicNode(name='TLE_1', input_nodes=['A', 'B'])
    TLE_2   = FaultTreeLogicNode(name='TLE_2', input_nodes=['A', 'B'])

    probability_nodes = [A, B]
    logic_nodes       = [TLE_1, TLE_2]

    model = BayesianFaultTree(dummy_name, probability_nodes, logic_nodes)

    with pytest.raises(Exception) as e:
        assert model.get_top_level_event_name()

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("fixture, expected_tle",
                        [("fixture_ft_elevator_model", "AND_TOP"),
                        ("fixture_ft_and_only_model", "AND"),
                        ("fixture_ft_or_only_model", "OR"),
                        ("fixture_ft_fatram_paper_model", "TOP"),
                        ("fixture_ft_mocus_book_model", "TLE_G1"),])
def test_get_top_level_event_name_returns_correct(fixture, expected_tle, request):
    model, _ = request.getfixturevalue(fixture)
    assert model.get_top_level_event_name() == expected_tle


def test_copy_returns_correct():
    model_name = "test"
    pbf_A = 2.78e-6
    pbf_B = 4.12e-4
    pbf_C = 8.81e-5

    A = FaultTreeProbNode(name='A'   , probability_of_failure=pbf_A)
    B = FaultTreeProbNode(name='B'   , probability_of_failure=pbf_B)
    C = FaultTreeProbNode(name='C'   , probability_of_failure=pbf_C)

    AND   = FaultTreeLogicNode(name='AND', input_nodes=['A', 'B', 'C'], logic_type="AND" )

    given_probability_nodes = [A, B, C]
    given_logic_nodes       = [AND]

    elem_names = set([elem.name for elem in given_probability_nodes + given_logic_nodes])
    correct_node_connections = set([('A', "AND"), ('B', "AND"), ('C', "AND")])
    correct_node_pbfs = set([("A", pbf_A), ("B", pbf_B), ("C", pbf_C)])

    model = BayesianFaultTree(model_name, given_probability_nodes, given_logic_nodes)

    copied_model = model.copy()

    assert copied_model.name == model_name
    assert all(isinstance(prob_node, FaultTreeProbNode) for prob_node in copied_model.probability_nodes)
    assert all(isinstance(logic_node, FaultTreeLogicNode) for logic_node in copied_model.logic_nodes)
    assert len(set(copied_model.model_elements.keys()) ^ elem_names) == 0
    assert len(set(copied_model.node_connections) ^ correct_node_connections) == 0
    assert len(set([(node.name, node.probability_of_failure) for node in copied_model.probability_nodes]) ^ correct_node_pbfs) == 0


def test_add_prob_node_success():
    model_name = "test"
    new_node_name = "NEW_NODE"
    new_node_pbf = 0.123
    new_node_time_dependency = True
    attached_to = "AND"

    A = FaultTreeProbNode(name='A'   , probability_of_failure=2.78e-6)
    B = FaultTreeProbNode(name='B'   , probability_of_failure=4.12e-4)
    C = FaultTreeProbNode(name='C'   , probability_of_failure=8.81e-5)

    AND   = FaultTreeLogicNode(name='AND', input_nodes=['A', 'B', 'C'], logic_type="AND" )

    original_probability_nodes = [A, B, C]
    original_logic_nodes       = [AND]

    model = BayesianFaultTree(model_name, original_probability_nodes, original_logic_nodes)

    expected_elem_names = set([elem.name for elem in original_probability_nodes + original_logic_nodes] + [new_node_name])
    expected_node_connections = set([('A', "AND"), ('B', "AND"), ('C', "AND")] + [(new_node_name, attached_to)])
    expected_inputs_for_target_node = set(["A", "B", "C", new_node_name])

    model.add_prob_node(node_name=new_node_name, input_to=attached_to, probability_of_failure=new_node_pbf, is_time_dependent=new_node_time_dependency)

    #check if new node got added to prob. nodes
    assert len([node for node in model.probability_nodes if node.name == new_node_name]) == 1

    #check if logic node got extended
    modified_inputs_for_logic_node = [node.input_nodes for node in model.logic_nodes if node.name == attached_to]
    assert len(modified_inputs_for_logic_node) == 1
    assert len(set(modified_inputs_for_logic_node[0]) ^ expected_inputs_for_target_node) == 0

    #check if added prob. node is part of model elements
    assert len(set(model.model_elements.keys()) ^ expected_elem_names) == 0

    #check if connections are correct
    assert len(set(model.node_connections) ^ expected_node_connections) == 0


def test_add_prob_node_ft_target_not_in_model_raises_exception(fixture_ft_and_only_model):
    expected_exc_substring = "is not part of this Fault Tree"
    new_node_name = "NEW_NODE"
    new_node_pbf = 0.123
    attached_to = "BAD_ATTACHMENT"

    model, _ = fixture_ft_and_only_model

    with pytest.raises(ValueError) as e:
        assert model.add_prob_node(node_name=new_node_name, input_to=attached_to, probability_of_failure=new_node_pbf)

    assert expected_exc_substring in str(e.value)


def test_add_prob_node_ft_target_no_logic_node_raises_exception(fixture_ft_and_only_model):
    expected_exc_substring = "is not a logic node"
    new_node_name = "NEW_NODE"
    new_node_pbf = 0.123
    attached_to = "A"

    model, _ = fixture_ft_and_only_model

    with pytest.raises(TypeError) as e:
        assert model.add_prob_node(node_name=new_node_name, input_to=attached_to, probability_of_failure=new_node_pbf)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("at_time, mod_frates, expeced_rates", [(0, None, [('A',2.78e-6), ('B',4.12e-4), ('C', 8.81e-5)]),
                                                                (5e32, None, [('A',2.78e-6), ('B',4.12e-4), ('C', 8.81e-5)]),
                                                                (100, {'B': 0.789}, [('A',2.78e-6), ('B',0.789), ('C', 8.81e-5)]) ])
def test_get_tree_at_time_returns_correct(at_time, mod_frates, expeced_rates, fixture_ft_and_only_model):
    model, _ = fixture_ft_and_only_model

    tree_at_time = model.get_tree_at_time(at_time, modified_frates=mod_frates)
    for node, rate in expeced_rates:
        assert tree_at_time.get_elem_by_name(node).probability_of_failure == rate


def test_get_tree_at_time_invalid_modfrate_node_type_raises_exception(fixture_ft_and_only_model):
    expected_exc_substring = "Check if node is a probability node."
    bad_mod_frates = {"AND":0.123} #logic node
    model, _ = fixture_ft_and_only_model

    with pytest.raises(TypeError) as e:
        assert model.get_tree_at_time(at_time=0, modified_frates=bad_mod_frates)

    assert expected_exc_substring in str(e.value)


def test_get_tree_at_time_invalid_modfrate_pbf_raises_exception(fixture_ft_and_only_model):
    expected_exc_substring = "Invalid modified frate"
    bad_mod_frates = {"A":789}
    model, _ = fixture_ft_and_only_model

    with pytest.raises(ValueError) as e:
        assert model.get_tree_at_time(at_time=0, modified_frates=bad_mod_frates)

    assert expected_exc_substring in str(e.value)


##### plotting functions
@patch("matplotlib.pyplot.show")
def test_plot_bayesian_fault_tree(mock_show, fixture_ft_and_only_model):
    model, _ = fixture_ft_and_only_model
    assert model.plot_bayesian_fault_tree(at_time=0) is None


@patch("matplotlib.pyplot.show")
def test_plot_time_simulation(mock_show, fixture_ft_and_only_model):
    model, _ = fixture_ft_and_only_model
    model.run_time_simulation(start_time=0, stop_time=1, simulation_steps=1, node_name=None, plot_simulation=True)
    mock_show.assert_called()


###### private methods

def test__verify_node_connections_unspecified_node_raises_exception():
    dummy_name = "test"
    missing_node = "C"
    expected_exc_substring = "Mismatch between node connections"

    A = FaultTreeProbNode(name='A'   , probability_of_failure=0.5)
    B = FaultTreeProbNode(name='B'   , probability_of_failure=0.1)
    TLE   = FaultTreeLogicNode(name='TLE', input_nodes=['A', 'B', missing_node])

    probability_nodes = [A, B]
    logic_nodes       = [TLE]

    with pytest.raises(Exception) as e:
        assert BayesianFaultTree(dummy_name, probability_nodes, logic_nodes)

    assert expected_exc_substring in str(e.value)


def test_run_query_for_individual_nodes_bad_node_raises_exception(fixture_ft_and_only_model):
    bad_node = "bad"
    expected_exc_substring = "Invalid scoped event"
    model, _ = fixture_ft_and_only_model

    with pytest.raises(Exception) as e:
        assert model.run_query_for_individual_nodes(individual_nodes=[bad_node], at_time=0, modified_frates=None)

    assert expected_exc_substring in str(e.value)


def test_run_query_for_individual_nodes_bad_modified_frate_raises_exception(fixture_ft_and_only_model):
    bad_frate = {}
    bad_frate["A"] = 123
    tle_node = "AND"

    expected_exc_substring = "Involid modified frate"
    model, _ = fixture_ft_and_only_model

    with pytest.raises(Exception) as e:
        assert model.run_query_for_individual_nodes(individual_nodes=[tle_node], at_time=0, modified_frates=bad_frate)

    assert expected_exc_substring in str(e.value)
