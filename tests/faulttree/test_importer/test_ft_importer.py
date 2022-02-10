import os
import pytest

from bayesiansafety.faulttree import BayesianFaultTree
from bayesiansafety.faulttree import FaultTreeImporter
from bayesiansafety.faulttree import FaultTreeProbNode
from bayesiansafety.faulttree import FaultTreeLogicNode

cur_dir_path = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.abspath(os.path.join(cur_dir_path, os.pardir, os.pardir, "test_data"))


@pytest.mark.parametrize("file_name", [ "openpsa_ft_gates.xml", "openpsa_bt_simple_model.xml", "openpsa_ft_time_dependent.xml"])
def test_ft_importer_correct_loading_of_ft_instance(file_name):

    test_file_path = os.path.abspath(os.path.join(test_data_dir, file_name))
    bay_ft = FaultTreeImporter().load(test_file_path)

    assert isinstance(bay_ft, BayesianFaultTree)


@pytest.mark.parametrize("file_name, ft_name, expected_gates, expected_events",
    [ ( "openpsa_ft_gates.xml"       , "GateTest" , ["AND", "OR_1", "OR_2"], ["Basic_event_1_1", "Basic_event_1_2", "Basic_event_2_1", "Basic_event_2_2"] ),
      ( "openpsa_bt_simple_model.xml", "FaultTree", ["OR_TLE", "AND", "OR_B4_5", "OR_B1_2"], ["Basic_event_1", "Basic_event_2", "Basic_event_3", "Basic_event_4", "Basic_event_5"]) ])
def test_ft_importer_correct_loading_of_element_main_type(file_name, ft_name, expected_gates, expected_events):

    test_file_path = os.path.abspath(os.path.join(test_data_dir, file_name))
    bay_ft = FaultTreeImporter().load(test_file_path)

    assert set(bay_ft.model_elements.keys()) == set(expected_gates).union(set(expected_events))

    for gate in expected_gates:
        assert isinstance(bay_ft.get_elem_by_name(gate), FaultTreeLogicNode)

    for basic_event in expected_events:
        assert isinstance(bay_ft.get_elem_by_name(basic_event), FaultTreeProbNode)


@pytest.mark.parametrize("file_name, static_events, time_dependent_events",
    [ ( "openpsa_ft_gates.xml"         , ["Basic_event_1_1", "Basic_event_1_2", "Basic_event_2_1", "Basic_event_2_2"], [] ),
      ( "openpsa_ft_time_dependent.xml", ["Basic_event_1_1", "Basic_event_1_2"],  ["Time_basic_event_2_1", "Time_basic_event_2_2"]) ])
def test_ft_importer_correct_loading_of_time_behaviour(file_name, static_events, time_dependent_events):

    test_file_path = os.path.abspath(os.path.join(test_data_dir, file_name))
    bay_ft = FaultTreeImporter().load(test_file_path)

    for static_event in static_events:
        instance = bay_ft.get_elem_by_name(static_event)
        assert instance.is_time_dependent is False

    for dependent_event in time_dependent_events:
        instance = bay_ft.get_elem_by_name(dependent_event)
        assert instance.is_time_dependent is True


@pytest.mark.parametrize("file_name, event_prob_tuples",
    [ ( "openpsa_ft_gates.xml"         , [("Basic_event_1_1", 1.23e-4), ("Basic_event_1_2", 5.67e-8), ("Basic_event_2_1", 4.32e-1), ("Basic_event_2_2", 9.76e-5)]),
      ( "openpsa_ft_time_dependent.xml", [("Basic_event_1_1", 1.23e-4), ("Basic_event_1_2", 5.67e-8), ("Time_basic_event_2_1", 1.23e-4), ("Time_basic_event_2_2", 5.67e-8)]),
      ( "openpsa_bt_simple_model.xml"  , [("Basic_event_1", 1.2e-3), ("Basic_event_2", 2.3e-4), ("Basic_event_3", 3.4e-5), ("Basic_event_4", 4.5e-6), ("Basic_event_5", 5.6e-7)]) ])
def test_ft_importer_correct_loading_of_failure_probabilities(file_name, event_prob_tuples):

    test_file_path = os.path.abspath(os.path.join(test_data_dir, file_name))
    bay_ft = FaultTreeImporter().load(test_file_path)

    for event_name, expected_prob in event_prob_tuples:
        instance = bay_ft.get_elem_by_name(event_name)
        assert instance.probability_of_failure == expected_prob
