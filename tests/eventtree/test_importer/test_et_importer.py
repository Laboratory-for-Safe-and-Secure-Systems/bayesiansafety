import os
import pytest
import networkx as nx

from bayesiansafety.eventtree import EventTreeImporter
from bayesiansafety.eventtree import BayesianEventTree

cur_dir_path = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.abspath(os.path.join(cur_dir_path, os.pardir, os.pardir, "test_data"))


##### creating a tree
@pytest.mark.parametrize("fixture, model_name",
    [ ("fixture_tree_causal_arc_param", "causal"),
      ("fixture_tree_consequence_arc_param", "consequence"),
      ("fixture_tree_dont_care_param", "dontcare")
     ])
def test_et_importer_build_tree_success(fixture, model_name, request):
    ref_tree, node_connections, data = request.getfixturevalue(fixture)(model_name)
    created_tree = EventTreeImporter().build_tree(node_connections, data)

    assert isinstance(created_tree, nx.DiGraph)
    assert nx.is_isomorphic(ref_tree, created_tree)
    assert ref_tree.nodes == created_tree.nodes


##### Loading from file

@pytest.mark.parametrize("file_name, model_name",
    [ ("openpsa_et_causal_arc_simp.xml", "CausalArcSimplification"),
      ("openpsa_et_consequence_arc_simp.xml", "ConsequenceArcSimplification"),
      ("openpsa_et_trail_derailment.xml", "TrainDerailment")])
def test_et_importer_load_correct_instance_from_file(file_name, model_name):

    test_file_path = os.path.abspath(os.path.join(test_data_dir, file_name))
    bay_et = EventTreeImporter().load(test_file_path)

    assert isinstance(bay_et, BayesianEventTree)
    assert bay_et.name == model_name


@pytest.mark.parametrize("file_name, expected_func_events",
    [ ( "openpsa_et_causal_arc_simp.xml"       ,  ["e1", "e2"] ),
      ( "openpsa_et_consequence_arc_simp.xml"  ,  ["e1", "e2"]),
      ( "openpsa_bt_heat_exchanger_accident.xml", ["Ignition", "Sprinkler", "Alarm"]),
      ( "openpsa_et_trail_derailment.xml", ["contained", "clear", "cess_adj", "falls", "hits", "collapse", "collision"]),
      ( "openpsa_et_tank.xml", ["Alarm", "Operator", "Transmitter", "ESDV"]) ])
def test_et_importer_load_correct_functional_events_from_file(file_name, expected_func_events):
    test_file_path = os.path.abspath(os.path.join(test_data_dir, file_name))
    bay_et = EventTreeImporter().load(test_file_path)

    nodes = set(bay_et.model_elements.keys())
    nodes.remove(bay_et.get_consequence_node_name())
    assert nodes == set(expected_func_events)


@pytest.mark.parametrize("file_name, expected_consequences",
    [ ( "openpsa_et_consequence_arc_simp.xml"  ,  ["c1", "c2"]),
      ( "openpsa_et_causal_arc_simp.xml"       ,  ["c1", "c2", "c3"] ),
      ( "openpsa_bt_heat_exchanger_accident.xml", ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]),
      ( "openpsa_et_trail_derailment.xml", ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12"]),
      ( "openpsa_et_tank.xml", ["Continue", "Shutdown", "Overflow"]) ])
def test_et_importer_load_correct_consequences_from_file(file_name, expected_consequences):
    test_file_path = os.path.abspath(os.path.join(test_data_dir, file_name))
    bay_et = EventTreeImporter().load(test_file_path)

    cnsqncs = bay_et.model_elements[bay_et.get_consequence_node_name()].state_names[bay_et.get_consequence_node_name()]
    assert set(cnsqncs) == set(expected_consequences)


##### Construction from user definition
@pytest.mark.parametrize("fixture, model_name",
    [ ("fixture_et_causal_arc_param", "causal"),
      ("fixture_et_consequence_arc_param", "consequence"),
      ("fixture_et_dont_care_param", "dontcare"),
      ("fixture_et_train_derailment_param", "derailment")  ])
def test_et_importer_construct_correct_instance_from_manual(fixture, model_name, request):
    bay_et, cnsq_probs = request.getfixturevalue(fixture)(model_name)

    assert isinstance(bay_et, BayesianEventTree)
    assert bay_et.name == model_name


@pytest.mark.parametrize("fixture, expected_func_events",
    [ ( "fixture_et_causal_arc_param"       ,  ["FE1", "FE2"] ),
      ( "fixture_et_consequence_arc_param"  ,  ["FE1", "FE2"]),
      ( "fixture_et_dont_care_param", ["FE1", "FE2"]),
      ( "fixture_et_train_derailment_param", ["Contained", "Clear", "Cess_Adj", "Falls", "Hits", "Collapse", "Collision"]) ])
def test_et_importer_construct_correct_functional_events_from_manual(fixture, expected_func_events, request):
    model_name = 'test'
    bay_et, cnsq_probs = request.getfixturevalue(fixture)(model_name)

    nodes = set(bay_et.model_elements.keys())
    nodes.remove(bay_et.get_consequence_node_name())
    assert nodes == set(expected_func_events)


@pytest.mark.parametrize("fixture, expected_consequences",
    [ ( "fixture_et_causal_arc_param"     ,  ["c1", "c2", "c3"] ),
      ( "fixture_et_consequence_arc_param",  ["c1", "c2"]),
      ( "fixture_et_dont_care_param"      , ["c1", "c2"]),
      ( "fixture_et_train_derailment_param", ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12"]) ])
def test_et_importer_construct_correct_consequences_from_manual(fixture, expected_consequences, request):
    model_name = 'test'
    bay_et, cnsq_probs = request.getfixturevalue(fixture)(model_name)

    cnsqncs = bay_et.model_elements[bay_et.get_consequence_node_name()].state_names[bay_et.get_consequence_node_name()]
    assert set(cnsqncs) == set(expected_consequences)