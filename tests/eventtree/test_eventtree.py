import os
import pytest
import numpy as np
import networkx as nx

from bayesiansafety.core import BayesianNetwork
from bayesiansafety.eventtree import EventTreeImporter
from bayesiansafety.eventtree import BayesianEventTree

cur_dir_path = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.abspath(os.path.join(cur_dir_path, os.pardir, "test_data"))


@pytest.mark.parametrize("fixture, model_name",
    [ ("fixture_tree_causal_arc_param", "causal"),
      ("fixture_tree_consequence_arc_param", "consequence"),
      ("fixture_tree_dont_care_param", "dontcare")
     ])
def test_bayesianeventtree_instantiate_success(fixture, model_name, request):
    tree, node_connections, data = request.getfixturevalue(fixture)(model_name)

    bay_et = BayesianEventTree(name=model_name, tree_obj=tree)

    assert isinstance(bay_et, BayesianEventTree)
    assert isinstance(bay_et.model, BayesianNetwork)
    assert bay_et.name == model_name
    assert bay_et.tree_obj == tree


def test_bayesianeventtree_copy_success(fixture_tree_causal_arc_param):
    tree, node_connections, data = fixture_tree_causal_arc_param("Test")

    bay_et = BayesianEventTree(name="TEST", tree_obj=tree)
    copy_model = bay_et.copy()

    assert isinstance(copy_model, BayesianEventTree)
    assert isinstance(copy_model.model, BayesianNetwork)
    assert copy_model.name == bay_et.name
    assert nx.is_isomorphic(bay_et.tree_obj, copy_model.tree_obj)
    assert bay_et.model_elements.keys() == copy_model.model_elements.keys()


def test_bayesianeventtree_get_consequence_node_name_success(fixture_tree_causal_arc_param):
    tree, node_connections, data = fixture_tree_causal_arc_param("Test")

    bay_et = BayesianEventTree(name="TEST", tree_obj=tree)

    assert bay_et.get_consequence_node_name() is not None


@pytest.mark.parametrize("fixture, model_name, is_import, outcome_probabilities",
    [ ("fixture_tree_causal_arc_param", "causal", False, {"c1":0.1, "c2":0.63, "c3":0.27}),
      ("fixture_tree_consequence_arc_param", "consequence", False,  {"c1":0.037, "c2":0.963}),
      ("fixture_tree_dont_care_param", "dontcare", False, {"c1":0.061, "c2":0.939}),
      ("openpsa_et_trail_derailment.xml", "TrainDerailment", True, {"d1":0.0, "d2":0.29, "d3":1349/20000, "d4":0.016019375, "d5":8.43125e-4,
                                                                   "d6":213/64000, "d7":1.05390625e-3, "d8":71/1280000, "d9":0.53116875,
                                                                   "d10":0.05901875, "d11":0.02795625, "d12":497/160000}),
      ("openpsa_et_tank.xml", "Tank", True, {"Continue":0.63, "Shutdown":0.298775, "Overflow":0.071225}),
      ("openpsa_bt_heat_exchanger_accident.xml", "EventTree", True, {"C1":0.8628768, "C2":351/312500, "C3":0.0359532, "C4":4.68e-5,
                                                                     "C5":93/1250, "C6":27/1250, "C7":31/10000, "C8":9/10000}),
     ])
def test_bayesianeventtree_get_consequence_likelihoods_success(fixture, model_name, is_import, outcome_probabilities, request):
    bay_et = None

    if is_import:
        test_file_path = os.path.abspath(os.path.join(test_data_dir, fixture))
        bay_et = EventTreeImporter().load(test_file_path)

    else:
        tree, node_connections, data = request.getfixturevalue(fixture)(model_name)
        bay_et = BayesianEventTree(name=model_name, tree_obj=tree)

    result = bay_et.get_consequence_likelihoods()

    for outcome, expected_prob in outcome_probabilities.items():
        res_val = result.get_value({bay_et.get_consequence_node_name():outcome})
        assert np.allclose(res_val, expected_prob, atol=1e-9)


def test_bayesianeventtree_get_elem_by_name_success(fixture_tree_causal_arc_param):
    tree, node_connections, data = fixture_tree_causal_arc_param("Test")

    bay_et = BayesianEventTree(name="TEST", tree_obj=tree)

    for name, element in bay_et.model_elements.items():
        assert bay_et.get_elem_by_name(name) == element


def test_get_elem_by_name_wrong_name_raises_exception(fixture_tree_causal_arc_param):
    bad_node = "BAD_NODE"
    expected_exc_substring = "Scoped element:"
    tree, node_connections, data = fixture_tree_causal_arc_param("Test")
    bay_et = BayesianEventTree(name="TEST", tree_obj=tree)

    with pytest.raises(Exception) as e:
        assert bay_et.get_elem_by_name(node_name=bad_node)

    assert expected_exc_substring in str(e.value)
