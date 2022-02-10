import pytest

from bayesiansafety.core import BayesianNetwork
from bayesiansafety.eventtree import BayesianEventTreeMapper


@pytest.mark.parametrize("fixture, model_name, expected_node_connections",
    [ ("fixture_tree_causal_arc_param", "causal"            , (("e1", "cnsq"), ("e2", "cnsq")) ),
      ("fixture_tree_consequence_arc_param", "consequence"  , (("e1", "e2"), ("e2", "cnsq")) ),
      ("fixture_tree_dont_care_param", "dontcare"           , (("e1", "e2"), ("e1", "cnsq"), ("e2", "cnsq"))),
      ("fixture_tree_train_derailment_param", "train"       , (("contained", "cnsq"), ("clear", "cnsq"), ("ca", "cnsq"),
                                                               ("falls", "hits"), ("falls", "cnsq"), ("hits", "cnsq"),
                                                               ("collapse", "cnsq"), ("collision", "cnsq")) ),

    ("fixture_tree_heat_exchanger_param", "heat"  , (("sprinkler", "alarm"), ("ignition", "alarm"), ("sprinkler", "cnsq"),
                                                     ("ignition", "cnsq"), ("alarm", "cnsq")) ),
     ])
def test_et_mapper_map_success(fixture, model_name, expected_node_connections, request):
    tree, node_connections, data = request.getfixturevalue(fixture)(model_name)

    mapper = BayesianEventTreeMapper()
    #use actual name of consequence node
    expected_node_connections = [ tuple([elem if elem != "cnsq" else mapper.consequence_node_name for elem in list(tup) ]) for tup in expected_node_connections]

    bn = mapper.map(tree)

    assert isinstance(bn, BayesianNetwork)
    assert set(expected_node_connections) == set(bn.node_connections)

