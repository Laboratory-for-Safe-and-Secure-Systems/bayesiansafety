from unittest.mock import patch
import pytest
import numpy as np

from bayesiansafety.core.inference import TwinNetwork
from bayesiansafety.core.inference import InferenceFactory, Backend
#fixtures provided via conftest
BACKENDS_TO_TEST = [Backend.PYAGRUM, Backend.PGMPY]
XFAIL_REASON = "PGMPY has buggy behaviour for indentifiable causal queries of this test. If backend=PYAGRUM tests are expected to pass."


@pytest.mark.parametrize("fixture", ["fixture_bn_confounder_param", "fixture_bn_independent_nodes_only_param"])
@pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
@patch('bayesiansafety.core.inference.InferenceFactory.get_configured_backend', autospec=True)
def test_instantiate_success(fn_default_backend, backend, fixture, request ):
    model_name = 'test'
    model, _ = request.getfixturevalue(fixture)(model_name)
    fn_default_backend.return_value = backend

    instance = InferenceFactory(model).get_engine()

    assert instance.model == model
    if backend == Backend.PGMPY:
        assert instance._PgmpyInference__inference_engine_inst is not None
        assert instance._PgmpyInference__internal_model  is not None
    else:
        assert instance._PyagrumInference__inference_engine_inst is not None
        assert instance._PyagrumInference__internal_model  is not None



@pytest.mark.parametrize("variables, evidence, fixture, expected",
                          [(["NODE_A"], None, "fixture_bn_confounder_param", [[0.1], [0.9]]) ,
                           (["NODE_B"], None, "fixture_bn_independent_nodes_only_param", [[0.987], [0.013]]),
                           (["NODE_C"], None, "fixture_bn_collider_param", [[0.71], [0.29]]),
                           (["NODE_A"], {"NODE_B":"STATE_NODE_B_Yes"}, "fixture_bn_independent_nodes_only_param", [[0.123], [0.877]]),
                           (["NODE_A", "NODE_B"], {"NODE_C":"STATE_NODE_C_Yes"}, "fixture_bn_independent_nodes_only_param",
                                                                [ ( {"NODE_A":"STATE_NODE_A_Yes", "NODE_B":"STATE_NODE_B_Yes"}, 0.1214) ,
                                                                  ( {"NODE_A":"STATE_NODE_A_Yes", "NODE_B":"STATE_NODE_B_No"},  0.0016),
                                                                  ( {"NODE_A":"STATE_NODE_A_No", "NODE_B":"STATE_NODE_B_Yes"}, 0.8656),
                                                                  ( {"NODE_A":"STATE_NODE_A_No", "NODE_B":"STATE_NODE_B_No"}, 0.0114) ] ) ])
@pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
@patch('bayesiansafety.core.inference.InferenceFactory.get_configured_backend', autospec=True)
def test_query_success(fn_default_backend, backend, variables, evidence, fixture, expected, request ):
    model_name = 'test'
    model, _ = request.getfixturevalue(fixture)(model_name)
    fn_default_backend.return_value = backend

    instance = InferenceFactory(model).get_engine()
    query_result =  instance.query(variables=variables, evidence=evidence)

    if len(variables) == 1:
        assert np.allclose(query_result.get_probabilities(), expected, atol=1e-3)

    else:
        for state_combo, expected_val in expected:
          assert np.isclose(query_result.get_value(state_combo), expected_val, atol=1e-4)


@pytest.mark.parametrize("variables, do, evidence, fixture, expected",
                          [(["NODE_A"], {"NODE_C":"STATE_NODE_C_Yes"}, None, "fixture_bn_collider_param", [[0.123], [0.877]]     ),                         # Do-Calc Rule 1 -> P(A)
                           (["NODE_B"], {"NODE_C":"STATE_NODE_C_No"},  None, "fixture_bn_collider_param", [[0.987],[ 0.013]]     ),                         # Do-Calc Rule 1 -> P(B)
                           (["NODE_C"], {"NODE_A":"STATE_NODE_A_Yes"}, None, "fixture_bn_collider_param", [[0.127329], [0.872671]] ),                     # Rule 2 -> P(C|A=Yes)
                           (["NODE_B"], {"NODE_A":"STATE_NODE_A_No"}, None,  "fixture_bn_collider_param", [[0.987],[ 0.013]]  ),                          # Rule 2 -> P(B)
                           (["NODE_C"], {"NODE_B":"STATE_NODE_B_Yes"}, {"NODE_A":"STATE_NODE_A_No"},  "fixture_bn_collider_param", [[0.789], [0.211]]), # Rule 2 -> P(C|A=No, B=Yes)

                           (["NODE_B"], {"NODE_A":"STATE_NODE_A_No"}, None,  "fixture_bn_confounder_param", [[0.3354896], [0.6645104]]),                 #Rule 2 -> P(B|A=No)


                           (["NODE_C"], {"NODE_B":"STATE_NODE_B_Yes"}, {"NODE_A":"STATE_NODE_A_No"}  , "fixture_bn_independent_nodes_only_param", [[0.456], [0.544]] ), ]) # Rule 1 -> P(C)
@pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
@patch('bayesiansafety.core.inference.InferenceFactory.get_configured_backend', autospec=True)
def test_interventional_query_simple_marginal_success(fn_default_backend, backend, variables, do, evidence, fixture, expected, request ):
    model_name = 'test'
    model, _ = request.getfixturevalue(fixture)(model_name)

    fn_default_backend.return_value = backend
    if backend is Backend.PGMPY:
        pytest.xfail(reason=XFAIL_REASON)

    instance = InferenceFactory(model).get_engine()
    query_result =  instance.interventional_query(variables=variables, do=do, evidence=evidence)


    assert np.allclose(query_result.get_probabilities(), expected, atol=1e-1)



@pytest.mark.parametrize("variables, do, evidence, fixture, expected",
                          [(["NODE_A"], {"NODE_D":"STATE_NODE_D_Yes"}, {"NODE_B":"STATE_NODE_B_Yes"} , "fixture_bn_causal_queries_param", [[0.123], [0.877]]        ), # Rule 1 -> P(A|B) - d.sep via D -> P(A)
                           (["NODE_B"], {"NODE_D":"STATE_NODE_D_No"}, {"NODE_C":"STATE_NODE_C_No"}    , "fixture_bn_causal_queries_param", [[0.987], [0.013]]        ), # Rule 1 -> P(B|C) - d.sep via D -> P(B)

                           (["NODE_D"], {"NODE_C":"STATE_NODE_C_Yes"}, None                            , "fixture_bn_causal_queries_param", [[0.131658], [0.868342]]  ), # Rule 2 -> P(D|C=Yes)
                           (["NODE_D"], {"NODE_C":"STATE_NODE_C_No"}, None                              , "fixture_bn_causal_queries_param", [[0.462903], [0.537097]]  ), # Rule 2 -> P(D|C=No)
                           (["NODE_D"], {"NODE_C":"STATE_NODE_C_No"}, {"NODE_B":"STATE_NODE_B_No"}    , "fixture_bn_causal_queries_param", [[0.987], [0.013]]        ), # Rule 2 -> P(D|B=No,C=No)

                           (["NODE_C"], {"NODE_B":"STATE_NODE_B_Yes"}, {"NODE_A":"STATE_NODE_A_No"}  , "fixture_bn_causal_queries_param", [[0.654], [0.346]]        ), ])  # Rule 2 -> P(C|B=Yes, A=No) ])
@pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
@patch('bayesiansafety.core.inference.InferenceFactory.get_configured_backend', autospec=True)
def test_interventional_query_complex_marginal_success(fn_default_backend, backend, variables, do, evidence, fixture, expected, request ):
    model_name = 'test'
    model = request.getfixturevalue(fixture)(model_name)
    fn_default_backend.return_value = backend

    if backend is Backend.PGMPY:
        pytest.xfail(reason=XFAIL_REASON)

    instance = InferenceFactory(model).get_engine()
    query_result =  instance.interventional_query(variables=variables, do=do, evidence=evidence)

    assert np.allclose(query_result.get_probabilities(), expected, atol=1e-3)


@pytest.mark.parametrize("variables, do, evidence, expected",
                          [(["NODE_B", "NODE_D"], {"NODE_C":"STATE_NODE_C_No"}, None, {"B_Yes#D_Yes":0.4500719887, "B_Yes#D_No":0.5369279899, "B_No#D_Yes":0.01283102136,  "B_No#D_No":1.689999968e-4} ),      # Rule 2 -> P(B,D|C=No)
                           (["NODE_A", "NODE_B"], {"NODE_D":"STATE_NODE_D_No"}, None, {"A_Yes#B_Yes":0.1175625252, "A_Yes#B_No":0.1305492425e-3, "A_No#B_Yes":0.8693846908, "A_No#B_No":0.01144729152} ), ])  # Rule 3 -> P(A,B)
@pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
@patch('bayesiansafety.core.inference.InferenceFactory.get_configured_backend', autospec=True)
def test_interventional_query_complex_factor_success(fn_default_backend, backend, variables, do, evidence, fixture_bn_causal_queries_param, expected):
    model_name = 'test'
    delimiter = "#"
    model = fixture_bn_causal_queries_param(model_name)
    fn_default_backend.return_value = backend

    if backend is Backend.PGMPY:
        pytest.xfail(reason=XFAIL_REASON)

    instance = InferenceFactory(model).get_engine()
    query_result =  instance.interventional_query(variables=variables, do=do, evidence=evidence)

    for scope, expected_val in expected.items():
        parts = scope.split(delimiter)
        node_states = {"NODE_" + state_ending[0]:"STATE_NODE_" + state_ending for state_ending in parts}

        assert np.isclose(query_result.get_value(node_states), expected_val, atol=1e-2)


@pytest.mark.parametrize("target, whatif, observed, expected",
                          [("NODE_C", {"NODE_D":"STATE_NODE_D_Yes"}, {"NODE_C":"STATE_NODE_C_No" , "NODE_D":"STATE_NODE_D_No" } ,  [[0.58212892], [0.41787108]]  ) , # twin-net // Rule 3 -> P(tw_C|C, D)
                           ("NODE_C", {"NODE_D":"STATE_NODE_D_No" }, {"NODE_C":"STATE_NODE_C_Yes", "NODE_D":"STATE_NODE_D_Yes"} ,  [[0.63255308], [0.36744692]]  ) , # twin-net // Rule 3 -> P(tw_C|C, D)
                           ("NODE_D", {"NODE_C":"STATE_NODE_C_Yes"}, {"NODE_D":"STATE_NODE_D_No" , "NODE_C":"STATE_NODE_C_No" } ,  [[0.12320956], [0.87679044]]  ) , # twin-net // Rule 2 -> P(tw_D | tw_C, C, D)
                           ("NODE_D", {"NODE_C":"STATE_NODE_C_No" }, {"NODE_D":"STATE_NODE_D_Yes", "NODE_C":"STATE_NODE_C_Yes"} ,  [[0.4973683 ], [0.5026317]]) ])    # twin-net // Rule 2 -> P(tw_D | tw_C, C, D)
@pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
@patch('bayesiansafety.core.inference.InferenceFactory.get_configured_backend', autospec=True)
def test_counterfactual_query_marginal_success(fn_default_backend, backend, target, whatif, observed, expected, fixture_bn_causal_queries_param ):
    model_name = 'test'
    model = fixture_bn_causal_queries_param(model_name)

    fn_default_backend.return_value = backend
    if backend is Backend.PGMPY:
        pytest.xfail(reason=XFAIL_REASON)

    instance = InferenceFactory(model).get_engine()
    query_result =  instance.counterfactual_query(target = target , do=whatif, observed=observed)

    assert np.allclose(query_result.get_probabilities(), expected, atol=1e-4)



PREFIX = TwinNetwork.TWIN_NET_PREFIX
@pytest.mark.parametrize("target, whatif, observed, expected",
                          [(["CS_1", "CS_3"], {"FOG":"YES"}, {"FOG":"NO", "SEASON":"WINTER"},  { (PREFIX+"CS_1", "YES", PREFIX+"CS_3", "YES"):0.05494,
                                                                                                      (PREFIX+"CS_1", "YES", PREFIX+"CS_3", "NO"):0.00706,
                                                                                                      (PREFIX+"CS_1", "NO", PREFIX+"CS_3", "YES"):0.80626,
                                                                                                      (PREFIX+"CS_1", "NO", PREFIX+"CS_3", "NO"):0.13174 }    ),

                          (["CS_4", "OR"], {"LIGHTING":"INTENSE"}, {"LIGHTING":"NORMAL", "TIME":"NIGHT"},  {  (PREFIX+"CS_4", "YES", PREFIX+"OR", "ACTIVE"):0.05,
                                                                                                                    (PREFIX+"CS_4", "YES", PREFIX+"OR", "INACTIVE"):0.0,
                                                                                                                    (PREFIX+"CS_4", "NO",  PREFIX+"OR", "ACTIVE"):0.11853089,
                                                                                                                    (PREFIX+"CS_4", "NO", PREFIX+"OR", "INACTIVE"):0.83146911 } ),
                          ])
@pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
@patch('bayesiansafety.core.inference.InferenceFactory.get_configured_backend', autospec=True)
def test_counterfactual_query_factor_success(fn_default_backend, backend, target, whatif, observed, expected, fixture_bn_paper1_chauffeur_twin_model):
    model_name = 'test'
    model = fixture_bn_paper1_chauffeur_twin_model

    fn_default_backend.return_value = backend

    instance = InferenceFactory(model).get_engine()
    query_result =  instance.counterfactual_query(target = target , do=whatif, observed=observed)

    for scope, expected_val in expected.items():
        node_states = {scope[0]:scope[1], scope[2]:scope[3]}

        assert np.isclose(query_result.get_value(node_states), expected_val, atol=2e-4)

##------

def test_interventinal_inference_overlapping_query_node_and_evidence_raises_exception(fixture_bn_confounder_param):
    expected_exc_substring = "Query contains evidence"
    model_name = 'test'
    queried_node = "NODE_C"
    overlapping_evidence = {"NODE_B":"STATE_NODE_B_Yes", queried_node:f"STATE_{queried_node}_Yes"}

    model, _ = fixture_bn_confounder_param(model_name)
    instance = InferenceFactory(model).get_engine()

    with pytest.raises(ValueError) as e:
        assert instance.interventional_query(variables=queried_node, evidence=overlapping_evidence)

    assert expected_exc_substring in str(e.value)


def test_interventinal_inference_overlapping_query_node_and_do_raises_exception(fixture_bn_confounder_param):
    expected_exc_substring = "Query contains do-variables"
    model_name = 'test'
    queried_node = "NODE_C"
    overlapping_do = {"NODE_B":"STATE_NODE_B_Yes", queried_node:f"STATE_{queried_node}_Yes"}

    model, _ = fixture_bn_confounder_param(model_name)
    instance = InferenceFactory(model).get_engine()

    with pytest.raises(ValueError) as e:
        assert instance.interventional_query(variables=queried_node, do=overlapping_do)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("bad_query", [None, 5,  {0:"NODE_A"}])
def test_interventional_inference_invalid_type_for_variables_raises_exception(bad_query, fixture_bn_confounder_param):
    expected_exc_substring = "Queried variable(s) need to be a string or list of strings but are"
    model_name = 'test'
    model, _ = fixture_bn_confounder_param(model_name)
    instance = InferenceFactory(model).get_engine()

    with pytest.raises(TypeError) as e:
        assert instance.interventional_query(bad_query)

    assert expected_exc_substring in str(e.value)
@pytest.mark.parametrize("bad_query, valid_do", [(None, {"NODE_C":"STATE_NODE_C_Yes"}), (5, {"NODE_C":"STATE_NODE_C_Yes"}), ({0:"NODE_A"}, {"NODE_C":"STATE_NODE_C_Yes"})])
def test_counterfactual_inference_invalid_type_for_variables_raises_exception(bad_query, valid_do, fixture_bn_causal_queries_param):
    expected_exc_substring = "Queried target(s) need to be a string or list of strings but are"
    model_name = 'test'
    model = fixture_bn_causal_queries_param(model_name)
    instance = InferenceFactory(model).get_engine()

    with pytest.raises(TypeError) as e:
        assert instance.counterfactual_query(target = bad_query, do=valid_do)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("bad_query", [None, 5,  {0:"NODE_A"}])
def test_inference_invalid_type_for_variables_raises_exception(bad_query, fixture_bn_confounder_param):
    expected_exc_substring = "Queried variable(s) need to be a string or list of strings but are"
    model_name = 'test'
    model, _ = fixture_bn_confounder_param(model_name)
    instance = InferenceFactory(model).get_engine()

    with pytest.raises(TypeError) as e:
        assert instance.query(bad_query)

    assert expected_exc_substring in str(e.value)


def test_inference_overlapping_query_node_and_evidence_raises_exception(fixture_bn_confounder_param):
    expected_exc_substring = "Query contains evidence"
    model_name = 'test'
    queried_node = "NODE_C"
    overlapping_evidence = {"NODE_B":"STATE_NODE_B_Yes", queried_node:f"STATE_{queried_node}_Yes"}

    model, _ = fixture_bn_confounder_param(model_name)
    instance = InferenceFactory(model).get_engine()

    with pytest.raises(ValueError) as e:
        assert instance.query(variables=queried_node, evidence=overlapping_evidence)

    assert expected_exc_substring in str(e.value)




########## old tests - formerly as part of BayesianNetwork -class
# @pytest.mark.parametrize("fixture", ["fixture_bn_confounder_param", "fixture_bn_collider_param"])
# def test_inference_marginal_returns_correct(fixture, request):
#     model_name = 'test'
#     EPSILON = 1e-3
#     model, marginals = request.getfixturevalue(fixture)(model_name)

#     for node_name in model.model_elements.keys():
#         marg_cpt = model.inference(node_name)
#         assert np.allclose(marg_cpt.get_probabilities(), marginals[node_name], atol=EPSILON)



# @pytest.mark.parametrize("fixture, query, evidence, isCPT, expected",
#                                         [ ("fixture_bn_independent_nodes_only_param", ["NODE_A"], None, True, [[0.123], [0.877]]),
#                                          ("fixture_bn_independent_nodes_only_param", ["NODE_A"], {"NODE_B":"STATE_NODE_B_Yes"}, True, [[0.123], [0.877]]),
#                                          ("fixture_bn_collider_param"                , ["NODE_B"], {"NODE_A":"STATE_NODE_A_Yes"}, True, [[0.987], [0.0130]]),
#                                          ("fixture_bn_confounder_param"              , ["NODE_B"], {"NODE_A":"STATE_NODE_A_Yes"}, True, [[0.120], [0.880]]),
#                                          ("fixture_bn_confounder_param"              , ["NODE_A"], {"NODE_B":"STATE_NODE_B_Yes",
#                                                                                                        "NODE_C":"STATE_NODE_C_No"},  True, [[3/1321], [1318/1321]]),
#                                          ("fixture_bn_independent_nodes_only_param", ["NODE_A", "NODE_B"], None, False, [0.0016]),
#                                          ("fixture_bn_independent_nodes_only_param", ["NODE_A", "NODE_B"], {"NODE_C":"STATE_NODE_C_No"} , False, [0.0016]),
#                                          ("fixture_bn_confounder_param"              , ["NODE_A", "NODE_B"], {"NODE_C":"STATE_NODE_C_Yes"}, False, [0.1103]),
#                                          ("fixture_bn_collider_param"                , ["NODE_A", "NODE_B"], {"NODE_C":"STATE_NODE_C_No"} , False, [2.99987e-3]),
#                                         ])
# def test_inference_returns_correct(fixture, query, evidence, isCPT, expected, request):
#     model_name = 'test'
#     EPSILON = 1e-3
#     discrete_factor_fixed_states = {"NODE_A":"STATE_NODE_A_Yes", "NODE_B":"STATE_NODE_B_No"}

#     model, _ = request.getfixturevalue(fixture)(model_name)

#     query_result = model.inference(variables=query, evidence=evidence)

#     assert isinstance(query_result, (DiscreteFactor, ConditionalProbabilityTable))

#     if isCPT:
#         assert isinstance(query_result, ConditionalProbabilityTable)
#         assert np.allclose(query_result.get_probabilities(), expected, atol=EPSILON)

#     else:
#         assert isinstance(query_result, DiscreteFactor)
#         assert np.allclose(query_result.get_value(discrete_factor_fixed_states), expected, atol=EPSILON)


# def test_interventional_inference_success(fixture_bn_confounder_param):
#     model_name = 'test'
#     model, _ = fixture_bn_confounder_param(model_name)
#     queried_node ="NODE_B"
#     do = {"NODE_A":"STATE_NODE_A_No"}
#     evidence = None

#     assert model.interventional_inference(variables=queried_node, do=do, evidence=evidence) != None

# def test_counterfactual_inference_success(fixture_bn_causal_queries_param):
#     model_name = 'test'
#     model = fixture_bn_causal_queries_param(model_name)
#     target ="NODE_D"
#     do = {"NODE_C":"STATE_NODE_C_No"}
#     observed = {"NODE_C":"STATE_NODE_C_Yes"}

#     assert model.counterfactual_inference(target=target, do=do, observed=observed) != None
