import pytest
import numpy as np
import networkx as nx

from bayesiansafety.core import BayesianNetwork
from bayesiansafety.core import ConditionalProbabilityTable

from bayesiansafety.faulttree import FaultTreeProbNode
from bayesiansafety.faulttree import FaultTreeLogicNode
from bayesiansafety.faulttree import BayesianFaultTree    

from bayesiansafety.eventtree import EventTreeImporter
from bayesiansafety.eventtree import InitiatingEvent, FunctionalEvent, Consequence, Path

from bayesiansafety.synthesis import Synthesis
from bayesiansafety.synthesis.hybrid import HybridConfiguration
from bayesiansafety.synthesis.functional import Behaviour, FunctionalConfiguration


############ Fault Trees
@pytest.fixture
def fixture_ft_and_only_model():
    # Book page 273
    A = FaultTreeProbNode(name='A'   , probability_of_failure=2.78e-6)
    B = FaultTreeProbNode(name='B'   , probability_of_failure=4.12e-4)
    C = FaultTreeProbNode(name='C'   , probability_of_failure=8.81e-5)


    AND   = FaultTreeLogicNode(name='AND', input_nodes=['A', 'B', 'C'], logic_type="AND" )
    
    correct_cutsets = [{"A", "B", "C"}]

    probability_nodes = [A, B, C] 
    logic_nodes       = [AND]

    return BayesianFaultTree("and_only_model", probability_nodes, logic_nodes), correct_cutsets


@pytest.fixture
def fixture_ft_or_only_model():
    # Book page 273
    A = FaultTreeProbNode(name='A'   , probability_of_failure=2.78e-6)
    B = FaultTreeProbNode(name='B'   , probability_of_failure=4.12e-4)
    C = FaultTreeProbNode(name='C'   , probability_of_failure=8.81e-5)


    OR   = FaultTreeLogicNode(name='OR', input_nodes=['A', 'B', 'C'], logic_type="OR" )
    
    correct_cutsets = [{"A"}, {"B"}, {"C"}]

    probability_nodes = [A, B, C] 
    logic_nodes       = [OR]

    return BayesianFaultTree("or_only_mode", probability_nodes, logic_nodes), correct_cutsets


@pytest.fixture
def fixture_ft_elevator_model():
    # Elevator example from Fehlerbaumanalyse in Theorie und Praxis,  F. Edler & M.Soden and R.Hankammer (2015) p.126 / 127 (ISBN: 978-3-662-48165-3)
    EMI_BS_FP           = FaultTreeProbNode(name='EMI_BS_FP'        , probability_of_failure=2.78e-7)
    EMI_LOG_FA          = FaultTreeProbNode(name='EMI_LOG_FA'       , probability_of_failure=2.78e-7)
    EMI_LS_FP           = FaultTreeProbNode(name='EMI_LS_FP'        , probability_of_failure=2.78e-7)
    EMI_SIG_BS_FP       = FaultTreeProbNode(name='EMI_SIG_BS_FP'    , probability_of_failure=2.78e-7)

    EMI_SIG_HI_FP       = FaultTreeProbNode(name='EMI_SIG_HI_FP'    , probability_of_failure=2.78e-7)
    EMI_SIG_LO_FP       = FaultTreeProbNode(name='EMI_SIG_LO_FP'    , probability_of_failure=2.78e-7)
    EMI_SIG_LS_FP       = FaultTreeProbNode(name='EMI_SIG_LS_FP'    , probability_of_failure=2.78e-7)
    IHF_BS_FP           = FaultTreeProbNode(name='IHF_BS_FP'        , probability_of_failure=1.998e-3  , is_time_dependent=True)

    IHF_LOG_FA          = FaultTreeProbNode(name='IHF_LOG_FA'       , probability_of_failure=1.199e-3  , is_time_dependent=True)
    IHF_LS_FP           = FaultTreeProbNode(name='IHF_LS_FP'        , probability_of_failure=7.997e-4  , is_time_dependent=True)
    IHF_SIG_BS_FP       = FaultTreeProbNode(name='IHF_SIG_BS_FP'    , probability_of_failure=9.95e-3   , is_time_dependent=True)
    IHF_SIG_HI_FP       = FaultTreeProbNode(name='IHF_SIG_HI_FP'    , probability_of_failure=1e-5      , is_time_dependent=True)

    IHF_SIG_LO_FP       = FaultTreeProbNode(name='IHF_SIG_LO_FP'    , probability_of_failure=1e-5      , is_time_dependent=True)
    IHF_SIG_LS_FP       = FaultTreeProbNode(name='IHF_SIG_LS_FP'    , probability_of_failure=1e-5      , is_time_dependent=True)
    SYS_LOG_FA          = FaultTreeProbNode(name='SYS_LOG_FA'       , probability_of_failure=0)


    # Fig. 5.4 on page 126
    AND_TOP         = FaultTreeLogicNode(name='AND_TOP'     , input_nodes=['OR_SIG_HI', 'OR_SIG_LO']  , logic_type='AND')
    OR_SIG_HI       = FaultTreeLogicNode(name='OR_SIG_HI'   , input_nodes=['IHF_SIG_HI_FP', 'EMI_SIG_HI_FP', 'OR_LOG']  )
    OR_SIG_LO       = FaultTreeLogicNode(name='OR_SIG_LO'   , input_nodes=['IHF_SIG_LO_FP', 'EMI_SIG_LO_FP', 'OR_LOG']  )
    OR_LOG          = FaultTreeLogicNode(name='OR_LOG'      , input_nodes=['IHF_LOG_FA', 'EMI_LOG_FA', 'SYS_LOG_FA', 'AND_LS_BS_SIG']  )

     # Fig. 5.5 on page 127
    AND_LS_BS_SIG   = FaultTreeLogicNode(name='AND_LS_BS_SIG', input_nodes=['OR_LS_SIG', 'OR_BS_SIG']  , logic_type='AND')
    OR_LS_SIG       = FaultTreeLogicNode(name='OR_LS_SIG'    , input_nodes=['IHF_SIG_LS_FP', 'EMI_SIG_LS_FP', 'OR_LS']  )
    OR_BS_SIG       = FaultTreeLogicNode(name='OR_BS_SIG'    , input_nodes=['IHF_SIG_BS_FP', 'EMI_SIG_BS_FP', 'OR_BS']  )
    OR_LS           = FaultTreeLogicNode(name='OR_LS'        , input_nodes=['IHF_LS_FP', 'EMI_LS_FP']  )
    OR_BS           = FaultTreeLogicNode(name='OR_BS'        , input_nodes=['IHF_BS_FP', 'EMI_BS_FP']  )


    probability_nodes = [EMI_BS_FP, EMI_LOG_FA, EMI_LS_FP, EMI_SIG_BS_FP, EMI_SIG_HI_FP, EMI_SIG_LO_FP, EMI_SIG_LS_FP, IHF_BS_FP, IHF_LOG_FA, IHF_LS_FP, IHF_SIG_BS_FP, IHF_SIG_HI_FP, IHF_SIG_LO_FP, IHF_SIG_LS_FP, SYS_LOG_FA] 
    logic_nodes       = [OR_SIG_HI, OR_SIG_LO, OR_LOG, AND_LS_BS_SIG, OR_LS_SIG, OR_BS_SIG, OR_LS, OR_BS, AND_TOP]

    correct_cutsets = [{"IHF_LOG_FA"}, {"EMI_LOG_FA"},  {"SYS_LOG_FA"}, {"IHF_LS_FP" , "IHF_BS_FP"},  {"IHF_LS_FP" , "IHF_SIG_BS_FP"},
                    {"IHF_SIG_LS_FP" , "IHF_SIG_BS_FP"},  {"IHF_SIG_LS_FP" , "IHF_BS_FP"},  {"EMI_LS_FP" , "IHF_SIG_BS_FP"},  {"EMI_SIG_LS_FP" , "IHF_SIG_BS_FP"},
                    {"EMI_LS_FP" , "IHF_BS_FP"},  {"EMI_SIG_LS_FP" , "IHF_BS_FP"},  {"IHF_LS_FP" , "EMI_BS_FP"},  {"IHF_LS_FP" , "EMI_SIG_BS_FP"},  {"IHF_SIG_HI_FP" , "IHF_SIG_LO_FP"},
                    {"IHF_SIG_LS_FP" , "EMI_BS_FP"}, {"IHF_SIG_LS_FP" , "EMI_SIG_BS_FP"},  {"EMI_SIG_HI_FP" , "IHF_SIG_LO_FP"},  {"IHF_SIG_HI_FP" , "EMI_SIG_LO_FP"},
                    {"EMI_LS_FP" , "EMI_BS_FP"},  {"EMI_LS_FP" , "EMI_SIG_BS_FP"},  {"EMI_SIG_LS_FP" , "EMI_BS_FP"}, {"EMI_SIG_LS_FP" , "EMI_SIG_BS_FP"}, {"EMI_SIG_HI_FP" , "EMI_SIG_LO_FP"}]

    return BayesianFaultTree("elevator_model", probability_nodes, logic_nodes), correct_cutsets


@pytest.fixture
def fixture_ft_fatram_paper_model():
    # Original example from paper FATRAM-A Core Efficient Cut-Set Algorithm (DOI: 10.1109/TR.1978.5220353)
    A = FaultTreeProbNode(name='A'   , probability_of_failure=2.78e-6)
    B = FaultTreeProbNode(name='B'   , probability_of_failure=4.12e-4)
    C = FaultTreeProbNode(name='C'   , probability_of_failure=8.81e-5)
    D = FaultTreeProbNode(name='D'   , probability_of_failure=2.190e-3)
    E = FaultTreeProbNode(name='E'   , probability_of_failure=7.15e-4)
    F = FaultTreeProbNode(name='F'   , probability_of_failure=1.0e-2)
    G = FaultTreeProbNode(name='G'   , probability_of_failure=1.0e-4)
    H = FaultTreeProbNode(name='H'   , probability_of_failure=3.48e-4)

    TOP   = FaultTreeLogicNode(name='TOP'     , input_nodes=['G1', 'G2'], logic_type='AND'   )
    G1    = FaultTreeLogicNode(name='G1'   , input_nodes=['A', 'G3'], logic_type='AND'  )
    G2    = FaultTreeLogicNode(name='G2'   , input_nodes=['B', 'E', 'G4']  )
    G3    = FaultTreeLogicNode(name='G3'   , input_nodes=['B', 'H', 'C']  )
    G4    = FaultTreeLogicNode(name='G4'   , input_nodes=['D', 'G5'], logic_type='AND')
    G5    = FaultTreeLogicNode(name='G5'   , input_nodes=['F', 'C', 'G']  )

    probability_nodes = [A, B, C, D, E, F, G, H] 
    logic_nodes       = [TOP, G1, G2, G3, G4, G5]

    correct_cutsets = [{"A", "H", "E"}, {"A", "H", "D", "F"}, {"A", "H", "D", "G"}, {"A", "B"}, {"A", "C", "E"}, {"A", "C", "D"}]

    return BayesianFaultTree("fatram_paper_model", probability_nodes, logic_nodes), correct_cutsets


@pytest.fixture
def fixture_ft_mocus_book_model():
    # MOCUS hands on example from Fehlerbaumanalyse in Theorie und Praxis,  F. Edler & M.Soden and R.Hankammer (2015) p.273 (ISBN: 978-3-662-48165-3)
    E1 = FaultTreeProbNode(name='E1'   , probability_of_failure=2.78e-6)
    E2 = FaultTreeProbNode(name='E2'   , probability_of_failure=4.12e-4)
    E3 = FaultTreeProbNode(name='E3'   , probability_of_failure=8.81e-5)
    E4 = FaultTreeProbNode(name='E4'   , probability_of_failure=1.0e-3)

    TLE_G1   = FaultTreeLogicNode(name='TLE_G1'     , input_nodes=['E1', 'G2'] )
    G2       = FaultTreeLogicNode(name='G2'   , input_nodes=['G3', 'G4'], logic_type='AND'  )
    G3       = FaultTreeLogicNode(name='G3'   , input_nodes=['E1', 'E2']  )
    G4       = FaultTreeLogicNode(name='G4'   , input_nodes=['E3', 'E4']  )
    
    correct_cutsets = [{"E1"}, {"E2", "E3"}, {"E2", "E4"}]

    probability_nodes = [E1, E2, E3, E4] 
    logic_nodes       = [TLE_G1, G2, G3, G4]

    return BayesianFaultTree("mocus_book_model", probability_nodes, logic_nodes), correct_cutsets


############ Bayesian Networks

@pytest.fixture
def fixture_bn_confounder_param():

    def _make_bn(bn_name):
        node_connections = [("NODE_A", "NODE_B"), ("NODE_A", "NODE_C")]
        bn = BayesianNetwork(bn_name, node_connections)

        NODE_A = ConditionalProbabilityTable(name="NODE_A", variable_card=2, values=[[0.1], [0.9]], evidence=None, evidence_card=None, state_names={"NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"]})
        NODE_B = ConditionalProbabilityTable(name="NODE_B", variable_card=2, values=[[0.12, 0.34], [0.88, 0.66]], evidence=["NODE_A"], evidence_card=[2], state_names={"NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"], "NODE_B": ["STATE_NODE_B_Yes", "STATE_NODE_B_No"]})
        NODE_C = ConditionalProbabilityTable(name="NODE_C", variable_card=2, values=[[0.98, 0.76], [0.02, 0.24]], evidence=["NODE_A"], evidence_card=[2], state_names={"NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"], "NODE_C": ["STATE_NODE_C_Yes", "STATE_NODE_C_No"]})

        bn.add_cpts(NODE_A, NODE_B, NODE_C)

        marginals = {"NODE_A":[[0.1], [0.9]], "NODE_B":[[0.318], [0.682]], "NODE_C":[[0.782], [0.218]]}

        return bn, marginals

    return _make_bn


@pytest.fixture
def fixture_bn_collider_param():

    def _make_bn(bn_name):
        node_connections = [("NODE_A", "NODE_C"), ("NODE_B", "NODE_C")]
        bn = BayesianNetwork(bn_name, node_connections)

        NODE_A = ConditionalProbabilityTable(name="NODE_A", variable_card=2, values=[[0.123], [0.877]], evidence=None, evidence_card=None, state_names={"NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"]})
        NODE_B = ConditionalProbabilityTable(name="NODE_B", variable_card=2, values=[[0.987], [0.013]], evidence=None, evidence_card=None, state_names={"NODE_B": ["STATE_NODE_B_Yes", "STATE_NODE_B_No"]})
        NODE_C = ConditionalProbabilityTable(name="NODE_C", variable_card=2, values=[[0.123, 0.456, 0.789, 0.987], [0.877, 0.544, 0.211, 0.013]], evidence=["NODE_A", "NODE_B"], evidence_card=[2, 2],         
                                            state_names={"NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"], 
                                                         "NODE_B": ["STATE_NODE_B_Yes", "STATE_NODE_B_No"], 
                                                         "NODE_C": ["STATE_NODE_C_Yes", "STATE_NODE_C_No"]})

        bn.add_cpts(NODE_A, NODE_B, NODE_C)

        marginals = {"NODE_A":[[0.123], [0.877]], "NODE_B":[[0.987],[ 0.013]], "NODE_C":[[0.71], [0.29]]}

        return bn, marginals

    return _make_bn


@pytest.fixture
def fixture_bn_independent_nodes_only_param():

    def _make_bn(bn_name):
        node_connections = []
        bn = BayesianNetwork(bn_name, node_connections)
        
        nodes = ["NODE_A","NODE_B","NODE_C"]
        for node in nodes:
            bn.add_node(node)

        NODE_A = ConditionalProbabilityTable(name="NODE_A", variable_card=2, values=[[0.123], [0.877]], evidence=None, evidence_card=None, state_names={"NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"]})
        NODE_B = ConditionalProbabilityTable(name="NODE_B", variable_card=2, values=[[0.987], [0.013]], evidence=None, evidence_card=None, state_names={"NODE_B": ["STATE_NODE_B_Yes", "STATE_NODE_B_No"]})
        NODE_C = ConditionalProbabilityTable(name="NODE_C", variable_card=2, values=[[0.456], [0.544]], evidence=None, evidence_card=None, state_names={"NODE_C": ["STATE_NODE_C_Yes", "STATE_NODE_C_No"]})

        bn.add_cpts(NODE_A, NODE_B, NODE_C)

        marginals = {"NODE_A":[[0.123], [0.877]], "NODE_B":[[0.987], [0.013]], "NODE_C":[[0.456], [0.544]]}

        return bn, marginals

    return _make_bn


@pytest.fixture
def fixture_bn_causal_queries_param():

    def _make_bn(bn_name):
        node_connections = [("NODE_A", "NODE_C"), ("NODE_B", "NODE_D"), ("NODE_C", "NODE_D")]
        bn = BayesianNetwork(bn_name, node_connections)

        NODE_A = ConditionalProbabilityTable(name="NODE_A", variable_card=2, values=[[0.123], [0.877]], evidence=None, evidence_card=None, state_names={"NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"]})
        NODE_B = ConditionalProbabilityTable(name="NODE_B", variable_card=2, values=[[0.987], [0.013]], evidence=None, evidence_card=None, state_names={"NODE_B": ["STATE_NODE_B_Yes", "STATE_NODE_B_No"]})
        
        NODE_C = ConditionalProbabilityTable(name="NODE_C", variable_card=2, values=[[0.321, 0.654], [0.679, 0.346]], evidence=["NODE_A"], evidence_card=[2], state_names={"NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"],
                                                                                                                                                "NODE_C": ["STATE_NODE_C_Yes", "STATE_NODE_C_No"]})

        NODE_D = ConditionalProbabilityTable(name="NODE_D", variable_card=2, values=[[0.123, 0.456, 0.789, 0.987], [0.877, 0.544, 0.211, 0.013]], evidence=["NODE_B", "NODE_C"], evidence_card=[2, 2],         
                                            state_names={"NODE_B": ["STATE_NODE_B_Yes", "STATE_NODE_B_No"], 
                                                         "NODE_C": ["STATE_NODE_C_Yes", "STATE_NODE_C_No"], 
                                                         "NODE_D": ["STATE_NODE_D_Yes", "STATE_NODE_D_No"]})

        bn.add_cpts(NODE_A, NODE_B, NODE_C, NODE_D)

        return bn

    return _make_bn


@pytest.fixture
def fixture_bn_paper1_chauffeur_twin_model():
    SEASON      = ConditionalProbabilityTable(name="SEASON", variable_card=4, values=[[0.25], [0.25], [0.25], [0.25]], evidence=None, evidence_card=None, state_names={"SEASON": ["SPRING", "SUMMER", "FALL", "WINTER"]})
    TIME        = ConditionalProbabilityTable(name="TIME", variable_card=4, values=[[0.20], [0.20], [0.20], [0.40]], evidence=None, evidence_card=None, state_names={"TIME": ["MORNING", "NOON", "EVENING", "NIGHT"]})
    FOG         = ConditionalProbabilityTable(name="FOG", variable_card=2, values=[[0.02, 0.01, 0.2, 0.05], [0.98, 0.99, 0.80, 0.95]], evidence=["SEASON"], evidence_card=[4], state_names={"FOG": ["YES", "NO"], "SEASON": ["SPRING", "SUMMER", "FALL", "WINTER"]})
    LIGHTING    = ConditionalProbabilityTable(name="LIGHTING", variable_card=2, values=[[0.005, 0.10, 0.005, 0.001, 0.10, 0.25, 0.15, 0.05, 0.005, 0.1, 0.005, 0.001, 0.001, 0.001, 0.001, 0.001],
                                                                                     [0.995, 0.90, 0.995, 0.999, 0.90, 0.75, 0.85, 0.95, 0.995, 0.9, 0.995, 0.999, 0.999, 0.999, 0.999, 0.999]], evidence=["TIME", "SEASON"], evidence_card=[4, 4], state_names={"LIGHTING": ["INTENSE", "NORMAL"], "TIME": ["MORNING", "NOON", "EVENING", "NIGHT"], "SEASON": ["SPRING", "SUMMER", "FALL", "WINTER"]})
    SNOW        = ConditionalProbabilityTable(name="SNOW", variable_card=2, values=[[0.02, 0.001, 0.03, 0.08], [0.98, 0.999, 0.97, 0.92]], evidence=["SEASON"], evidence_card=[4], state_names={"SNOW": ["YES", "NO"], "SEASON": ["SPRING", "SUMMER", "FALL", "WINTER"]})

    CS_1        = ConditionalProbabilityTable(name="CS_1", variable_card=2, values=[[0.20, 0.05], [0.80, 0.95]], evidence=["SNOW"], evidence_card=[2], state_names={"CS_1": ["YES", "NO"], "SNOW": ["YES", "NO"]})
    CS_3        = ConditionalProbabilityTable(name="CS_3", variable_card=2, values=[[0.99, 0.85, 0.20, 0.01], [0.01, 0.15, 0.80, 0.99]], evidence=["FOG", "SNOW"], evidence_card=[2, 2], state_names={"CS_3": ["YES", "NO"], "FOG": ["YES", "NO"], "SNOW": ["YES", "NO"]}) 
    CS_4        = ConditionalProbabilityTable(name="CS_4", variable_card=2, values=[[0.05, 0.001], [0.95, 0.999]], evidence=["LIGHTING"], evidence_card=[2], state_names={"CS_4": ["YES", "NO"], "LIGHTING": ["INTENSE", "NORMAL"]})

    OR          = ConditionalProbabilityTable(name="OR", variable_card=2, values=[  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0 ], 
                                                                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0 ]], 
                                                                                    evidence=["CS_1", "CS_3", "CS_4"], evidence_card=[2, 2, 2], state_names={"OR": ["ACTIVE", "INACTIVE"], "CS_1": ["YES", "NO"], "CS_3": ["YES", "NO"], "CS_4": ["YES", "NO"]})


    node_connections      = [("SEASON", "FOG"), ("SEASON", "SNOW"), ("SEASON", "LIGHTING"), ("TIME", "LIGHTING"),
                            ("FOG", "CS_3"), ("SNOW", "CS_3"), ("SNOW", "CS_1"), ("LIGHTING", "CS_4"),                           
                            ("CS_3", "OR"), ("CS_1", "OR"), ("CS_4", "OR")]

    bn = BayesianNetwork("Chauffeur", node_connections)
    bn.add_cpts(SEASON, TIME, FOG ,LIGHTING ,SNOW,CS_1 ,CS_3 ,CS_4, OR)

    return bn


############ Event Trees

@pytest.fixture
def fixture_et_causal_arc_param():
    '''Defines the example Event Tree from Bearfield and Marsh 2005 for causal arc simplification.
        See https://doi.org/10.1007/11563228_5, section 3.4
    '''
    def _make_et(et_name):
        FE_1 = FunctionalEvent("FE1", ["e11", "e12"])
        FE_2 = FunctionalEvent("FE2", ["e21", "e22"])
        CNSQ = Consequence("CNSQ",["c1", "c2", "c3"])

        paths = [ ( (FE_1, "e11", 0.7), (FE_2, "e21", 0.1), (CNSQ, "c1") ),
                  ( (FE_1, "e11", 0.7), (FE_2, "e22", 0.9), (CNSQ, "c2") ),
                  ( (FE_1, "e12", 0.3), (FE_2, "e21", 0.1), (CNSQ, "c1") ),
                  ( (FE_1, "e12", 0.3), (FE_2, "e22", 0.9), (CNSQ, "c3") ) ]

        et_model = EventTreeImporter().construct(et_name, paths)
        consequence_probabilities = {"c1":0.1, "c2":0.63, "c3":0.27}

        return et_model, consequence_probabilities

    return _make_et


@pytest.fixture
def fixture_et_consequence_arc_param():
    '''Defines the example Event Tree from Bearfield and Marsh 2005 for consequence arc simplification.
        See https://doi.org/10.1007/11563228_5, section 3.3
    '''
    def _make_et(et_name):
        FE_1 = FunctionalEvent("FE1", ["e11", "e12"])
        FE_2 = FunctionalEvent("FE2", ["e21", "e22"])
        CNSQ = Consequence("CNSQ",["c1", "c2"])

        paths = [ ( (FE_1, "e11", 0.7), (FE_2, "e21", 0.01), (CNSQ, "c1") ),
                  ( (FE_1, "e11", 0.7), (FE_2, "e22", 0.99), (CNSQ, "c2") ),
                  ( (FE_1, "e12", 0.3), (FE_2, "e21", 0.10), (CNSQ, "c1") ),
                  ( (FE_1, "e12", 0.3), (FE_2, "e22", 0.90), (CNSQ, "c2") ) ]

        et_model = EventTreeImporter().construct(et_name, paths)
        consequence_probabilities = {"c1":0.037, "c2":0.963}

        return et_model, consequence_probabilities

    return _make_et

@pytest.fixture
def fixture_et_dont_care_param():
    '''Defines the example Event Tree from Bearfield and Marsh 2005 for don't care paths.
        See https://doi.org/10.1007/11563228_5, Fig. 1
    '''
    def _make_et(et_name):
        FE_1 = FunctionalEvent("FE1", ["e11", "e12", "e13"])
        FE_2 = FunctionalEvent("FE2", ["e21", "e22"])
        CNSQ = Consequence("CNSQ",["c1", "c2"])

        paths = [ ( (FE_1, "e11", 0.1), (FE_2, "e21", 0.01), (CNSQ, "c1") ),
                  ( (FE_1, "e11", 0.1), (FE_2, "e22", 0.99), (CNSQ, "c2") ),
                  ( (FE_1, "e12", 0.2), (FE_2, "e21", 0.70), (CNSQ, "c2") ),
                  ( (FE_1, "e12", 0.2), (FE_2, "e22", 0.30), (CNSQ, "c1") ),
                  ( (FE_1, "e13", 0.7), (CNSQ, "c2")) ]

        et_model = EventTreeImporter().construct(et_name, paths)
        consequence_probabilities = {"c1":0.061, "c2":0.939}

        return et_model, consequence_probabilities

    return _make_et

@pytest.fixture
def fixture_et_train_derailment_param():
    '''Defines the example Event Tree from Bearfield and Marsh 2005 for the train derailment case study
        See https://doi.org/10.1007/11563228_5, section 4.1
    '''
    def _make_et(et_name):
        Contained = FunctionalEvent("Contained", ["yes", "no"])
        Clear = FunctionalEvent("Clear", ["yes", "no"])
        Cess_Adj = FunctionalEvent("Cess_Adj", ["cess", "adj"])
        Falls = FunctionalEvent("Falls", ["yes", "no"])
        Hits = FunctionalEvent("Hits", ["yes", "no"])
        Collapse = FunctionalEvent("Collapse", ["yes", "no"])
        Collision = FunctionalEvent("Collision", ["yes", "no"])

        CNSQ = Consequence("CNSQ",["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12"])
        paths = [ ( (Contained, "yes", 0.0), (CNSQ, "d1") ),
                  ( (Contained, "no",  1.0), (Clear, "yes", 0.29), (CNSQ, "d2") ),
                  ( (Contained, "no",  1.0), (Clear, "no",  0.71),  (Cess_Adj, "cess", 0.125), (Falls, "no",  0.95), (Hits, "no",  0.80), (CNSQ, "d3") ),
                  ( (Contained, "no",  1.0), (Clear, "no",  0.71),  (Cess_Adj, "cess", 0.125), (Falls, "no",  0.95), (Hits, "yes", 0.20), (Collapse, "no",  0.95), (CNSQ, "d4") ),
                  ( (Contained, "no",  1.0), (Clear, "no",  0.71),  (Cess_Adj, "cess", 0.125), (Falls, "no",  0.95), (Hits, "yes", 0.20), (Collapse, "yes", 0.05), (CNSQ, "d5") ),
                  ( (Contained, "no",  1.0), (Clear, "no",  0.71),  (Cess_Adj, "cess", 0.125), (Falls, "yes", 0.05), (Hits, "no",  0.75), (CNSQ, "d6") ),
                  ( (Contained, "no",  1.0), (Clear, "no",  0.71),  (Cess_Adj, "cess", 0.125), (Falls, "yes", 0.05), (Hits, "yes", 0.25), (Collapse, "no",  0.95), (CNSQ, "d7") ),
                  ( (Contained, "no",  1.0), (Clear, "no",  0.71),  (Cess_Adj, "cess", 0.125), (Falls, "yes", 0.05), (Hits, "yes", 0.25), (Collapse, "yes", 0.05), (CNSQ, "d8") ),
                  ( (Contained, "no",  1.0), (Clear, "no",  0.71),  (Cess_Adj, "adj",  0.875), (Falls, "no",  0.95), (Collision, "no", 0.90), (CNSQ, "d9") ),
                  ( (Contained, "no",  1.0), (Clear, "no",  0.71),  (Cess_Adj, "adj",  0.875), (Falls, "no",  0.95), (Collision, "yes", 0.1), (CNSQ, "d10") ),
                  ( (Contained, "no",  1.0), (Clear, "no",  0.71),  (Cess_Adj, "adj",  0.875), (Falls, "yes", 0.05), (Collision, "no", 0.90), (CNSQ, "d11") ),
                  ( (Contained, "no",  1.0), (Clear, "no",  0.71),  (Cess_Adj, "adj",  0.875), (Falls, "yes", 0.05), (Collision, "yes", 0.1), (CNSQ, "d12") )
                ]

        et_model = EventTreeImporter().construct(et_name, paths)
        consequence_probabilities = {"d1":0.0, "d2":0.29 , "d3":1349/20000, "d4":0.016019375, "d5":8.43125e-4 , "d6":213/64000, 
                                     "d7":1.05390625e-3, "d8":71/128000, "d9":0.53116875, "d10":0.05901875, "d11":0.02795625, "d12":497/160000}

        return et_model, consequence_probabilities

    return _make_et


############ Bayesian Safety (Combination of FTs and BNs)

@pytest.fixture 
def fixture_bsafe_extended_elevator(fixture_ft_elevator_model, fixture_bn_collider_param, fixture_bn_confounder_param, fixture_bn_independent_nodes_only_param):
    ft_elevator, _ = fixture_ft_elevator_model
    trees = {ft_elevator.name:ft_elevator}

    bn_a, _ = fixture_bn_collider_param("BN_A")
    bn_b, _ = fixture_bn_confounder_param("BN_B")
    bn_c, _ = fixture_bn_independent_nodes_only_param("BN_C")

    baynets = { "BN_A":bn_a, "BN_B":bn_b, "BN_C":bn_c}

    shared_nodes = {"BN_A":["NODE_A"],  
                    "BN_B":["NODE_B", "NODE_C"],
                    "BN_C":["NODE_C"] }

    ft_coupling_points = {ft_elevator.name: {"BN_A":[( "NODE_A", 'AND_TOP')] , 
                                             "BN_B":[( "NODE_B", 'OR_SIG_HI'), ( "NODE_C", 'OR_SIG_LO')],
                                             "BN_C":[( "NODE_C", 'OR_SIG_LO')] }}

    pbf_states =  { "BN_A": [('NODE_A', "STATE_NODE_A_Yes")], 
                    "BN_B": [('NODE_B', "STATE_NODE_B_No"), ('NODE_C', "STATE_NODE_C_Yes")],
                    "BN_C": [('NODE_C', "STATE_NODE_C_Yes")],}

    return Synthesis(fault_trees=trees, bayesian_nets=baynets, shared_nodes=shared_nodes,  ft_coupling_points=ft_coupling_points, pbf_states=pbf_states)



############ Functionals and Hybrids

@pytest.fixture
def fixture_functional_config():

    def _make_functional(behaviour, name=None, time_func=np.exp, func_params={"x":123}):
        config = None
        name = name if name != None else str(behaviour)

        node_instance = FaultTreeProbNode(name, 1.7e-3, True)

        if behaviour ==  Behaviour.ADDITION:
            config = FunctionalConfiguration(node_instance=node_instance, 
                                environmental_factors={ "BN_A": [("NODE_A", "STATE_NODE_A_Yes")]}, 
                                thresholds={"BN_A": [("NODE_A", 0.123)]}, 
                                weights={"BN_A": [("NODE_A", 0.5)] }, 
                                time_func=time_func, 
                                func_params=func_params, 
                                behaviour=behaviour)
        
        if behaviour ==  Behaviour.REPLACEMENT:
            config = FunctionalConfiguration(node_instance=node_instance, 
                                environmental_factors={ "BN_A": [("NODE_A", "STATE_NODE_A_Yes")]}, 
                                thresholds={"BN_A": [("NODE_A", 0.123)]}, 
                                weights=None, # due to replacement weight for node will be 1
                                time_func=time_func, 
                                func_params=func_params, 
                                behaviour=behaviour)
        
        if behaviour ==  Behaviour.OVERLAY:
            config = FunctionalConfiguration(node_instance=node_instance, 
                                environmental_factors={ "BN_A": [("NODE_A", "STATE_NODE_A_Yes")],  "BN_B" :[("NODE_B", "STATE_NODE_B_No")], "BN_C" :[("NODE_C", "STATE_NODE_C_Yes")] } , 
                                thresholds= {"BN_A": [("NODE_A", 0.1)], "BN_B": [("NODE_B", 0.2)], "BN_C": [("NODE_C", 0.3)] }, 
                                weights={"BN_A": [("NODE_A", 1)], "BN_B": [("NODE_B", 1)], "BN_C": [("NODE_C", 1)] }, 
                                time_func=time_func, 
                                func_params=func_params, 
                                behaviour=behaviour)
        
        if behaviour ==  Behaviour.FUNCTIONAL:
            config = FunctionalConfiguration(node_instance=node_instance, 
                                environmental_factors={ "BN_A": [("NODE_A", "STATE_NODE_A_Yes")]},   
                                thresholds={"BN_A": [("NODE_A", 0.123)]}, 
                                weights=None, # due to replacement weight for node will be 0
                                time_func=time_func, 
                                func_params=func_params, 
                                behaviour=behaviour)
        
        if behaviour ==  Behaviour.RATE:
            config = FunctionalConfiguration(node_instance=node_instance, 
                                environmental_factors={ "BN_A": [("NODE_A", "STATE_NODE_A_Yes")]}, 
                                thresholds={"BN_A": [("NODE_A", 0.123)]}, 
                                weights=None, # due to replacement weight for node will be 0
                                time_func=time_func, 
                                func_params=func_params, 
                                behaviour=behaviour)

        if behaviour ==  Behaviour.PARAMETER:
            config = FunctionalConfiguration(node_instance=node_instance, 
                                environmental_factors={ "BN_A": [("NODE_A", "STATE_NODE_A_Yes")]}, 
                                thresholds={"BN_A": [("NODE_A", 0.123)]}, 
                                weights={"BN_A": [("NODE_A", 0.5)] },
                                time_func=time_func, 
                                func_params=func_params, 
                                behaviour=behaviour)

        return config, node_instance

    return _make_functional


@pytest.fixture
def fixture_hybrid_config():

    def _make_hybrid_conf(name):
        shared_nodes = {"BN_A":["NODE_A", "NODE_B"]}
        couplings = {"BN_A":[("NODE_A", "AND_TOP"), ("NODE_B", "OR_SIG_HI")]}
        pbf_states = {"BN_A":[("NODE_A", "STATE_NODE_A_No"), ("NODE_B", "STATE_NODE_B_Yes")]}

        config = HybridConfiguration(name=name, shared_nodes=shared_nodes, ft_coupling_points=couplings, pbf_states=pbf_states)
        return config 

    return _make_hybrid_conf


############ Tree objects

@pytest.fixture
def fixture_tree_causal_arc_param():
    '''Defines the example Event Tree from Bearfield and Marsh 2005 for causal arc simplification as actual Tree.
        See https://doi.org/10.1007/11563228_5, section 3.4
    '''

    def _make_tree(tree_name):
        init_event = InitiatingEvent(name="init_event")
        c1_a = Consequence(name="c1")
        c2 = Consequence(name="c2")
        c1_b = Consequence(name="c1")
        c3 = Consequence(name="c3")
        e1 = FunctionalEvent(name="e1")
        e2_a = FunctionalEvent(name="e2")
        e2_b = FunctionalEvent(name="e2")
        path_1 = Path(name="path_1", state="init", probability=1.0, f_event_name="init_event")  #"Init_e1",    
        path_2 = Path(name="path_2", state="e11" , probability=0.7, f_event_name="e1")  #"e1_e11" 
        path_3 = Path(name="path_3", state="e21" , probability=0.1, f_event_name="e2")  #"e2_21" 
        path_4 = Path(name="path_4", state="e22" , probability=0.9, f_event_name="e2")  #"e2_e22" 
        path_5 = Path(name="path_5", state="e12" , probability=0.3, f_event_name="e1")  #"e1_e12" 
        path_6 = Path(name="path_6", state="e21" , probability=0.1, f_event_name="e2")  #"e2_21"  
        path_7 = Path(name="path_7", state="e22" , probability=0.9, f_event_name="e2")  #"e2_e22" 


        node_connections = ( ("init_event", "path_1"), ("path_1", "e1"), ("e1", "path_2"), ("path_2", "e2_a"), ("e2_a", "path_3"), ("path_3", "c1_a"),
                             ("init_event", "path_1"), ("path_1", "e1"), ("e1", "path_2"), ("path_2", "e2_a"), ("e2_a", "path_4"), ("path_4", "c2"),
                             ("init_event", "path_1"), ("path_1", "e1"), ("e1", "path_5"), ("path_5", "e2_b"), ("e2_b", "path_6"), ("path_6", "c1_b"),
                             ("init_event", "path_1"), ("path_1", "e1"), ("e1", "path_5"), ("path_5", "e2_b"), ("e2_b", "path_7"), ("path_7", "c3"),    )

        data = {"init_event":init_event, "c1_a":c1_a, "c2":c2, "c1_b":c1_b, "c3":c3, "e1":e1, "e2_a":e2_a, "e2_b":e2_b, 
                "path_1":path_1, "path_2":path_2, "path_3":path_3, "path_4":path_4, "path_5":path_5, "path_6":path_6, "path_7":path_7}
        

        tree = nx.DiGraph()
        tree.add_edges_from(node_connections)
        tree = nx.bfs_tree(tree, source="init_event")

        node_attributes = {node_name:{"data":node} for node_name, node in data.items()}
        nx.set_node_attributes(tree, node_attributes)

        return tree, node_connections, data

    return _make_tree


@pytest.fixture
def fixture_tree_consequence_arc_param():
    '''Defines the example Event Tree from Bearfield and Marsh 2005 for consequence arc simplification.
        See https://doi.org/10.1007/11563228_5, section 3.3
    '''

    def _make_tree(tree_name):
        init_event = InitiatingEvent(name="init_event")
        c1_a = Consequence(name="c1")
        c2_a = Consequence(name="c2")
        c1_b = Consequence(name="c1")
        c2_b = Consequence(name="c2")
        e1 = FunctionalEvent(name="e1")
        e2_a = FunctionalEvent(name="e2")
        e2_b = FunctionalEvent(name="e2")
        path_1 = Path(name="path_1", state="init", probability=1.0, f_event_name="init_event")  #"Init_e1",    
        path_2 = Path(name="path_2", state="e11" , probability=0.7, f_event_name="e1")  #"e1_e11" 
        path_3 = Path(name="path_3", state="e21" , probability=0.01, f_event_name="e2")  #"e2_21" 
        path_4 = Path(name="path_4", state="e22" , probability=0.99, f_event_name="e2")  #"e2_e22" 
        path_5 = Path(name="path_5", state="e12" , probability=0.3, f_event_name="e1")  #"e1_e12" 
        path_6 = Path(name="path_6", state="e21" , probability=0.1, f_event_name="e2")  #"e2_21"  
        path_7 = Path(name="path_7", state="e22" , probability=0.9, f_event_name="e2")  #"e2_e22" 


        node_connections = ( ("init_event", "path_1"), ("path_1", "e1"), ("e1", "path_2"), ("path_2", "e2_a"), ("e2_a", "path_3"), ("path_3", "c1_a"),
                             ("init_event", "path_1"), ("path_1", "e1"), ("e1", "path_2"), ("path_2", "e2_a"), ("e2_a", "path_4"), ("path_4", "c2_a"),
                             ("init_event", "path_1"), ("path_1", "e1"), ("e1", "path_5"), ("path_5", "e2_b"), ("e2_b", "path_6"), ("path_6", "c1_b"),
                             ("init_event", "path_1"), ("path_1", "e1"), ("e1", "path_5"), ("path_5", "e2_b"), ("e2_b", "path_7"), ("path_7", "c2_b"),    )

        data = {"init_event":init_event, "c1_a":c1_a, "c2_a":c2_a, "c1_b":c1_b, "c2_b":c2_b, "e1":e1, "e2_a":e2_a, "e2_b":e2_b, 
                "path_1":path_1, "path_2":path_2, "path_3":path_3, "path_4":path_4, "path_5":path_5, "path_6":path_6, "path_7":path_7}
        
        tree = nx.DiGraph()
        tree.add_edges_from(node_connections)
        tree = nx.bfs_tree(tree, source="init_event")

        node_attributes = {node_name:{"data":node} for node_name, node in data.items()}
        nx.set_node_attributes(tree, node_attributes)

        return tree, node_connections, data

    return _make_tree


@pytest.fixture
def fixture_tree_dont_care_param():
    '''Defines the example Event Tree from Bearfield and Marsh 2005 for consequence arc simplification.
        See https://doi.org/10.1007/11563228_5, section 3.3
    '''

    def _make_tree(tree_name):
        init_event = InitiatingEvent(name="init_event")
        c1_a = Consequence(name="c1")
        c2_a = Consequence(name="c2")
        c1_b = Consequence(name="c1")
        c2_b = Consequence(name="c2")
        c2_c = Consequence(name="c2")
        e1 = FunctionalEvent(name="e1")
        e2_a = FunctionalEvent(name="e2")
        e2_b = FunctionalEvent(name="e2")
        path_1 = Path(name="path_1", state="init", probability=1.0, f_event_name="init_event")  #"Init_e1",    
        path_2 = Path(name="path_2", state="e11" , probability=0.1, f_event_name="e1")  #"e1_e11" 
        path_3 = Path(name="path_3", state="e21" , probability=0.01, f_event_name="e2")  #"e2_21" 
        path_4 = Path(name="path_4", state="e22" , probability=0.99, f_event_name="e2")  #"e2_e22" 
        path_5 = Path(name="path_5", state="e12" , probability=0.2, f_event_name="e1")  #"e1_e12" 
        path_6 = Path(name="path_6", state="e21" , probability=0.7, f_event_name="e2")  #"e2_21"  
        path_7 = Path(name="path_7", state="e22" , probability=0.3, f_event_name="e2")  #"e2_e22" 
        path_8 = Path(name="path_8", state="e13" , probability=0.7, f_event_name="e1")  #"e1_e13" 


        node_connections = ( ("init_event", "path_1"), ("path_1", "e1"), ("e1", "path_2"), ("path_2", "e2_a"), ("e2_a", "path_3"), ("path_3", "c1_a"),
                             ("init_event", "path_1"), ("path_1", "e1"), ("e1", "path_2"), ("path_2", "e2_a"), ("e2_a", "path_4"), ("path_4", "c2_a"),
                             ("init_event", "path_1"), ("path_1", "e1"), ("e1", "path_5"), ("path_5", "e2_b"), ("e2_b", "path_6"), ("path_6", "c2_b"),
                             ("init_event", "path_1"), ("path_1", "e1"), ("e1", "path_5"), ("path_5", "e2_b"), ("e2_b", "path_7"), ("path_7", "c1_b"),
                             ("init_event", "path_1"), ("path_1", "e1"), ("e1", "path_8"), ("path_8", "c2_c")   )

        data = {"init_event":init_event, "c1_a":c1_a, "c2_a":c2_a, "c1_b":c1_b, "c2_b":c2_b, "c2_c":c2_c, "e1":e1, "e2_a":e2_a, "e2_b":e2_b, 
                "path_1":path_1, "path_2":path_2, "path_3":path_3, "path_4":path_4, "path_5":path_5, "path_6":path_6, "path_7":path_7, "path_8":path_8}
        
        tree = nx.DiGraph()
        tree.add_edges_from(node_connections)
        tree = nx.bfs_tree(tree, source="init_event")

        node_attributes = {node_name:{"data":node} for node_name, node in data.items()}
        nx.set_node_attributes(tree, node_attributes)

        return tree, node_connections, data

    return _make_tree


@pytest.fixture
def fixture_tree_train_derailment_param():
    '''Defines the example Event Tree from Bearfield and Marsh 2005 for the train derailment case study
        See https://doi.org/10.1007/11563228_5, section 4.1
    '''

    def _make_tree(tree_name):
        init_event = InitiatingEvent(name="init_event")
        cd1 = Consequence(name="d1")
        cd2 = Consequence(name="d2")
        cd3 = Consequence(name="d3")
        cd4 = Consequence(name="d4")
        cd5 = Consequence(name="d5")
        cd6 = Consequence(name="d6")    
        cd7 = Consequence(name="d7")
        cd8 = Consequence(name="d8")
        cd9 = Consequence(name="d9")    
        cd10 = Consequence(name="d10")
        cd11 = Consequence(name="d11")
        cd12 = Consequence(name="d12")

        contained = FunctionalEvent(name="contained")
        clear = FunctionalEvent(name="clear")
        ca = FunctionalEvent(name="ca") #cess/adj
        falls_a = FunctionalEvent(name="falls")
        falls_b = FunctionalEvent(name="falls")
        hits_a = FunctionalEvent(name="hits")
        hits_b = FunctionalEvent(name="hits")
        collapse_a = FunctionalEvent(name="collapse")
        collapse_b = FunctionalEvent(name="collapse")
        collision_a = FunctionalEvent(name="collision")
        collision_b = FunctionalEvent(name="collision")

        path_1 = Path(name="path_1", state="init", probability=1.0, f_event_name="init_event")  #"Init_contained",    
        path_2 = Path(name="path_2", state="yes" , probability=0.0, f_event_name="contained")  #contained_yes
        path_3 = Path(name="path_3", state="no"  , probability=1.0, f_event_name="contained")  #contained_no
        path_4 = Path(name="path_4", state="yes" , probability=0.29, f_event_name="clear")  #clear_yes
        path_5 = Path(name="path_5", state="no"  , probability=0.71, f_event_name="clear")  #clear_no
        path_6 = Path(name="path_6", state="cess", probability=0.125, f_event_name="ca")  #ca_cess
        path_7 = Path(name="path_7", state="adj" , probability=0.875, f_event_name="ca")  #ca_adj
        path_8 = Path(name="path_8", state="no"  , probability=0.95, f_event_name="falls")   #falls_a_no
        path_9 = Path(name="path_9", state="yes" , probability=0.05, f_event_name="falls")   #falls_a_yes
        path_10 = Path(name="path_10", state="no"  , probability=0.95, f_event_name="falls")   #falls_b_no
        path_11 = Path(name="path_11", state="yes" , probability=0.05, f_event_name="falls")   #falls_b_yes
        path_12 = Path(name="path_12", state="no"  , probability=0.80, f_event_name="hits")   #hits_a_no
        path_13 = Path(name="path_13", state="yes" , probability=0.20, f_event_name="hits")   #hits_a_yes
        path_14 = Path(name="path_14", state="no"  , probability=0.75, f_event_name="hits")   #hits_b_no
        path_15 = Path(name="path_15", state="yes" , probability=0.25, f_event_name="hits")   #hits_b_yes
        path_16 = Path(name="path_16", state="no"  , probability=0.95, f_event_name="collapse")   #collapse_a_no
        path_17 = Path(name="path_17", state="yes" , probability=0.05, f_event_name="collapse")   #collapse_a_yes
        path_18 = Path(name="path_18", state="no"  , probability=0.95, f_event_name="collapse")   #collapse_b_no
        path_19 = Path(name="path_19", state="yes" , probability=0.05, f_event_name="collapse")   #collapse_b_yes
        path_20 = Path(name="path_20", state="no"  , probability=0.90, f_event_name="collision")   #collision_a_no
        path_21 = Path(name="path_21", state="yes" , probability=0.10, f_event_name="collision")   #collision_a_yes
        path_22 = Path(name="path_22", state="no"  , probability=0.90, f_event_name="collision")   #collision_b_no
        path_23 = Path(name="path_23", state="yes" , probability=0.10, f_event_name="collision")   #collision_b_yes

        node_connections = ( ("init_event", "path_1"), ("path_1", "contained"), ("contained", "path_2"), ("path_2", "d1"),
                             ("init_event", "path_1"), ("path_1", "contained"), ("contained", "path_3"), ("path_3", "clear"), ("clear", "path_4"), ("path_4", "d2"),
                             ("init_event", "path_1"), ("path_1", "contained"), ("contained", "path_3"), ("path_3", "ca"), ("ca", "path_6"), ("path_6", "falls_a"), ("falls_a", "path_8"), ("path_8", "hits_a"), ("hits_a", "path_12"), ("path_12", "d3"),
                            ("init_event", "path_1"), ("path_1", "contained"), ("contained", "path_3"), ("path_3", "ca"), ("ca", "path_6"), ("path_6", "falls_a"), ("falls_a", "path_8"), ("path_8", "hits_a"), ("hits_a", "path_13"), ("path_13", "collapse_a"), ("collapse_a", "path_16"),  ("path_16", "d4"),
                            ("init_event", "path_1"), ("path_1", "contained"), ("contained", "path_3"), ("path_3", "ca"), ("ca", "path_6"), ("path_6", "falls_a"), ("falls_a", "path_8"), ("path_8", "hits_a"), ("hits_a", "path_13"), ("path_13", "collapse_a"), ("collapse_a", "path_17"),  ("path_17", "d5"),
                            ("init_event", "path_1"), ("path_1", "contained"), ("contained", "path_3"), ("path_3", "ca"), ("ca", "path_6"), ("path_6", "falls_a"), ("falls_a", "path_9"), ("path_9", "hits_b"), ("hits_b", "path_14"), ("path_14", "d6"),
                            ("init_event", "path_1"), ("path_1", "contained"), ("contained", "path_3"), ("path_3", "ca"), ("ca", "path_6"), ("path_6", "falls_a"), ("falls_a", "path_9"), ("path_9", "hits_b"), ("hits_b", "path_15"), ("path_15", "collapse_b"), ("collapse_b", "path_18"),  ("path_18", "d7"),
                            ("init_event", "path_1"), ("path_1", "contained"), ("contained", "path_3"), ("path_3", "ca"), ("ca", "path_6"), ("path_6", "falls_a"), ("falls_a", "path_9"), ("path_9", "hits_b"), ("hits_b", "path_15"), ("path_15", "collapse_b"), ("collapse_b", "path_19"),  ("path_19", "d8"),
                            ("init_event", "path_1"), ("path_1", "contained"), ("contained", "path_3"), ("path_3", "ca"), ("ca", "path_7"), ("path_7", "falls_b"), ("falls_b", "path_10"), ("path_10", "collision_a"), ("collision_a", "path_20"), ("path_20", "d9"),
                            ("init_event", "path_1"), ("path_1", "contained"), ("contained", "path_3"), ("path_3", "ca"), ("ca", "path_7"), ("path_7", "falls_b"), ("falls_b", "path_10"), ("path_10", "collision_a"), ("collision_a", "path_21"), ("path_21", "d10"),
                            ("init_event", "path_1"), ("path_1", "contained"), ("contained", "path_3"), ("path_3", "ca"), ("ca", "path_7"), ("path_7", "falls_b"), ("falls_b", "path_11"), ("path_11", "collision_b"), ("collision_b", "path_22"), ("path_22", "d11"),
                            ("init_event", "path_1"), ("path_1", "contained"), ("contained", "path_3"), ("path_3", "ca"), ("ca", "path_7"), ("path_7", "falls_b"), ("falls_b", "path_11"), ("path_11", "collision_b"), ("collision_b", "path_23"), ("path_23", "d12"),
                            )

        data = {"init_event":init_event,
                "d1":cd1, "d2":cd2, "d3":cd3, "d4":cd4, "d5":cd5, "d6":cd6, "d7":cd7, "d8":cd8, "d9":cd9, "d10":cd10, "d11":cd11, "d12":cd12, 
                "contained":contained, "clear":clear, "ca":ca, "falls_a":falls_a, "falls_b":falls_b, "hits_a":hits_a, "hits_b":hits_b, 
                "collapse_a":collapse_a, "collapse_b":collapse_b, "collision_a":collision_a, "collision_b":collision_b, 
                "path_1":path_1, "path_2":path_2, "path_3":path_3, "path_4":path_4, "path_5":path_5, "path_6":path_6, "path_7":path_7, 
                "path_8":path_8, "path_9":path_9, "path_10":path_10, "path_11":path_11, "path_12":path_12, "path_13":path_13, "path_14":path_14, 
                "path_15":path_15, "path_16":path_16, "path_17":path_17, "path_18":path_18, "path_19":path_19, "path_20":path_20, 
                "path_21":path_21, "path_22":path_22, "path_23":path_23 }

        tree = nx.DiGraph()
        tree.add_edges_from(node_connections)
        tree = nx.bfs_tree(tree, source="init_event")

        node_attributes = {node_name:{"data":node} for node_name, node in data.items()}
        nx.set_node_attributes(tree, node_attributes)

        return tree, node_connections, data

    return _make_tree


@pytest.fixture
def fixture_tree_heat_exchanger_param():
    '''Defines the example Event Treefrom Khakzad et al. 2013 for the heat exchanger accident scenario.
       See https://doi.org/10.1016/j.psep.2012.01.005, section 4 -->
    '''

    def _make_tree(tree_name):
        init_event = InitiatingEvent(name="init_event")
        c1 = Consequence(name="c1")
        c2 = Consequence(name="c2")
        c3 = Consequence(name="c3")
        c4 = Consequence(name="c4")
        c5 = Consequence(name="c5")
        c6 = Consequence(name="c6")
        c7 = Consequence(name="c7")
        c8 = Consequence(name="c8")

        ignition = FunctionalEvent(name="ignition")
        sprinkler_a = FunctionalEvent(name="sprinkler")
        sprinkler_b = FunctionalEvent(name="sprinkler")

        alarm_a = FunctionalEvent(name="alarm")
        alarm_b = FunctionalEvent(name="alarm")
        alarm_c = FunctionalEvent(name="alarm")
        alarm_d = FunctionalEvent(name="alarm")

        path_1 = Path(name="path_1", state="init", probability=1.0, f_event_name="init_event")  #"Init_iginition",    
        path_2 = Path(name="path_2", state="working" , probability=0.9, f_event_name="ignition")  #ignition_working
        path_3 = Path(name="path_3", state="failing" , probability=0.1, f_event_name="ignition")  #ignition_failing
        path_4 = Path(name="path_4", state="working" , probability=0.96, f_event_name="sprinkler")  #sprinkler_a_working
        path_5 = Path(name="path_5", state="failing" , probability=0.04, f_event_name="sprinkler")  #sprinkler_a_failing
        path_6 = Path(name="path_6", state="working" , probability=0.96, f_event_name="sprinkler")   #sprinkler_b_working
        path_7 = Path(name="path_7", state="failing" , probability=0.04, f_event_name="sprinkler")  #sprinkler_b_failing
        path_8 = Path(name="path_8", state="working" , probability=0.9987, f_event_name="alarm") #alarm_a_working
        path_9 = Path(name="path_9", state="failing" , probability=0.0013, f_event_name="alarm") #alarm_a_failing 
        path_10 = Path(name="path_10", state="working" , probability=0.9987, f_event_name="alarm") #alarm_b_working
        path_11 = Path(name="path_11", state="failing" , probability=0.0013, f_event_name="alarm") #alarm_b_failing
        path_12 = Path(name="path_12", state="working" , probability=0.775, f_event_name="alarm") #alarm_c_working
        path_13 = Path(name="path_13", state="failing" , probability=0.225, f_event_name="alarm") #alarm_c_failing
        path_14 = Path(name="path_14", state="working" , probability=0.775, f_event_name="alarm") #alarm_d_working
        path_15 = Path(name="path_15", state="failing" , probability=0.225, f_event_name="alarm") #alarm_d_failing



        node_connections = ( ("init_event", "path_1"), ("path_1", "ignition"), ("ignition", "path_2"), ("path_2", "sprinkler_a"), ("sprinkler_a", "path_4"), ("path_4", "alarm_a"), ("alarm_a", "path_8"), ("path_8", "c1"),
                             ("init_event", "path_1"), ("path_1", "ignition"), ("ignition", "path_2"), ("path_2", "sprinkler_a"), ("sprinkler_a", "path_4"), ("path_4", "alarm_a"), ("alarm_a", "path_9"), ("path_9", "c2"),
                             ("init_event", "path_1"), ("path_1", "ignition"), ("ignition", "path_2"), ("path_2", "sprinkler_a"), ("sprinkler_a", "path_5"), ("path_5", "alarm_b"), ("alarm_b", "path_10"), ("path_10", "c3"),
                             ("init_event", "path_1"), ("path_1", "ignition"), ("ignition", "path_2"), ("path_2", "sprinkler_a"), ("sprinkler_a", "path_5"), ("path_5", "alarm_b"), ("alarm_b", "path_11"), ("path_11", "c4"),

                             ("init_event", "path_1"), ("path_1", "ignition"), ("ignition", "path_3"), ("path_3", "sprinkler_b"), ("sprinkler_b", "path_6"), ("path_6", "alarm_c"), ("alarm_c", "path_12"), ("path_12", "c5"),
                             ("init_event", "path_1"), ("path_1", "ignition"), ("ignition", "path_3"), ("path_3", "sprinkler_b"), ("sprinkler_b", "path_6"), ("path_6", "alarm_c"), ("alarm_c", "path_13"), ("path_13", "c6"),
                             ("init_event", "path_1"), ("path_1", "ignition"), ("ignition", "path_3"), ("path_3", "sprinkler_b"), ("sprinkler_b", "path_7"), ("path_7", "alarm_d"), ("alarm_d", "path_14"), ("path_14", "c7"),
                             ("init_event", "path_1"), ("path_1", "ignition"), ("ignition", "path_3"), ("path_3", "sprinkler_b"), ("sprinkler_b", "path_7"), ("path_7", "alarm_d"), ("alarm_d", "path_15"), ("path_15", "c8"), 
                             )

        data = {"init_event":init_event, "c1":c1, "c2":c2, "c3":c3, "c4":c4, "c5":c5, "c6":c6, "c7":c7, "c8":c8,
                "ignition":ignition, "sprinkler_a":sprinkler_a, "sprinkler_b":sprinkler_b, "alarm_a":alarm_a, "alarm_b":alarm_b, "alarm_c":alarm_c, "alarm_d":alarm_d ,
                "path_1":path_1, "path_2":path_2, "path_3":path_3, "path_4":path_4, "path_5":path_5, "path_6":path_6, "path_7":path_7, "path_8":path_8, "path_9":path_9, 
                "path_10":path_10, "path_11":path_11, "path_12":path_12, "path_13":path_13, "path_14":path_14, "path_15":path_15 
                }

        tree = nx.DiGraph()
        tree.add_edges_from(node_connections)
        tree = nx.bfs_tree(tree, source="init_event")

        node_attributes = {node_name:{"data":node} for node_name, node in data.items()}
        nx.set_node_attributes(tree, node_attributes)

        return tree, node_connections, data

    return _make_tree