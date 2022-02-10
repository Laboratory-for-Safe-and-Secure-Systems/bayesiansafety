"""Helper class to generate the dummy networks for the examples.
"""
from bayesiansafety.core.BayesianNetwork import BayesianNetwork
from bayesiansafety.core.ConditionalProbabilityTable import ConditionalProbabilityTable
from bayesiansafety.faulttree.FaultTreeProbNode import FaultTreeProbNode
from bayesiansafety.faulttree.FaultTreeLogicNode import FaultTreeLogicNode
from bayesiansafety.faulttree.BayesianFaultTree import BayesianFaultTree


def make_fault_tree(tree_name):
    """Helper function to generate the elevator example from Fehlerbaumanalyse in Theorie und Praxis,  F. Edler & M.Soden and R.Hankammer (2015) p.126 / 127 (ISBN: 978-3-662-48165-3)

    Args:
        tree_name (str): Name of this Fault Tree.

    Returns:
        BayesianFaultTree: Instance of this Faul Tree
    """
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

    return BayesianFaultTree(tree_name, probability_nodes, logic_nodes)


def make_bn(bn_name, bn_type ="collider"):
    """Helper function to create a simple collider or confounder BN.

    Args:
        bn_name (str): Name of the BN
        bn_type (str, optional): Weather the node configuration should be a collider or confounder

    Returns:
        BayesianNetwork: Instance of a BN.
    """
    if bn_type == "confounder":
        node_connections = [("CPT_A", "CPT_B"), ("CPT_A", "CPT_C")]
        bn = BayesianNetwork(bn_name, node_connections)

        CPT_A = ConditionalProbabilityTable(name="CPT_A", variable_card=2, values=[[0.1], [0.9]], evidence=None, evidence_card=None, state_names={"CPT_A": ["A_Yes", "A_No"]})
        CPT_B = ConditionalProbabilityTable(name="CPT_B", variable_card=2, values=[[0.12, 0.34], [0.88, 0.66]], evidence=["CPT_A"], evidence_card=[2], state_names={"CPT_A": ["A_Yes", "A_No"], "CPT_B": ["B_Yes", "B_No"]})
        CPT_C = ConditionalProbabilityTable(name="CPT_C", variable_card=2, values=[[0.98, 0.76], [0.02, 0.24]], evidence=["CPT_A"], evidence_card=[2], state_names={"CPT_A": ["A_Yes", "A_No"], "CPT_C": ["C_Yes", "C_No"]})

        bn.add_cpts(CPT_A, CPT_B, CPT_C)
        return bn

    if bn_type == "collider":
        node_connections = [("CPT_A", "CPT_C"), ("CPT_B", "CPT_C")]
        bn = BayesianNetwork(bn_name, node_connections)

        CPT_A = ConditionalProbabilityTable(name="CPT_A", variable_card=2, values=[[0.123], [0.877]], evidence=None, evidence_card=None, state_names={"CPT_A": ["A_Yes", "A_No"]})
        CPT_B = ConditionalProbabilityTable(name="CPT_B", variable_card=2, values=[[0.987], [0.013]], evidence=None, evidence_card=None, state_names={"CPT_B": ["B_Yes", "B_No"]})
        CPT_C = ConditionalProbabilityTable(name="CPT_C", variable_card=2, values=[[0.123, 0.456, 0.789, 0.987], [0.877, 0.544, 0.211, 0.013]], evidence=["CPT_A", "CPT_B"], evidence_card=[2, 2],
                                            state_names={"CPT_A": ["A_Yes", "A_No"],
                                                         "CPT_B": ["B_Yes", "B_No"],
                                                         "CPT_C": ["C_Yes", "C_No"]})
        CPT_TEST = ConditionalProbabilityTable(name="CPT_TEST", variable_card=3, values=[[0.20], [0.30], [0.50]], evidence=None, evidence_card=None, state_names={"CPT_TEST": ["TEST_1", "TEST_2", "TEST_3"]})

        bn.add_node("CPT_TEST")
        bn.add_cpts(CPT_A, CPT_B, CPT_C, CPT_TEST)
        return bn
