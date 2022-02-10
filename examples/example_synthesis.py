"""Example showing the capabilities of the  "Synthesis" class (experimental).
    That is:
        - Management of multiple FTs and BNs
        - Linking multiple BNs (and multiple nodes of each) to FTs
        - Full time evaluation of networks (Faul Tree + associated BNs)
"""
import os
import sys
cur_dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(cur_dir_path, os.pardir)))

from example_networks_provider import make_fault_tree, make_bn

from bayesiansafety.synthesis import Synthesis
from bayesiansafety.synthesis.functional import FunctionalConfiguration, Behaviour
from bayesiansafety.synthesis.hybrid import HybridConfiguration

if __name__ == '__main__':

    print("-- The following is currently in an experimental state --")

    #########################################################################
    ################   Create the basic stuff - our models ##################
    ft_elevator = make_fault_tree(tree_name="Elevator_1")
    trees = {"Elevator_1":ft_elevator}
    baynets = { "BN_A":make_bn(bn_name="BN_A", bn_type="collider"),
                "BN_B":make_bn(bn_name="BN_B", bn_type="confounder")}

    bay_safety = Synthesis(fault_trees=trees, bayesian_nets=baynets)


    #########################################################################
    ################   Create functional configurations #####################
    env_factors_a =  { "BN_A": [('CPT_B', "B_Yes")]}

    env_factors_b =  { "BN_A": [('CPT_B', "B_No")],
                       "BN_B": [('CPT_B', "B_No"), ('CPT_C', "C_Yes")]}

    ## thresholds are currently ignored
    threshs_a = { "BN_A": [('CPT_B', 0.0)]}
    threshs_b = { "BN_A": [('CPT_B', 0.0)],
                  "BN_B": [('CPT_B', 0.05), ('CPT_C', 0.12)]}

    functional_configs =    {   "FUNC_ID_1":[ FunctionalConfiguration(ft_elevator.get_elem_by_name("IHF_BS_FP"), environmental_factors=env_factors_a, thresholds=threshs_a, weights=None, time_func=None, func_params=None, behaviour=Behaviour.ADDITION),
                                              FunctionalConfiguration(ft_elevator.get_elem_by_name("EMI_SIG_HI_FP"), environmental_factors=env_factors_b, thresholds=threshs_b, weights=None, time_func=None, func_params=None, behaviour=Behaviour.OVERLAY)],

                                "FUNC_ID_2":[ FunctionalConfiguration(ft_elevator.get_elem_by_name("IHF_SIG_BS_FP"), environmental_factors=env_factors_a, thresholds=threshs_a, weights=None, time_func=None, func_params=None, behaviour=Behaviour.REPLACEMENT)]
                            }

    #########################################################################
    ################   Create hybrid configurations #########################
    shared_nodes = {"BN_A":["CPT_B"],
                    "BN_B":["CPT_B", "CPT_C"]}

    coupling_nodes_1 = {"BN_A":[( "CPT_B", 'AND_TOP')] ,
                      "BN_B":[( "CPT_B", 'OR_SIG_HI'), ( "CPT_C", 'OR_SIG_LO')] }


    coupling_nodes_2 = {"BN_A":[( "CPT_B", 'AND_TOP')] ,
                        "BN_B":[( "CPT_B", 'OR_LOG'), ( "CPT_C", 'AND_TOP')] }

    pbf_states =  { "BN_A": [('CPT_B', "B_Yes")],
                    "BN_B": [('CPT_B', "B_No"), ('CPT_C', "C_Yes")]}

    hybrid_configs =    {   "HYBRID_ID_1":HybridConfiguration(name="HYBRID_ID_1", shared_nodes=shared_nodes, ft_coupling_points=coupling_nodes_1, pbf_states=pbf_states),
                            "HYBRID_ID_2":HybridConfiguration(name="HYBRID_ID_2", shared_nodes=shared_nodes, ft_coupling_points=coupling_nodes_2, pbf_states=pbf_states)
                        }

    #########################################################################
    ###################### Evaluate all the stuff ###########################
    observations = {"BN_A":[("CPT_A", "A_Yes")]}

    func_time_scales = {  "Elevator_1": (0 , 1e5, 50, "Eval_func")}
    hybrid_time_scales = {  "Elevator_1": (1e4 , 1e6, 10, "Eval_hybrid")}

    bay_safety.set_hybrid_configurations( {"Elevator_1": hybrid_configs} )
    bay_safety.set_functional_configurations( {"Elevator_1": functional_configs} )

    bay_safety.evaluate_functional_fault_trees(ft_name="Elevator_1", bn_observations=observations, ft_time_scales=func_time_scales)
    bay_safety.evaluate_hybrid_fault_trees(ft_name="Elevator_1", bn_observations=observations, ft_time_scales=hybrid_time_scales)
