"""Example showing environmental effects on a Fault Tree and how an observed event in the Fault Tree
    affects associated, contributing BNs (posterior probabilitites)
    Note: The idea here is to extend a Faul Tree with different nodes from BNs. Each coupled node has a defined state which
            is treated as a constant probability of failure inside the Faul Tree (basic event). This relates to a functional
            modification with behavior "REPLACEMENT" and is fundamentally different from functional modifications (R*(t)).
"""
import os
import sys
cur_dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(cur_dir_path, os.pardir)))
from example_networks_provider import make_fault_tree, make_bn

from bayesiansafety.core.inference import InferenceFactory
from bayesiansafety.faulttree import Evaluation
from bayesiansafety.synthesis.hybrid import HybridConfiguration, HybridManager


if __name__ == '__main__':
    ft_name = "Elevator_1"

    ft_elevator = make_fault_tree(tree_name=ft_name)
    trees = {ft_name:ft_elevator}
    baynets = { "BN_A":make_bn(bn_name="BN_A", bn_type="collider"),
                "BN_B":make_bn(bn_name="BN_B", bn_type="confounder")}

    shared_nodes = {"BN_A":["CPT_A"],
                    "BN_B":["CPT_B", "CPT_C"]}


    coupling_nodes = {"BN_A":[( "CPT_A", 'AND_TOP')] ,
                      "BN_B":[( "CPT_B", 'OR_SIG_HI'), ( "CPT_C", 'OR_SIG_LO')] }

    pbf_states =  { "BN_A": [('CPT_A', "A_Yes")],
                    "BN_B": [('CPT_B', "B_No"), ('CPT_C', "C_Yes")]}


    configs = {"ID_1":HybridConfiguration(name=ft_name, shared_nodes=shared_nodes, ft_coupling_points=coupling_nodes, pbf_states=pbf_states)}

    manager = HybridManager(ft_inst=ft_elevator, bayesian_nets=baynets, configurations=configs)


    print(">> EVALUATING ORIGINAL FAULT TREE (STATIC)")
    Evaluation(ft_elevator).evaluate_fault_tree()


    print("+"*120)
    print("\n>> EVALUATING EXTENDED FAULT TREE (STATIC)")
    ext_ft = manager.build_extended_ft(config_id="ID_1")
    Evaluation(ext_ft).evaluate_fault_tree(plot_dir="No_observations")

    print("-"*120)
    print("\n>> EVALUATING EXTENDED FAULT TREE WITH OBSERVATIONS IN ASSOCIATED BAYESIAN NETWORKS (STATIC)")

    print("#"*120)
    print("\n>> EVALUATING EXTENDED FAULT TREE (DYNAMIC - WITH ALL COUPLED BAYESIAN NETWORKS)")
    hybrid_networks = manager.build_hybrid_networks(config_id="ID_1", at_time=0, fix_other_bns=False)
    full_hybrid = hybrid_networks[list(hybrid_networks.keys())[0]]
    full_hybrid.plot_graph()
    ft_evidence_node = "AND_TOP"
    queried_CPT = "BN_A_CPT_C"


    print(f"Prior probs for {queried_CPT}:  {InferenceFactory(full_hybrid).get_engine().query(queried_CPT).values }")
    print(f"Posterior probs for {queried_CPT} with {ft_evidence_node} = faulty : {InferenceFactory(full_hybrid).get_engine().query(queried_CPT, evidence=[(ft_evidence_node, 1)]).values } \n")


    print("#"*120)
    print("\n>> EVALUATING EXTENDED FAULT TREE (DYNAMIC - WITH ALL BUT ONE FIXED BAYESIAN NETWORKS)")
    for eval_time in [0, 500, 2500, 5000]:
        print("#"*120)
        print(f"Retrospective query for time: {eval_time}")

        hybrid_networks = manager.build_hybrid_networks(config_id="ID_1", at_time=eval_time, fix_other_bns=True)

        for bn_name, bn_inst in hybrid_networks.items():
            print(f"Evaluation for associated (dynamic) BN: {bn_name}")
            linked_bns = ["BN_A", "BN_B"]
            ft_evidence_node = "AND_TOP"

            for linked_bn in linked_bns:
                if linked_bn in bn_name:
                    queried_CPT = f"{linked_bn}_CPT_C"
                    print(f"Prior probs for {queried_CPT}:  {InferenceFactory(bn_inst).get_engine().query(queried_CPT).values }")
                    print(f"Posterior probs for {queried_CPT} with {ft_evidence_node} = faulty : {InferenceFactory(bn_inst).get_engine().query(queried_CPT, evidence=[(ft_evidence_node, 1)]).values } \n")

            bn_inst.plot_graph()
