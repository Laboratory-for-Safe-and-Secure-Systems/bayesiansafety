"""Example showing environmental effects on the reliability functions of a basis event.
    There can be differnt "behaviours" on how this environmental influence manifests, namely:
        REPLACEMENT # R(t)   -> P(env)
        ADDITION    # R(t)   -> w_0 *R(t) + w_1*P(env)
        OVERLAY     # R(t)   -> w_0* R(t) * /prod w_n*P_n
        RATE        # R(t,l) -> R(t,l*)     # l = /lambda
        FUNCTIONAL  # R(t,l) -> R*(t, X)    # X = set of parameters
        PARAMETER   # R(t)   -> R*(t, P(env))  # special case of RATE or the more generic FUNCTIONAL
"""
import os
import sys
cur_dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(cur_dir_path, os.pardir)))
from example_networks_provider import make_fault_tree, make_bn

from bayesiansafety.faulttree import Evaluation
from bayesiansafety.synthesis.functional import FunctionalConfiguration, Behaviour, FunctionalManager


if __name__ == '__main__':
    ft_elevator = make_fault_tree(tree_name="Elevator_1")

    baynets = { "BN_A":make_bn(bn_name="BN_A", bn_type="collider"),
                "BN_B":make_bn(bn_name="BN_B", bn_type="confounder")}


    env_factors_a =  { "BN_A": [('CPT_B', "B_Yes")]}

    env_factors_b =  { "BN_A": [('CPT_A', "A_Yes")],
                       "BN_B": [('CPT_B', "B_No"), ('CPT_C', "C_Yes")]}

    # are currently ignored
    threshs_a = { "BN_A": [('CPT_B', 0.0)]}

    threshs_b = { "BN_A": [('CPT_A', 0.0)],
                  "BN_B": [('CPT_B', 0.05), ('CPT_C', 0.12)]}


    configs = {"ID_1":[ FunctionalConfiguration(ft_elevator.get_elem_by_name("IHF_BS_FP"), environmental_factors=env_factors_a, thresholds=threshs_a, weights=None, time_func=None, func_params=None, behaviour=Behaviour.ADDITION),
                        FunctionalConfiguration(ft_elevator.get_elem_by_name("EMI_SIG_HI_FP"), environmental_factors=env_factors_b, thresholds=threshs_b, weights=None, time_func=None, func_params=None, behaviour=Behaviour.OVERLAY)]}


    manager = FunctionalManager(ft_inst=ft_elevator,  bayesian_nets=baynets, configurations=configs)
    functional_ft = manager.build_functional_fault_tree(config_id="ID_1", bn_observations = None)

    ft_elevator = Evaluation(functional_ft)

    ft_elevator.evaluate_fault_tree(start_time = 0, stop_time= 10**4, simulation_steps=50, include_risk_worths=True)
    results = functional_ft.run_time_simulation(start_time = 0, stop_time= 10**3.7, simulation_steps=50)
    functional_ft.plot_bayesian_fault_tree(at_time=316.227766)
