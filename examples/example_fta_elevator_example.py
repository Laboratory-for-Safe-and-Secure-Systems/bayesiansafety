"""Example showing the core capabilitites of the "BayesianFaultTree" class.
    Which are:
        - Easy instantiation of a Fault Tree (see ./example_networks_provider.py)
        - Time analyis of a Faul Tree including plotting
        - Single time stamp analyis of a Faul Tree including plotting
"""
import os
import sys
cur_dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(cur_dir_path, os.pardir)))

from example_networks_provider import make_fault_tree
from bayesiansafety.faulttree import Evaluation

if __name__ == '__main__':
    BayFTA = make_fault_tree(tree_name="Elevator_example")
    ft_evaluator = Evaluation(BayFTA)
    ft_evaluator.evaluate_fault_tree(start_time = 0, stop_time= 10**4, simulation_steps=50, plot_dir=r"Plots", include_risk_worths=True)

    results = BayFTA.run_time_simulation(start_time = 0, stop_time= 10**3.7, simulation_steps=50)
    BayFTA.plot_bayesian_fault_tree(at_time=316.227766)
