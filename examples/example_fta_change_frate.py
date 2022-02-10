"""Example showing the manipulation of time behaviour via the "BayesianFaultTree" class.
    This includes:
        - providing a parameterized time behaviour function (as replacement for R(t) = exp(-lambda t))
        - resetting the time behaviour to the initial one (const. or exponential)
"""
import os
import sys
import numpy as np
cur_dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(cur_dir_path, os.pardir)))

from example_networks_provider import make_fault_tree


def SigmoidLike(time):
    """Dummy time function showing a sigmoid like behaviour.

    Args:
        time (float): Time stamp at which the function should be evaluated.

    Returns:
        float: Probability of failure at time t.
    """
    factor = 3e-4  # similiar to the classic frate lambda

    if time == 0:
        return 0

    return 1 - 1 / (np.exp(factor * time) + 1 )


def Trigonometric(time, kind="sin"):
    """Dummy time function showing a trigonometric behaviour.

    Args:
        time (float): Time stamp at which the function should be evaluated.
        kind (str, optional): Kind of trigonometric behaviour.

    Returns:
        float: Probability of failure at time t.
    """
    val = 0
    if time == 0:
        return val

    if kind == "sin":
        val = np.sin(time)

    elif kind == "cos":
        val = np.cos(time)

    else:
        val = np.tan(time)

    return np.abs(val)


if __name__ == '__main__':
    BayFTA = make_fault_tree(tree_name="Functional_frate_example")

    model_elem = BayFTA.get_elem_by_name("IHF_SIG_HI_FP")
    model_elem.change_time_behaviour(fn_behaviour=SigmoidLike, params={})

    model_elem = BayFTA.get_elem_by_name("IHF_BS_FP")
    myParams = {"kind":"cos"}
    model_elem.change_time_behaviour(fn_behaviour=Trigonometric, params=myParams)

    BayFTA.run_time_simulation(start_time=0, stop_time=10**5, simulation_steps=50, node_name="IHF_SIG_HI_FP", plot_simulation=True)
    BayFTA.run_time_simulation(start_time=0, stop_time=10**4, simulation_steps=50, node_name="IHF_BS_FP", plot_simulation=True)

    print("\n>>>>>>>> Reset")
    BayFTA.get_elem_by_name("IHF_SIG_HI_FP").reset_time_behaviour()
    BayFTA.get_elem_by_name("IHF_BS_FP").reset_time_behaviour()

    BayFTA.run_time_simulation(start_time=0, stop_time=10**4, simulation_steps=50, node_name="IHF_SIG_HI_FP", plot_simulation=True)
    BayFTA.run_time_simulation(start_time=0, stop_time=10**4, simulation_steps=50, node_name="IHF_BS_FP", plot_simulation=True)
