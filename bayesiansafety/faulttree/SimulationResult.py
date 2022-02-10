"""
This class acts as a storage for time-related query results of a Fault Tree (helper class).
"""
from typing import Any
from dataclasses import dataclass

@dataclass(eq=True)
class SimulationResult():
    """Convenience class to store results of a single timestep for a single node
        of a time analyis.

    Attributes:
        cpt (ConditionalProbabilityTable): CPT containing the current [prob. of no failure,
                                           prob. of failure] for this model element
        node_name (str): Name of the model element.
        simulation_time (float): Time step at which the model element was evaluated
    """
    node_name: str = ""
    simulation_time: float = 0.0
    cpt: Any = None
