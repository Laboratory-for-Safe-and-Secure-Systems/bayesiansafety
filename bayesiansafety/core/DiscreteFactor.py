"""
This class is a wrapper for the internal representation of discrete factors to
seperate the actual implementaiton from the usage througout the package.

"""
import copy
from typing import Dict, List, Optional, Union

import numpy as np
from numpy import ndarray


class DiscreteFactor:

    """This class is a wrapper for the internal representation of discrete factors to
    seperate the actual implementaiton from the usage througout the package.
    Implementation is oriented to comply to pgmpy.factors.discrete.DiscreteFactor.

    Attributes:
        cardinalities (list, array_like): List of cardinalities/no.of states of each variable. `cardinality` array must have a value corresponding to each variable in `variables`.
        name (str): Name of this factor.
        scope (list, array_like): List of variables on which the factor is to be defined i.e. scope of the factor.
        state_names (dict<str, list(str)): Dictionary with key as variable and key as list of sate names (e.g. {'A': ['a1', 'a2'], 'B': ['b1', 'b2']}.
        values (list<float>): List of values of factor. A DiscreteFactor's values are stored in a row vector in the value using an ordering such that the left-most variables
                              as defined in `variables` cycle through their values the fastest
    """

    name = None
    scope = None  # pgmpy: variables
    cardinalities = None  # pgmpy: cardinality:
    values = None
    state_names = None

    __index_mapping = None
    __str_repr = None

    def __init__(self, name: str, scope: List[str], cardinalities: Union[ndarray, List[int]], values: Union[ndarray, List[Union[float, int]]], state_names: Optional[Dict[str, List[str]]] = None) -> None:
        """Ctor of the DiscreteFactor class.

        Args:
            name (str): Name of this factor.
            scope (list, array_like): List of variables on which the factor is to be defined i.e. scope of the factor.
            cardinalities (list, array_like): List of cardinalities/no.of states of each variable. `cardinality` array must have a value corresponding to each variable in `variables`.
            values (list<float>): List of values of factor. A DiscreteFactor's values are stored in a row vector in the value using an ordering such that the left-most variables
                                  as defined in `variables` cycle through their values the fastest
            state_names (dict<str, list(str)): Dictionary with key as variable and key as list of sate names (e.g. {'A': ['a1', 'a2'], 'B': ['b1', 'b2']}.
        """
        self.name = name
        self.scope = scope
        self.cardinalities = cardinalities
        self.values = np.array(values).reshape(cardinalities)
        self.state_names = state_names

        self.__build_index_mapping()

    def copy(self) -> "DiscreteFactor":
        """Get a deep copy of this instance.

        Returns:
            DiscreteFactor: Deep copy of this instance.
        """
        return copy.deepcopy(self)

    def __build_index_mapping(self) -> None:
        """Helper to build a mapping between the explicit state names/variables and the values of the underlying factor.
        """
        name_to_no = {}
        if self.state_names:
            for key, values in self.state_names.items():
                name_to_no[key] = {name: no for no,
                                   name in enumerate(self.state_names[key])}

        else:
            name_to_no = {var: {i: i for i in range(
                int(self.cardinalities[index]))} for index, var in enumerate(self.scope)}

        self.__index_mapping = name_to_no

    def get_probabilities(self) -> ndarray:
        """Get the actual value table as array.
            Note: The order of the values is not fixed across instantiation for the same given parameters.
                  This means that the variable positions and therefore the order of variable combinations of the factor is random at object creation.
            Note: This method is named like it's counterpart in the core.ConditionalProbabilityTable class for consistency although a factor does not
                  directly store probabilities (0...1 values).

        Returns:
            array-like: Actual value table as 2-D array.
        """
        return np.array(self.values)

    def get_value(self, node_states: Dict[str, str]) -> float:
        """Get the value for one specific combination of contributors (factor variables/states).

        Args:
            node_states (dict<str, str>): Dictionary describing the contributorwhere key = factor variable name, value = factor variable state name.

        Returns:
            float: Factor value for the given combination of contributors.

        Raises:
            ValueError: Raised when variables or states do not comply with the factor.
        """
        # node_states <dict<str, str>>, key = name_of_contributor, val=state_of_contributor
        if len(set(node_states.keys()) ^ set(self.state_names.keys())) != 0:
            raise ValueError(f"Your given variables: {node_states.keys()} do not match with the factors named vars {self.state_names.keys()}")

        if not all(state in self.state_names[var] for var, state in node_states.items()):
            raise ValueError(f"Invalid states {node_states.values()} are not part of the listed state names: {self.state_names.values()}.")

        index = []
        for var in self.scope:
            try:
                index.append(self.__index_mapping[var][node_states[var]])
            except KeyError:
                index.append(node_states[var])

        #corr_shaped_values = np.reshape(np.ravel(np.array(self.values)), (self.cardinalities) )
        return float(self.values[tuple(index)])

    def __str__(self) -> str:
        """Customized version of the objects converstion to string.
            Allows usage as print( self ).

        Raises:
            NotImplementedError: Currently not implemented
        """
        #raise NotImplementedError
        if self.__str_repr:
            return self.__str_repr

        return str(self.values)
