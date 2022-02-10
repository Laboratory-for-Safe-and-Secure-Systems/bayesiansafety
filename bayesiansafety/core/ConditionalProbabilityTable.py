"""
This class is a wrapper for the internal representation of Tabular CPTs to
seperate the actual implementaiton from the usage througout the package.

"""
import copy
from typing import Dict, List, Optional, Union

import numpy as np
from numpy import ndarray


class ConditionalProbabilityTable:

    """This class is a wrapper for the internal representation of Tabular CPTs to
        seperate the actual implementaiton from the usage througout package.
        Implementation is oriented to comply to pgmpy.factors.discrete.CPD.TabularCPD

    Attributes:
        evidence (array-like): List of variables in evidences(if any) w.r.t. which CPD is defined.
        evidence_card (array-like): cardinality/no. of states of variables in `evidence`(if any)
        name (str): Name of this variable
        state_names (dict<str, list(str)): dictionary with key as variable and key as list of sate names (e.g. {'A': ['a1', 'a2'], 'B': ['b1', 'b2']}
        values (2D array, 2D list or 2D tuple): Values for the CPD table. Please refer the example for the exact format needed.
        variable_card (integer): Cardinality/no. of states of `variable`
    """

    name = None
    variable_card = None
    values = None
    evidence = None
    evidence_card = None
    state_names = None

    __index_mapping = None
    __scope = None
    __cardinalities = None
    __str_repr = None

    def __init__(self, name: str, variable_card: int, values: Union[List[ndarray], ndarray, List[List[Union[float, int]]]], evidence: Optional[List[str]] = None, evidence_card: Optional[Union[ndarray, List[int]]] = None, state_names: Optional[Dict[str, List[str]]] = None) -> None:
        """Ctor of the ConditionalProbabilityTable class.

        Args:
            name (str): Name of this variable
            variable_card (integer): Cardinality/no. of states of `variable`
            values (2D array, 2D list or 2D tuple): Values for the CPD table. Please refer the example for the exact format needed.
            evidence (array-like): List of variables in evidences(if any) w.r.t. which CPD is defined.
            evidence_card (array-like): cardinality/no. of states of variables in `evidence`(if any)
            state_names (dict<str, list(str)): dictionary with key as variable and key as list of sate names (e.g. {'A': ['a1', 'a2'], 'B': ['b1', 'b2']}

        """
        self.name = name
        self.variable_card = variable_card
        self.values = values
        self.evidence = evidence
        self.evidence_card = evidence_card
        self.state_names = state_names

        self.__scope = [self.name] + \
            self.evidence if self.evidence is not None else [self.name]
        self.__cardinalities = [*[self.variable_card], *self.evidence_card] if self.evidence_card is not None else [self.variable_card, -1]
        self.__build_index_mapping()

    def copy(self) -> "ConditionalProbabilityTable":
        """Get a deep copy of this instance.

        Returns:
            ConditionalProbabilityTable: Deep copy of this instance.
        """
        return copy.deepcopy(self)

    def get_probabilities(self) -> ndarray:
        """Get the actual probability table as array.

        Returns:
            array-like: Actual probability table as 2-D array.
        """
        return np.array(self.values)

    def __build_index_mapping(self) -> None:
        """Helper to build a mapping between the explicit state names/variables and the values of the underlying CPT.
        """
        name_to_no = {}
        if self.state_names:
            for key, values in self.state_names.items():
                name_to_no[key] = {name: no for no,
                                   name in enumerate(self.state_names[key])}

        else:
            name_to_no = {var: {i: i for i in range(
                int(self.__cardinalities[index]))} for index, var in enumerate(self.__scope)}

        self.__index_mapping = name_to_no

    def get_value(self, node_states: Dict[str, Union[str, int]]) -> float:
        """Get the conditional probability for one specific combination of contributors (parents/states).

        Args:
            node_states (dict<str, str>): Dictionary describing the conditionals where key = parent name, value = parent state name.

        Returns:
            float: Conditional probability for given combination of parents.

        Raises:
            ValueError: Raised if given combination doesn't match with the CPT.
        """
        if len(set(node_states.keys()) ^ set(self.state_names.keys())) != 0:
            raise ValueError(f"Your given variables: {node_states.keys()} do not match with the CPTs named vars {self.state_names.keys()}")

        if not all(state in self.state_names[var] for var, state in node_states.items()):
            raise ValueError(f"Invalid states {node_states.values()} are not part of the listed state names: {self.state_names.values()}.")

        index = []
        for var in self.__scope:
            try:
                index.append(self.__index_mapping[var][node_states[var]])
            except KeyError:
                index.append(node_states[var])

        corr_shaped_values = np.reshape(
            np.ravel(np.array(self.values)), (self.__cardinalities))
        return float(corr_shaped_values[tuple(index)])

    def get_index_of_state(self, state_name: str) -> int:
        """Get the index of a state for this CPT. This means the index compared to the cardinality of this node.
            E.g.: CPT for "Season" has a cardinality of 4: {"Winter", "Spring", "Summer", "Autumn"} with Summer having index 2.
            The index is relevant for interpreting probability arrays.

        Args:
            state_name (str): State name for which the index compared to the cardinality should be queried.

        Returns:
            int: Index of the state name.

        Raises:
            TypeError: Raised if given state name is not a string.
            ValueError: Raised if either states are not named or if given state name is not specified for this CPT.
        """
        if not self.state_names:
            raise ValueError("Conditional probability table does not contain state names.")

        if not isinstance(state_name, str):
            raise TypeError(f"Given state name: {state_name} is not a string: {type(state_name)}.")

        if state_name not in self.state_names[self.name]:
            raise ValueError(f"Given state name: {state_name} is not specified for this variable ({self.name})")

        idx = [i for i in range(self.variable_card)
               if self.state_names[self.name][i] == state_name][0]

        return idx

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
