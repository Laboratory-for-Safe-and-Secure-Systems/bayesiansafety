"""Class for representing logic nodes (boolean logic gates) where the inputs can have multiple states.
Combination of "bad states" dictates logical behaviour. E.g.: If one input state is a "bad state"
and logic is OR set the current CPT entry to 1. If the logic is AND,  CPT entry is set to 1
if all input states are currently in combination "bad states".
Note: The cardinality of the logic node is fixed to 2.
"""
from typing import Dict, List, Optional

import numpy as np
from numpy import ndarray
from bayesiansafety.core import ConditionalProbabilityTable


class MultistateLogicNode:
    """Class for representing logic nodes (boolean logic gates) where the inputs can have multiple states.
        Combination of "bad states" dictates logical behaviour. E.g.: If one input state is a "bad state"
        and logic is OR set the current CPT entry to 1. If the logic is AND,  CPT entry is set to 1
        if all input states are currently in combination "bad states".
        Note: The cardinality of the logic node is fixed to 2.


    Attributes:
        bad_states (dict<str, int>): Dictionary with key=input node name, value = state index (0...card-1) of the respective "pbf" state
        cardinalities (list<int>): List of cardinalities for each input node
        cpt (ConditionalProbabilityTable): Associated CPT for this node.
        input_nodes (list<str>): Associated inputs for this logic gate.
        name (str): Name of the node.

    """

    name = ""
    input_nodes = None
    cardinalities = None
    bad_states = None
    input_state_names = None
    cpt = None

    __default_card_inputs = 2
    __default_pbf_state = 1
    # logic_type (str): Boolean logic type for this gate (AND, OR (default))
    __node_type = None
    __default_state_names = ["working", "failing"]

    def __init__(self, name: str, input_nodes: List[str], input_state_names: Dict[str, List[str]], cardinalities: Optional[List[int]] = None, bad_states: Optional[Dict[str, int]] = None, logic_type: Optional[str] = 'OR') -> None:
        """Ctor for this class.

        Args:
            name (str): Name of the node.
            input_nodes (list<str>): Associated inputs for this logic gate.
            input_state_names (dict<str, list<str>>): Dictionary with key = node name, values = list of state names.
            cardinalities (list<int>, optional): List of cardinalities for each input node. If none are given inputs are considered as binary.
            bad_states (dict<str, int>, optional): Dictionary with key=input node name, value = state index (0...card-1) of the respective "pbf" state.
                                                 If None, idx 1 is considered as bad state.
            logic_type (str, optional): Boolean logic type for this gate (AND, OR (default))
        """
        self.name = name
        self.__node_type = logic_type
        self.input_nodes = input_nodes
        self.input_state_names = input_state_names

        cardinalities = np.repeat([self.__default_card_inputs], len(
            input_nodes)) if cardinalities is None else cardinalities
        bad_states = dict([(node_name, self.__default_pbf_state)
                           for node_name in input_nodes]) if bad_states is None else bad_states

        self.cardinalities = cardinalities
        self.bad_states = bad_states

        self.__create_cpt()

    def __create_cpt(self) -> None:
        """Helper method to instantiate a core.ConditionalProbabilityTable based on the passed node_type (OR, AND) and set the member self.cpt (inted for internal use).

        Raises:
            ValueError: Raised if less than two inputs or an unsupported logic type is provided.
        """

        if len(self.input_nodes) < 2:
            raise ValueError(f"Invalid input nodes: {self.input_nodes}. You need to specify at least two nodes.")

        if not isinstance(self.__node_type, str) or self.__node_type not in ["OR", "AND"]:
            raise ValueError(f"Unsupported logic type: {self.__node_type} of type: {type(self.__node_type)}. Valid types are 'OR' and 'AND'.")

        pbf_entries = self.__get_pbf_arr(nodes_order=self.input_nodes, cards=self.cardinalities,
                                         bad_states=self.bad_states, logic_type=self.__node_type)

        nf_entries = np.invert(pbf_entries).astype(int)
        pbf_entries = pbf_entries.astype(int)

        state_names = self.input_state_names
        state_names[self.name] = self.__default_state_names

        self.cpt = ConditionalProbabilityTable(name=self.name, variable_card=2,
                                               values=[nf_entries,         # NF
                                                       pbf_entries],       # PBF
                                               evidence=self.input_nodes,
                                               evidence_card=self.cardinalities,
                                               state_names=state_names)

    def __get_pbf_arr(self, nodes_order: List[str], cards: List[int], bad_states: Dict[str, int], logic_type: Optional[str] = "OR") -> ndarray:
        """Helper method to create the "pbf" part of the CPT.

        Args:
            nodes_order (list<str>): Associated inputs for this logic gate.
            cards (list<int>): List of cardinalities for each input node
            bad_states (dict<str, int>): Dictionary with key=input node name, value = state index (0...card-1) of the respective "pbf" state
            logic_type (str, optional): Boolean logic type for this gate (AND, OR (default))

        Returns:
            TYPE: Description
        """
        # OR: Variable bad state gets set if one input state is "bad"
        #       A_good          A_bad
        #  B_good  B_bad    B_good  B_bad
        # -------------------------------------
        #    1       0        0       0  | good
        #    0       1        1       1  | bad
        #######################################
        # AND: Variable bad state gets set if all input states are "bad"
        #       A_good          A_bad
        #  B_good  B_bad    B_good  B_bad
        # -------------------------------------
        #    1       1        1       0  | good
        #    0       0        0       1  | bad

        nr_of_entries = np.prod(cards)
        arrs = []
        for idx, node in enumerate(nodes_order):
            state_reps_node = np.zeros(cards[idx], dtype=bool)
            state_reps_node[bad_states[node]] = True
            state_reps_node = np.tile(np.repeat(state_reps_node, np.prod(
                cards[idx+1:])), int(nr_of_entries/(np.prod(cards[idx:]))))
            arrs.append(state_reps_node)

        pbf_arr = None
        if logic_type == "OR":
            pbf_arr = np.logical_or.reduce(tuple(arrs))

        elif logic_type == "AND":
            pbf_arr = np.logical_and.reduce(tuple(arrs))

        return pbf_arr

    def get_node_type(self) -> str:
        """Getter for type of node.

        Returns:
            str: Returns either "OR" or "AND" to indicate specify the associated boolean logic behaviour.
        """
        return self.__node_type.upper()
