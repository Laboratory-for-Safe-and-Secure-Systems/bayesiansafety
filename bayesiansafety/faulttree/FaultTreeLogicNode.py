"""Class for representing binary logic nodes in a bayesian Fault Tree.
"""
from typing import Any, List, Union, Optional

import numpy as np

from bayesiansafety.core import ConditionalProbabilityTable


class FaultTreeLogicNode:
    """Class for representing logic nodes (boolean logic gates).

    Attributes:
        cpt (ConditionalProbabilityTable): Associated CPT for this node.
        input_nodes (list<str>): Associated inputs for this logic gate.
        name (str): Name of the node.

    Deleted Attributes:
        logic_type (str): Boolean logic type for this gate (AND, OR (default))
    """

    name = ""
    input_nodes = []
    cpt = None

    __node_type = None
    __default_state_names = ["working", "failing"]

    def __init__(self, name: str, input_nodes: List[Union[Any, str]], logic_type: Optional[str] = 'OR') -> None:
        """Ctor for this class.

        Args:
            name (str): Name of the node.
            input_nodes (list<str>): Associated inputs for this logic gate.
            logic_type (str): Boolean logic type for this gate (AND, OR (default))
        """
        self.name = name
        self.__node_type = logic_type.upper()
        self.input_nodes = input_nodes
        self.__create_cpt()

    def get_node_type(self) -> str:
        """Getter for type of node.

        Returns:
            str: Returns either "OR" or "AND" to indicate specify the associated boolean logic behaviour.
        """
        return self.__node_type.upper()

    def __create_cpt(self) -> None:
        """Helper method to instantiate a ConditionalProbabilityTable based on the passed node_type (OR, AND) and set the member self.cpt (inted for internal use).

        Raises:
            ValueError: Raised if less than two inputs are specified.
        """

        if len(self.input_nodes) < 2:
            raise ValueError(f"Invalid input nodes: {self.input_nodes}. You need to specify at least two nodes.")


        evidence_card = (2 * np.ones(len(self.input_nodes), dtype=np.integer)).tolist()
        nr_of_entries = np.cumprod(evidence_card)[-1]

        # this is a problem since we might re-write explict state names when working with hybrid networks
        state_names = {
        name: self.__default_state_names for name in self.input_nodes}
        state_names[self.name] = self.__default_state_names

        if self.__node_type == 'OR':
            nf_entries = np.zeros(nr_of_entries)
            nf_entries[0] = 1

            pbf_entries = np.ones(nr_of_entries)
            pbf_entries[0] = 0
            self.cpt = ConditionalProbabilityTable(name=self.name, variable_card=2,
                                                values=[nf_entries,         # NF
                                                               pbf_entries],       # PBF
                                                evidence=self.input_nodes,
                                                evidence_card=evidence_card,
                                                state_names=state_names)

        elif self.__node_type == 'AND':
            nf_entries = np.ones(nr_of_entries)
            nf_entries[-1] = 0

            pbf_entries = np.zeros(nr_of_entries)
            pbf_entries[-1] = 1
            self.cpt = ConditionalProbabilityTable(name=self.name, variable_card=2,
                                                       values=[nf_entries,         # NF
                                                               pbf_entries],       # PBF
                                                       evidence=self.input_nodes,
                                                       evidence_card=evidence_card,
                                                       state_names=state_names)
        else:
            raise ValueError(f"Unsupported logic type: {self.__node_type}. Valid types are 'OR' and 'AND'.")
