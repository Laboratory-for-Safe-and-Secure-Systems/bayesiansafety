"""
This class allows defining and working with Event Trees.
"""
import copy
from typing import Union

from networkx.classes.digraph import DiGraph

from bayesiansafety.core.inference import InferenceFactory
from bayesiansafety.core import ConditionalProbabilityTable
from bayesiansafety.faulttree import FaultTreeLogicNode
from bayesiansafety.faulttree import FaultTreeProbNode
from bayesiansafety.eventtree.BayesianEventTreeMapper import BayesianEventTreeMapper


class BayesianEventTree:

    """Main class managing a (bayesian) Event Tree.

    Attributes:
        model (BayesianNetwork): Event Tree model represented as DiGraph instance (networX DiGraph)
        model_elements (TYPE): Description
        name (str): Name of this Event Tree
        node_connections (list<tuple<str, str>>): List of tuples defining the edges between model elements.
        tree_obj (networkx.DiGraph): Parsed Event Tree as real tree structure. Nodes.data contain objects
            of type BayesianEventTree.EventTreeObjects describing branching elements (i.e. functional event)
            path elements (i.e. branching probabilities) and consequences (i.e. possible outcome events).
    """
    model = None
    model_elements = None
    name = None
    node_connections = None
    tree_obj = None

    def __init__(self, name: str, tree_obj: DiGraph) -> None:
        """Ctor of the BayesianEventTree class.

        Args:
            name (str): Name of this Event Tree instance
            tree_obj (networkx.DiGraph): Parsed Event Tree as real tree structure. Nodes.data contain objects
                of type BayesianEventTree.EventTreeObjects describing branching elements (i.e. functional event)
                path elements (i.e. branching probabilities) and consequences (i.e. possible outcome events).
        """
        self.name = name
        self.tree_obj = tree_obj
        self.__build_model()

    def __build_model(self) -> None:
        """Setup method to initialize a BayesianEventTree.
        """
        self.model = BayesianEventTreeMapper().map(self.tree_obj, name=self.name)
        self.model_elements = self.model.model_elements
        self.node_connections = self.model.node_connections

    def copy(self) -> "BayesianEventTree":
        """Helper method to make a deep copy of this instance.

        Returns:
            BayesianEventTree: Returns deep copy of this instance.
        """
        return copy.deepcopy(self)

    def get_consequence_node_name(self) -> str:
        """Get the name of the consequence node of the Bayesian Network representation.

        Returns:
            str: Name of the consequence node of the Bayesian Network representation.

        Raises:
            ValueError: Raised if the consequence node is ambiguous.
        """
        leaves = self.model.get_leaf_node_names()
        if len(leaves) != 1:
            raise ValueError(f"There are multiple leave nodes ({leaves}). This indicates that the Event Tree might not be valid.")

        return leaves[0]

    def get_consequence_likelihoods(self) -> ConditionalProbabilityTable:
        """Returns the prior marginal probabilities of the consequences

        Returns:
            ConditionalProbabilityTable: Factor containing the prior marginal probabilities of the consequence node.
        """
        inference_engine = InferenceFactory(self.model).get_engine()
        return inference_engine.query(variables=self.get_consequence_node_name())

    def get_elem_by_name(self, node_name: str) -> Union[FaultTreeProbNode, FaultTreeLogicNode]:
        """Getter to access a model element (instance of FaultTree*Node class) by name.

        Args:
            node_name (str): Name of the queried node.

        Returns:
            <FaultTree*Node>: Model element as an instance of FaultTree*Node class.

        Raises:
            ValueError: Raised if scoped element is not part of this Fault Tree.
        """
        if node_name not in self.model_elements:
            raise ValueError(f"Scoped element: {node_name} could not be found in given model elements: {self.model_elements.keys()}.")

        return self.model_elements[node_name]
