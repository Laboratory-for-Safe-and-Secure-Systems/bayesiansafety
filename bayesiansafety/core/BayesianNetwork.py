"""
This class is a wrapper for the internal representation of Bayesian Networks.
It is intended to seperate the model representation from the actual library (e.g. pgmpy)
as well as adding some useful functions and members for easier handling of BayesianSafety matters.
Currently this is only done half assed since it is just a wrapper around pgmpy and networkx.
"""
import copy
from typing import Dict, List, Optional, Set, Any, Tuple

import networkx as nx
import matplotlib.pyplot as plt

from bayesiansafety.core.ConditionalProbabilityTable import ConditionalProbabilityTable


class BayesianNetwork(nx.DiGraph):

    """This class is a wrapper for the internal representation of Bayesian Networks.
        It is intended to seperate the model representation from the actual library (e.g. pgmpy)
        as well as adding some useful functions and members for easier handling of BayesianSafety matters.
        Currently this is only done half assed since it is just a wrapper around pgmpy and networkx.

    Attributes:
        model_elements (dict<str, ConditionalProbabilityTable>): Lookup dictionary keyed by CPT name, value is instance of ConditionalProbabilityTable
        name (str): Name of this Bayesian Network.
        node_connections (list<tuple<str, str>>): List of tuples defining the edges between model elements.
    """

    name = None
    node_connections = None
    model_elements = None
    __roots = None
    __leafs = None

    def __init__(self, name: str, node_connections: List[Tuple[str, str]]) -> None:
        """Ctor of the BayesianNetwork class.

        Args:
            name (str): Name of this Bayesian Network.
            node_connections (list<tuple<str, str>>): List of tuples defining the edges between model elements.
        """
        self.name = name
        self.node_connections = node_connections
        self.__create_nodes_lookup()
        self.__build_model(node_connections)
        self.__leafs_and_roots()

    def __create_nodes_lookup(self) -> None:
        """Setup method to build a dictionary of the managed model elements (keys)
        """
        src_nodes = [src for src, dest in self.node_connections]
        dest_nodes = [dest for src, dest in self.node_connections]
        uniques = set(src_nodes + dest_nodes)
        self.model_elements = dict.fromkeys(uniques, None)

    def __build_model(self, node_connections: List[Tuple[str, str]]) -> None:
        """Setup method to actually instantiate the model by calling the ctor of the internal representation (e.g. pgmpy)

        Args:
            node_connections (list<tuple<str, str>>): List of node name tuples specifying the graphs nodes and edges.
                                                      Tuples are of the type src node name -> dest node name

        Raises:
            ValueError: Raised if the BN contains cycles. BNs are defined as directed acyclic graphs.
        """
        super(BayesianNetwork, self).__init__(node_connections)
        try:
            cycles = list(nx.find_cycle(self))
        except nx.NetworkXNoCycle:
            pass
        else:
            out_str = "Cycles are not allowed in a DAG."
            out_str += "\nEdges indicating the path taken for a loop: "
            out_str += "".join([f"({u},{v}) " for (u, v) in cycles])
            raise ValueError(out_str)

        self.cpds = []

    def __leafs_and_roots(self) -> None:
        """Setup method to build a list of current leaf and root node names.
        """
        self.__leafs = [node for node in self.model_elements if super(
            BayesianNetwork, self).out_degree(node) == 0]
        self.__roots = [node for node in self.model_elements if super(
            BayesianNetwork, self).in_degree(node) == 0]

    def get_root_node_names(self) -> List[str]:
        """Get a list of all root (i.e. no ingoing edges) node names..

        Returns:
            list<str>: List of current root node names.
        """
        return self.__roots

    def get_leaf_node_names(self) -> List[str]:
        """Get a list of all leaf (i.e. no outgoing edges) node names..

        Returns:
            list<str>: List of current leaf node names.
        """
        return self.__leafs

    def copy(self) -> 'BayesianNetwork':
        """Returns a deep copy of this instnce.

        Returns:
            BayesianNetwork: Deep copy of this instance
        """
        return copy.deepcopy(self)

    def add_node(self, node_name: str) -> None:
        """Helper method to extend the graph by adding a single new node.

        Args:
            node_name (str): Node (name) to be added.
        """
        super(BayesianNetwork, self).add_node(node_name)
        if node_name not in self.model_elements.keys():
            self.model_elements[node_name] = None
        self.__leafs_and_roots()

    def add_edges_from(self, edges: List[Tuple[str, str]]) -> None:
        """Helper method to add a new edges including the src and dest nodes to the graph.

        Args:
            edges (list<tuple<str, str>>): List of src -> dest node name tuples.
        """
        super(BayesianNetwork, self).add_edges_from(edges)
        for src, dest in edges:
            if (src, dest) not in self.node_connections:
                self.node_connections.append((src, dest))
            self.add_node(src)
            self.add_node(dest)

    def add_cpts(self, *cpds) -> None:
        """Helper method to add ConditionalProbabilityTable's to the network.

        Raises:
            TypeError: Raised if given CPTs are not instances of core.ConditionalProbabilityTable

        Args:
            *cpds: Collection of CPTs to add.
        """
        if not all(isinstance(elem, ConditionalProbabilityTable) for elem in cpds):
            raise TypeError(f"CPTs to add must be instantiations of {type(ConditionalProbabilityTable)}. Given CPTs are of type {[type(elem) for elem in cpds]}")

        for elem in cpds:
            self.model_elements[elem.name] = elem  # internal representation
        self.__leafs_and_roots()

    def get_independent_nodes(self) -> Set[str]:
        """Helper method to get all nodes that are independent. Returns all nodes that are independent - meaning they neither have ingoing nor outgoing edges

        Returns:
            set<str>: Set of node names.
        """
        return set(self.__leafs).intersection(set(self.__roots))

    def plot_graph(self, title: Optional[str] = None, edge_labels: Optional[Dict[Tuple[str, str], str]] = None, options: Optional[Dict[str, Any]] = None) -> None:
        """Plot and display the current graph. Labels for individual edges can be provided.
            The edge_labels and options argument is identical to networkX.draw(...) and networkX.draw_networkx_edge_labels(...)

        Args:
            title (str, optional): Title of the plot
            edge_labels (dict<tuple<str, str>, str>, optional): Dictionary with key = tuple of src -> dest node names specified the edge
                                                                and value = edge value to display
            options (dict<str, obj>, optional): Dictionary parameteriziing how the style of the plot looks like. Key = option, value= option value.
        """
        if not options:
            options = {
                # "font_size": 7,
                # "node_size": 4000,
                "node_color": "lightblue",  # white
                # "edgecolors": "black",
                # "linewidths": 4,
                # "width": 4,
                # "node_shape": 's' #string (default=’o’)
            }

        title = self.name if title == "" or title is None else title
        fig, ax = plt.subplots()
        ax.set_title(title)

        # twopi - radial layouts, after Graham Wills 97. Nodes are placed on concentric circles depending their distance from a given root node.
        # dot - “hierarchical” or layered drawings of directed graphs. This is the default tool to use if edges have directionality.

        layout = None
        try:
            import pygraphviz
            layout = nx.nx_agraph.graphviz_layout(self, prog="dot")
        except ImportError as err:
            print(f"ImportError: {str(err)}. \nTo support proper layout consider installing pygraphviz http://pygraphviz.github.io/")

        layout = layout if layout is not None else nx.spring_layout(self)
        nx.draw(self, pos=layout, with_labels=True, ax=ax, **options)

        if edge_labels:
            nx.draw_networkx_edge_labels(
                self, pos=layout, edge_labels=edge_labels, ax=ax)
        plt.show()

    @staticmethod
    def get_prefixed_copy(bn_inst: 'BayesianNetwork', prefix_str: str) -> 'BayesianNetwork':
        """Static helper method to deep copy a BN instance. A prefix string needs to be provided which will be used to rename the
            node names as well as the BN name of the copy. State names are not affected.

        Args:
            bn_inst (BayesianNetwork): BN instance to make a renamed copy from.
            prefix_str (str): String that will be used as prefix when renaming the copies elements.

        Returns:
            BayesianNetwork: Deep copy of the passed instance.

        Raises:
            TypeError: Raised if either the passed BN instance is not a core.BayesianNetwork instance or the prefix is not a string.
        """
        if not isinstance(bn_inst, BayesianNetwork):
            raise TypeError(f"Passed BN instance must be an instantiation of the class: {type(BayesianNetwork)}.")

        if not isinstance(prefix_str, str):
            raise TypeError(f"Passed prefix string must be string object and not an instantiation of: {type(prefix_str)}.")

        independent_nodes = bn_inst.get_independent_nodes()
        new_node_connections = [(f"{prefix_str}{src}", f"{prefix_str}{dest}") for src, dest in bn_inst.node_connections]
        new_bn = BayesianNetwork(f"{prefix_str}{bn_inst.name}", new_node_connections)

        for old_name, old_cpd in bn_inst.model_elements.items():
            new_cpd_name = f"{prefix_str}{old_cpd.name}"
            new_cpd_evidence = [f"{prefix_str}{evi}" for evi in old_cpd.evidence] if old_cpd.evidence is not None else None
            new_cpd_state_names = {f"{prefix_str}{var}": states for var, states in old_cpd.state_names.items()} if old_cpd.state_names is not None else None

            if old_name in independent_nodes:
                new_bn.add_node(new_cpd_name)

            new_bn.add_cpts(ConditionalProbabilityTable(name=new_cpd_name, variable_card=old_cpd.variable_card, values=old_cpd.values,
                                                        evidence=new_cpd_evidence, evidence_card=old_cpd.evidence_card, state_names=new_cpd_state_names))

        return new_bn
