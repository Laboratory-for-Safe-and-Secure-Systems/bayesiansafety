"""This class is intended as a container to tie shared nodes of multiple BNs, their    relevant
    states (which will later be treated as probability of failure) to gates in a scoped Fault Tree.
"""
from typing import Dict, List, Tuple


class HybridConfiguration:

    """This class is intended as a container to tie shared nodes of multiple BNs, their    relevant
        states (which will later be treated as probability of failure) to gates in a scoped Fault Tree.

    Attributes:
        name (str): Name of the configuration
        ft_coupling_points ft_coupling_points (dict<str, list<tuple<str, str>>>): Specifies shared nodes where BN nodes are mounted in a Faul Tree.
                    Dictionary with key = bn name, value = list of tuple of: bn node name and connected ft logic gate name.
        pbf_states (dict<str, list<tuple<str, str>>>): Specifies which state of a shared node is treated as PBF (basis event) in an associated Faul Tree.
                    Dictionary Key = bn name, value = list of bn node name and "bad" state of this node.
        shared_nodes (dict<str, list<str>>): Specifies which nodes of an associated BN are shared with a Fault Tree. Basically for sanity checking.
                    Dictionary with Key = bn name, value = list of shared bn node names.
    """

    name = None
    ft_coupling_points = None
    pbf_states = None
    shared_nodes = None

    def __init__(self, name: str, shared_nodes: Dict[str, List[str]], ft_coupling_points: Dict[str, List[Tuple[str, str]]], pbf_states: Dict[str, List[Tuple[str, str]]]) -> None:
        """Ctor of the HybridConfiguration class.

        Args:
            name (str): Name of the configuration
            ft_coupling_points ft_coupling_points (dict<str, list<tuple<str, str>>>): Specifies shared nodes where BN nodes are mounted in a Faul Tree.
                        Dictionary with key = bn name, value = list of tuple of: bn node name and connected ft logic gate name.
            pbf_states (dict<str, list<tuple<str, str>>>): Specifies which state of a shared node is treated as PBF (basis event) in an associated Faul Tree.
                        Dictionary Key = bn name, value = list of bn node name and "bad" state of this node.
            shared_nodes (dict<str, list<str>>): Specifies which nodes of an associated BN are shared with a Fault Tree. Basically for sanity checking.
                        Dictionary with Key = bn name, value = list of shared bn node names.
        """
        self.name = name
        self.ft_coupling_points = ft_coupling_points
        self.pbf_states = pbf_states
        self.shared_nodes = shared_nodes

        self.__verify_valid_configuration()

    def __verify_valid_configuration(self) -> None:
        """Setup method to check if the given configurations is valid.
            This means if the combination of cross referenced nodes and BNs match together and are reasonable.

        Raises:
            ValueError: Raised if a sanity check for a paramter fails.
        """
        #  BN mounting points must be contained in shared nodes
        for bn_name, connections in self.ft_coupling_points.items():
            bn_mounting_nodes = [
                bn_node_name for bn_node_name, logic_node_name in connections]
            invalid_nodes = set(bn_mounting_nodes) - \
                set(self.shared_nodes[bn_name])
            if len(invalid_nodes) != 0:
                raise ValueError(f"Mounting nodes for BN: {bn_name} contain invalid nodes {invalid_nodes} that are not part of the shared nodes for this BN.")

        # pbf nodes must be contained in shared nodes
        for bn_name, selections in self.pbf_states.items():
            scoped_shared_nodes = self.shared_nodes[bn_name]
            common_vars = set(dict(selections).keys()).intersection(
                set(scoped_shared_nodes))

            if len(common_vars) != len(dict(selections).keys()):
                raise ValueError(f"Invalid PBF nodes. Not all nodes are part of the shared nodes. Missing: {common_vars ^ set(dict(selections).keys())}")
