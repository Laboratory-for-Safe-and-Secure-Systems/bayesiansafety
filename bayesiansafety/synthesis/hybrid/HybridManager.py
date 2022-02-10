"""Manager to actually create a hybrid Fault Tree out of provided HybridConfigurations and associated BNs.
This means pulling the relevant shared nodes and their designated states representing a const. probability of
failure and stitch them to associated Faul Tree gates (mounting points).
"""
from typing import Dict, List, Optional, Tuple

from bayesiansafety.core.BayesianNetwork import BayesianNetwork
from bayesiansafety.faulttree import FaultTreeLogicNode
from bayesiansafety.faulttree import BayesianFaultTree
from bayesiansafety.synthesis.hybrid import HybridBuilder
from bayesiansafety.synthesis.hybrid import HybridConfiguration


class HybridManager:
    """Manager to actually create a hybrid Fault Tree out of provided HybridConfigurations and associated BNs.
        This means pulling the relevant shared nodes and their designated states representing a const. probability of
        failure and stitch them to associated Faul Tree gates (mounting points).

    Attributes:
        bayesian_nets (dict<str, BayesianNetwork>): Dictionary of all associated BNs needed to build the hybrids for each
                    possible given configurations with key = bn name, value = bn instance.
        configurations (dict<str, HybridConfiguration>): Dictionary of managed, individual configurations that can be applied on the given Faul Tree.
                    Key = configuration ID and value =  Hybrid configuration we want to apply on this Faul Tree.
        ft_inst (BayesianFaultTree): Fault Free instance which serves as basis for building hybrids.
    """

    ft_inst = None
    configurations = None
    bayesian_nets = None

    # dict<str, list<str>: Dictionary with key = bn_name, value = list of node names for every original BN provided at manager construction.
    __bn_nodes = None
    # dict<str, HybridBuilder>: Dictionary with key = configuration ID, value = hybrid builder instance thats responsible for the scoped configuration.
    __builder = None

    # dict<str, set(str)>: contains all managed shared nodes defined by the given configurations. Dictionary with key= bn_name, value = set of node names
    __all_shared_nodes = None
    __all_pbf_states = None  # dict<str, set<tuple<str, str>>: contains all managed pbf node states defined by the given configurations. Dictionary with key= bn_name, value = set of tuples <node name, "bad" state>
    __all_ft_coupling_points = None  # dict<str, set<tuple<str, str>>: contains all coupling odes defined by the given configurations Dictionary with key = bn name, value = set op tuples <bn node name, connected ft logic gate name>

    def __init__(self, ft_inst: BayesianFaultTree, bayesian_nets: Dict[str, BayesianNetwork], configurations: Dict[str, HybridConfiguration]) -> None:
        """Ctor of the HybridManager class.

        Args:
            ft_inst (BayesianFaultTree): Fault Free instance which serves as basis for building hybrids.
            bayesian_nets (dict<str, BayesianNetwork>): Dictionary of all associated BNs needed to build the hybrids for each
                        possible given configurations with key = bn name, value = bn instance.
            configurations (dict<str, HybridConfiguration>): Dictionary of managed, individual configurations that can be applied on the given Faul Tree.
                        Key = configuration ID and value =  Hybrid configuration we want to apply on this Faul Tree.
        """

        self.ft_inst = ft_inst
        self.configurations = configurations
        self.bayesian_nets = bayesian_nets

        self.__create_lookup_dicts()
        self.__verify_valid_configuration()
        self.__setup_builder()

    def __create_lookup_dicts(self) -> None:
        """Setup method to create helpful lookup dictionaries that ease some accesses in other methods.
        """
        self.__bn_nodes = {bn_name: bn_inst.model_elements.keys()
                           for bn_name, bn_inst in self.bayesian_nets.items()}

        self.__all_shared_nodes = {}
        self.__all_pbf_states = {}
        self.__all_ft_coupling_points = {}

        for config in self.configurations.values():
            for bn_name, shared_nodes in config.shared_nodes.items():
                if self.__all_shared_nodes.get(bn_name, None) is None:
                    self.__all_shared_nodes[bn_name] = set(shared_nodes)
                else:
                    self.__all_shared_nodes[bn_name].update(set(shared_nodes))

            for bn_name, node_states in config.pbf_states.items():
                if self.__all_pbf_states.get(bn_name, None) is None:
                    self.__all_pbf_states[bn_name] = set(node_states)
                else:
                    self.__all_pbf_states[bn_name].update(set(node_states))

            for bn_name, couplings in config.ft_coupling_points.items():
                if self.__all_ft_coupling_points.get(bn_name, None) is None:
                    self.__all_ft_coupling_points[bn_name] = set(couplings)
                else:
                    self.__all_ft_coupling_points[bn_name].update(
                        set(couplings))

    def __verify_valid_configuration(self) -> None:
        """Setup method to check if the given configurations is valid.
            This means if the combination of cross referenced nodes and BNs match together and are reasonable.

        Raises:
            ValueError: Raised if a sanity check for a paramter fails.
            TypeError: Raised if scoped Faul Tree nodes are not logic gates.
        """
        for bn_name, scoped_shared_nodes in self.__all_shared_nodes.items():
            invalid_nodes = set(scoped_shared_nodes) - \
                set(self.__bn_nodes[bn_name])
            if len(invalid_nodes) != 0:
                raise ValueError(f"Shared nodes for BN: {bn_name} contain invalid nodes {invalid_nodes} that are not part of the BN.")

        for bn_name, connections in self.__all_ft_coupling_points.items():
            bn_mounting_nodes = [
                bn_node_name for bn_node_name, logic_node_name in connections]
            ft_logic_nodes = [logic_node_name for bn_node_name,
                              logic_node_name in connections]

            #  BN mounting points must be contained in BN nodes
            invalid_nodes = set(bn_mounting_nodes) - \
                set(self.__bn_nodes[bn_name])
            if len(invalid_nodes) != 0:
                raise ValueError(f"Mounting nodes for BN: {bn_name} contain invalid nodes {invalid_nodes} that are not part of the BN.")

            #  Faul Tree mouting points must be contained in Faul Tree nodes
            invalid_nodes = set(ft_logic_nodes) - \
                set(self.ft_inst.model_elements.keys())
            if len(invalid_nodes) != 0:
                raise ValueError(f"Mounting nodes for Faul Tree: {self.ft_inst.name} contain invalid nodes {invalid_nodes} that are not part of the Fault Tree.")

            #  Faul Tree mounting points must be logical nodes (AND or OR)
            for logic_node_name in ft_logic_nodes:
                cur_ft_elem = self.ft_inst.get_elem_by_name(logic_node_name)
                if not isinstance(cur_ft_elem, FaultTreeLogicNode):
                    raise TypeError(f"Given mounting node: {logic_node_name} for Fault Tree: {self.ft_inst.name} must be an instance of FaultTreeLogicNode.")

        for bn_name, selections in self.__all_pbf_states.items():
            for bn_node_name, sel_state in selections:
                #  selected "pbf state" must be valid state of scoped BN node
                bn = self.bayesian_nets[bn_name]
                node = bn.model_elements[bn_node_name]
                if sel_state not in node.state_names[bn_node_name]:
                    raise ValueError(f"Invalid PBF state: {sel_state} for node: {bn_node_name} in BN: {bn_name}")

    def __setup_builder(self) -> None:
        """Setup method to initialize a group of builders for each unique set of configurations.
        """
        self.__builder = {}

        for config_id, config in self.configurations.items():
            associated_bns = {bn_name: bn_inst for bn_name, bn_inst in self.bayesian_nets.items(
            ) if bn_name in config.shared_nodes.keys()}
            self.__builder[config_id] = HybridBuilder(
                ft_inst=self.ft_inst, configuration=config, associated_bns=associated_bns)

    def build_extended_ft(self, config_id: str, bn_observations: Optional[Dict[str, List[Tuple[str, str]]]] = None) -> BayesianFaultTree:
        """Create an extended Fault Tree. An extended Fault Tree is the scoped Faul Tree extended by all associated BN nodes which are represented as static probability nodes.
            Additionally observation for associated BNs can be provided. This will affect the marginal distributions of shared nodes (posterior CPTs).
        Args:
            config_id (str): Unique identifier to select a given hybrid configurations that should be applied.
            bn_observations (dict<str, list<tuple<str, str>>, optional): Evidences for each provided BN. This is relevant when the PBF for a shared node is evaluated.
                        Key = bn name, value = list of tuples (bn node name, bn state).
        Returns:
            BayesianFaultTree: Extended Fault Tree

        Raises:
            ValueError: Raised if requested configuration is not a managed one.
        """
        if config_id not in self.__builder:
            raise ValueError(f"There is no managed hybrid configuration for given config id: {config_id}.")

        return self.__builder[config_id].get_extended_fault_tree(bn_observations=bn_observations)

    def build_hybrid_networks(self, config_id: str, at_time: Optional[float] = 0, fix_other_bns: Optional[bool] = True) -> Dict[str, BayesianNetwork]:
        """Generates all hybrid networks (Faul Tree + associated networks(s)) for this Faul Tree.
            Depending on the "fix_other_bns" flag, hybrids consisting of the Faul Tree and each individual associated BN (True) are generated.

        Args:
            config_id (str): Unique identifier to select a given hybrid configurations that should be applied.
            at_time (int, optional): Evaluation is done at a specific time stamp. This means all PBFs of the Faul Tree are evaluated at this time.
            fix_other_bns (bool, optional): Flag indicating if one "full" hybrid model (Faul Tree + all associated BNs) (False) or if
                                            a hybrid for each assocated BNs should be created (True, default)

        Returns:
            dict<str, BayesianNetwork>: Dictionary consisting of a generated BN name (key) and the generated hybrid BN instance (value).

        Raises:
            ValueError: Raised if requested configuration is not a managed one.
        """
        if config_id not in self.__builder:
            raise ValueError(f"There is no managed hybrid configuration for given config id: {config_id}.")

        return self.__builder[config_id].get_hybrid_networks(at_time=at_time, fix_other_bns=fix_other_bns)
