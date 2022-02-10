"""Builder like class to assemble associated BNs and their shared nodes into "hybird" networks.
    Note: Hybrid here is meant as a mix of a BayesianFaultTree and one or more BayesianNetworks
            and NOT like often in literature a BN as a hybrid of continous and discrete nodes.
"""
from typing import Dict, List, Optional, Tuple

from bayesiansafety.core.BayesianNetwork import BayesianNetwork
from bayesiansafety.core.inference import InferenceFactory
from bayesiansafety.core import MultistateLogicNode
from bayesiansafety.faulttree import FaultTreeLogicNode
from bayesiansafety.faulttree import BayesianFaultTree
from bayesiansafety.synthesis.hybrid import HybridConfiguration


class HybridBuilder:

    """Builder like class to assemble associated BNs and their shared nodes into "hybird" networks.
        Note: Hybrid here is meant as a mix of a BayesianFaultTree and one or more BayesianNetworks
            and NOT like often in literature a BN as a hybrid of continous and discrete nodes.

    Attributes:
        associated_bns (dict<str, BayesianNetwork>): Dictionary of all currently associated BNs needed to build a hybrid for each configuration.
                    Dictionary consists of key= bn name, value = bn instance.
        configuration (HybridConfiguration): Configuration specifying how the scoped Faul Tree can be extended.
        ft_inst (BayesianFaultTree): Scoped Fault Tree instance which will be extended by associated BNs depending on a given configuration.
    """

    ft_inst = None
    configuration = None
    associated_bns = None

    def __init__(self, ft_inst: BayesianFaultTree, configuration: HybridConfiguration, associated_bns: Dict[str, BayesianNetwork]) -> None:
        """Ctor of the HybridBuilder class.

        Args:
            associated_bns (dict<str, BayesianNetwork>): Dictionary of all currently associated BNs needed to build a hybrid for each configuration.
                        Dictionary consists of key= bn name, value = bn instance.
            configuration (HybridConfiguration): Configuration specifying how the scoped Faul Tree can be extended.
            ft_inst (BayesianFaultTree): Scoped Fault Tree instance which will be extended by associated BNs depending on a given configuration.
        """
        self.ft_inst = ft_inst
        self.configuration = configuration
        self.associated_bns = associated_bns

    def get_extended_fault_tree(self, bn_observations: Optional[Dict[str, List[Tuple[str, str]]]] = None) -> BayesianFaultTree:
        """Create an extended Fault Tree. An extended Fault Tree is the scoped Faul Tree extended by all associated BN nodes which are represented as static probability nodes.
            Additionally observation for associated BNs can be provided. This will affect the marginal distributions of shared nodes (posterior CPTs).
        Args:
            bn_observations (dict<str, list<tuple<str, str>>, optional): Evidences for each provided BN. This is relevant when the PBF for a shared node is evaluated.
                        Key = bn name, value = list of tuples (bn node name, bn state).

        Returns:
            BayesianFaultTree: Extended Fault Tree
        """
        mod_fault_tree = self.ft_inst.copy()
        mod_fault_tree.name = "MOD_" + str(self.ft_inst.name)

        # 1) add nodes to Fault Tree at the correct position
        for bn_name, connections in self.configuration.ft_coupling_points.items():
            bn_inst = self.associated_bns[bn_name]
            inf_engine = InferenceFactory(bn_inst).get_engine()
            bn_pbf_states = dict(self.configuration.pbf_states[bn_name])
            evidence = bn_observations.get(
                bn_name, None) if bn_observations else None

            # 2) get marginal and pull state value designated as prob. of failure
            for bn_node_name, logic_node_name in connections:
                marg_cpt = inf_engine.query(bn_node_name, evidence)
                pbf = marg_cpt.get_value(
                    {bn_node_name: bn_pbf_states[bn_node_name]})
                mod_fault_tree.add_prob_node(node_name=f"{bn_name}_{bn_node_name}", input_to=logic_node_name, probability_of_failure=pbf, is_time_dependent=False)
        return mod_fault_tree

    def get_hybrid_networks(self, at_time: Optional[float] = 0, fix_other_bns: Optional[bool] = True) -> Dict[str, BayesianNetwork]:
        """Generates all hybrid networks (Faul Tree + associated networks(s)) for a given Fault Tree.
            Depending on the "fix_other_bns" flag, hybrids consisting of the Faul Tree and each individual associated BN (True) are generated.

        Args:
            at_time (int, optional): Evaluation is done at a specific time stamp. This means all PBFs of the Faul Tree are evaluated at this time.
            fix_other_bns (bool, optional): Flag indicating if one "full" hybrid model (Faul Tree + all associated BNs) (False) or if
                                            a hybrid for each assocated BNs should be created (True, default)

        Returns:
            dict<str, BayesianNetwork>: Dictionary consisting of a generated BN name (key) and the generated hybrid BN instance (value).

        Raises:
            ValueError: Raised if requested time stamp is invalid (negativ).
        """
        if at_time < 0:
            raise ValueError(f"A time stamp for evaluation can not be negative but was requested at time: {at_time}.")

        # 1) get Faul Tree at time t (updated pbf's for this evaluation)
        extended_ft = self.get_extended_fault_tree()
        if at_time != 0:
            extended_ft = extended_ft.get_tree_at_time(
                at_time=at_time, modified_frates=None)

        # 2) stitch them together by the shared nodes of the BN
        hybrid_networks = {}

        if fix_other_bns:
            hybrid_name_template = f"{self.ft_inst.name}_at_time_{at_time}"
            hybrid_networks = self.__get_fixed_hybrid_networks(
                hybrid_name=hybrid_name_template, extended_ft=extended_ft)

        else:
            hybrid_name = f"Full_hybrid_{self.ft_inst.name}_at_time_{at_time}"
            hybrid_bn = self.__get_full_hybrid_network(
                hybrid_name=hybrid_name, extended_ft=extended_ft)
            hybrid_networks[hybrid_bn.name] = hybrid_bn

        return hybrid_networks

    def __get_fixed_hybrid_networks(self, hybrid_name: str, extended_ft: BayesianFaultTree) -> Dict[str, BayesianNetwork]:
        """Helper method called by get_hybrid_networks(...) to actually build all partial hybrid networks.
            This means a hybrid consisting of the scoped Faul Tree and each individual associated BN.

        Args:
            hybrid_name (str): Template name for the generated hybrids. The given template will be extended by
                                '_with_<bn name>' for each processed BN
            extended_ft (BayesianFaultTree): Scoped and already extended Fault Tree that serves as the basis for building the hybrids.

        Returns:
            dict<str, BayesianNetwork>: Dictionary of the generated hybrids where key= hybrid bn name and value = hybrid bn instance.
        """
        hybrid_networks = {}

        for bn_name in self.associated_bns.keys():
            # 3.a) collect all node connections of the hybrid network
            #      due to the logic of the Faul Tree the edge direction is BN ->> Faul Tree
            cur_bn_inst = self.associated_bns[bn_name]

            renamed_associated_bn = BayesianNetwork.get_prefixed_copy(bn_inst=cur_bn_inst, prefix_str=f"{bn_name}_")
            renamed_coupling_nodes = [(f"{bn_name}_{bn_node_name}", logic_node_name) for bn_node_name, logic_node_name in self.configuration.ft_coupling_points[bn_name]]
            cur_bn_renamed_node_names = renamed_associated_bn.model_elements.keys()

            node_connections = extended_ft.node_connections + \
                renamed_associated_bn.node_connections + renamed_coupling_nodes

            # 3.b) Create the current hybrid network
            name = f"{hybrid_name}_with_{bn_name}"
            hybrid_bn = BayesianNetwork(
                name=name, node_connections=node_connections)

            # 3.c) Populate current hybrid network with CPTs
            for ft_node_name, ft_node_inst in extended_ft.model.model_elements.items():
                # Note: Logic nodes with shared nodes as input must now be treated as multistate inputed
                # Note: A shared node from a BN that is not currently scoped is still treated as a (binary) FaultTreeProbNode
                # Note: Associated BN nodes where renamed as f"{bn_name}_{bn_node_name}" to make them unique in the ext. Faul Tree.
                #        But now we need to work with the original name (bn_node_name) - that is the name of the node that's managed by the BN.

                ft_model_elem = extended_ft.model_elements[ft_node_name]

                if not isinstance(ft_model_elem, FaultTreeLogicNode):
                    # This is a prob. node, either a regular one from the Faul Tree or from an associated BN
                    cur_shared_nodes = self.configuration.shared_nodes[bn_name]
                    corrected_node_name = ft_node_name.replace(
                        bn_name + "_", "")

                    # Check if shared node:  part of the Faul Tree or another BN  ?  ft_node_inst  :  node of scoped BN (shared node, use bn_node_inst)
                    inst_to_add = ft_node_inst if corrected_node_name not in cur_shared_nodes else renamed_associated_bn.model_elements[
                        ft_node_name]
                    hybrid_bn.add_cpts(inst_to_add)

                else:
                    # Logic node, check if at least one of the inputs is part of the scoped BN
                    common_vars = set(cur_bn_renamed_node_names).intersection(
                        set(ft_model_elem.input_nodes))

                    if len(common_vars) == 0:
                        # all inputs are from the Faul Tree itself or different BNs - nothing to do
                        hybrid_bn.add_cpts(ft_node_inst)

                    else:
                        # At least one node from scoped BN is an input
                        # We need to change the current binary logic node to a multistate input logic node
                        # first - find out the cards and bad states for each input node
                        cardinalities = []
                        bad_states = {}
                        # f"{bn_name}_{name}"
                        state_names = dict(ft_model_elem.cpt.state_names.items())

                        for input_var in ft_model_elem.input_nodes:
                            if input_var in cur_bn_renamed_node_names:
                                input_var_card = renamed_associated_bn.model_elements[
                                    input_var].variable_card
                                cardinalities.append(input_var_card)

                                state_names[input_var] = renamed_associated_bn.model_elements[input_var].state_names[input_var]

                                input_var_bad_state_name = dict(self.configuration.pbf_states[bn_name])[
                                    input_var.replace(bn_name + "_", "")]
                                bad_state_idx = renamed_associated_bn.model_elements[input_var].get_index_of_state(
                                    input_var_bad_state_name)
                                bad_states[input_var] = bad_state_idx

                            else:
                                # Input var is either part of the Faul Tree or another BN, treat as regular binary prob. node with states: [working, failing]
                                cardinalities.append(2)
                                bad_states[input_var] = 1

                        # second - instantiate a multistate input logic node
                        mstate_logic_node = MultistateLogicNode(name=ft_node_name, input_nodes=ft_model_elem.input_nodes, input_state_names=state_names, cardinalities=cardinalities,
                                                                bad_states=bad_states, logic_type=ft_model_elem.get_node_type())

                        # third - add the CPT to the hybrid network
                        hybrid_bn.add_cpts(mstate_logic_node.cpt)

            # 3.d) Add remaining CPTs of scoped BN (un-shared nodes)
            #      Note: We need to handle independent nodes (no ingoing or outoing edges) in the scoped BN as well
            independent_node_names = renamed_associated_bn.get_independent_nodes()
            if len(independent_node_names) != 0:
                for ind_node_name in independent_node_names:
                    hybrid_bn.add_node(ind_node_name)

            common_nodes = set(cur_bn_renamed_node_names).intersection(
                set(extended_ft.model_elements.keys()))
            missing_bn_nodes = set(cur_bn_renamed_node_names) ^ common_nodes
            for missing_node in missing_bn_nodes:
                hybrid_bn.add_cpts(
                    renamed_associated_bn.model_elements[missing_node])

            # 3.e) Add current hybrid network to dictionary of networks
            hybrid_networks[name] = hybrid_bn

        return hybrid_networks

    def __get_full_hybrid_network(self, hybrid_name: str, extended_ft: BayesianFaultTree) -> Dict[str, BayesianNetwork]:
        """Helper method called by get_hybrid_networks(...) to actually build all a full hybrid network.
            This means a hybrid consisting of the scoped Faul Tree and each associated BN merged to one.

        Args:
            hybrid_name (str): Template name for the generated full hybrid.
            extended_ft (BayesianFaultTree): Scoped and already extended Fault Tree that serves as the basis for building the hybrid.

        Returns:
            dict<str, BayesianNetwork>: Dictionary of the generated full hybridswhere key= hybrid bn name and value = hybrid bn instance.
                                        A dictionary is returned to comply to the signature of the calling get_hybrid_networks function.
        """

        # 3.a) collect all node connections for all associated BNs
        #      due to the logic of the Faul Tree the edge direction is BN ->> Faul Tree
        #      Note: Node names might not be unique across BNs - renaming
        renamed_associated_bns = {}
        for bn_name, bn_inst in self.associated_bns.items():
            renamed_associated_bns[bn_name] = BayesianNetwork.get_prefixed_copy(bn_inst=bn_inst, prefix_str=f"{bn_name}_")

        node_connections = extended_ft.node_connections
        for bn_name, bn_inst in renamed_associated_bns.items():
            renamed_coupling_nodes = [(f"{bn_name}_{bn_node_name}", logic_node_name) for bn_node_name, logic_node_name in self.configuration.ft_coupling_points[bn_name]]

            node_connections += bn_inst.node_connections + renamed_coupling_nodes

        # 3.b) Create the current hybrid network
        print(f">> Instantiating full hybrid: {hybrid_name}")
        print(f">> with node node_connections: {node_connections}")

        hybrid_bn = BayesianNetwork(
            name=hybrid_name, node_connections=set(node_connections))

        # dict<str, str> with key=renamed node_name, val=related bn
        renamed_shared_nodes = {}
        for bn_name, bn_shared_nodes in self.configuration.shared_nodes.items():
            for bn_node_name in bn_shared_nodes:
                renamed_shared_nodes[f"{bn_name}_{bn_node_name}"] = bn_name

        # 3.c) Populate the hybrid network with CPTs
        for ft_node_name, ft_node_inst in extended_ft.model.model_elements.items():
            # Note: Logic nodes with shared nodes as input must now be treated as multistate inputed
            # Note: Associated BN nodes where renamed as f"{bn_name}_{bn_node_name}" to make them unique in the hybrid network
            #       But we need to work with the original name (bn_node_name) - when pulling the CPTs

            ft_model_elem = extended_ft.model_elements[ft_node_name]

            if not isinstance(ft_model_elem, FaultTreeLogicNode):
                # This is a prob. node, either a regular one from the Faul Tree or from an associated BN
                # Check if shared node:  part of the Faul Tree  ?  ft_node_inst  :  node of associated BN (shared node, use bn_node_inst)

                if ft_node_name not in renamed_shared_nodes:
                    hybrid_bn.add_cpts(ft_node_inst)

                else:
                    bn_inst = renamed_associated_bns[renamed_shared_nodes[ft_node_name]]
                    bn_node_inst = bn_inst.model_elements[ft_node_name]
                    hybrid_bn.add_cpts(bn_node_inst)

            else:
                # Logic node, check if at least one of the inputs is part of the associated BNs
                common_vars = set(renamed_shared_nodes.keys()).intersection(
                    ft_model_elem.input_nodes)

                if len(common_vars) == 0:
                    # all inputs are from the Faul Tree itself
                    hybrid_bn.add_cpts(ft_node_inst)

                else:
                    # At least one input is from an associated BN
                    # We need to change the current binary logic node to a multistate input logic node
                    # first - find out the cards and bad states for each input node
                    cardinalities = []
                    bad_states = {}
                    # f"{bn_name}_{name}"
                    state_names = dict(ft_model_elem.cpt.state_names.items())

                    for input_var in ft_model_elem.input_nodes:
                        if input_var in renamed_shared_nodes.keys():
                            associated_bn_name = renamed_shared_nodes[input_var]
                            bn_inst = renamed_associated_bns[associated_bn_name]

                            input_var_card = bn_inst.model_elements[input_var].variable_card
                            cardinalities.append(input_var_card)

                            original_input_var_name = input_var.replace(f"{associated_bn_name}_", "")
                            input_var_bad_state_name = dict(self.configuration.pbf_states[associated_bn_name])[
                                original_input_var_name]
                            bad_state_idx = bn_inst.model_elements[input_var].get_index_of_state(
                                input_var_bad_state_name)
                            bad_states[input_var] = bad_state_idx

                            state_names[input_var] = bn_inst.model_elements[input_var].state_names[input_var]

                        else:
                            # Input var part of the Faul Tree, treat as regular binary prob. node with states: [prob. no failure, prob. of failure]
                            cardinalities.append(2)
                            bad_states[input_var] = 1

                    # second - instantiate a multistate input logic node
                    mstate_logic_node = MultistateLogicNode(name=ft_node_name, input_nodes=ft_model_elem.input_nodes, input_state_names=state_names, cardinalities=cardinalities,
                                                            bad_states=bad_states, logic_type=ft_model_elem.get_node_type())

                    # third - add the CPT to the hybrid network
                    hybrid_bn.add_cpts(mstate_logic_node.cpt)

        # 3.d) Add remaining CPTs of associated BNs (un-shared nodes)
        # #      Note: We need to handle independent nodes (no ingoing or outoing edges) in the scoped BN as well
        for bn_name, bn_inst in renamed_associated_bns.items():
            independent_node_names = bn_inst.get_independent_nodes()
            if len(independent_node_names) != 0:
                for ind_node_name in independent_node_names:
                    hybrid_bn.add_node(ind_node_name)

            cur_bn_node_names = set([f"{bn_name}_{node_name}" for node_name in self.associated_bns[bn_name].model_elements.keys()])
            cur_shared_nodes = set([f"{bn_name}_{node_name}" for node_name in self.configuration.shared_nodes[bn_name]])
            missing_bn_nodes = cur_bn_node_names ^ cur_shared_nodes
            for missing_node in missing_bn_nodes:
                hybrid_bn.add_cpts(bn_inst.model_elements[missing_node])

        return hybrid_bn
