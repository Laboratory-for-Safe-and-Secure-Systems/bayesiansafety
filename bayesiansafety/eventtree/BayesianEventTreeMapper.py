"""This class allows top map a parsed Event Tree DiGraph-object into a BayesianNetwork.
"""
import copy
import itertools
from typing import Optional, List, Tuple, Dict

import numpy as np
import networkx as nx
from networkx.classes.digraph import DiGraph

from bayesiansafety.utils.utils import get_root_and_leaves, remove_duplicate_tuples
from bayesiansafety.core import BayesianNetwork
from bayesiansafety.core import ConditionalProbabilityTable
from bayesiansafety.eventtree.EventTreeObjects import EtElement, FunctionalEvent, Path


class BayesianEventTreeMapper:

    """This class allows mapping a parsed Event Tree DiGaph-object into a Bayesian Network.

    Attributes:
        consequence_node_name (str): Default name of the consequence node.
        consequences (dict<name, list<tuple<clearview_path, id_path>>>): Dictionary with key = consequence name,
                    value = list of path tuples(clearview path, id path) leading to the individual consequences (outcome events).
                    The first tuple element describes a path consisting of readable names, the second tuple element describes the same path as a series of tree node ids.
        functional_events (dict<name, list<node_id>):  Dictionary with key = event name, value = list of node ids (UUID) of the parsed tree object.
        model (core.BayesianNetwork): Event Tree model represented as DiGraph instance (networX DiGraph)
        name (str): Name of this Event Tree
        node_connections (list<tuple<str, str>>): List of node name tuples defining the edges between model elements.
        paths_to_leaves (list< tuple<clearview_path, id_path>): List of path tuples. The first tuple element describes a path consisting of readable names,
                    the second tuple element describes the same path as a series of tree node ids.
        states_and_probabilities (dict<func_event_name, dict<state_name, list<probs>>): Dictionary where key = functional event name,
                    value = dictionary with key = state name and value = list of probabilities for that given state (branching probabilites).
    """

    consequence_node_name = "CNSQNC"
    consequences = None
    paths_to_leaves = None
    functional_events = None
    states_and_probabilities = None

    node_connections = None
    model = None
    name = None

    __default_tree_name = "EventTree"
    __tree_obj = None  # nx.DiGraph


    def map(self, tree_obj: DiGraph, name: Optional[str] = None) -> BayesianNetwork:
        """Maps a given parsed Event Tree object (nx.DiGraph) into a Bayesian Network.

        Args:
            tree_obj (networkx.DiGraph): Parsed Event Tree as real tree structure. Nodes.data contain objects
                        of type BayesianEventTree.EventTreeObjects describing branching elements (i.e. functional event)
                        path elements (i.e. branching probabilities) and consequences (i.e. possible outcome events).
            name (str, optional): Name of the to be created Event Tree instance. If no name is given, the default name "EventTree"
                                  will be used instead.

        Returns:
            BayesianNetwork: Event Tree model as a Bayesian Network.
        """
        self.__tree_obj = tree_obj
        self.name = name if name else self.__default_tree_name
        self.parse_basic_containers()
        self.map_tree_structure()
        self.build_bn_model()
        return self.model

    def parse_basic_containers(self) -> None:
        """Traverse and collect all neccesary elements of the Event Tree. This includes mapping the occurance of functional events and
            probabilities to paths leading to the different outcome events (consequences).
        """
        self.paths_to_leaves = self.parse_all_paths(self.__tree_obj)
        self.consequences = self.parse_all_consequences(self.paths_to_leaves)
        self.functional_events = self.parse_all_functional_events(
            self.__tree_obj)
        self.states_and_probabilities = self.parse_all_probabilities(
            self.__tree_obj, self.functional_events)

    def parse_all_paths(self, tree_obj: DiGraph) -> List[Tuple[List[str], List[str]]]:
        """Collect all paths from the root of the given tree to each leaf.
           Paths through the tree are stored as ID paths (i.e. elements along the path
           consist only of node IDs) and clearview paths (i.e. elemnts along the path
           consist only of human-readable names)

        Args:
            tree_obj (DiGraph): Given Event Tree as DiGraph object (real tree)

        Returns:
            List[Tuple[List[str], List[str]]]: Collected paths from the root node to each leaf node.
        """
        paths_to_leaves = []
        root, leaves = get_root_and_leaves(tree_obj)

        for id_path in nx.all_simple_paths(tree_obj, source=root, target=leaves):
            clearview_path = self.get_filtered_clearview_path(id_path)
            path_tuple = (clearview_path, id_path)

            if path_tuple not in paths_to_leaves:
                paths_to_leaves.append(path_tuple)

        return paths_to_leaves

    def parse_all_consequences(self, paths_to_leaves: List[Tuple[List[str], List[str]]]) -> Dict[str, List[Tuple[List[str], List[str]]]]:
        """Collect all possible, unique consequences (leafs) and all unique paths starting at the
            root node leading to them.

        Args:
            paths_to_leaves (List[Tuple[List[str], List[str]]]): All paths from the root of a given tree to each leaf.

        Returns:
            Dict[str, List[Tuple[List[str], List[str]]]]: Dictionary of all consequences and paths leading to them.
        """
        consequences = {}

        for path_tuple in paths_to_leaves:
            clearview_path, id_path = path_tuple
            consequence = clearview_path[-1]
            if consequence not in consequences:
                consequences[consequence] = [path_tuple]

            else:
                consequences[consequence].append(path_tuple)

        return consequences

    def parse_all_functional_events(self, tree_obj: DiGraph) -> Dict[str, List[str]]:
        """Collect all functional events and the nodes representing them in the given
            DiGraph tree object.

        Args:
            tree_obj (DiGraph): Given Event Tree as DiGraph object (real tree)

        Returns:
            Dict[str, List[str]]: Mapping of all functional events in the Event Tree to the unique node IDs
                representing them in the DiGraph tree object.
        """
        func_events = {}

        for node_id, node_attributes in tree_obj.nodes(data=True):
            et_object = node_attributes["data"]

            if isinstance(et_object, FunctionalEvent):
                if et_object.name not in func_events:
                    func_events[et_object.name] = [node_id]
                else:
                    func_events[et_object.name].append(node_id)

        return func_events

    def parse_all_probabilities(self, tree_obj: DiGraph, func_events: Dict[str, List[str]]) -> Dict[str, Dict[str, List[float]]]:
        """Collect all branching/outcome probabilities for a given Event Tree.

        Args:
            tree_obj (DiGraph): Given Event Tree as DiGraph object (real tree)
            func_events (Dict[str, List[str]]): Mapping of all functional events in the Event Tree to the unique node IDs
                representing them in the DiGraph tree object.

        Returns:
            Dict[str, Dict[str, List[float]]]: Outcome probabilities for each functional event and its
                respective outcomes/states/branching options in the tree.
        """
        states_and_probabilities = {}

        for func_event_name, nodes in func_events.items():
            if func_event_name not in states_and_probabilities:
                states_and_probabilities[func_event_name] = {}

            for node_id in nodes:
                for child_id in tree_obj.successors(node_id):
                    child_et_object = tree_obj.nodes[child_id]["data"]

                    prob_to_add = child_et_object.probability if child_et_object.probability else 0.0

                    if child_et_object.state not in states_and_probabilities[func_event_name].keys():
                        states_and_probabilities[func_event_name][child_et_object.state] = [
                            prob_to_add]
                    else:
                        if prob_to_add not in states_and_probabilities[func_event_name][child_et_object.state]:
                            states_and_probabilities[func_event_name][child_et_object.state].append(
                                prob_to_add)

        return states_and_probabilities

    def map_tree_structure(self) -> None:
        """Main method to actually convert the tree to a BayesianSafety BN (structure only).
            First the given tree object is evaluated by traversing all paths.
            Based on the pre-processed information, BN-nodes are created and linked based on the tree structure and
            branching probabilities. This method creates the topology / skeleton of the BN.
        """
        primitive_connections = self.build_primitive_node_connections(
            self.paths_to_leaves)
        causal_connections = self.build_causal_arcs(
            self.__tree_obj, self.paths_to_leaves, self.functional_events, self.states_and_probabilities)

        primitive_connections.extend(causal_connections)
        node_connections = remove_duplicate_tuples(primitive_connections)
        node_connections = self.simplify_causal_arcs(
            self.__tree_obj, node_connections, self.paths_to_leaves, self.functional_events, self.states_and_probabilities)

        consequence_connections = self.build_consequence_arcs(
            self.functional_events)
        consequence_connections = self.simplify_consequence_arcs(
            consequence_connections,  self.paths_to_leaves, self.functional_events, self.consequences)

        node_connections.extend(consequence_connections)
        self.node_connections = remove_duplicate_tuples(node_connections)

    def build_primitive_node_connections(self, paths_to_leaves: List[Tuple[List[str], List[str]]]) -> List[Tuple[str, str]]:
        """Build the skeleton/topology of the BN.
        This is done based on the functional events and their connections.

        Args:
            paths_to_leaves (List[Tuple[List[str], List[str]]]): All paths from the root of a given tree to each leaf.

        Returns:
            List[Tuple[str, str]]: List of all primitive/trivial node-connections.
        """
        primitive_connections = []

        for clearview_path, id_path in paths_to_leaves:
            events_path = self.get_filtered_clearview_path(
                id_path, filter_by=FunctionalEvent)

            for idx in range(len(events_path) - 1):
                new_connection = (events_path[idx], events_path[idx + 1])

                if new_connection not in primitive_connections:
                    primitive_connections.append(new_connection)

        return primitive_connections

    def build_causal_arcs(self, tree_obj: DiGraph, paths_to_leaves: List[Tuple[List[str], List[str]]], func_events: Dict[str, List[str]], states_and_probabilities: Dict[str, Dict[str, List[float]]]) -> List[Tuple[str, str]]:
        """Build the causal arcs (connections between functional events that actually share a cause-effect relationsship)
        This is based on Bearfield and Marsh 2005 (https://doi.org/10.1007/11563228_5) section 3

        Args:
            tree_obj (DiGraph): Given Event Tree as DiGraph object (real tree)
            paths_to_leaves (List[Tuple[List[str], List[str]]]): All paths from the root of a given tree to each leaf.
            func_events (Dict[str, List[str]]):  Mapping of all functional events in the Event Tree to the unique node IDs
                representing them in the DiGraph tree object.
            states_and_probabilities (Dict[str, Dict[str, List[float]]]): Outcome probabilities for each functional event and its
                respective outcomes/states/branching options in the tree.

        Returns:
            List[Tuple[str, str]]: List of causal node connections in the BayesianNetwork representation of the Event Tree.
        """

        # we need to compare the downstream probabilities of functional event states (see Sec. 3.4 Causal Arc Elemination, Bearfield/Marsh 2005)
        # if we find the probs to be equal -> we can delete the ingoing arc to a functional event in the BN
        # for all nodes where the probs differ, we need to add arcs from all previous, contributing nodes to the BN
        # If a state for a functional event has multiple probabilities then paths leading there are causal contributors (traversed functional nodes are causal parents in the BN)

        causal_node_connections = []

        for func_event_name in func_events:

            if all(len(state_probabilities) == 1 for state_probabilities in states_and_probabilities[func_event_name].values()):
                continue

            # key = contributing func event, value = list<contributing states of that func event>
            contributing_func_events_and_states = {}

            for state in states_and_probabilities[func_event_name]:
                scope = f"{func_event_name}.{state}"

                for clearview_path, id_path in paths_to_leaves:
                    branches = [f"{tree_obj.nodes[node_id]['data'].f_event_name}.{tree_obj.nodes[node_id]['data'].state}" for node_id in id_path if isinstance(tree_obj.nodes[node_id]["data"], Path)]
                    ids = [node_id for node_id in id_path if isinstance(
                        tree_obj.nodes[node_id]["data"], Path)]

                    if scope in branches:
                        upper_pos = id_path.index(ids[branches.index(scope)])
                        contributing_func_events = self.get_filtered_clearview_path(
                            id_path, filter_by=FunctionalEvent, max_index=upper_pos)

                        for contributing_func_event in contributing_func_events:
                            event_pos = clearview_path.index(
                                contributing_func_event)
                            # follow up element is a path object
                            state_node_id = id_path[event_pos + 1]
                            event_state = tree_obj.nodes[state_node_id]["data"].state

                            if contributing_func_event not in contributing_func_events_and_states:
                                contributing_func_events_and_states[contributing_func_event] = [
                                    event_state]

                            else:
                                if event_state not in contributing_func_events_and_states[contributing_func_event]:
                                    contributing_func_events_and_states[contributing_func_event].append([
                                                                                                        event_state])

            # we need to check if multiple choices (e.g. yes, no...) for a contributing func node are present, if yes it is a causal parent
            for contributing_func_event, states in contributing_func_events_and_states.items():
                if len(states) > 1 and func_event_name != contributing_func_event:
                    new_connection = (contributing_func_event, func_event_name)
                    if new_connection not in causal_node_connections:
                        causal_node_connections.append(new_connection)

        return causal_node_connections

    def simplify_causal_arcs(self, tree_obj: DiGraph, node_connections: List[Tuple[str, str]], paths_to_leaves: List[Tuple[List[str], List[str]]], func_events: Dict[str, List[str]], states_and_probabilities: Dict[str, Dict[str, List[float]]]) -> List[Tuple[str, str]]:
        """Remove unneccesary causal arcs.

        Args:
            tree_obj (DiGraph): Given Event Tree as DiGraph object (real tree)
            node_connections (List[Tuple[str, str]]): Causal node connections to check for simplification.
            paths_to_leaves (List[Tuple[List[str], List[str]]]): All paths from the root of a given tree to each leaf.
            func_events (Dict[str, List[str]]):  Mapping of all functional events in the Event Tree to the unique node IDs
                representing them in the DiGraph tree object.
            states_and_probabilities (Dict[str, Dict[str, List[float]]]): Outcome probabilities for each functional event and its
                respective outcomes/states/branching options in the tree.

        Returns:
            List[Tuple[str, str]]: Minimum necc. causal node connections in the BayesianNetwork representation of the Event Tree.
        """
        for func_event_name in func_events:
            if not any(len(state_probabilities) > 1 for state_probabilities in states_and_probabilities[func_event_name].values()):

                # 1) get all node connections that contain the currently scoped func even (dest node)
                affected_connections = [
                    tup for tup in node_connections if tup[1] == func_event_name]

                if len(affected_connections) == 0:
                    continue

                # 2)collect all paths that lead to the current func_event
                paths_to_cur_event = []
                offset_func_event_to_outcome = 2

                for clearview_path, id_path in paths_to_leaves:
                    if func_event_name in clearview_path:
                        upper_pos = clearview_path.index(func_event_name)
                        path_tup = (clearview_path[:upper_pos + offset_func_event_to_outcome],
                                    id_path[:upper_pos + offset_func_event_to_outcome])
                        if path_tup not in paths_to_cur_event:
                            paths_to_cur_event.append(path_tup)

                for src, func_event_name in affected_connections:
                    # 3) Compare conditional probabilities p(dest_x | ..., src, ....)
                    #   this is possible if all paths have the same length == same conditioning set
                    longest_path = len(
                        max((clear for clear, ids in paths_to_cur_event), key=len))
                    if all(len(tup[0]) == longest_path for tup in paths_to_cur_event):

                        paths_to_outcome = {}
                        for clearview_path, id_path in paths_to_cur_event:
                            state = tree_obj.nodes[id_path[-1]]["data"].state
                            if state not in paths_to_outcome:
                                paths_to_outcome[state] = [
                                    (clearview_path, id_path)]
                            else:
                                paths_to_outcome[state].append(
                                    (clearview_path, id_path))

                        # b) paths represent conditional probabilites p(dest_x | ..., src, ....)
                        # all outcome probabilities across paths within a dest state must be equal -> cond. independence
                        outcome_probs = []
                        for state, paths in paths_to_outcome.items():
                            outcome_probs.append(
                                set([tree_obj.nodes[id_path[-1]]["data"].probability for _, id_path in paths]))

                        if all(len(outcome) == 1 for outcome in outcome_probs):
                            # cond. independence -> remove arc
                            node_connections = [
                                tup for tup in node_connections if tup != (src, func_event_name)]

        return node_connections

    def build_consequence_arcs(self, func_events: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        """Build all connections from a functional event to the consequence node in the BayesianNetwork
            representation of the Event Tree.

        Args:
            func_events (Dict[str, List[str]]): Mapping of all functional events in the Event Tree to the unique node IDs
                representing them in the DiGraph tree object.

        Returns:
            List[Tuple[str, str]]: List of all node connections between a functional event and the consequence
                node in the BN version of the Event Tree.
        """
        consequence_connections = []
        # 1) Add all possible consequence connections
        for func_event_name in func_events:
            new_connection = (str(func_event_name), self.consequence_node_name)
            if new_connection not in consequence_connections:
                consequence_connections.append(new_connection)

        return consequence_connections

    def simplify_consequence_arcs(self, consequence_connections: List[Tuple[str, str]], paths_to_leaves: List[Tuple[List[str], List[str]]], func_events: Dict[str, List[str]], consequences: Dict[str, List[Tuple[List[str], List[str]]]]) -> List[Tuple[str, str]]:
        """Build the consequence arcs (connections between a functional event and the consequence node that actually have a cause-effect relationsship)
        This is based on Bearfield and Marsh 2005 (https://doi.org/10.1007/11563228_5) section 3

        Args:
            consequence_connections (List[Tuple[str, str]]): List of all node connections between a functional event and the consequence
                node in the BN version of the Event Tree.
            paths_to_leaves (List[Tuple[List[str], List[str]]]): All paths from the root of a given tree to each leaf.
            func_events (Dict[str, List[str]]): Mapping of all functional events in the Event Tree to the unique node IDs
                representing them in the DiGraph tree object.
            consequences (Dict[str, List[Tuple[List[str], List[str]]]]): Dictionary of all consequences and paths leading to them.

        Returns:
            List[Tuple[str, str]]: List of valid node connections between the functional events and the consequence
                node in the BN version of the Event Tree (i.e. unnecessary arcs removed).
        """
        # Arcs can be deleted if they fullfill:
        # REQ_1) ALL consequences need at least two paths in the parsed tree that lead to it
        # REQ_2) Paths for a consequence need to share the SAME order of functional events that lead to that consequence
        # REQ_3) If REQ_1 and REQ_2: a left side (near the root in the Event Tree) candidate functional event needs to be mitigated by the SAME follow up functional event in ALL paths
        #          if thats the case - the left side candidate func event is irrelevant -> consequence arc can be deleted

        arcs_can_be_simplified = True

        # check REQ_1
        arcs_can_be_simplified = all(
            len(path_options) >= 2 for path_options in consequences.values())

        if arcs_can_be_simplified:
            # check REQ_2
            for _, path_options in consequences.items():
                _, ref_id_path = path_options[0]
                ref_events_path = self.get_filtered_clearview_path(
                    ref_id_path, filter_by=FunctionalEvent)

                for pos in range(1, len(path_options)):
                    _, optional_id_path = path_options[pos]

                    optional_events_path = self.get_filtered_clearview_path(
                        optional_id_path, filter_by=FunctionalEvent)

                    if optional_events_path != ref_events_path:
                        arcs_can_be_simplified = False
                        break

        if arcs_can_be_simplified:
            # check REQ_3
            all_func_events = [str(elem) for elem in func_events]
            for idx, func_event in enumerate(all_func_events):
                if idx == len(all_func_events)-1:
                    break

                arcs_can_be_simplified = True
                next_func_event = all_func_events[idx+1]

                for clearview_path, id_path in paths_to_leaves:
                    events_path = self.get_filtered_clearview_path(
                        id_path, filter_by=FunctionalEvent)

                    if func_event in clearview_path and next_func_event in clearview_path:
                        are_adjacent = (events_path.index(
                            func_event) + 1) == events_path.index(next_func_event)

                        if not are_adjacent:
                            arcs_can_be_simplified = False
                            break

                if arcs_can_be_simplified:
                    obsolete_connection = (
                        func_event, self.consequence_node_name)
                    if obsolete_connection in consequence_connections:
                        consequence_connections = [
                            tup for tup in consequence_connections if tup != obsolete_connection]

        return consequence_connections

    def get_filtered_clearview_path(self, raw_id_path, filter_by: Optional[EtElement] = None, max_index: Optional[int] = None) -> List[str]:
        """Convenience method to get a clearview path (human-readable path element names of the Event Tree)
            from a given ID path (unique ID based element names of the Event Tree-Tree object).

        Args:
            raw_id_path (list<str>): ID path (list of IDs describing a path from the init event to a defined Event Tree object)
            filter_by (Optional[EtElement], optional): Filter based on class derivates of EtElement (i.e. Event Tree objects).
                If applied, the resulting clearview path will only contain objects of a specific type (e.g. only the names of functional events).
            max_index (Optional[int], optional): Step parameter to clip the given raw ID path.

        Returns:
            List[str]: List of human-readable path element names (i.e. clearview path)
        """
        filter_by = filter_by if filter_by is not None and issubclass(
            filter_by, EtElement) else EtElement
        max_index = max_index if max_index is not None else len(raw_id_path)
        result = [self.__tree_obj.nodes[node_id]["data"].name for node_id in raw_id_path[:max_index]
                  if isinstance(self.__tree_obj.nodes[node_id]["data"], filter_by)]
        return result

    def build_bn_model(self) -> None:
        """This method actually parameterizes the BayesianNetwork (i.e creating the CPTs).
            So far we only built the skeleton/topology of the transformed model.
        """
        self.model = BayesianNetwork(
            name=self.name, node_connections=self.node_connections)
        root_cpts = self.build_root_cpts(
            self.model, self.states_and_probabilities)
        conditional_cpts = self.build_conditional_cpts(
            self.model, self.node_connections, self.__tree_obj, self.paths_to_leaves, self.states_and_probabilities)
        consequence_cpts = self.build_consequence_cpt(
            self.node_connections, self.__tree_obj, self.paths_to_leaves, self.consequences, self.states_and_probabilities)

        self.model.add_cpts(
            *(root_cpts | conditional_cpts | consequence_cpts).values())

    def build_root_cpts(self, model: BayesianNetwork,  states_and_probabilities: Dict[str, Dict[str, List[float]]]) -> Dict[str, ConditionalProbabilityTable]:
        """Create all CPTs for root nodes (parentless nodes).

        Raises:
            ValueError: Raised if in pre-processing a dependent node was falsely labled as root.

        Args:
            model (BayesianNetwork): BayesianNetwork representation of the Event Tree (only connections defined).
            states_and_probabilities (Dict[str, Dict[str, List[float]]]): Outcome probabilities for each functional event and its
                respective outcomes/states/branching options in the tree.

        Returns:
            Dict[str, ConditionalProbabilityTable]: Conditional Probability Tables for each root node of the BayesianNetwork
        """
        root_cpts = {}
        for root_name in model.get_root_node_names():
            state_names = []
            values = []
            for state, probabilities in states_and_probabilities[root_name].items():
                state_names.append(state)
                if len(probabilities) != 1:
                    raise ValueError(f"Invalid number of probabilities ({len(probabilities)}) for root variable {root_name} and state {state}.")

                values.extend(probabilities)

            values = np.array(values).reshape(len(state_names), -1)

            root_cpts[root_name] = ConditionalProbabilityTable(name=root_name, variable_card=len(state_names),
                                                               values=values,
                                                               evidence=None, evidence_card=None,
                                                               state_names={root_name: state_names})
        return root_cpts

    def build_conditional_cpts(self, model: BayesianNetwork, node_connections: List[Tuple[str, str]],  tree_obj: DiGraph, paths_to_leaves: List[Tuple[List[str], List[str]]], states_and_probabilities: Dict[str, Dict[str, List[float]]]) -> Dict[str, ConditionalProbabilityTable]:
        """Create all conditional CPTs for functional events that depend on other functional events.

        Args:
            model (BayesianNetwork): BayesianNetwork representation of the Event Tree (connections defined).
            node_connections (List[Tuple[str, str]]): List of (src, dest) tuples describing the node connections
                in the BayesianNetwork representation of the Event Tree.
            tree_obj (DiGraph): Given Event Tree as DiGraph object (real tree)
            paths_to_leaves (List[Tuple[List[str], List[str]]]): All paths from the root of a given tree to each leaf.
            states_and_probabilities (Dict[str, Dict[str, List[float]]]): Outcome probabilities for each functional event and its
                respective outcomes/states/branching options in the tree.

        Returns:
            Dict[str, ConditionalProbabilityTable]: CPTs for the conditional nodes in the Event Tree.
        """
        conditional_cpts = {}
        conditional_nodes = set(model.model_elements.keys()) ^ set(model.get_root_node_names())
        conditional_nodes.remove(self.consequence_node_name)

        for conditional_node_name in conditional_nodes:
            conditional_states = list(states_and_probabilities[conditional_node_name].keys())
            state_names = {conditional_node_name: conditional_states}

            parents = []
            parent_cards = []
            for src, dest in node_connections:
                if dest == conditional_node_name:
                    if src not in parents:
                        parents.append(src)
                        state_names[src] = list(
                            states_and_probabilities[src].keys())
                        parent_cards.append(
                            len(states_and_probabilities[src].keys()))

            # 1) collect all paths containing the parents up to the conditional node
            relevant_paths = []
            for clearview_path, id_path in copy.deepcopy(paths_to_leaves):
                if conditional_node_name in clearview_path:
                    relevant_paths.append((clearview_path, id_path))

            # 2) filter candidate paths to have unique paths up to the conditional nodes state
            for cnt, paths in enumerate(relevant_paths):
                clearview_path, id_path = paths

                indices = [(parent, clearview_path.index(
                    parent) if parent in clearview_path else float(np.inf)) for parent in parents]
                first_func_event_idx = min(indices, key=lambda t: t[1])

                start_idx = first_func_event_idx[1]
                # +1 since we need the path element behind the func event
                end_idx = clearview_path.index(conditional_node_name) + 1

                # +1 since upper bound is not included in slicing
                relevant_paths[cnt] = (
                    clearview_path[start_idx:end_idx+1], id_path[start_idx:end_idx+1])

            unique_paths = [next(g) for _, g in itertools.groupby(
                relevant_paths, key=lambda x:x[0])]

            # 3) extract the probabilities for the different parent state configurations as index tuple
            #                        e.g.                 ( (0            ,          2         ...          1         ,               0)   ,  0.7)
            # in the form           tuple< tuple<conditional_node_state_x, parent x_i_state_x , parent_x_j_state_x ... parent_x_m_state_x>,  prob >
            probability_combinations = []

            for clearview_path, id_path in unique_paths:
                # <conditional_node_state_x, parent x_i_state_x , parent_x_j_state_x ... parent_x_m_state_x>
                state_config = []
                dont_care_placeholder = -999

                for parent in parents:
                    if parent not in clearview_path:
                        # dummy value as this a don't care node -> tuple needs to be duplicated later on
                        state_config.append(dont_care_placeholder)
                        continue

                    parent_pos = clearview_path.index(parent)
                    parent_state_name = tree_obj.nodes[id_path[parent_pos + 1]
                                                       ]["data"].state
                    parent_state_index = state_names[parent].index(parent_state_name)
                    state_config.append(parent_state_index)

                conditional_node_pos = clearview_path.index(conditional_node_name)
                conditional_state_name = tree_obj.nodes[id_path[conditional_node_pos + 1]]["data"].state
                conditional_state_index = state_names[conditional_node_name].index(conditional_state_name)

                conditional_prob = tree_obj.nodes[id_path[conditional_node_pos + 1]]["data"].probability
                conditional_prob = conditional_prob if conditional_prob else 0.0
                state_config.insert(0, conditional_state_index)

                if dont_care_placeholder in state_config:
                    for dont_care_pattern in self.possible_patterns(state_config, placeholder=dont_care_placeholder):
                        probability_combinations.append(
                            (tuple(dont_care_pattern), conditional_prob))
                else:
                    probability_combinations.append((tuple(state_config), conditional_prob))

            conditional_cpts[conditional_node_name] = self.generate_cpt(probabilities=dict(probability_combinations),
                                                                        cpt_name=conditional_node_name,
                                                                        cpt_card=len(conditional_states),
                                                                        evidence=parents,
                                                                        ev_cards=parent_cards,
                                                                        state_names=state_names,
                                                                        fill_in=True)

        return conditional_cpts

    def possible_patterns(self, data: List[int], fill_in: Optional[List[int]] = None, placeholder: Optional[int] = -999) -> List[int]:
        """Helper function to create all combinations for a given list of values, where some values are fixed.
            Variable elements are filled in the generated combinatorics with the given fill-in values.

            E.g.: data = [0, -999, 1] with fill-ins [a, b] generates [0, a, 1] and [0, b, 1]

        Args:
            data (List[int]): List of values with fixed elements and missing entries that can be filled
                              in via combinatorics
            fill_in (List[int], optional): Fill-in values. These are used to fill-in the variable elements
                              of the provided data.
            placeholder (int, optional): Flag indicating which entries of data are variable

        Yields:
            List[int]: Valid, filled-in combination for missing data.
        """

        if fill_in is None:
            fill_in = [0, 1]

        bad_indices = [i for i, bit in enumerate(data) if bit == placeholder]

        for replacement in itertools.product(fill_in, repeat=len(bad_indices)):
            for index, bit in zip(bad_indices, replacement):
                data[index] = bit
            yield data

    def build_consequence_cpt(self, node_connections: List[Tuple[str, str]], tree_obj: DiGraph,  paths_to_leaves: List[Tuple[List[str], List[str]]], consequences: Dict[str, List[Tuple[List[str], List[str]]]], states_and_probabilities: Dict[str, Dict[str, List[float]]]) -> Dict[str, ConditionalProbabilityTable]:
        """Create the CPT for the consequence node. Consequences of the Event Tree are states of this node.

        Args:
            node_connections (List[Tuple[str, str]]): List of (src, dest) tuples describing the node connections
                in the BayesianNetwork representation of the Event Tree.
            tree_obj (DiGraph): Given Event Tree as DiGraph object (real tree)
            paths_to_leaves (List[Tuple[List[str], List[str]]]): All paths from the root of a given tree to each leaf.
            consequences (Dict[str, List[Tuple[List[str], List[str]]]]): Dictionary of all consequences and paths leading to them.
            states_and_probabilities (Dict[str, Dict[str, List[float]]]): Outcome probabilities for each functional event and its
                respective outcomes/states/branching options in the tree.

        Returns:
            Dict[str, ConditionalProbabilityTable]: CPT for the consequence node in the Event Tree.
        """
        consequence_cpts = {}
        consequences = list(consequences.keys())
        state_names = {self.consequence_node_name: consequences}
        parents = []
        parent_cards = []

        for src, dest in node_connections:
            if dest == self.consequence_node_name:
                parents.append(src)
                state_names[src] = list(states_and_probabilities[src].keys())
                parent_cards.append(len(states_and_probabilities[src].keys()))

        # 1) generate all possible parent state configurations that lead to a consequence
        #                        e.g.                 ( (0            ,          2         ...          1         ,               0)   ,  0.7)
        # in the form           tuple< tuple<conditional_node_state_x, parent x_i_state_x , parent_x_j_state_x ... parent_x_m_state_x>,  prob >
        #      fix the index of a participating func event for the index configuration tuple instead of cycling over the full range
        #      fix the state of the specific consequence
        probability_combinations = []

        for clearview_path, id_path in paths_to_leaves:
            contributing_parents = []
            contributing_states = []

            # -1 since the last element is a consequence
            # step 2 since the path element order will be: func event, path obj, func event, path obj...
            start_offset = [id_path.index(node_id) for node_id in id_path if isinstance(tree_obj.nodes[node_id]["data"], FunctionalEvent)][0]
            for idx in range(start_offset, len(clearview_path) - 1, 2):
                cur_func_event = clearview_path[idx]
                if cur_func_event in parents:
                    contributing_parents.append(cur_func_event)

                    func_event_state_name = tree_obj.nodes[id_path[idx+1]
                                                           ]["data"].state
                    func_event_state_index = state_names[cur_func_event].index(
                        func_event_state_name)
                    contributing_states.append(func_event_state_index)

            # now we know which contributing parents we need to fix in which state
            # generate all relevant state index combinations
            ranges = [[consequences.index(clearview_path[-1])]]
            for parent in parents:
                if parent in contributing_parents:
                    ranges.append([contributing_states[contributing_parents.index(parent)]])
                else:
                    ranges.append(range(parent_cards[parents.index(parent)]))

            for state_combination in itertools.product(*ranges):
                probability_combinations.append((tuple(state_combination), 1.0))

        consequence_cpts[self.consequence_node_name] = self.generate_cpt(probabilities=dict(probability_combinations),
                                                                         cpt_name=self.consequence_node_name,
                                                                         cpt_card=len(
                                                                             consequences),
                                                                         evidence=parents,
                                                                         ev_cards=parent_cards,
                                                                         state_names=state_names,
                                                                         fill_in=False)

        return consequence_cpts

    def generate_cpt(self, probabilities: Dict[Tuple[int, ...], float],  cpt_name: str, cpt_card: int, evidence: List[str], ev_cards: List[int], state_names: Dict[str, List[str]], fill_in: Optional[bool] = False) -> ConditionalProbabilityTable:
        """Helper method to instantiate and populate a CPT based on (some)
            probability values for specific state index combinations.

        Args:
            probabilities (Dict[Tuple[int, ...], float]): Dictionary that maps a state combination (as indices) of the CPT to a probability.
            cpt_name (str): Name of the to be created CPT.
            cpt_card (int): Cardinality of the to be created CPT.
            evidence (List[str]): List of parents/evidence for this CPT.
            ev_cards (List[int]): List of parent/evidence cardinalities for this CTP.
            state_names (Dict[str, List[str]]): Mapping of the participating variables to their respective state names.
            fill_in (Optional[bool], optional): Indicating if a missing probability value should be treated as
                0.0 or "don't care" value calculated as non-informative prior.

        Returns:
            ConditionalProbabilityTable: Generated and populated CPT.
        """
        values = []
        state_combinations = self.generate_all_state_combinations(
            cpt_card=cpt_card, parent_cards=ev_cards)

        not_found_fill_in = 1 / cpt_card if fill_in else 0.0

        for state_combination in state_combinations:
            index_probability = probabilities.get(
                state_combination, not_found_fill_in)
            values.append(index_probability)

        # reshape the values array to conform to the CPT layout as specified by the cardinality
        values = np.array(values).reshape(cpt_card, -1)

        return ConditionalProbabilityTable(name=cpt_name, variable_card=cpt_card,
                                           values=values, evidence=evidence,
                                           evidence_card=ev_cards,
                                           state_names=state_names)

    def generate_all_state_combinations(self, cpt_card: int, parent_cards: List[int]) -> List[Tuple[int, ...]]:
        """Generate a list of all possible state index combinations for a given cardinality.
            The resulting combinatorics define the look-up table for a factor/CPT as each element will
            index a probability for a given state configuration.
            With this approach we cycle through the first parent card the slowest and the last parent card the fastest.

        Args:
            cpt_card (int): Cardinality of the factor/cpt table.
            parent_cards (List[int]): List of cardinalities for contributing factors.

        Returns:
            List[Tuple[int, ...]]: All possible combinations of indices for given cardinalities.
        """
        cards = [cpt_card, *parent_cards]
        combination_ranges = [range(elem) for elem in cards]
        state_combinations = list(itertools.product(*combination_ranges))

        return state_combinations
