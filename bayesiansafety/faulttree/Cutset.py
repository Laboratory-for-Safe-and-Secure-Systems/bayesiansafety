"""This class allows in conjunction with FaultTree*Nodes and BayesianFaultTree to
generte minimal cutests for Fault Tree analysis.
"""
import collections
from typing import List, Set, Dict, Tuple
import networkx as nx

from bayesiansafety.faulttree.FaultTreeLogicNode import FaultTreeLogicNode
from bayesiansafety.faulttree.FaultTreeProbNode import FaultTreeProbNode
from bayesiansafety.faulttree.BayesianFaultTree import BayesianFaultTree


class Cutset:

    """This class allows in conjunction with FaultTree*Nodes and BayesianFaultTree to
        generte minimal cutests for Fault Tree analysis.

    Attributes:
        bayFTA (BayesianFaultTree): Instance of the associated BayesianFaultTree class
        cutsets (tuple<str, list>): Evaluated cutsets given as (used algorithm, list of cutsets)
    """

    bayFTA = None
    cutsets = None
    __graph = None

    def __init__(self, bayFTA: BayesianFaultTree) -> None:
        """Ctor of this class.

        Args:
            bayFTA (BayesianFaultTree): Instance of the associated BayesianFaultTree class
        """
        self.bayFTA = bayFTA
        self.__graph = nx.DiGraph(bayFTA.node_connections)

    def get_minimal_cuts(self, algorithm: str = "fatram") -> List[Set[str]]:
        """Main methode to generate minimal cutsets. Currently two algorithms are implemented:
            Method of obtaining cutsets ('mocus') or Fault Tree Reduction Algorithm ('fatram') (default).
            Fatram algorithm is derived from DOI: 10.1109/TR.1978.5220353.

        Args:
            algorithm (str, optional): Cutset algorithm to use. 'fatram' (default) should be used due to mem. and time saving.

        Returns:
            tuple<str, list<set<str>>: Evaluated cutsets given as tuple (used algorithm, list of cutsets).

        Raises:
            ValueError: Raised if requested algorithm is not implement. Currently only 'mocus' and 'fatram' are supported.

        """

        if algorithm.lower() not in ["mocus", "fatram"]:
            raise ValueError(f"Unsupported algorithm: {algorithm.lower()}. Currently only 'fatram' and 'mocus' are supported")

        cutsets = []
        unresolved_OR_gates = []
        idx_node_occurance_in_cutsets = collections.defaultdict(set)
        indexer = 0

        logic_nodes_hierarchy, sub_branches = self.__get_hierarchical_layout()

        # add TLE as a starting point
        cutsets.append(set([self.bayFTA.get_top_level_event_name()]))
        idx_node_occurance_in_cutsets[self.bayFTA.get_top_level_event_name()].add(
            indexer)
        indexer += 1

        for level, nodes in logic_nodes_hierarchy.items():

            if algorithm.lower() == 'fatram':
                # FATRAM Rule 2) we need to resolve AND gates first
                nodes.sort(
                    key=lambda x: self.bayFTA.model_elements[x].get_node_type(), reverse=False)

            for logic_node in nodes:
                if self.bayFTA.model_elements[logic_node].get_node_type() == 'OR':

                    if algorithm.lower() == "fatram":
                        if all(isinstance(self.bayFTA.model_elements[sub], FaultTreeProbNode) for sub in sub_branches[logic_node]):
                            # FATRAM Rule 2) if an OR gate only contains basic events - discard it for later
                            unresolved_OR_gates.append(logic_node)
                            continue

                    # Duplicate scoped gate with all combinations
                    copy_positions = (
                        len(sub_branches[logic_node])-1) * list(idx_node_occurance_in_cutsets[logic_node])
                    for pos in copy_positions:
                        current_set_to_copy = cutsets[pos]
                        cutsets.append(current_set_to_copy.copy())
                        idx_node_occurance_in_cutsets[logic_node].add(indexer)
                        indexer += 1

                    # Replace scoped gate with sub branch elements
                    reps = int(
                        len(idx_node_occurance_in_cutsets[logic_node]) / len(sub_branches[logic_node]))
                    replacements = sorted(sub_branches[logic_node] * reps)
                    set_replacements = dict(
                        zip(idx_node_occurance_in_cutsets[logic_node], replacements))

                    for pos, replacement_value in set_replacements.items():
                        cutsets[pos].remove(logic_node)
                        cutsets[pos].add(replacement_value)

                        # the duplicate step also duplicates other than the scoped elements - and we need to update their index
                        for node in cutsets[pos]:
                            idx_node_occurance_in_cutsets[node].add(pos)

                if self.bayFTA.model_elements[logic_node].get_node_type() == 'AND':
                    # Replace scoped gate by expansion with sub branch elements
                    for pos in idx_node_occurance_in_cutsets[logic_node]:
                        cutsets[pos].remove(logic_node)
                        cutsets[pos] = cutsets[pos].union(
                            set(sub_branches[logic_node]))

                        # add indizes for newly added elements
                        for set_elem in cutsets[pos]:
                            idx_node_occurance_in_cutsets[set_elem].add(pos)

                # gate was process - remove scoped gate
                del idx_node_occurance_in_cutsets[logic_node]

        if algorithm.lower() == "fatram":
            # FATRAM Rule 3) remove all supersets
            cutsets = self.__get_minimal_subsets(cutsets)
            idx_node_occurance_in_cutsets, indexer = self.__rebuild_indexer_fatram(
                cutsets)

            # we should only be left with unresolved OR gates
            # this means OR gates with basic events as inputs only
            repeated_basic_events = []
            for gate in unresolved_OR_gates:
                repeated_basic_events += sub_branches[gate]

            # we only want unique values ;)
            repeated_basic_events = set(repeated_basic_events)
            for basic_event in repeated_basic_events:
                if isinstance(self.bayFTA.model_elements[basic_event], FaultTreeLogicNode):
                    raise ValueError(f"Logic node: {basic_event} in unresolved basic events!")

                # FATRAM Rule 4.a) The repeated event replaces all unresolved gates of which it is an input to form new sets
                for unresolved_gate in unresolved_OR_gates:
                    if basic_event in sub_branches[unresolved_gate]:
                        copy_positions = idx_node_occurance_in_cutsets[unresolved_gate]
                        for pos in copy_positions:

                            # FATRAM Rule 4.b) These new sets are added to the collection
                            current_set_to_copy = cutsets[pos]
                            cutsets.append(current_set_to_copy.copy())
                            cutsets[indexer].remove(unresolved_gate)
                            cutsets[indexer] = cutsets[indexer].union(
                                set([basic_event]))
                            indexer += 1

                        # FATRAM Rule 4.c) This event is removed as an input from the appropiate gate(s)
                        sub_branches[unresolved_gate] = [
                            value for value in sub_branches[unresolved_gate] if value != basic_event]

                        # FATRAM Rule Rule 4.d) Supersets are removed
                        cutsets = self.__get_minimal_subsets(cutsets)
                        idx_node_occurance_in_cutsets, indexer = self.__rebuild_indexer_fatram(
                            cutsets)

            # FATRAM Rule 5) Resolve the remaining OR gates
            cutsets = [sub_set for sub_set in cutsets if len(
                sub_set.intersection(set(unresolved_OR_gates))) == 0]

        # reduce to minimal sets and we are done
        self.cutsets = (algorithm, sorted(
            self.__get_minimal_subsets(cutsets), key=len))
        return self.cutsets[1]

    def __get_hierarchical_layout(self) -> Dict[int, List[str]]:
        """Helper method to get hierarchical layout of Fault Tree.
            This class will basically collect nodes for each layer of the Faul Tree with levl 0 being the top level event.

        Returns:
            dict<int, list<str>>: Dictionary keyed by the level and values being a list of nodes on this level.
        """
        sub_branches = {}

        # traverse tree once according to it's hierarchy and store order of logic nodes
        # we need to reverse the tree to make the top level event the first node
        logic_nodes_hierarchy = nx.single_source_shortest_path_length(
            self.__graph.reverse(), self.bayFTA.get_top_level_event_name())

        # kick out all basis events
        logic_nodes_hierarchy = {elem: level for elem, level in logic_nodes_hierarchy.items(
        ) if isinstance(self.bayFTA.get_elem_by_name(elem), FaultTreeLogicNode)}

        for node in logic_nodes_hierarchy.keys():
            sub_branches[node] = list(elem for elem in self.__graph.predecessors(node))

        inversed = collections.defaultdict(list)
        for node, level in logic_nodes_hierarchy.items():
            inversed[level].append(node)

        logic_nodes_hierarchy = inversed

        return logic_nodes_hierarchy, sub_branches

    def __rebuild_indexer_fatram(self, cutsets: List[Set[str]]) -> Tuple[Dict[str, Set[str]], int]:
        """Helper method for the FATRAM algortihm. Due to the algo it is neccessary to get minimal sets cyclically.
            The algo is implementd to use an indexer pointing for each node at the cutset it is contained in.
            The reduction step destroys these positions - the indexer needs to be rebuilt.

        Args:
            cutsets (list<set>): List of current cutsets.

        Returns:
            tuple<dict<str, set>, int>: Dictionary keyed by node name with a list of positions at which this node can be found in the list of given cutsets
                and new indexer position.
        """
        idx_node_occurance_in_cutsets = collections.defaultdict(set)
        for pos, sub_set in enumerate(cutsets):
            for node in sub_set:
                idx_node_occurance_in_cutsets[node] = idx_node_occurance_in_cutsets[node].union(set([pos]))

        return idx_node_occurance_in_cutsets, len(cutsets)

    def __get_minimal_subsets(self, sets: List[Set[str]]) -> List[Set[str]]:
        """Helper method to get minimal sets from list of candidate sets

        Args:
            sets (list<sets>): List of candidate sets.

        Returns:
            list<set>: List of minimal subsets.
        """
        sets = sorted(map(set, sets), key=len)
        minimal_subsets = []
        for s in sets:
            if not any(minimal_subset.issubset(s) for minimal_subset in minimal_subsets):
                minimal_subsets.append(s)

        return minimal_subsets
