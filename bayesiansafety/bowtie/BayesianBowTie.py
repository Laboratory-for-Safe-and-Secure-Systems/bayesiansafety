"""
This class allows defining and working with Bow-Tie Diagrams
consisting of one Faul Tree feeding into one Event Tree.
"""
import copy
from typing import Dict, Optional

import numpy as np
from bayesiansafety.core import ConditionalProbabilityTable
from bayesiansafety.core.inference import InferenceFactory
from bayesiansafety.eventtree import BayesianEventTree
from bayesiansafety.faulttree import BayesianFaultTree
from bayesiansafety.faulttree import FaultTreeLogicNode



class BayesianBowTie:
    """Main class managing a (bayesian) Bow-Tie  Diagram.

    Attributes:
        causal_arc_et_nodes (dict<str, str>): Dictionary with key = name of node that is also affected from the pivot node.
                                              These nodes require a causal arc between them and the pivot node.
                                              Value = name of state the node will be in in case of the initiating event not happenening ("failing/off-state").
        et_model (BayesianEventTree): Event Tree instance representing the right side of the Bow-Tie
        ft_model (BayesianFaultTree): Fault Tree instance representing the left side of the Bow-Tie
        model (BayesianNetwork): Bow-Tie model represented as DiGraph instance (networX DiGraph)
        model_elements (dict<str, object>): Lookup dictionary containing all nodes as instances of either FaultTree*Node
                                                    or Event Tree node keyed by their name.
        name (str): Name of this Bow-Tie model
        node_connections (list<tuple<str, str>>): List of tuples defining the edges between model elements.
        pivot_node (str): Pivot node (i.e. trigger for the initiating event). If no pivot node is given
                                        by the user the Top-Level event of the Fault Tree is assumed as pivot node.
    """
    causal_arc_et_nodes = None
    et_model = None
    ft_model = None
    model = None
    model_elements = None
    name = None
    node_connections = None
    pivot_node = None
    triggering_state = None

    def __init__(self, name: str, bay_ft: BayesianFaultTree, bay_et: BayesianEventTree, pivot_node: Optional[str] = None, triggering_state: Optional[str] = None, causal_arc_et_nodes: Optional[Dict[str, str]] = None) -> None:
        """Ctor of the BayesianBowTie class.

        Args:
            name (str): Name of this Bow-Tie instance
            bay_ft (BayesianFaultTree): Fault Tree instance representing the left side of the Bow-Tie
            bay_et (BayesianEventTree): Event Tree instance representing the right side of the Bow-Tie
            pivot_node (str,, optional): Pivot node name (i.e. trigger for the initiating event).
                                        If no pivot node is given the Top-Level event of the Fault Tree is assumed as pivot node.
            triggering_state (str, optional): State of the pivot node that triggers the initiating event. In case of the TLE "failing"
                                        is used as default state. If for other nodes no state is given, last state of the pivot node is used.
            causal_arc_et_nodes (dict<str, str>): Dictionary with key = name of node that is also affected from the pivot node.
                                        These nodes require a causal arc between them and the pivot node.
                                        Value = name of state the node will be in in case of the initiating event not happenening ("failing/off-state").
        """
        self.name = name
        self.pivot_node = pivot_node if pivot_node else bay_ft.get_top_level_event_name()
        self.triggering_state = triggering_state
        self.ft_model = bay_ft
        self.et_model = bay_et
        self.causal_arc_et_nodes = causal_arc_et_nodes
        self.__build_model()

    def __build_model(self):
        """Setup method to initialize instances of BayesianBowTie.
            Fault Tree and Event Tree need to be merged. A pivot node (node in the Faul Tree)
            serves as linking element and triggers the initiating event in the Event Tree.
            As described by Khakzad et al. 2013 (https://doi.org/10.1016/j.psep.2012.01.005),
            a causal arc between the pivot node and the consequence node of the Event Tree needs to be added.
            This renders the pivot node a parent of the consequence node and requries adjusting the
            laters CPT. This includes adding an additional "safe" state.

        Raises:
            ValueError: Raised if the used pivot node is not part of the given Faul Tree instance.

        """
        if self.pivot_node in self.ft_model.model_elements.keys():
            bowtie_bn = self.ft_model.model.copy()
            bowtie_bn.add_edges_from(self.et_model.model.node_connections)
            bowtie_bn.add_cpts(*self.et_model.model.model_elements.values())

            consequence_node = self.et_model.get_consequence_node_name()

            # this should be done for all potential "new" causal dependencies that result from linking the pivot node (i.e. effects on functional events)
            # for now we expected that P(func_ev_i | pivot_node) == P(func_ev_i | ~pivot_node) (See Khakzad 2013, section 2.3.3)
            # therfore we only add the causal arc from the pivot node to the consequence node and adjust the CPT of the later
            updated_cpt = self.__adjust_dependent_node(triggering_cpt=self.ft_model.model_elements[self.pivot_node].cpt.copy(
            ), affected_cpt=self.et_model.model_elements[consequence_node].copy())
            causal_edge = (self.pivot_node, consequence_node)
            bowtie_bn.add_edges_from([causal_edge])
            bowtie_bn.model_elements[consequence_node] = updated_cpt

            self.node_connections = [causal_edge]
            self.node_connections.extend(self.et_model.model.node_connections)
            self.node_connections.extend(self.ft_model.model.node_connections)
            self.model = bowtie_bn

            # we want to keep the reference to the actual objects of each tree - not just the CPTs
            self.model_elements = copy.deepcopy(
                self.ft_model.model_elements | self.et_model.model_elements)
            self.model_elements[consequence_node].cpt = bowtie_bn.model_elements[consequence_node]

            # handle additional nodes
            #  P(func_ev_i | pivot_node) != P(func_ev_i | ~pivot_node) (See Khakzad 2013, section 2.3.3)
            if self.causal_arc_et_nodes:
                for additonal_node, failing_state_affected_node in self.causal_arc_et_nodes.items():
                    updated_cpt = self.__adjust_dependent_node(triggering_cpt=self.ft_model.model_elements[self.pivot_node].cpt.copy(),
                                                               affected_cpt=self.et_model.model_elements[additonal_node].copy(
                    ),
                        failing_state_affected_node=failing_state_affected_node)
                    causal_edge = (self.pivot_node, additonal_node)
                    bowtie_bn.add_edges_from([causal_edge])
                    bowtie_bn.model_elements[additonal_node] = updated_cpt

                    self.model_elements[additonal_node].cpt = bowtie_bn.model_elements[additonal_node]

        else:
            raise ValueError(f"Bow-Tie can not be built. Pivot node {self.pivot_node} not found in the given Fault Tree instance.")

    def copy(self) -> "BayesianBowTie":
        """Helper method to make a deep copy of this instance.

        Returns:
            BayesianBowTie: Returns deep copy of this instance.
        """
        return copy.deepcopy(self)

    def get_elem_by_name(self, node_name: str) -> FaultTreeLogicNode:
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


    def get_consequence_likelihoods(self) -> ConditionalProbabilityTable:
        """Returns the prior marginal probabilities of the consequences

        Returns:
            DiscreteFactor: Factor containing the prior marginal probabilities of the consequence node.
        """
        inference_engine = InferenceFactory(self.model).get_engine()
        return inference_engine.query(variables=self.et_model.get_consequence_node_name())

    def __adjust_dependent_node(self, triggering_cpt: ConditionalProbabilityTable, affected_cpt: ConditionalProbabilityTable, failing_state_affected_node: Optional[str] = None):
        """By linking a pivot node (trigger) to the Event Tree a causal arc needs to be added
            between the pivot node and the consequence node of the BN.
            This represents the influence of the pivot node acting as trigger.
            Due to this, the pivot node acts as parent of the consequence node
            and therefore an adjustment of the laters CPT is required.
            Additionally a "safe" state is added, indicating the absence of a triggering event.

            Currently we only assume TLE -> consequence_node.
            As described by Khakzad et al., if the pivot node affects not only the initiating event
            additional causal arcs need to be added (pivot node -> functional even_i / safety barrier_i).
            These nodes are given at construction via self.causal_arc_et_nodes.

        Args:
            triggering_cpt (ConditionalProbabilityTable): CPT of the pivot node (in the Faul Tree)
            affected_cpt (ConditionalProbabilityTable): CPT of the consequence/target node (in the Event Tree)
            failing_state_affected_node (str, optional): Name of state the affected node that will take, in case of the initiating event not happenening ("failing/off-state").

        Returns:
            ConditionalProbabilityTable: Modified CPT of the affected node.

        Raises:
            ValueError: Raised if triggering node (pivot node) not in Fault Tree or triggering state (of the initiating event) is invalid.
        """
        # The trigger node cycles the slowest in the modified CPT
        # the self.triggering_state leads (default="failing") to the original Event Tree CPT entries
        # the state "working"/"no TLE" leads to the "safe state"

        if triggering_cpt.name not in self.ft_model.model_elements.keys():
            raise ValueError(f"Requested pivot node {triggering_cpt.name} not in Fault Tree - can't add causal link {str(triggering_cpt.name) + ',' + str(affected_cpt.name) } to Event Tree.")

        if self.triggering_state and self.triggering_state not in triggering_cpt.state_names[triggering_cpt.name]:
            raise ValueError(f"Given triggering state {self.triggering_state} for {triggering_cpt.name} not valid.")

        if affected_cpt.name is self.et_model.get_consequence_node_name():
            new_var_card = affected_cpt.variable_card + 1  # for the safe state
            new_evidence = affected_cpt.evidence
            new_evidence.insert(0, triggering_cpt.name)

            new_ev_card = affected_cpt.evidence_card
            new_ev_card.insert(0, triggering_cpt.variable_card)

            new_state_names = affected_cpt.state_names
            new_state_names[triggering_cpt.name] = triggering_cpt.state_names.get(
                triggering_cpt.name, None)
            new_state_names[affected_cpt.name].append("safe")

            # currently the values of a Faul Tree node represent[ [working], [failing]]
            # and need to be updated to [000....0, orig values], .... [111....1, orig values] for the newly added state "safe"
            new_values = affected_cpt.values
            new_values = np.vstack([new_values, np.zeros(new_values.shape[1])])
            extension = np.array(affected_cpt.values)
            extension[:] = 0
            extension = np.vstack([extension, np.ones(extension.shape[1])])
            new_values = np.concatenate((extension, new_values), axis=1)

            return ConditionalProbabilityTable(name=affected_cpt.name,
                                               variable_card=new_var_card,
                                               values=new_values,
                                               evidence=new_evidence,
                                               evidence_card=new_ev_card,
                                               state_names=new_state_names)
        else:
            # we deal with an additionally affected (event) node
            new_var_card = affected_cpt.variable_card
            new_evidence = affected_cpt.evidence if affected_cpt.evidence else []
            new_evidence.insert(0, triggering_cpt.name)

            new_ev_card = affected_cpt.evidence_card if affected_cpt.evidence_card else []
            new_ev_card.insert(0, triggering_cpt.variable_card)

            new_state_names = affected_cpt.state_names
            new_state_names[triggering_cpt.name] = triggering_cpt.state_names.get(
                triggering_cpt.name, None)
            new_values = affected_cpt.values
            extension = np.array(affected_cpt.values)
            extension[:] = 0
            idx_state_affected_node = extension.shape[0]-1 if not failing_state_affected_node else affected_cpt.state_names[affected_cpt.name].index(
                failing_state_affected_node)
            extension[idx_state_affected_node::] = 1

            # based on self.triggering_state the concatenation changes
            # currently we expect binary nodes in the Fault Tree only - so switching is easy since bayesianeventtree.FaultTree*Nodes have state "failing" by construction
            pos_triggering_state = triggering_cpt.state_names[triggering_cpt.name].index(
                self.triggering_state) if self.triggering_state else triggering_cpt.state_names[triggering_cpt.name].index("failing")
            if pos_triggering_state == 1:
                new_values = np.concatenate((extension, new_values), axis=1)
            else:
                new_values = np.concatenate((new_values, extension), axis=1)

            return ConditionalProbabilityTable(name=affected_cpt.name,
                                               variable_card=new_var_card,
                                               values=new_values,
                                               evidence=new_evidence,
                                               evidence_card=new_ev_card,
                                               state_names=new_state_names)
