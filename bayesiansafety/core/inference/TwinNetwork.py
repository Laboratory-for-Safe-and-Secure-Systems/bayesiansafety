"""This class is intended to bundle methodologies for twin-network creation.
The main goal is to build a twin-network from a fully specified
core.BayesianNetwork.BayesianNetwork instance.
Additional support of node-merging and other optimaziations are planned.
(e.g. see https://api.semanticscholar.org/CorpusID:211029267)

Currently all root nodes are considered as exogenous.
For more information on twin-networks see Pearl 2009, Ch. 7.
"""
import copy
from typing import List

from bayesiansafety.core import BayesianNetwork
from bayesiansafety.core import ConditionalProbabilityTable


class TwinNetwork:
    """This class is intended to bundle methodologies for twin-network creation.
    The main goal is to build a twin-network from a fully specified
    core.BayesianNetwork.BayesianNetwork instance.
    Additional support of node-merging and other optimaziations are planned.
    (e.g. see https://api.semanticscholar.org/CorpusID:211029267)

    Currently all root nodes are considered as exogenous.
    For more information on twin-networks see Pearl 2009, Ch. 7.

    Attributes:
        TWIN_NET_PREFIX (str): Prefix string (default = "tw_") that will be used to re-name twin-network nodes.
    """

    TWIN_NET_PREFIX = "tw_"

    @staticmethod
    def build_twin_network(model: BayesianNetwork) -> BayesianNetwork:
        """Create a twin-network based on the passed model.
            Root nodes are interpreted as exogenous variables.

        Args:
            model (core.BayesianNetwork): Network for which a twin should be generated.

        Returns:
            core.BayesianNetwork: Twin-network of the given model.
        """
        twin_model = model.copy()

        root_vars = twin_model.get_root_node_names()
        endogenous_nodes = [twin_model.model_elements[endo]
                            for endo in twin_model.model_elements.keys() if endo not in root_vars]

        twin_nodes = [TwinNetwork.create_twin_node(
            endo, root_vars) for endo in endogenous_nodes]
        twin_connections = []
        for src, dest in twin_model.node_connections:

            if src in root_vars:
                twin_connections.append(
                    (src, TwinNetwork.TWIN_NET_PREFIX + dest))
            else:
                twin_connections.append(
                    (TwinNetwork.TWIN_NET_PREFIX + src, TwinNetwork.TWIN_NET_PREFIX + dest))

        twin_model.add_edges_from(twin_connections)
        twin_model.add_cpts(*twin_nodes)

        return twin_model

    @staticmethod
    def create_twin_node(node: ConditionalProbabilityTable, exo_evidence: List[str] = []) -> ConditionalProbabilityTable:
        """Create a twin-node for the given node.
            The twin node as well as it's endogenous evidences will be prefixed with the
            current value of TwinNetwork.TWIN_NET_PREFIX.

        Args:
            node (core.ConditionalProbabilityTable): Node of a network (i.e. CPT) which should be duplicated.
            exo_evidence (list, optional): All evidence vars are considered endogenous and will be prefixed in the copy,
                since it is expected to have these twin evidence nodes present in a twin-network.
                Since exogenous nodes are singletons in a twin-network (won't get duplicated), variables depending on them
                need to keep the original evidence names. "exo_evidence" acts as a whitelist to specify which evidence vars
                will not be prefixed in the copy.

        Returns:
            twin_node (core.ConditionalProbabilityTable): Duplicated "twin" of the given node.
        """
        node_evidence = copy.deepcopy(
            node.evidence) if node.evidence is not None else []
        node_evidence_card = copy.deepcopy(
            node.evidence_card) if node.evidence_card is not None else []

        twin_state_names = {}
        twin_state_names[TwinNetwork.TWIN_NET_PREFIX +
                         node.name] = copy.deepcopy(node.state_names[node.name])
        for evidence in node_evidence:
            if evidence in (set(exo_evidence) ^ set(node_evidence)):
                twin_state_names[TwinNetwork.TWIN_NET_PREFIX +
                                 str(evidence)] = copy.deepcopy(node.state_names[evidence])
            else:
                twin_state_names[evidence] = copy.deepcopy(
                    node.state_names[evidence])

        # Note order of evidence is important!
        # CPT values are based on this order - if a set is used the twin-node evidence order might get messed up
        # leading to wrong results
        twin_evidence = []
        for evidence in node_evidence:
            if evidence in exo_evidence:
                twin_evidence.append(evidence)
            else:
                twin_evidence.append(
                    str(TwinNetwork.TWIN_NET_PREFIX + evidence))

        twin_node = ConditionalProbabilityTable(name=TwinNetwork.TWIN_NET_PREFIX+str(node.name), variable_card=copy.deepcopy(
            node.variable_card), values=copy.deepcopy(node.values), evidence=twin_evidence, evidence_card=node_evidence_card, state_names=twin_state_names)
        return twin_node
