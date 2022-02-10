"""Manager to actually create a modified Fault Tree out of provided FunctionalConfigurations.
This means pulling the marginals from all associated networks and according to the 'Behaviour'
instantiate or change the time behaviour of Faul Tree nodes via a FunctionalBuilder.
"""
from typing import Dict, List, Optional, Tuple, Set

from bayesiansafety.utils.utils import flatten_list_of_lists
from bayesiansafety.core.inference import InferenceFactory
from bayesiansafety.core import BayesianNetwork
from bayesiansafety.faulttree import BayesianFaultTree
from bayesiansafety.synthesis.functional import FunctionalBuilder
from bayesiansafety.synthesis.functional import FunctionalConfiguration


class FunctionalManager:

    """Manager to actually create a modified Fault Tree out of provided FunctionalConfigurations.
        This means pulling the marginals from all associated networks and according to the 'Behaviour'
        instantiate or change the time behaviour of Faul Tree nodes via a FunctionalBuilder.

    Attributes:
        bayesian_nets (dict<str, BayesianNetwork>): Dictionary of scoped/relevant BNs that are needed to build each functional.
                                Key = bn name, value = instance of BayesianNetwork
        configurations (dict<str, list<FunctionalConfiguration>>): Dictionary of a set of simultaneous functional configurations
                                with key = unique configuration ID for bundle of configs and value = list of functional configurations
                                for the affected nodes.
        ft_inst (BayesianFaultTree): Scoped Fault Tree instance on which we want to apply the configurations on.
    """

    ft_inst = None
    configurations = None
    bayesian_nets = None

    __builder = None  # dict<str, dict<str, FunctionalBuilder>>  # dict of builders with key= config id, value = dict of builders for config with key = node name, value=FunctionalBuilder

    def __init__(self, ft_inst: BayesianFaultTree, bayesian_nets: Dict[str, BayesianNetwork], configurations: Dict[str, List[FunctionalConfiguration]]) -> None:
        """Ctor for the FunctionalManager class.

        Args:
            bayesian_nets (dict<str, BayesianNetwork>): Dictionary of scoped/relevant BNs that are needed to build each functional.
                                    Key = bn name, value = instance of BayesianNetwork
            configurations (dict<str, list<FunctionalConfiguration>>): Dictionary of a set of simultaneous functional configurations
                                    with key = unique configuration ID for bundle of configs and value = list of functional configurations
                                    for the affected nodes.
            ft_inst (BayesianFaultTree): Scoped Fault Tree instance on which we want to apply the configurations on.
        """
        self.ft_inst = ft_inst
        self.configurations = configurations
        self.bayesian_nets = bayesian_nets

        self.__setup_builder()

    def __setup_builder(self) -> None:
        """Setup method to initialize a group of builders for each unique set of configurations.
        """
        self.__builder = {}

        for config_id, cur_configs in self.configurations.items():
            self.__builder[config_id] = {conf.node_instance: FunctionalBuilder(
                self.ft_inst.get_elem_by_name(conf.node_instance.name), conf) for conf in cur_configs}

    def build_functional_fault_tree(self, config_id: str, bn_observations: Optional[Dict[str, List[Tuple[str, str]]]] = None) -> BayesianFaultTree:
        """Build a functionally modified Fault Tree (new reliability functions for one or more probability nodes).

        Args:
            config_id (str): Unique identifier to select a given set of functional configurations that should be applied.
            bn_observations (dict<str, list<tuple<str, str>>, optional): Evidences for each provided BN. This is relevant when the PBF for
                 an environment node is evaluated. Key = bn name, value = list of tuples: bn node name, bn state.

        Returns:
            BayesianFaultTree: Functionally modified Fault Tree instance.

        Raises:
            ValueError: Raised if requested configuration ID is invalid for managed configs.
        """

        if config_id not in self.__builder:
            raise ValueError(f"There is no managed functional configuration for given config id: {config_id}.")

        functional_ft = self.ft_inst.copy()
        relevant_queries = self.__evaluate_relevant_queries(config_id)

        # 1) Infere all marginals
        # (dict<str, dict<str, ConditionalProbabilityTable>>>)  /// dict<bn_name, dict<bn_node, marginalCPT>>
        associated_marginals = dict.fromkeys(relevant_queries.keys(), {})

        for bn_name in relevant_queries.keys():
            bn_inst = self.bayesian_nets[bn_name]
            inf_engine = InferenceFactory(bn_inst).get_engine()
            evidence = bn_observations.get(
                bn_name, None) if bn_observations else None

            for relevant_node in relevant_queries[bn_name]:
                associated_marginals[bn_name][relevant_node] = inf_engine.query(
                    relevant_node, evidence)

        # 2) Build functionals
        functionals = {}
        scoped_builder = self.__builder[config_id]

        for conf in self.configurations[config_id]:
            builder = scoped_builder[conf.node_instance]

            # associated_probabilities = dict.fromkeys(conf.environmental_factors.keys(), []) ## (dict<str, list<tuple<str, float>>>): Provides the marginal probabilities (pbf) for each node. # key = bn name, value = list of bn node name and marginal prob ("P(env)")
            associated_probabilities = {}

            for bn_name in conf.environmental_factors.keys():
                for node_name, state in conf.environmental_factors[bn_name]:
                    marg_cpt = associated_marginals[bn_name][node_name]
                    pbf = marg_cpt.get_value({node_name: state})
                    associated_probabilities.setdefault(
                        bn_name, []).append((node_name, pbf))
                    #associated_probabilities[bn_name].append((node_name, pbf))

            func, params = builder.create_functional(associated_probabilities)
            functionals[conf.node_instance] = (func, params)

        # 3) replace original Faul Tree node's behaviour with the correct functional
        for scoped_ft_node in functionals.keys():
            environmental_fn, fn_parms = functionals[scoped_ft_node]
            functional_ft.get_elem_by_name(scoped_ft_node.name).change_time_behaviour(
                fn_behaviour=environmental_fn, params=fn_parms)

        return functional_ft

    def __evaluate_relevant_queries(self, config_id: str) -> Dict[str, Set[str]]:
        """Internal helper function to check which marginals will be requested when building the functionals.
            The intention is to query each nodes marginal only once even though it might be requested in multiple configuraitons.

        Args:
            config_id (str): Unique identifier to select a given set of functional configurations that should be applied.

        Returns:
            dict<str, set<str>>: Dictionary of needed queries with key = bn name, value = list of bn nodes for which marginals are needed.
        """
        relevant_bns = set(flatten_list_of_lists(
            [conf.environmental_factors.keys() for conf in self.configurations[config_id]]))
        # dict<str, set<str> with key= bn_name and value = set<bn_node_name>
        relevant_queries = dict.fromkeys(relevant_bns, set())

        for conf in self.configurations[config_id]:
            for bn_name in relevant_bns:
                if bn_name in conf.environmental_factors.keys():
                    relevant_queries[bn_name].update(
                        set([node_name for node_name, state in conf.environmental_factors[bn_name]]))

        return relevant_queries
