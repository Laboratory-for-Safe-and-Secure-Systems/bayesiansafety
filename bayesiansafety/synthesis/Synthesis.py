"""
This class acts as hyper manager for handling the different BNs and FTs.
It controls the correct execution and instantiation of models and nodes.
It also dynamically extends the given Fault Trees (adding the BN nodes) and therefore
ecapsulated the differnt flavors (hybrid, functional...).
"""
import os
from typing import Dict, List, Optional, Tuple, Union

from bayesiansafety.core.BayesianNetwork import BayesianNetwork
from bayesiansafety.synthesis.hybrid import HybridManager
from bayesiansafety.synthesis.functional import FunctionalManager
from bayesiansafety.synthesis.functional import FunctionalConfiguration
from bayesiansafety.synthesis.hybrid import HybridConfiguration

from bayesiansafety.faulttree import BayesianFaultTree
from bayesiansafety.faulttree import Evaluation


class Synthesis:

    """This class acts as hyper manager for handling the different BNs and FTs.
        It controls the correct execution and instantiation of models and nodes.
        It also dynamically extends the given Fault Trees (adding the BN nodes) and therefore
        ecapsulated the differnt flavors (hybrid, functional...).

    Attributes:
        bayesian_nets (dict<str, BayesianNetwork>): Dictionary of scoped bayesian networks. Key = bn name, value = bn instance.
        fault_trees (dict<str, BayesianFaultTree>): Dictionary of scoped Fault Trees . Key = ft name, value = Fault Tree instance.
    """

    fault_trees = None
    bayesian_nets = None

    # dict<str, HybridManager> with key = ft_name, value = HybridManager instance for this Faul Tree
    __hybrid_managers = None
    # dict<str, FunctionalManager> with key = ft_name, value = FunctionalManager instance for this Faul Tree
    __functional_managers = None

    __models = None  # dict with key = name, val = obj (instance of BN or Faul Tree)

    def __init__(self, fault_trees: Dict[str, BayesianFaultTree], bayesian_nets: Dict[str, BayesianNetwork]) -> None:
        """Ctor of the BayesianSafety class.

        Args:
            fault_trees (dict<str, BayesianFaultTree>): Dictionary of scoped Fault Trees . Key = ft name, value = Fault Tree instance.
            bayesian_nets (dict<str, BayesianNetwork>): Dictionary of scoped bayesian networks. Key = bn name, value = bn instance.
        """
        self.fault_trees = fault_trees
        self.bayesian_nets = bayesian_nets

        self.__verify_valid_initializer_types()

        self.__models = {**fault_trees, **bayesian_nets}

    def __verify_valid_initializer_types(self) -> None:
        """Setup method that checks if provided ctor argument types are valid.

        Raises:
            TypeError: Raised if a given parameter has an invalid type.
        """
        params = [self.fault_trees, self.bayesian_nets]
        if not all(params):
            str_params = '\n'.join([str(par) for par in params])
            raise TypeError(f"Given parameters can't be default or none.\n{str_params}")

        for ft_name, ft_inst in self.fault_trees.items():
            if not isinstance(ft_inst, BayesianFaultTree):
                raise TypeError(f"All given Fault Trees must be an instantiation of {type(BayesianFaultTree)}.")

        for bn_name, bn_inst in self.bayesian_nets.items():
            if not isinstance(bn_inst, BayesianNetwork):
                raise TypeError(f"All given Bayesian Networks must be an instantiation of {type(BayesianNetwork)}.")

    def __check_managers_available(self, manager_type: Optional[str] = "hybrid") -> None:
        """Helper method to validate if managers got instantiated.
            Instantiation is done by calling one of the set_*_configuration(...) methods.
            The setter approach was chosen to ease runtime and memory consumption a little.

        Args:
            manager_type (str, optional): Which managers should be available.

        Raises:
            Exception: Raised if requested managers are not available.
        """
        if manager_type.lower() == "hybrid":
            if not self.__hybrid_managers:
                raise Exception("No hybrid managers are instantiated. Make sure to set the hybrid configurations before requesting associated services.")

        else:
            if not self.__functional_managers:
                raise Exception("No functional managers are instantiated. Make sure to set the functional configurations before requesting associated services.")

    def set_hybrid_configurations(self, ft_configurations: Dict[str, Dict[str, HybridConfiguration]]) -> None:
        """Set the hybrid configurations. Only if this method is called - related functionality is available.

        Args:
            ft_configurations (dict<str, dict<str, HybridConfiguration>>): Dictionary containing all configurations for the various Fault Trees.
                    Key=ft_name, value = dictionary with key = configuration ID and value = hybrid configuration.
        """
        self.__hybrid_managers = dict.fromkeys(ft_configurations.keys())

        for ft_name, hybrid_configs in ft_configurations.items():
            self.__hybrid_managers[ft_name] = HybridManager(
                ft_inst=self.fault_trees[ft_name], bayesian_nets=self.bayesian_nets, configurations=hybrid_configs)

    def set_functional_configurations(self, ft_configurations: Dict[str, Dict[str, List[FunctionalConfiguration]]]) -> None:
        """Set the fucntional configurations. Only if this method is called - related functionality is available.

        Args:
            ft_configurations (dict<str, dict<str, list<FunctionalConfiguration>>): Dictionary containing all configurations for the various Fault Trees.
                    Key = ft_name, value = dictionary with key = configuration ID and value = list of functional configurations
        """
        self.__functional_managers = dict.fromkeys(ft_configurations.keys())

        for ft_name, functional_configs in ft_configurations.items():
            self.__functional_managers[ft_name] = FunctionalManager(
                ft_inst=self.fault_trees[ft_name], bayesian_nets=self.bayesian_nets, configurations=functional_configs)

    def get_model_by_name(self, name: str) -> Union[BayesianFaultTree, BayesianNetwork]:
        """Getter to access a managed model (instance of BayesianNetwork or BayesianFaultTree class) by name.

        Args:
            name (str): Name of the queried model.

        Returns:
            <BayesianNetwork or BayesianFaultTree>: Model as an instance of either BayesianNetwork or BayesianFaultTree class.

        Raises:
            ValueError: Raised if requested model element is not part of the currently managed configuration.
        """
        if name not in self.__models:
            raise ValueError(f"Scoped element: {name} could not be found in given models: {self.__models.keys()}.")

        return self.__models[name]



    def get_extended_fault_tree(self, ft_name: str, config_id: str, bn_observations: Optional[Dict[str, List[Tuple[str, str]]]] = None) -> BayesianFaultTree:
        """Returns an extended Fault Tree. An extended Fault Tree is the scoped Faul Tree extended by all associated BN nodes which are represented as static probability nodes.

        Args:
            ft_name (str): Fault tree that should be extended.
            config_id (str): Identifier of the hybrid configuration that should be used.
            bn_observations (dict<str, list<tuple<str, str>>, optional): Evidences for each provided BN. This is relevant when the PBF for a shared node is evaluated.
                                                                         Key = bn name, value = list of tuples (bn node name, bn state).

        Returns:
            BayesianFaultTree: Extended Fault Tree.
        """

        self.__check_managers_available("hybrid")
        return self.__hybrid_managers[ft_name].build_extended_ft(config_id=config_id, bn_observations=bn_observations)

    def get_hybrid_networks(self, ft_name: str, config_id: str, at_time: Optional[float] = 0, fix_other_bns: Optional[bool] = True) -> Dict[str, BayesianNetwork]:
        """Generates all hybrid networks (Faul Tree + associated networks(s)) for a given Fault Tree.
            Depending on the "fix_other_bns" flag, hybrids consisting of the Faul Tree and each individual associated BN (True) are generated.

        Args:
            ft_name (str): Name of the Fault Tree to extend.
            config_id (str): Identifier of the hybrid configuration that should be used.
            at_time (int, optional): Evaluation is done at a specific time stamp. This means all PBFs of the Faul Tree are evaluated at this time.
            fix_other_bns (bool, optional): Flag indicating if one "full" hybrid model (Faul Tree + all associated BNs) (False) or if
                                            a hybrid for each assocated BNs should be created (True, default)

        Returns:
            dict<str, BayesianNetwork>: Dictionary consisting of a generated BN name (key) and the generated BN instance (value).
        """

        self.__check_managers_available("hybrid")
        return self.__hybrid_managers[ft_name].build_hybrid_networks(config_id=config_id, at_time=at_time, fix_other_bns=fix_other_bns)

    def get_functional_fault_tree(self, ft_name: str, config_id: str, bn_observations: Optional[Dict[str, Tuple[str, str]]] = None) -> BayesianFaultTree:
        """Build a functionally modified Fault Tree (new reliability functions for one or more probability nodes).

        Args:
            ft_name (str): Name of the Fault Tree to modify.
            config_id (str): Unique identifier to select a given set of functional configurations that should be applied.
            bn_observations (dict<str, list<tuple<str, str>>, optional): Evidences for each provided BN. This is relevant when the PBF for
                 an environment node is evaluated. Key = bn name, value = list of tuples: bn node name, bn state.

        Returns:
            BayesianFaultTree: Functionally modified Fault Tree instance.

        """
        self.__check_managers_available("functional")
        return self.__functional_managers[ft_name].build_functional_fault_tree(config_id=config_id, bn_observations=bn_observations)

    def evaluate_hybrid_fault_trees(self, ft_name: str, bn_observations: Optional[Dict[str, Tuple[str, str]]] = None, ft_time_scales: Optional[Dict[str, Tuple[float, float, int, str]]] = None) -> None:
        """## Convenience function to evaluate all managed configurations for one specified Fault Tree in their initial state (prior probabilites).
            ## Fault trees are evaluated considering their associated BNs.
            ### A static evidence for any BN can be provided. This will influence the fault rates/evaluation of the shared BN nodes in all FTs (posterior probabilitites).
            ### Note: the shared nodes will be interpreded as marginals
            ### For each Faul Tree an individual time frame can be provided.

        Args:
            ft_name (str): Name of the Fault Tree to that should be evaluated.
            bn_observations (dict<str, list<tuple<str, str>>, optional): Evidences for each provided BN. This is relevant when the PBF for a shared node is evaluated.
                                                                         Key = bn_name, value = list of tuples (bn node name, bn state).
            ft_time_scales (dict<str, tuple<float, float, int, path>>, optional): Individual time analyis constraints for a specific Faul Tree.
                                                                         The given tuple need to satisfy mostly the signature of BayesianFaultTree.evaluate_fault_tree(...).
                                                                         Key = ft name, value = tuple(start_time, stop_time, simulation_steps, plot_dir)
        """

        self.__check_managers_available("hybrid")
        hybrid_manager = self.__hybrid_managers[ft_name]

        for config_id, config in hybrid_manager.configurations.items():
            print(f"Processing configuration: {config_id} for scoped Fault Tree: {ft_name}")

            ext_ft = hybrid_manager.build_extended_ft(
                config_id=config_id, bn_observations=bn_observations)
            ft_evaluator = Evaluation(ext_ft)

            if ft_time_scales:
                if ft_time_scales.get(ft_name, None):
                    start_time, stop_time, simulation_steps, plot_dir = ft_time_scales[ft_name]
                    ft_evaluator.evaluate_fault_tree(start_time=start_time, stop_time=stop_time, simulation_steps=simulation_steps, plot_dir=os.path.join(plot_dir, f"{config_id}"))

                    continue

            # eval Faul Tree with default values
            ft_evaluator.evaluate_fault_tree()

    def evaluate_functional_fault_trees(self, ft_name: str, bn_observations: Optional[Dict[str, Tuple[str, str]]] = None, ft_time_scales: Optional[Dict[str, Tuple[float, float, int, str]]] = None) -> None:
        """
        Args:
            ft_name (str): Name of the Fault Tree to that should be evaluated.
            bn_observations (dict<str, list<tuple<str, str>>, optional): Evidences for each provided BN. This is relevant when the PBF for a shared node is evaluated.
                                                                         Key = bn_name, value = list of tuples (bn node name, bn state).
            ft_time_scales (dict<str, tuple<float, float, int, path>>, optional): Individual time analyis constraints for a specific Faul Tree.
                                                                         The given tuple need to satisfy mostly the signature of BayesianFaultTree.evaluate_fault_tree(...).
                                                                         Key = ft name, value = tuple(start_time, stop_time, simulation_steps, plot_dir)
        """
        self.__check_managers_available("functional")
        functional_manager = self.__functional_managers[ft_name]

        for config_id, config in functional_manager.configurations.items():
            print(f"Processing configuration: {config_id} for scoped Fault Tree: {ft_name}")

            ext_ft = functional_manager.build_functional_fault_tree(
                config_id=config_id, bn_observations=bn_observations)
            ft_evaluator = Evaluation(ext_ft)

            if ft_time_scales:
                if ft_time_scales.get(ft_name, None):
                    start_time, stop_time, simulation_steps, plot_dir = ft_time_scales[ft_name]
                    ft_evaluator.evaluate_fault_tree(start_time=start_time, stop_time=stop_time, simulation_steps=simulation_steps, plot_dir=os.path.join(plot_dir, f"{config_id}"))

                    continue

            # eval Faul Tree with default values
            ft_evaluator.evaluate_fault_tree()

    # def evaluate_bns_in_retrospective(self, ft_configs=None):
        """Summary

        Args:
            ft_configs (None, optional): Description
        """
        # retrospective handler
        # this means we observed the TLE (or another event) and now we try to
        # calculate the posterior cpts for all affected BNs.
        # Association flows from the Faul Tree via the shared nodes to the connected BNs

        # ft_configs = dict<str, tuple<float, list<str>> ## key=ft_name, value= tuple(at_time, list<active 'faulty' logic nodes>)
        # if ft_configs = None :: evaluate all ft's with only TLE active at time '0'

        # for ft_name, ft_config in ft_configs.items():
        #    at_time, active_nodes = ft_config
    #    pass
