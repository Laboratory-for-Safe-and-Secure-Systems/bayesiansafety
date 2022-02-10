"""Environment effects can change the reliability function R(t) of a component in different ways.
This class is intended as a container to bundle these properties as well as thresholds and
additional parameters for the exact calculation of the modified function R*(t).
"""
from typing import Callable, Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np

from bayesiansafety.faulttree import FaultTreeProbNode


class Behaviour(Enum):

    """Enum class for a consistent use of functional "behaviours".

    Attributes:
        REPLACEMENT (int)   :     R(t)   -> P(env)
        ADDITION (int)      :     R(t)   -> w_0 *R(t) + w_1*P(env)
        OVERLAY (int)       :     R(t)   -> w_0* R(t) * /prod w_n*P_n
        RATE (int)          :     R(t,l) -> R(t,l*)        # l = /lambda
        FUNCTIONAL (int)    :     R(t,l) -> R*(t, X)    # X = set of parameters
        PARAMETER (int)     :     R(t)   -> R*(t, P(env))  # special case of RATE or the more generic FUNCTIONAL, not supported.
    """

    REPLACEMENT = 0
    ADDITION = 1
    OVERLAY = 2
    RATE = 3
    FUNCTIONAL = 4
    PARAMETER = 5


class FunctionalConfiguration:

    """"Environment effects can change the reliability function R(t) of a component in different ways.
        This class is intended as a container to bundle these properties as well as thresholds and
        additional parameters for the exact calculation of the modified function R*(t).


    Attributes:
        behaviour (Behaviour): Instance of enum 'Behaviour'
        environmental_factors (dict<str, list<tuple<str, str>>>): Specifies which state of an environmental node is treated as PBF (basis event).
                                    Depending on the 'behaviour' these factors will be used to build the new reliability function.
                                    Key = bn name, value = list of tuples: bn node name and "bad" state of this node.
        func_params (dict<str, obj>): Dictionary containing the keyword arguments for a new reliability function in the form key= str(<keyword>), val = value.
        node_instance (FaultTreeProbNode): Basis event in a Fault Tree - needed to access the original R(t)
        thresholds dict<str, list<tuple<str, float>>>): Specifies for each associated BN node when it starts contributing to R*(t).
                                    Threshold is defined for the marginal state probability that's used as PBF.
                                    A threshold of 0 indicates that a node always contributes. Key = bn_name, value = list of tuple: bn node name and threshold level
        time_func (callable): Function representing the new reliability function. First parameter of this func. must be time. Only considered if Behaviour is FUNCTIONAL.
        weights (dict<str, list<tuple<str, float>>>): Specifies for each contributing BN node how it should be weighted when combined. Weights will limit the
                                    combination of max. probs to 1 and might be therfore adjusted. If no weights are passed - each of the  n-contributing BN nodes
                                    is weigheted equally with 1/(n+1). The weight for a potentially used, original R(t) w_0 is defaulted to 1.0 and automatically adjusted
                                    based on the passed or calculated weights depending on the behaviour.
                                    Dictionary with key = bn name, value = list of tuples: bn node name, weight
    """

    node_instance = None
    behaviour = None
    environmental_factors = None
    thresholds = None
    weights = None
    time_func = None
    func_params = None

    __orig_rfunc_weight = 1.0

    def __init__(self, node_instance: FaultTreeProbNode, environmental_factors: Dict[str, List[Tuple[str, str]]], thresholds: Dict[str, List[Tuple[str, float]]], weights: Optional[Dict[str, List[Tuple[str, float]]]] = None, time_func: Optional[Callable] = None, func_params: Optional[Dict[str, Any]] = None, behaviour: Optional[Behaviour] = Behaviour.REPLACEMENT) -> None:
        """Ctor of the FunctionalConfiguration class.
            Note: Thresholds are currently ignored.

        Args:
        behaviour (Behaviour, optional): Instance of enum 'Behaviour'
        environmental_factors (dict<str, list<tuple<str, str>>>): Specifies which state of an environmental node is treated as PBF (basis event).
                                    Depending on the 'behaviour' these factors will be used to build the new reliability function.
                                    Key = bn name, value = list of tuples: bn node name and "bad" state of this node.
        func_params (dict<str, obj>, optional): Dictionary containing the keyword arguments for a new reliability function in the form key= str(<keyword>), val = value.
        node_instance (FaultTreeProbNode): Basis event in a Fault Tree - needed to access the original R(t), maybe name is sufficient
        thresholds dict<str, list<tuple<str, float>>>): Specifies for each associated BN node when it starts contributing to R*(t).
                                    Threshold is defined for the marginal state probability that's used as PBF.
                                    A threshold of 0 indicates that a node always contributes. Key = bn_name, value = list of tuple: bn node name and threshold level
        time_func (callable, optional): Function representing the new reliability function. First parameter of this func. must be time. Only considered if Behaviour is FUNCTIONAL.
        weights (dict<str, list<tuple<str, float>>>, optional): Specifies for each contributing BN node how it should be weighted when combined. Weights will limit the
                                    combination of max. probs to 1 and might be therfore adjusted. If no weights are passed - each of the  n-contributing BN nodes
                                    is weigheted equally with 1/(n+1). The weight for a potentially used, original R(t) w_0 is defaulted to 1.0 and automatically adjusted
                                    based on the passed or calculated weights depending on the behaviour.
                                    Dictionary with key = bn name, value = list of tuples: bn node name, weight
        """
        self.node_instance = node_instance
        self.behaviour = behaviour
        self.environmental_factors = environmental_factors
        self.thresholds = thresholds
        self.weights = weights
        self.time_func = time_func
        self.func_params = func_params

        self.__verify_valid_configuration()
        self.__verify_weights()

    def __verify_valid_configuration(self) -> None:
        """Setup function to check if the given configurations is valid.
            This means if the combination of cross referenced nodes and BNs match together and are reasonable.

        Raises:
            ValueError: Raised if a sanity check for a paramter fails.
        """
        if self.behaviour in [Behaviour.REPLACEMENT, Behaviour.ADDITION]:
            if len(self.environmental_factors.keys()) != 1:
                raise ValueError(f"Due to the specified behaviour: {self.behaviour}, only one fixed BN is allowed.")

            first_bn = next(iter(self.environmental_factors))
            if len(self.environmental_factors[first_bn]) != 1:
                raise ValueError(f"Due to the specified behaviour: {self.behaviour}, only one fixed node is allowed.")

        if len(set(self.environmental_factors.keys()) ^ set(self.thresholds.keys())) != 0:
            raise ValueError(f"For each given contributing BN: {self.environmental_factors.keys()} probability thresholds need to be defined - but are given for: {self.thresholds.keys()}")

        for bn_name in self.environmental_factors.keys():
            contributing_nodes = [bn_node_name for bn_node_name,
                                  state in self.environmental_factors[bn_name]]
            thresholded_nodes = [
                bn_node_name for bn_node_name, thr in self.thresholds[bn_name]]
            if not set(contributing_nodes).issubset(set(thresholded_nodes)):
                raise ValueError("For each contributing node of each BN a probability threshold needs to be defined.")

            for bn_node_name, thr_value in self.thresholds[bn_name]:
                if thr_value > 1.0 or thr_value < 0.0:
                    raise ValueError(f"Probability threshold for node: {bn_node_name} of BN: {bn_name} out of bounds (0...1.0) with value:{thr_value}.")

        if self.behaviour in [Behaviour.RATE, Behaviour.FUNCTIONAL, Behaviour.PARAMETER]:
            if not all([callable(self.time_func), self.func_params]):
                raise ValueError(f"Due to the specified behaviour: {self.behaviour}, a callable function and associated parameters need to be provided. ")

    def __verify_weights(self) -> None:
        """Setup function to check the given weights according to a given Behaviour.
            If no weights are given - they are calcuated to equally weight each contributor.
        """
        if self.behaviour in [Behaviour.REPLACEMENT, Behaviour.ADDITION, Behaviour.OVERLAY]:
            tmp_weights = {}

            if not self.weights:
                # no weights are passed - each of the  n-contributors is weigheted equally with 1/(n +1)
                # if ADDITION or OVERLAY is specified, the weight w_0 for the original R(t) is re-weighted to 1/(n+1) as well - otherwise (REPLACEMENT) set to 0.0
                contributor_count = sum(len(val) for val in self.environmental_factors.values())
                contributor_count = contributor_count + \
                    1 if self.behaviour in [
                        Behaviour.ADDITION, Behaviour.OVERLAY] else contributor_count
                self.__orig_rfunc_weight = 1 / \
                    contributor_count if self.behaviour in [
                        Behaviour.ADDITION, Behaviour.OVERLAY] else 0.0

                for bn_name, contributors in self.environmental_factors.items():
                    for bn_node_name, state in contributors:
                        #tmp_weights[bn_name].append( (bn_node_name, 1/contributor_count) )
                        tmp_weights.setdefault(bn_name, []).append(
                            (bn_node_name, 1/contributor_count))

            else:
                # we got weights!
                # check if weights limit combination of probs to a maximum of 1.
                if self.behaviour is Behaviour.REPLACEMENT:
                    self.__orig_rfunc_weight = 0.0

                    bn_name = next(iter(self.environmental_factors))
                    bn_node_name, cur_weight = self.weights[bn_name][0]
                    cur_weight = cur_weight if 0.0 < cur_weight <= 1.0 else 1.0

                    tmp_weights.setdefault(bn_name, []).append(
                        (bn_node_name, cur_weight))
                    #tmp_weights[bn_name].append((bn_node_name , cur_weight))

                else:
                    # this is either ADDITION or OVERLAY
                    given_weights = []
                    for bn_name, contributors in self.weights.items():
                        given_weights += [cur_weight for bn_node_name,
                                          cur_weight in contributors]

                    total_weight = np.sum(
                        given_weights) if self.behaviour is Behaviour.ADDITION else np.prod(given_weights)

                    if total_weight >= 1.0:
                        total_weight += self.__orig_rfunc_weight
                        norm_factor = 1.0 / total_weight

                        self.__orig_rfunc_weight = self.__orig_rfunc_weight * norm_factor

                        for bn_name, contributors in self.weights.items():
                            for bn_node_name, cur_weight in contributors:
                                #tmp_weights[bn_name].append( (bn_node_name, cur_weight * norm_factor ) )
                                tmp_weights.setdefault(bn_name, []).append(
                                    (bn_node_name, cur_weight * norm_factor))
                    else:
                        # weights are fine - let's adjust w_0 for the original R(t)
                        self.__orig_rfunc_weight = 1.0 - total_weight
                        tmp_weights = self.weights

            self.weights = tmp_weights

        else:
            # no weights needed
            # maybe initialize them to 0?
            pass

    def get_weight_orig_fn(self) -> float:
        """Access the current weight w0 for the original reliability function.

        Returns:
            float: Current weight for the original reliability function.
        """
        return self.__orig_rfunc_weight
