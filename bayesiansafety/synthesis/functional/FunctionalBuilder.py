"""Builder like class to assemble the marginals of given environment states and functions according to a given 'Behaviour' and other parameters
        provided by an instance of bayesiansafety.functional.FunctionalConfiguration. to generate a new reliability function R*(t) that
        can be executed inside a Faul Tree simulating an environmentally affected node.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from bayesiansafety.faulttree import FaultTreeProbNode
from bayesiansafety.synthesis.functional.FunctionalConfiguration import FunctionalConfiguration, Behaviour


# Via FunctionalConfiguration provided  thresholds are currently ignored
class FunctionalBuilder:

    """Builder like class to assemble the marginals of given environment states and functions according to a given 'Behaviour' and other parameters
        provided by an instance of bayesiansafety.functional.FunctionalConfiguration. to generate a new reliability function R*(t) that
        can be executed inside a Faul Tree simulating an environmentally affected node.

    Attributes:
        associated_probabilities (dict<str, list<<tuple<str, float>>>): Dictionary of relevant nodes and their marginal probabilities to build the funcional.
                    Dictionary key = bn name, value = list of tuples of bn node name and marginal probability for the it's designated state.
        configuration (FunctionalConfiguration): Functional configuraiton for this node.
        ft_node_inst (FaultTreeProbNode): Original Faul Tree node which will be modified (Needed to access it's origial reliability func R(t)).
    """

    ft_node_inst = None
    configuration = None
    associated_probabilities = None

    def __init__(self, ft_node_inst: FaultTreeProbNode, configuration: FunctionalConfiguration) -> None:
        """Ctor of the FunctionalBuilder class.

        Args:
            configuration (FunctionalConfiguration): Functional configuration for this node.
            ft_node_inst (FaultTreeProbNode): Original Faul Tree node which will be modified (Needed to access it's original reliability func R(t)).

        Raises:
             TypeError: Raised if provided Faul Tree node instance is not a probability node.
        """
        if not isinstance(ft_node_inst, FaultTreeProbNode):
            raise TypeError(f"Fault tree node instance must be of type: {type(FaultTreeProbNode)} - yet an element of type: {type(ft_node_inst)} was provided.")

        self.ft_node_inst = ft_node_inst.copy()
        self.configuration = configuration

    def create_functional(self, associated_probabilities: Dict[str, List[Tuple[str, float]]], new_configuration: Optional[FunctionalConfiguration] = None) -> Tuple[Callable, Dict[str, Any]]:
        """Create a new reliability function and it's parameters from a configuration and from current environmental effects (state probabilities).

        Args:
            associated_probabilities (dict<str, list<<tuple<str, float>>>): Dictionary of relevant nodes and their marginal probabilities to build the funcional.
                    Dictionary key = bn name, value = list of tuples of bn node name and marginal probability for the it's designated state.
            new_configuration (FunctionalConfiguration, optional): New functional configuraiton for this node. If provided, the member variable "configuration" will be updated.

        Returns:
             callable, dict<str, obj>: Returns the new reliability function with a dictionary of it's parameters.
        """
        # intended use is to re-use this builder but for another configuration or updated marginals
        self.associated_probabilities = associated_probabilities
        self.configuration = new_configuration if new_configuration is not None else self.configuration

        return self.__create_functional()

    def __create_functional(self) -> Tuple[Callable, Dict[str, Any]]:
        """Internal helper method to actually build the funcional from the configuration

        Returns:
            callable, dict<str, obj>: Returns the new reliability function with a dictionary of it's parameters.

        Raises:
             NotImplementedError: Raised if the behaviour "PARAMETER" is specified.
        """
        if self.configuration.behaviour is Behaviour.REPLACEMENT:
            # Note: we return a constant value which is by definition not callable passing an argument would
            # therefore raise an excepton. By allowing an abritrary number of parameters which are ignored we hacked this.
            first_bn = next(iter(self.associated_probabilities))
            node_name, pbf = self.associated_probabilities[first_bn][0]

            # R(t)    -> P(env)
            fn_replacement = lambda *args, **kwargs: pbf

            return fn_replacement, {}

        orig_fn, orig_params = self.ft_node_inst.get_time_behaviour()
        orig_fn = orig_fn if self.ft_node_inst.is_time_dependent else lambda *args, **kwargs: self.ft_node_inst.probability_of_failure

        if self.configuration.behaviour is Behaviour.ADDITION:
            first_bn = next(iter(self.associated_probabilities))
            node_name, pbf = self.associated_probabilities[first_bn][0]
            node_name, node_weight = self.configuration.weights[first_bn][0]

            def fn_addition(at_time):
                """Functional behaviour of "ADDITION".

                Args:
                    at_time (float): Time stamp at which this func should be evaluated.

                Returns:
                    callable, dict<str, obj>: Returns the new reliability function with a dictionary of it's parameters.
                """
                # R(t)    -> w_0 *R(t) + w_1*P(env)

                w0 = self.configuration.get_weight_orig_fn()
                w1 = node_weight

                return w0*orig_fn(at_time) + w1*pbf

            return fn_addition, {"at_time": None}

        elif self.configuration.behaviour is Behaviour.OVERLAY:

            def fn_overlay(at_time):
                """Functional behaviour of "OVERLAY".

                Args:
                    at_time (float): Time stamp at which this func should be evaluated.

                Returns:
                    callable, dict<str, obj>: Returns the new reliability function with a dictionary of it's parameters.
                """
                # R(t)    -> w_0* R(t) * \prod w_n*P_n
                w0 = self.configuration.get_weight_orig_fn()
                result = w0 * orig_fn(at_time)

                for bn_name in self.associated_probabilities.keys():
                    cur_weights = dict(self.configuration.weights[bn_name])
                    cur_pbfs = dict(self.associated_probabilities[bn_name])
                    matched_weight_pbf = [(weight, cur_pbfs[node_name])
                                          for node_name, weight in cur_weights.items()]
                    result *= np.prod(matched_weight_pbf)

                return result

            return fn_overlay, {"at_time": None}

        elif self.configuration.behaviour is Behaviour.RATE:
            # R(t,l) -> R(t,l*)      # l = \lambda

            def fn_rate(at_time, frate):
                """Functional behaviour of "RATE".

                Args:
                    at_time (float): Time stamp at which this func should be evaluated.
                    frate (float): New fault rate /lambda.

                Returns:
                    callable, dict<str, obj>: Returns the new reliability function with a dictionary of it's parameters.
                """
                tmp_node_inst = self.ft_node_inst.copy()
                tmp_node_inst.reset_time_behaviour()
                tmp_node_inst.change_frate(frate)
                orig_fn, orig_params = tmp_node_inst.get_time_behaviour()

                return orig_fn(at_time)

            return fn_rate, {"at_time": None, "frate": None}

        elif self.configuration.behaviour is Behaviour.FUNCTIONAL:
            # R(t,l) -> R*(t, X)     # X = set of parameters

            return self.configuration.time_func, self.configuration.func_params

        else:  # Behaviour.PARAMETER
            # R(t)    -> R*(t, P(env))
            raise NotImplementedError(
                "Requested behaviour: {self.configuration.behaviour} is currently not supported.")
