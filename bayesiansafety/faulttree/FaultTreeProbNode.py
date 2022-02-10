"""Class for representing binary probability nodes in a bayesian Fault Tree.
"""
import copy
from typing import Any, Callable, Dict, Tuple, Union, Optional

import numpy as np

from bayesiansafety.core import ConditionalProbabilityTable


class FaultTreeProbNode:
    """Class for representing probability nodes (basic events).

    Attributes:
        cpt (ConditionalProbabilityTable): Associated CPT for this node.
        is_time_dependent (bool): Flag indicating if the probability of failure for this node is constant (False) or time dependent (True).
        name (str): Name of the node.
        probability_of_failure (float): Probability of failure for this node (PBF). If the node is time dependet this will represent the fault rate /lambda.
        probability_of_no_failure (float): Probability of no failure for this node (1-PBF).
        values (TYPE): Description
    """

    name = ""
    cpt = None
    is_time_dependent = False

    probability_of_failure = -999
    probability_of_no_failure = -999
    values = None

    __node_type = 'PROB'
    __time_func = None
    __time_func_params = None
    __default_pbf = None
    __default_is_time_dependent = None
    __default_state_names = ["working", "failing"]

    def __init__(self, name: str, probability_of_failure: Union[int, float], is_time_dependent: Optional[bool] = False) -> None:
        """Ctor for this class.

        Args:
            name (str): Name of the node.
            probability_of_failure (float): Probability of failure for this node (PBF). If the node is time dependet this will represent the fault rate /lambda.
            is_time_dependent (bool, optional): Flag indicating if the probability of failure for this node is constant (False) or time dependent (True).
        """
        self.name = name
        self.is_time_dependent = is_time_dependent

        self.__default_pbf = probability_of_failure
        self.__default_is_time_dependent = is_time_dependent
        self.__time_func = self.__default_time_func
        self.__time_func_params = {}

        self.change_frate(probability_of_failure)

    def get_node_type(self) -> str:
        """Getter for type of node.

        Returns:
            str: Returns "PROB" to indicate that this is a probability node (basic event).
        """
        return self.__node_type.upper()

    def get_time_behaviour(self) -> Tuple[Callable, Dict[Any, Any]]:
        """Get the current reliability function and a dictionary of associated paramters for it.

        Returns:
            callable, dict<str, obj>: Returns the current reliability function with a dictionary of it's parameters.
        """
        return self.__time_func, self.__time_func_params

    def copy(self) -> "FaultTreeProbNode":
        """Returns a deep copy of the this instance.

        Returns:
            FaultTreeProbNode: Deep copy of this instance.
        """
        return copy.deepcopy(self)

    def change_frate(self, probability_of_failure: Union[float, int]) -> None:
        """Setter for the probability of failure. This will also set the associated member self.cpt (at time = 0).

        Args:
            probability_of_failure (float): New probability of failure for this node.

        Raises:
            TypeError: Raised if probability of failure is not a number.
            ValueError: Invalid probability of failure. Must be between 0 and 1.
        """
        if not isinstance(probability_of_failure, (int, float)):
            raise TypeError(f"Probability of failure must be a number and not a: {type(probability_of_failure)}")

        if probability_of_failure < 0.0 or probability_of_failure > 1.0:
            raise ValueError(f"Invalid probability of failure: {probability_of_failure}. Must be between 0 and 1")

        self.cpt = self.__create_cpt(current_pbf=probability_of_failure)
        self.probability_of_failure = probability_of_failure
        self.probability_of_no_failure = 1 - probability_of_failure
        self.values = np.array(
            [self.probability_of_no_failure, self.probability_of_failure], np.float64)

    def change_time_behaviour(self, fn_behaviour: Callable, params: Dict[str, Any]) -> None:
        """Allows to change the time behaviour of this node. This means that the usual time behaviour of const or exp(- /lambda * t)
            will be replaced by this function.

        Args:
            fn_behaviour (callable): Function calculating the time dependent behaviour of this node. First parameter of this func. must be time.
            params (dict<str, obj>): Dictionary containing the keyword arguments for this function in the form key= str(<keyword>), val = value.

        Raises:
            TypeError: Uncallable function or invalid kwargs given.
        """
        if not callable(fn_behaviour):
            raise TypeError(f"Invalid object of type: {type(fn_behaviour)} passed as functional behaviour. Must be a callable object.")

        if not isinstance(params, dict):
            raise TypeError(f"Invalid params of type: {type(params)} passed as functional behaviour. Must be a dictionary.")

        self.__time_func = fn_behaviour
        self.__time_func_params = params

        if self.__time_func != self.__default_time_func:
            self.is_time_dependent = True
        else:
            self.is_time_dependent = self.__default_is_time_dependent

        self.change_frate(
            probability_of_failure=self.__calc_pbf_at_time(at_time=0))

    def reset_time_behaviour(self) -> None:
        """Allows to reset the time behaviour of this node. This means that the usual time behaviour of const or exp(- /lambda * t)
        will be restored including the original (at time of object construction) probability of failure.
        """
        self.probability_of_failure = self.__default_pbf
        self.change_time_behaviour(self.__default_time_func, {})

    def get_cpt_at_time(self, at_time: Optional[float] = 0) -> ConditionalProbabilityTable:
        """Get the CPT of this node at a time stamp containing the [prob. of no failure, prob. of failure]

        Args:
            at_time (float, optional): Time stamp at which the node shall be evaluated (default 0).

        Returns:
            ConditionalProbabilityTable: CPT evaluated at specified time.

        Raises:
            ValueError: Invalid time. Time needs to be a postive value.
        """
        if not self.is_time_dependent:
            return self.cpt

        if at_time is None or at_time < 0:
            raise ValueError(f"Invalid time: {at_time}. Time needs to be a postive value.")

        return self.__create_cpt(current_pbf=self.__calc_pbf_at_time(at_time=at_time))

    def __create_cpt(self, current_pbf: float) -> ConditionalProbabilityTable:
        """Helper method to instantiate a ConditionalProbabilityTable (intended for internal use)

        Args:
            current_pbf (float): Probability of failure

        Returns:
            ConditionalProbabilityTable: Instantiated CPT containing the [prob. of no failure, prob. of failure]
        """
        return ConditionalProbabilityTable(name=self.name, variable_card=2, values=[[1 - current_pbf], [current_pbf]], state_names={self.name: self.__default_state_names})

    def __calc_pbf_at_time(self, at_time: float) -> float:
        """Helper method to calculate the time dependent probability of failure for this node (intended for internal use).

        Args:
            at_time (float): Time stamp at which the node shall be evaluated.

        Returns:
            float: Evaluated probability of failure.
        """
        return self.__evaluate_time_func(at_time)

    def __default_time_func(self, at_time: float) -> float:
        """Represents the default time behaviour of a time dependent probability node inside a Fault Tree which is
            R(t) = 1 - exp(-/lambda * t) where /lambda is the fault rate.

        Args:
            at_time (float): Time stamp where this function should be evaluated

        Returns:
            float: Evaluated reliability function giving the current probability of failure for this component.
        """
        # What's the correct definition for at_time = 0?
        if at_time == 0:
            return self.probability_of_failure

        return 1 - np.exp(-1.0 * self.probability_of_failure * at_time)

    def __evaluate_time_func(self, *args) -> float:
        """Helper method to evaluate the current (anonymous) time dependent pbf function.
            Potential kwargs are pulled from the given parameters when setting the function via change_time_behaviour(...).

        Args:
            *args: Regular args for the current (anonymous) time dependent pbf function

        Returns:
            float: Evaluated probability of failure for a given timestamp

        Raises:
            ValueError: Raised if evaluation of time funciton failed. Indicates problems with passed arguments.
        """
        try:
            return self.__time_func(*args, **self.__time_func_params)
        except TypeError:
            try:
                return self.__time_func(**self.__time_func_params)
            except TypeError:
                try:
                    return self.__time_func(*args)
                except TypeError:
                    raise ValueError(
                        "Evaluation of time function failed. Problems with passed arguments.")
