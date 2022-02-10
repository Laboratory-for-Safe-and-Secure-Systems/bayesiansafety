"""
Interface-like definition for BayesianSafety inference classes.
"""
# ---from pgmpy docs on distinction between CPT and Factor:
# [...] returns an equivalent factor with the same variables, cardinality, values as that of the CPD.
# Since factor doesn't distinguish between conditional and non-conditional distributions,
# evidence information will be lost. ----
# this means we can only create a CPT if the returned factor is a marginal (len(variables) == 1)
import abc
from typing import Union, Any, List, Optional, Tuple


class IInference(metaclass=abc.ABCMeta):

    """Interface-like class to define mandatory methods and their signature for customized (i.e. wrapping
        a third party backend) implementations of an inference class.
    """

    @classmethod
    def __subclasshook__(cls, subclass: Any) -> bool:
        """Define the set of mandatory methods for this interface.
        """
        return (hasattr(subclass, 'query') and
                callable(subclass.query) and
                hasattr(subclass, 'interventional_query') and
                callable(subclass.interventional_query) and
                hasattr(subclass, 'counterfactual_query') and
                callable(subclass.counterfactual_query) or
                NotImplemented)

    @abc.abstractmethod
    def query(self, variables: Union[str, List[str]], evidence: Optional[List[Tuple[str, str]]] = None):
        """This method can be used to run associational inference on the current BN instance.
            Depending on the queried variables either a ConditionalProbabilityTable (only one var.) or a DiscreteFactor (multiple vars.) instance is returned.

        Args:
            variables (str or list<str>): Queried variables
            evidence (list<tuple<str, str>>, optional): Observations in the form of a list of tuples with node name and observed node state.

        Raises:
            NotImplementedError: Default error raised if this method is not implement outside the base class.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def interventional_query(self, variables: Union[str, List[str]], do: Optional[List[Tuple[str, str]]] = None, evidence: Optional[List[Tuple[str, str]]] = None):
        """This method can be used to run interventional inference on the current BN instance.
            Depending on the queried variables either a ConditionalProbabilityTable (only one var.) or a DiscreteFactor (multiple vars.) instance is returned.

        Args:
            variables (str or list<str>): Queried variables
            do (list<tuple<str, str>, optional): List of do-nodes and their active states.
            evidence (list<tuple<str, str>>, optional): Observations in the form of a list of tuples with node name and observed node state.

        Raises:
            NotImplementedError: DescripDefault error raised if this method is not implement outside the base class.tion
        """
        raise NotImplementedError

    @abc.abstractmethod
    def counterfactual_query(self, target: Union[str, List[str]], do: List[Tuple[str, str]], observed: Optional[List[Tuple[str, str]]] = None):
        """This method can be used to run counterfactual inference on the current BN instance.
            Internally a twin-network is used  (see Pearl 2009, Ch. 7).
            Root nodes are treated as exogenous variables.

        Args:
            target (str or list<str>): Variable or list of variables of interest where we want to calculate the effects of the counterfactual on.
            do (list<tuple<str, str>): List variables and their counterfactual states (what we imagine/differs from reality)
            observed (list<tuple<str, str>, optional): Optional list of variables and their observed states (what acutally happened)

        Raises:
            NotImplementedError: Default error raised if this method is not implement outside the base class.
        """
        raise NotImplementedError
