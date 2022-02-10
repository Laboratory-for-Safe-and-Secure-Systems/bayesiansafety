"""
This class is a wrapper for the internal representation of Inference engines to
seperate the actual implementaiton from the usage througout the package.
This implementation wraps the library "pgmpy".
See https://github.com/pgmpy/pgmpy
"""
from typing import List, Optional, Tuple, Union, Any

import numpy as np

from pgmpy.models import BayesianNetwork as pgmpyBayesianModel
from pgmpy.inference import VariableElimination as pgmpyVariableElimination
from pgmpy.inference import CausalInference as pgmpyCausalInference
from pgmpy.factors.discrete import TabularCPD as pgmpyTabularCPD

from bayesiansafety.core import ConditionalProbabilityTable
from bayesiansafety.core import DiscreteFactor
from bayesiansafety.core import BayesianNetwork
from bayesiansafety.core.inference.IInference import IInference
from bayesiansafety.core.inference.TwinNetwork import TwinNetwork


class PgmpyInference(IInference):

    """This class is a wrapper for the internal representation of Inference engines to
        seperate the actual implementaiton from the usage througout the package.
        This implementation wraps the library "pgmpy".
        See https://github.com/pgmpy/pgmpy

    Attributes:
        model (BayesianNetwork): Instance of a BaySafety-BN on which queries should be run.
    """

    model = None
    __inference_engine_inst = None  # pgmpy VariableElemination
    __causal_engine_inst = None  # pgmpy CausalInference
    __internal_model = None  # pgmpy BayesianModel

    def __init__(self, model: BayesianNetwork) -> None:
        """Ctor of the Inference class for backend pgmpy.

        Args:
            model (TYPE): Description
        """
        self.model = model
        self.__build_internal_model()
        self.__inference_engine_inst = pgmpyVariableElimination(
            self.__internal_model)
        self.__causal_engine_inst = pgmpyCausalInference(self.__internal_model)

    def __build_internal_model(self) -> None:
        """Setup method to instantiate a model that is compatible with the internally used, encapsulated inference algorithm.
        """
        self.__internal_model = pgmpyBayesianModel(self.model.node_connections)
        for node in self.model.get_independent_nodes():
            self.__internal_model.add_node(node)

        for cpt in self.model.model_elements.values():
            pgmpy_cpt = pgmpyTabularCPD(variable=cpt.name, variable_card=cpt.variable_card, values=cpt.values,
                                        evidence=cpt.evidence, evidence_card=cpt.evidence_card, state_names=cpt.state_names)
            self.__internal_model.add_cpds(pgmpy_cpt)

    def query(self, variables: Union[str, List[str]], evidence: Optional[List[Tuple[str, str]]] = None) -> Union[ConditionalProbabilityTable, DiscreteFactor]:
        """This method can be used to run associational inference on the current BN instance.
            Depending on the queried variables either a ConditionalProbabilityTable (only one var.) or a DiscreteFactor (multiple vars.) instance is returned.

        Args:
            variables (list<str>): Queried variables
            evidence (list<tuple<str, str>>, optional): Observations in the form of a list of tuples with node name and observed node state.

        Returns:
             ConditionalProbabilityTable or DiscreteFactor: Return type is depending on number of queried variables.
        Raises:
             TypeError: Raised if queried variables are not a string or list of strings.
             ValueError: Raised if evidence variables and query variables intersect.
         """
        if not isinstance(variables, (list, str)):
            raise TypeError(f"Queried variable(s) need to be a string or list of strings but are {type(variables)}.")

        variables = variables if isinstance(variables, list) else [variables]

        evidence = dict(evidence) if isinstance(evidence, list) else evidence
        common_vars = set(evidence if evidence is not None else [
        ]).intersection(set(variables))

        if common_vars:
            raise ValueError(f"Query contains evidence: {common_vars} that is part of the scoped variables:{variables}.")

        query_result = self.__inference_engine_inst.query(
            variables=variables, evidence=evidence, joint=True, show_progress=False, elimination_order="MinFill")

        if len(query_result.cardinality) == 1:
            return self.__build_bay_safety_result(query_result, is_cpt=True)

        return self.__build_bay_safety_result(query_result, is_cpt=False)

    def interventional_query(self, variables: Union[str, List[str]], do: Optional[List[Tuple[str, str]]] = None, evidence: Optional[List[Tuple[str, str]]] = None) -> Union[ConditionalProbabilityTable, DiscreteFactor]:
        """This method can be used to run interventional inference on the current BN instance.
            Depending on the queried variables either a ConditionalProbabilityTable (only one var.) or a DiscreteFactor (multiple vars.) instance is returned.

        Args:
            variables (list<str>): Queried variables
            do (list<tuple<str, str>, optional): List of do-nodes and their active states.
            evidence (list<tuple<str, str>>, optional): Observations in the form of a list of tuples with node name and observed node state.

        Returns:
            ConditionalProbabilityTable or DiscreteFactor: Return type is depending on number of queried variables.

        Raises:
            TypeError: Raised if queried variables are not a string or list of strings.
            ValueError: Raised if evidence variables and query variables intersect.
        """
        evidence = dict(evidence) if isinstance(evidence, list) else evidence
        do = dict(do) if isinstance(do, list) else do

        if not isinstance(variables, (list, str)):
            raise TypeError(f"Queried variable(s) need to be a string or list of strings but are {type(variables)}.")

        variables = variables if isinstance(variables, list) else [variables]
        common_vars = set(evidence if evidence is not None else [
        ]).intersection(set(variables))

        if common_vars:
            raise ValueError(f"Query contains evidence: {common_vars} that is part of the scoped variables:{variables}.")

        common_vars = set(do if do is not None else []
                          ).intersection(set(variables))

        if common_vars:
            raise ValueError(f"Query contains do-variables: {common_vars} that are part of the scoped variables:{variables}.")

        query_result = self.__causal_engine_inst.query(
            variables=variables, do=do, evidence=evidence, adjustment_set=None, inference_algo='ve', show_progress=False)

        if len(query_result.cardinality) == 1:
            return self.__build_bay_safety_result(query_result, is_cpt=True)

        return self.__build_bay_safety_result(query_result, is_cpt=False)

    def counterfactual_query(self, target: Union[str, List[str]], do: List[Tuple[str, str]], observed: Optional[List[Tuple[str, str]]] = None) -> Union[ConditionalProbabilityTable, DiscreteFactor]:
        """This method can be used to run counterfactual inference on the current BN instance.
            Internally a twin-network is used  (see Pearl 2009, Ch. 7).
            Root nodes are treated as exogenous variables.

        Args:
            target (str or list<str>): Variable or list of variables of interest where we want to calculate the effects of the counterfactual on.
            do (list<tuple<str, str>): List variables and their counterfactual states (what we imagine/differs from reality)
            observed (list<tuple<str, str>, optional): Optional list of variables and their observed states (what acutally happened)

        Returns:
            ConditionalProbabilityTable or DiscreteFactor: Return type is depending on number of queried variables.

        Raises:
            TypeError: Raised if queried variables are not a string or list of strings.
        """

        do = dict(do) if isinstance(do, list) else do

        if not isinstance(target, (list, str)):
            raise TypeError(f"Queried target(s) need to be a string or list of strings but are {type(target)}.")

        target = target if isinstance(target, list) else [target]

        prefixed_do = {TwinNetwork.TWIN_NET_PREFIX +
                       var: state for (var, state) in do.items()}
        twin_model = TwinNetwork.build_twin_network(self.model)
        inf = PgmpyInference(twin_model)

        query_result = inf.interventional_query(variables=[
                                                TwinNetwork.TWIN_NET_PREFIX + var for var in target], do=prefixed_do, evidence=observed)

        return query_result

    def __build_bay_safety_result(self, res_factor: Any, is_cpt: Optional[bool] = True) -> Union[ConditionalProbabilityTable, DiscreteFactor]:
        """Method to convert the internal (pgmpy) representation of a factor/cpt
            into a BaySafety-CPT or BaySafety-DiscretFactor.
            This method is needed to encapsulate backend representation of results
            from the rest of BaySafety.

        Args:
            res_factor (pgmpy.factor.DiscreteFactor or pgmpy.factor.TabularCPD): Result of a query.
            is_cpt (bool, optional): Flag indicating if the result should be interpreted as
                    a CPT or DiscreteFactor.

        Returns:
            core.ConditionalProbabilityTable or core.DiscreteFactor: BaySafety representation
                    of query results.
        """

        if is_cpt:
            bay_safety_obj = ConditionalProbabilityTable(name=res_factor.variables[0], variable_card=int(res_factor.cardinality),
                                                         values=res_factor.values.reshape(
                                                             np.prod(int(res_factor.cardinality)), 1),
                                                         evidence=None, evidence_card=None,
                                                         state_names=res_factor.state_names)

            bay_safety_obj._ConditionalProbabilityTable__str_repr = str(
                res_factor)
            return bay_safety_obj

        else:
            new_name = '_'.join(res_factor.variables)
            bay_safety_obj = DiscreteFactor(name=new_name, scope=res_factor.variables, cardinalities=res_factor.cardinality,
                                            values=res_factor.values, state_names=res_factor.state_names)
            bay_safety_obj._DiscreteFactor__str_repr = str(res_factor)
            return bay_safety_obj
