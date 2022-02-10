"""
This class is a wrapper for the internal representation of Inference engines to
seperate the actual implementaiton from the usage througout the package.
This implementation wraps the library "PyAgrum".
See  https://agrum.gitlab.io/  and   https://gitlab.com/agrumery/aGrUM
"""
import copy
from typing import List, Optional, Union, Tuple, Any
import numpy as np

import pyAgrum.causal as pyagrumCausal
from pyAgrum import BayesNet as pyagrumBayesNet
from pyAgrum import LazyPropagation as pyagrumLazyPropagation
from pyAgrum import LabelizedVariable as pyagrumLabelizedVariable

from bayesiansafety.utils.utils import remove_duplicate_tuples
from bayesiansafety.core import ConditionalProbabilityTable
from bayesiansafety.core import DiscreteFactor
from bayesiansafety.core import BayesianNetwork
from bayesiansafety.core.inference.IInference import IInference
from bayesiansafety.core.inference.TwinNetwork import TwinNetwork


class PyagrumInference(IInference):

    """This class is a wrapper for the internal representation of Inference engines to
        seperate the actual implementaiton from the usage througout the package.
        This implementation wraps the library "PyAgrum".
        See  https://agrum.gitlab.io/  and   https://gitlab.com/agrumery/aGrUM

    Attributes:
        model (core.BayesianNetwork): Instance of a BaySafety-BN on which queries should be run.
    """

    model = None
    __inference_engine_inst = None  # pyagrum LazyPropagation
    __internal_model = None  # pyagrum BayesModel

    def __init__(self, model: BayesianNetwork) -> None:
        """Ctor of the Inference class for backend pyagrum.

        Args:
            model (core.BayesianNetwork): Instance of a BaySafety-BN on which queries should be run.
        """
        self.model = model
        self.__build_internal_model()
        self.__inference_engine_inst = pyagrumLazyPropagation(
            self.__internal_model)

    def __build_internal_model(self) -> None:
        """Setup method to instantiate a model that is compatible with the internally used, encapsulated inference algorithm.
        """
        self.__internal_model = pyagrumBayesNet(self.model.name)

        def bay_safety_cpt_to_agrum_potential(bay_arr, var_card, evidence_card):
            """Custom helper to transform the a BaySafety-CPT to a pyagrum CPT

            Args:
                bay_arr (core.ConditionalProbabilityTable.values): Values (2-D array like) that represent the conditional
                        probabilities of the current node we want to convert.
                var_card (int): Cardinality of the current node (number of states) - needed to reshape the passed values.
                evidence_card (list<int>): Evidence cardinalities - needed to reshape the passed values.

            Returns:
                res_arr (2-D like array): Converted/reshaped values representing the value layout for a pyagrum CPT.
            """
            total_card = np.prod(evidence_card)
            flat_bay_arr = np.ravel(bay_arr)
            res_arr = []
            for i in range(total_card):
                res_arr.append(flat_bay_arr[i::total_card])

            target_shape = evidence_card + [var_card]
            res_arr = np.reshape(res_arr, target_shape)

            return res_arr

        # add nodes
        for node_name, node in self.model.model_elements.items():
            states = node.state_names[node_name] if node.state_names is not None and node_name in node.state_names.keys(
            ) else node.variable_card
            self.__internal_model.add(
                pyagrumLabelizedVariable(node_name, node_name, states))

        # add connections
        for link in remove_duplicate_tuples(self.model.node_connections):
            self.__internal_model.addArc(*link)

        # add cpts
        def correct_evidence_ordering(orig_evidence, orig_cardinalities, other_evidence):
            """Order of evidence in CPT may differ after converstion from BaySafety to pyagrum CPT which affects how the
                given values (e.g. conditional probabilities) are interpreted.
                This method is used to adjust the evidence ordering to comply to the converted values.

            Args:
                orig_evidence (list<str>): Original list of evidence for this node (BaySafety-CPT).
                orig_cardinalities (list<int>): Original list of cardinalities for the given orig_evidence (BaySafety-CPT).
                other_evidence (list<str>): New list of evidence for the converted CPT (pyagrum-CPT).

            Returns:
                corrected_evidence_card, mapping (list<int>, list<int>): Corrected list of evidence cardinalities, mapping of which indices need to be switched.


            Example:
                adjust ordering of evidence
                e.g. orig BaySafety order was A,B,C    and pyagrum order was B, A, C
                e.g. orig BaySafety card  was 2,3,4  generated pyagrum order 3, 2, 4
                mapping to get this (indices)                               [1, 0, 2]
            """

            corrected_evidence_card = copy.deepcopy(orig_cardinalities)
            mapping = []
            for ref in orig_evidence:
                for comp in other_evidence:
                    if comp == ref:
                        corrected_evidence_card[other_evidence.index(
                            ref)] = orig_cardinalities[orig_evidence.index(ref)]
                        mapping.append(other_evidence.index(ref))

            return corrected_evidence_card, mapping

        for node_name, node in self.model.model_elements.items():
            if node.evidence_card is not None and len(node.evidence_card) >= 1:
                corrected_evidence_card, mapping = correct_evidence_ordering(
                    node.evidence, node.evidence_card, self.__internal_model.cpt(node_name).names[::-1])
                # in pyagrum the last axis of the CPT contains the conditional probs - this axis needs to be added
                mapping.append(-1)

                # 1) transform BaySafety CPT to pyagrum assuming the order of evidence vars is correct
                reshaped = bay_safety_cpt_to_agrum_potential(
                    node.values, node.variable_card, node.evidence_card)

                # 2) Swap entries to give the acutal evidence ordering
                corrected_values = np.transpose(reshaped, mapping)

                # 3) Add the CPT to the pyagrum network
                self.__internal_model.cpt(node_name)[:] = corrected_values
            else:
                self.__internal_model.cpt(node_name).fillWith(
                    np.ravel(node.values).tolist())

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

        # query_result is a pyagrum.Potential
        evidence = dict(evidence) if evidence is not None else {}
        self.__inference_engine_inst.setEvidence(evidence)

        if len(variables) <= 1:
            self.__inference_engine_inst.makeInference()
            query_result = self.__inference_engine_inst.posterior(
                str(variables[0]))

            return self.__build_bay_safety_result(query_result, is_cpt=True)

        else:
            self.__inference_engine_inst.addJointTarget(set(variables))
            self.__inference_engine_inst.makeInference()
            query_result = self.__inference_engine_inst.jointPosterior(
                set(variables))

            return self.__build_bay_safety_result(query_result, is_cpt=False)

    def interventional_query(self, variables: Union[str, List[str]], do: Optional[List[Tuple[str, str]]] = None, evidence: Optional[List[Tuple[str, str]]] = None) -> Union[ConditionalProbabilityTable, DiscreteFactor]:
        """This method can be used to run interventional inference on the current BN instance.
            Depending on the queried variables either a ConditionalProbabilityTable (only one var.) or a DiscreteFactor (multiple vars.) instance is returned.
            Internally pyAgrum uses do-calculus or *door-criterias to build the estimation formula.
            Due to this the estimand, estimate and logic used to create the estimand are available as return values.

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

        params_on = set(variables)
        params_do = dict(do) if do is not None else {}
        params_knowing = dict(evidence) if evidence is not None else {}
        params_values = {**params_do, **params_knowing}

        formula, query_result, explanation = pyagrumCausal.causalImpact(cm=pyagrumCausal.CausalModel(self.__internal_model),
                                                                        on=params_on, doing=params_do.keys(),
                                                                        knowing=params_knowing.keys(), values=params_values)
        print(f"Formula: {formula}")
        print(f"Explanation: {explanation}")
        print(f"Query Result: {query_result}")
        #latex_forumula = formula.toLatex()

        if len(variables) <= 1:
            return self.__build_bay_safety_result(query_result, is_cpt=True)

        return self.__build_bay_safety_result(query_result, is_cpt=False)

    def counterfactual_query(self, target: Union[str, List[str]], do: List[Tuple[str, str]], observed: Optional[List[Tuple[str, str]]] = None) -> Union[ConditionalProbabilityTable, DiscreteFactor]:
        """This method can be used to run counterfactual inference on the current BN instance.
            Internally a twin-network is used  (see Pearl 2009, Ch. 7).
            Root nodes are treated as exogenous variables.

            pyAgrum supports the 3-step approach via internal methods (Abduction, Action, Prediction).
            Due to current implementation progress the custom twin-network approach is used.

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

        # Current approach via custom twin-network
        target = target if isinstance(target, list) else [target]
        prefixed_do = {TwinNetwork.TWIN_NET_PREFIX +
                       var: state for (var, state) in do.items()}
        twin_model = TwinNetwork.build_twin_network(self.model)

        inf = PyagrumInference(twin_model)
        query_result = inf.interventional_query(variables=[
                                                TwinNetwork.TWIN_NET_PREFIX + var for var in target], do=prefixed_do, evidence=observed)

        # pyAgrum original method - untested
        # params_profile = observed if observed is not None else {}
        # params_whatif = dict(do) if do is not None else {}
        # params_on = set( target if isinstance(target, list) else [target] )
        # params_values = dict(do)

        # query_result = pyagrumCausal.counterfactual( cm = pyagrumCausal.CausalModel(self.__internal_model),
        #                                 profile = params_profile, whatif = params_whatif.keys(),
        #                                 on = params_on, values = params_whatif )

        # if len(params_on) <= 1:
        #     return self.__build_bay_safety_result(query_result, is_cpt=True)

        # else:
        #     return self.__build_bay_safety_result(query_result, is_cpt=False)

        return query_result

    def __build_bay_safety_result(self, res_potential: Any, is_cpt: Optional[bool] = True) -> Union[ConditionalProbabilityTable, DiscreteFactor]:
        """Method to convert the internal (pyagrum) representation of a factor/cpt
            into a BaySafety-CPT or BaySafety-DiscretFactor.
            This method is needed to encapsulate backend representation of results
            from the rest of BaySafety.

        Args:
            res_potential (pyAgrum.Potential): Result of a query.
            is_cpt (bool, optional): Flag indicating if the result should be interpreted as
                    a CPT or DiscreteFactor.

        Returns:
            core.ConditionalProbabilityTable or core.DiscreteFactor: BaySafety representation
                    of query results.
        """
        if is_cpt:
            var_card = res_potential.shape[0]
            ev_card = res_potential.shape[1:] if len(
                res_potential.shape) > 1 else None
            evidence = res_potential.names[1:] if len(
                res_potential.names) > 1 else None
            var_name = res_potential.names[0]
            values = res_potential.toarray()
            state_names = {var_name: list(
                self.__internal_model.variable(str(var_name)).labels())}

            bay_safety_obj = ConditionalProbabilityTable(name=var_name, variable_card=var_card,
                                                         values=np.reshape(values, (var_card, -1)), evidence=evidence,
                                                         evidence_card=ev_card, state_names=state_names)

            bay_safety_obj._ConditionalProbabilityTable__str_repr = str(
                res_potential)
            return bay_safety_obj

        else:
            var_card = res_potential.shape[0]
            var_name = res_potential.names[0]
            values = res_potential.toarray()
            state_names = {}
            for var in res_potential.names:
                state_names[var] = list(
                    self.__internal_model.variable(str(var)).labels())

            bay_safety_obj = DiscreteFactor(name=var_name, scope=res_potential.names[::-1], cardinalities=res_potential.shape[::-1],
                                            values=np.reshape(values, (var_card, -1)), state_names=state_names)

            bay_safety_obj._DiscreteFactor__str_repr = str(res_potential)
            return bay_safety_obj
