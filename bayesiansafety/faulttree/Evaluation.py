"""
This class allows evaluation of Fault Trees.
"""
from typing import Dict, Optional
import numpy as np

from bayesiansafety.utils.utils import pprint_map

from bayesiansafety.faulttree.Cutset import Cutset
from bayesiansafety.faulttree.BayesianFaultTree import BayesianFaultTree


class Evaluation:
    """Class for evaluating a Fault Tree. This includes generating minimal cut sets, calculation of risk worths,
        and importances.

    Attributes:
        ft_inst (BayesianFaultTree):

    Raises:
            TypeError: Raised if an invalid Fault Tree is passed.
    """

    ft_inst = None

    def __init__(self, ft_inst: BayesianFaultTree) -> None:
        if not isinstance(ft_inst, BayesianFaultTree):
            raise TypeError(f"Given Faul Tree instance is of type {type(ft_inst)} but must be of type {BayesianFaultTree}.")

        self.ft_inst = ft_inst

    def set_ft(self, new_ft_inst: BayesianFaultTree) -> None:
        """Sets a new Fault Tree instance for evaluation.

        Args:
            new_ft_inst (BayesianFaultTree): New Fault Tree instance on which all future evaluations will be run on.

        Raises:
            TypeError: Raised if an invalid Fault Tree is passed.
        """
        if not isinstance(new_ft_inst, BayesianFaultTree):
            raise TypeError(f"Given Faul Tree instance is of type {type(new_ft_inst)} but must be of type {BayesianFaultTree}.")

        self.ft_inst = new_ft_inst

    def get_importances(self, at_time: Optional[float] = 0, importance_type: Optional[str] = "birnbaum") -> Dict[str, float]:
        """Calculate importance metrics (Birnbaum Important ('birnbaum') or "Fussell-Vesely-Important ('fussel-vesely") for
            each node in the Fault Tree.

        Args:
            at_time (float, optional): Time stamp at which the Fault Tree shall be evaluated (default 0).
            importance_type (str, optional): Type of the importance metric to use ('birnbaum' (default) or 'fussel-vesely')

        Returns:
            dict<str, float>: Dictionary of importances keyed by the node name sorted by highest value first.

        Raises:
            ValueError: Raised if requested importance type is not implemented. Currently valid requests are 'birnbaum' and 'fussel_vesely'.
        """

        # Die Birnbaum-Importanz (BI) bestimmt die maximale Erhoehung der Eintretenswahrscheinlichkeit
        # fuer den Ausfall einer Komponente im Vergleich dazu, dass die Komponente nicht
        # ausfaellt. Umgekehrt ausgedrueckt bezeichnet die Importanz, welche Verbesserung sich einstellen
        # wuerde, wenn die Komponente nicht mehr ausfaellt (z.B. durch Reparatur).

        # Die Fussel-Vesely-Importanz (FVI) gibt an, welches prozentuale Gewicht die Minimalschnitte
        # an der Eintretenswahrscheinlichkeit des untersuchten Haupt- oder Zwischenereignisses
        # haben, die das jeweilige Basisereignis beinhalten.
        # Die FVI zeigt auf, bei welchen Primaerereignissen eine Reduzierung der Wahrscheinlichkeiten
        # den groeßten Nutzen bringen wuerde, also die Eintretenswahrscheinlichkeit des
        # Hauptereignissen maßgeblich reduziert wuerde.

        if importance_type.lower() not in ['birnbaum', 'fussel_vesely']:
            raise ValueError(f"Invalid importance metric specified: {importance_type} but must be {['birnbaum', 'fussel_vesely']}.")

        cutset_probabilities = self.get_cutsets_with_probabilities(
            at_time=at_time)
        simulation_results = self.ft_inst.run_time_simulation(
            start_time=at_time, stop_time=at_time, simulation_steps=1, plot_simulation=False)

        # first accumulate the cutset probs
        importances = dict.fromkeys(
            [x.name for x in self.ft_inst.probability_nodes], 0)
        first_order_nodes = []
        for node in importances:
            for cur_set, set_prob in cutset_probabilities.items():
                if node in cur_set:
                    importances[node] += set_prob
                if len(cur_set) == 1:
                    first_order_nodes.append(node)

        if importance_type.lower() == 'birnbaum':
            # secon divide by the individual prob
            for node, sum_prob in importances.items():
                if sum_prob == 0.0 and node in first_order_nodes:
                    importances[node] = 1.0
                else:
                    divisor = simulation_results[node][0].cpt.get_probabilities()[
                        1]
                    importances[node] = importances[node] / \
                        divisor if divisor != 0 else np.nan

        if importance_type.lower() == 'fussel_vesely':
            sum_prob_all_cutsets = sum(cutset_probabilities.values())

            for node, sum_prob in importances.items():
                importances[node] = importances[node] / sum_prob_all_cutsets

        return sorted(importances.items(), key=lambda x: x[1], reverse=True)

    def get_risk_worths(self, method: Optional[str] = 'rrw', at_time: Optional[float] = 0, scoped_event: Optional[str] = None) -> Dict[str, float]:
        """Calculate risk worth metrics (Risk ReductionWorth ('rrw') (default) or Risk Achievement Worth ('raw') for
            each node in the Fault Tree.

        Args:
            method (str, optional): Type of the risk worth metric to use ('rrw' (default) or 'raw')
            at_time (float, optional): Time stamp at which the Fault Tree shall be evaluated (default 0).
            scoped_event (str, optional): Node to which the risk worth refers to. If no node is given
                all evaluation will refere to the top level event.

        Returns:
            dict<str, float>: Dictionary of risk worths keyed by the node name sorted by highest value first.

        Raises:
            ValueError: Raised if requested risk worth is not implemented. Currently valid requests are 'rrw' and 'raw'
        """

        # Risk ReductionWorth (RRW): Gibt die relative aenderung der Eintretenswahrscheinlichkeit des untersuchten
        # Haupt- oder Zwischenereignisses, indem die Wahrscheinlichkeit eines darunter
        # verknuepften Primaerereignisses auf null gesetzt wird an.
        # Der resultierende Wert gibt also an, um welchen Faktor sich die Zuverlaessigkeit des
        # untersuchten Systems bzw. Haupt- oder Zwischenereignisses erhoehen wuerde, wenn das
        # Primaerereignis eliminiert werden koennte (z. B. Fehlerausschluss, perfekte Diagnose etc.).

        # Risk Achievement Worth (RAW): Gibt die relative aenderung der Eintretenswahrscheinlichkeit des untersuchten
        # Haupt- oder Zwischenereignisses, indem die Wahrscheinlichkeit eines darunter
        # verknuepften Primaerereignisses auf eins gesetzt wird.
        # Der resultierende Wert gibt also an, um welchen Faktor sich die UNzuverlaessigkeit des
        # untersuchten Systems bzw. Haupt- oder Zwischenereignisses erhoehen wuerde, wenn das
        # Primaerereignis mit Sicherheit eintritt (z. B. Ausfall ist gewiss, keine Diagnosedeckung etc.).

        if method.lower() not in ["rrw", "raw"]:
            raise ValueError(f"Invalid risk analysis specified: {method} but must be {['rrw', 'raw']}.")

        modification_value = 0.0 if method.lower() == "rrw" else 1.0

        scoped_event = self.ft_inst.get_top_level_event_name(
        ) if scoped_event is None else scoped_event if scoped_event in self.ft_inst.model_elements.keys() else self.ft_inst.get_top_level_event_name()
        orig_scoped_prob = self.ft_inst.run_query_for_individual_nodes(
            individual_nodes=[scoped_event], at_time=at_time)[scoped_event].get_probabilities()[1]

        risk_worths = dict.fromkeys(
            [x.name for x in self.ft_inst.probability_nodes], 0.0)

        for node in risk_worths:
            modified_frates = dict.fromkeys([node], modification_value)
            modified_scoped_prob = self.ft_inst.run_query_for_individual_nodes(individual_nodes=[
                                                                               scoped_event], at_time=at_time, modified_frates=modified_frates)[scoped_event].get_probabilities()[1]

            if method.lower() == "rrw":
                risk_worths[node] = orig_scoped_prob / \
                    modified_scoped_prob if modified_scoped_prob != 0 else np.nan

            else:
                risk_worths[node] = modified_scoped_prob / \
                    orig_scoped_prob if orig_scoped_prob != 0 else np.nan

        return sorted(risk_worths.items(), key=lambda x: x[1], reverse=True)

    def get_cutsets_with_probabilities(self, at_time: Optional[float] = 0, algorithm: Optional[str] = "fatram") -> Dict[frozenset, float]:
        """Calculate all minimal cutsets. Additionally calculate for each set the product of the fault probabilites of its contained nodes.

        Args:
            at_time (float, optional): Time stamp at which the Fault Tree shall be evaluated (default 0).
            algorithm (str, optional): Cutset algorithm to use. Can either be Method of obtaining cutsets ('mocus') or Fault Tree Reduction Algorithm ('fatram') (default).


        Returns:
            dict<frozenset, float>: Dictionary of cutsets and their prod. probability keyed by the node name.
        """
        min_cuts = Cutset(self.ft_inst).get_minimal_cuts(algorithm=algorithm)
        simulation_results = self.ft_inst.run_time_simulation(
            start_time=at_time, stop_time=at_time, simulation_steps=1, plot_simulation=False)

        node_probabilities = {}
        for node, results in simulation_results.items():
            node_probabilities[node] = results[0].cpt.get_probabilities()[1]

        cutset_probabilities = {}
        for cutset in min_cuts:
            cutset_prob = 1.0
            for node in cutset:
                cutset_prob *= node_probabilities[node]
            cutset_probabilities[frozenset(cutset)] = cutset_prob

        return cutset_probabilities

    def evaluate_fault_tree(self, start_time: Optional[int] = 0, stop_time: Optional[int] = 1e5, simulation_steps: Optional[int] = 50, plot_dir: Optional[str] = None, include_risk_worths: Optional[bool] = False) -> None:
        """Convenience method to do a full analysis of the Fault Tree.
            All currently available metrics for the Fault Tree are evaluated (with reference to TLE in case of risk worths).
            If a plot directory is given, generate and save a plot of the time behaviour of the probability of fault for a all nodes.


        Args:
            start_time (float, optional): First time stamp of analysis.
            stop_time (float, optional): Last time stamp of analysis.
            simulation_steps (int, optional): Number of equidistant steps between first and last time stamp.
            plot_dir (path, optional): If a directory is given it indicate that the probability plots of all nodes should be saved. No plotting otherwises
            include_risk_worths (bool, optional): Flag indicating if risk worths (RRW, RAW) should be calculated.
                                        This has a drastic impact on runtime performance. Risk worth might give the same quantitative results as BI/FVI.
        """
        save_plots = True if plot_dir  is not None else False

        if save_plots:
            print("\n>> Running time analysis...")
            simulation_results = self.ft_inst.run_time_simulation(
                start_time=start_time, stop_time=stop_time, simulation_steps=simulation_steps, plot_simulation=False)

            for node in simulation_results.keys():
                print(f">> Generating plot for node: {node}")
                self.ft_inst.plot_time_simulation(
                    simulation_results=simulation_results, node_name=node, plot_dir=plot_dir)

        # first calculate and print cutset
        print("\n>> Calculating Cutsets...")
        cutsets = self.get_cutsets_with_probabilities(
            at_time=0, algorithm="fatram")

        print("\n>> Minimal cutsets (sorted by prod. of probabilities)")
        length_longest_set = len(max((str(set(key)) for key in cutsets.keys()), key=len))

        for cutset, prob in sorted(cutsets.items(), key=lambda x: x[1], reverse=True):
            str_cutset = str(set(cutset)).ljust(length_longest_set, " ")
            print(f"{str_cutset} : {'{:4e}'.format(float(prob))}")

        # second calculate and print metrics
        print("\n>> Calculating Metrics...")
        birnbaum_importances = self.get_importances(
            at_time=0, importance_type='birnbaum')
        fussell_vesely_importances = self.get_importances(
            at_time=0, importance_type='fussel_vesely')

        rrw_worths, raw_worths = None, None

        if include_risk_worths:
            rrw_worths = self.get_risk_worths(method='rrw', at_time=0)
            raw_worths = self.get_risk_worths(method='raw', at_time=0)

        print("\n>> Metrics ")
        joint = birnbaum_importances + fussell_vesely_importances
        joint = joint + rrw_worths + raw_worths if include_risk_worths else joint

        header = ['Node', 'Birnbaum Importance', 'Fussel-Vesely Importance']
        header = header + ['Risk Reduction Worth',
                           'Risk Achievement Worth'] if include_risk_worths else header

        tmp = {}
        for k, v in joint:
            tmp.setdefault(k, [k]).append(v)
        grouped = map(tuple, tmp.values())

        pprint_map(map_obj=grouped, header=header)
