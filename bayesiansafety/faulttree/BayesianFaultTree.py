"""
This class allows defining and managing binary Fault Trees.
"""
import sys
import copy
import collections
from typing import Dict, List, Union, Optional
from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt

from bayesiansafety.utils.utils import create_dir
from bayesiansafety.core import BayesianNetwork
from bayesiansafety.core import ConditionalProbabilityTable
from bayesiansafety.core.inference import InferenceFactory
from bayesiansafety.faulttree.FaultTreeLogicNode import FaultTreeLogicNode
from bayesiansafety.faulttree.FaultTreeProbNode import FaultTreeProbNode
from bayesiansafety.faulttree.SimulationResult import SimulationResult


class BayesianFaultTree:

    """Main class managing a (bayesian) binary Fault Tree.
        The class allows cutset and risk worth analysis as well as
        fault evaluation at different time steps.

    Attributes:
        logic_nodes (list<FaultTreeLogicNode>): List of all logic nodes (AND, OR) of the Fault Tree as instantiations of the FaultTreeLogicNode class.
        model (BayesianNetwork): Fault Tree represented as DiGraph instance (networX DiGraph)
        model_elements (dict<str, FaultTree*Node>): Lookup dictionary containing all nodes as instances of either FaultTreeLogicNode
                                                    or FaultTreeProbNode keyed by their name.
        name (str): Name of this Fault Tree
        node_connections (list<tuple<str, str>>): List of tuples defining the edges between model elements.
        probability_nodes (list<FaultTreeProbNode>): List of all probability nodes of the Fault Tree as instantiations of the FaultTreeProbNode class.
    """
    name = None
    probability_nodes = None
    logic_nodes = None
    node_connections = None

    model = None
    model_elements = None
    __tle_name = None

    def __init__(self, name: str, probability_nodes: List[FaultTreeProbNode], logic_nodes: List[FaultTreeLogicNode]) -> None:
        """Ctor of the BayesianFaultTree class.

        Args:
            name (str): Name of this Fault Tree instance
            probability_nodes (list<FaultTreeProbNode>): List of all probability nodes of the Fault Tree as instantiations of the FaultTreeProbNode class.
            logic_nodes (list<FaultTreeLogicNode>): List of all logic nodes (AND, OR) of the Fault Tree as instantiations of the FaultTreeLogicNode class.
        """
        self.probability_nodes = probability_nodes
        self.logic_nodes = logic_nodes
        self.model_elements = dict([(node.name, node) for node in self.probability_nodes + self.logic_nodes])
        self.name = name
        self.__build_model()

    def __build_model(self) -> None:
        """Setup method to initialize instances of BayesianFaultTree.
        """
        self.__build_node_connections()
        self.__verify_node_connections()
        self.model = BayesianNetwork(
            name=self.name, node_connections=self.node_connections)
        self.__poplulate_model()

    def __poplulate_model(self) -> None:
        """Setup method to add the CPTs to the network.

        Raises:
            TypeError: Raised if an element that is not an instance of FaultTree*Node is provided.
        """
        # model (Bayesian Network currently is only a skeleton)
        # we need to add the CPTs
        for model_element in self.model_elements.values():
            if isinstance(model_element, FaultTreeProbNode):
                self.model.add_cpts(model_element.get_cpt_at_time(at_time=0))

            elif isinstance(model_element, FaultTreeLogicNode):
                self.model.add_cpts(model_element.cpt)
            else:
                raise TypeError(f"Unsupported type {type(model_element)} for model element: {model_element.name}")

    def __build_node_connections(self) -> None:
        """Generate list of tuples defining the edges between model elements.
        """
        self.node_connections = []

        # only logic nodes connect model elements in a Fault Tree
        for logic_node in self.logic_nodes:
            input_nodes = logic_node.cpt.evidence
            self.node_connections += list(zip(input_nodes,
                                              [logic_node.name]*len(input_nodes)))

    def __verify_node_connections(self) -> None:
        """Verify all nodes are part of the edges between model elements.

        Raises:
            ValueError: Raised when there is a mismatch between model elements and given node connections.
        """
        connected_nodes = set()
        for connection in self.node_connections:
            connected_nodes = connected_nodes.union(set(list(connection)))

        if len(connected_nodes ^ set(self.model_elements.keys())) != 0:
            raise ValueError(f"Mismatch between node connections: {self.node_connections} and provided model elements: {self.model_elements.keys()}.")

    def add_prob_node(self, node_name: str, input_to: str, probability_of_failure: float, is_time_dependent: Optional[bool] = False) -> None:
        """Helper method to add probability nodes to the Fault tree.

        Args:
            node_name (str): Name of the probability node
            input_to (str): Name of the logic gate where this node is an input to.
            probability_of_failure (float): The probabilit of failure for this node. If this node is time dependet it specifies the fault rate lambda.
            is_time_dependent (bool, optional): Flag indicating if this node has a static pbf (False, default) or if True an exponential time behaviour.

        Raises:
            TypeError: Raised if the targeted gate is not a logic node.
            ValueError: Raised if the targeted gate does not exist.
        """
        if input_to not in self.model_elements.keys():
            raise ValueError(f"Target Faul Tree element: {input_to} where the node is an input to is not part of this Fault Tree.")

        if not isinstance(self.model_elements[input_to], FaultTreeLogicNode):
            raise TypeError(f"Target Faul Tree element: {input_to} is not a logic node.")

        new_prob_node = FaultTreeProbNode(
            name=node_name, probability_of_failure=probability_of_failure, is_time_dependent=is_time_dependent)
        mod_logic_node = FaultTreeLogicNode(name=input_to, input_nodes=self.model_elements[input_to].input_nodes + [
                                            str(node_name)], logic_type=self.model_elements[input_to].get_node_type())

        # update lists
        self.logic_nodes = [
            logic_node for logic_node in self.logic_nodes if logic_node.name != input_to] + [mod_logic_node]
        self.probability_nodes.append(new_prob_node)
        self.model_elements[node_name] = self.probability_nodes[-1]
        self.model_elements[input_to] = mod_logic_node

        self.__build_model()

    def copy(self) -> 'BayesianFaultTree':
        """Helper method to make a deep copy of this instance.

        Returns:
            BayesianFaultTree: Returns deep copy of this instance.
        """
        return copy.deepcopy(self)

    def get_elem_by_name(self, node_name: str) -> Union[FaultTreeProbNode, FaultTreeLogicNode]:
        """Getter to access a model element (instance of FaultTree*Node class) by name.

        Args:
            node_name (str): Name of the queried node.

        Returns:
            <FaultTree*Node>: Model element as an instance of FaultTree*Node class.

        Raises:
            ValueError: Raised if scoped element is not part of this Fault Tree.
        """
        if node_name not in self.model_elements:
            raise ValueError(f"Scoped element: {node_name} could not be found in given model elements: {self.model_elements.keys()}.")

        return self.model_elements[node_name]



    def get_top_level_event_name(self) -> str:
        """Getter to acces the name of the top level event.

        Returns:
            str: Name of the top level event.

        Raises:
            ValueError: Raised if TLE can not be uniquely identified.
        """
        if self.__tle_name is None:
            leaves = list((v for v, d in self.model.out_degree() if d == 0))

            if len(leaves) != 1:
                raise ValueError(f"Tree contains more than one leaf: {leaves}. The top hazard should be the only leaf. Something is modeled wrong.")

            self.__tle_name = leaves[0]

        return self.__tle_name

    def get_tree_at_time(self, at_time: Optional[float] = 0, modified_frates: Optional[Dict[str, float]] = None) -> 'BayesianFaultTree':
        """Get a copy of this Fault Tree with it's probabilities of failure evaluated and fixed at a given time.
            For this evaluation individual fault rates can be modified.

        Args:
            at_time (int, optional): Time at which this Fault Tree is evaluated.
            modified_frates (dict<str, float>, optional): Dictionary where key = node name, value = modified fault rate for the evaluation.

        Returns:
            BayesianFaultTree: New BayesianFaultTree instance with all nodes pbf's fixed and evaluated at given time.

        Raises:
            TypeError: Raised if a scoped node for fault rate modification is a logic gate.
            ValueError: Raised if a given modified fault rate is not a valid probability (0...1)
        """
        # modified_frates see docs run_query_for_individual_nodes
        if modified_frates is not None:
            for mod_node, mod_rate in modified_frates.items():
                if not isinstance(self.get_elem_by_name(mod_node), FaultTreeProbNode):
                    raise TypeError(f"Invalid given node: {mod_node}. Check if node is a probability node.")
                if mod_rate is None or mod_rate > 1.0 or mod_rate < 0.0:
                    raise ValueError(f"Invalid modified frate: {mod_rate} for given node: {mod_node}")
        else:
            modified_frates = {}

        tmp_model = self.model.copy()
        tmp_model_elements = copy.deepcopy(self.model_elements)

        for model_element in tmp_model_elements.values():
            if model_element.name in modified_frates.keys():
                model_element.change_frate(
                    probability_of_failure=modified_frates[model_element.name])

            if isinstance(model_element, FaultTreeProbNode):
                tmp_model.add_cpts(
                    model_element.get_cpt_at_time(at_time=at_time))

            elif isinstance(model_element, FaultTreeLogicNode):
                tmp_model.add_cpts(model_element.cpt)

        tree_at_time = BayesianFaultTree(
            name=self.name, probability_nodes=self.probability_nodes, logic_nodes=self.logic_nodes)
        tree_at_time.model = tmp_model
        tree_at_time.model_elements = tmp_model_elements
        return tree_at_time

    def run_time_simulation(self, start_time: Optional[int] = 0, stop_time: Optional[int] = 1e5, simulation_steps: Optional[int] = 50, node_name: Optional[str] = None, plot_simulation: Optional[bool] = True) -> Dict[str, List[SimulationResult]]:
        """Time analysis of the Fault Tree. Execution will update the member variable self.simulation_results.
            The intended (external) use is to generate a plot of the time behaviour of the probability of fault for a specified node.
            If no node is specified the top level event is used as the default node.

        Args:
            start_time (float, optional): First time stamp of analysis.
            stop_time (float, optional): Last time stamp of analysis.
            simulation_steps (int, optional): Number of equidistant steps between first and last time stamp.
            node_name (str, optional): Name of the node for which its time behaviour should be plotted.
            plot_simulation (bool, optional): Flag to indicate if a plot should be generated.


        Returns:
            dict<str, list<SimulationResult>>: Dictionary containing the prior CPTs of all nodes for the executed time simulation.
                Keys are node names, values is a list of instances of the SimulationResult class for each time stamp.
        """
        start_time = start_time if start_time > 0 else 0
        stop_time = stop_time if stop_time > start_time else start_time
        simulation_steps = simulation_steps if simulation_steps > 0 else 10

        time_steps = np.linspace(
            start=start_time, stop=stop_time, num=simulation_steps)

        simulation_results = collections.defaultdict(list)

        for time in tqdm(time_steps, desc="Running simulation", bar_format='{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} steps', file=sys.stdout):
            prior_cpts = self.run_query_for_individual_nodes(
                individual_nodes=self.model_elements.keys(), at_time=time, modified_frates=None)
            for node, prior in prior_cpts.items():
                simulation_results[node].append(SimulationResult(
                    node_name=node, simulation_time=time, cpt=prior))

        if plot_simulation:
            self.plot_time_simulation(
                simulation_results=simulation_results, node_name=node_name)

        return simulation_results

    def plot_bayesian_fault_tree(self, at_time: Optional[float] = 0) -> None:
        """Plot the Fault Tree as networkX - DiGraph evaluated at a specified time stamp.
            The edges give the current probability of failure for the node they emerge from.

        Args:
            at_time (float, optional): Time stamp at which the Fault Tree shall be evaluated (default 0).
        """
        drawing_model = self.model.copy()
        simulation_results = self.run_time_simulation(
            start_time=at_time, stop_time=at_time, simulation_steps=1, plot_simulation=False)
        edge_labels = collections.defaultdict(str)

        # calc values for original Fault Tree
        for node_connection in self.node_connections:
            edge_labels[node_connection] = "{:.2e}".format(
                float(simulation_results[node_connection[0]][0].cpt.get_probabilities()[1]))

        # to display the PBF of the top hazard we need to add an artificial edge
        top_hazard_name = self.get_top_level_event_name()
        drawing_model.add_edges_from([(top_hazard_name, "DUMMY_FOR_DISP")])
        edge_labels[(top_hazard_name, "DUMMY_FOR_DISP")] = "{:2e}".format(
            float(simulation_results[top_hazard_name][0].cpt.get_probabilities()[1]))

        title = f"{drawing_model.name} evaluation for timestamp: {round(at_time, 2)}"
        drawing_model.plot_graph(title=title, edge_labels=edge_labels)

    def run_query_for_individual_nodes(self, individual_nodes: List[str], at_time: Optional[float] = 0, modified_frates: Optional[Dict[str, float]] = None) -> Dict[str, ConditionalProbabilityTable]:
        """Convenience method to query the prior cpts of nodes for a specified time stamp (intended for internal use).
            This method also allows to evaluate the network with temporarily modified fault rates for some probability nodes (instances of FaultTreeProbNode).

        Args:
            individual_nodes (list<str>): List of node names for which the prior CPTs should be calculated.
            at_time (float, optional): Time stamp at which the Fault Tree shall be evaluated (default 0).
            modified_frates (dict<str, float>, optional): Dictionary of nodes for which their probability of failure should temporarily be modified. Elements are
            keyed by the node name, values give the modified probability of failure.

        Returns:
            dict<str, ConditionalProbabilityTable>: Dictionary of prior CPTs keyed by the node name.

        Raises:
            ValueError: Raised if scoped individual node is not part of this Fault Tree or if an invalid modified fault rate is provided.
        """
        for node in individual_nodes:
            if node is None or node not in self.model_elements:
                raise ValueError(f"Invalid scoped event: {node}.")

        if modified_frates is not None:
            for mod_node, mod_rate in modified_frates.items():
                if mod_node is None or mod_node not in self.model_elements or mod_rate is None or mod_rate > 1.0 or mod_rate < 0.0:
                    raise ValueError(f"Involid modified frate: {mod_rate} for given node: {mod_node}")
        else:
            modified_frates = {}

        tree_at_time = self.get_tree_at_time(at_time=at_time, modified_frates=modified_frates)
        prior_cpts = {}
        inference_engine = InferenceFactory(tree_at_time.model).get_engine()

        for node in individual_nodes:
            prior_cpts[node] = inference_engine.query(node)

        return prior_cpts

    def plot_time_simulation(self, simulation_results: Dict[str, List[SimulationResult]], node_name: Optional[str] = None, plot_dir: Optional[str] = None) -> None:
        """Helper method to plot the time behaviour of the probability of fault for a specified node.
            This method is meant to be used internally. It is called via BayesianFaultTree.run_time_simulation(...)
            If a path to a "plot_dir" is given, save the figure instead of showing it.

        Args:
            node_name (str, optional): Name of the node for which its time behaviour should be plotted.
            plot_dir (path, optional): If a directory is given figure is saved instead of shown.
            simulation_results (dict<str, list<SimulationResult>>): Dictionary containing the prior CPTs of all nodes for the last executed time simulation
            (via BayesianFaultTree.run_time_simulation(...)). Keys are node names, values is a list of instances of the SimulationResult class.

        """
        scale = 'log'
        fault_prob = []
        timestamps = []
        node_name = node_name if node_name is not None else self.get_top_level_event_name()
        save_fig = True if plot_dir is not None else False

        for time_step_result in simulation_results[node_name]:
            fault_prob.append(time_step_result.cpt.get_probabilities()[1])
            timestamps.append(time_step_result.simulation_time)

        plt.plot(timestamps, fault_prob, label=f"PBF for node: {node_name} of type {self.get_elem_by_name(node_name).get_node_type()}")
        plt.xlabel("Time (log)")
        plt.ylabel("Probability of failure")
        plt.legend()
        plt.xscale(scale)

        if save_fig:
            create_dir(plot_dir)
            model_name_tag = self.name + "_" if self.name != "" else ""
            img_path = plot_dir + f"/{model_name_tag}{self.get_elem_by_name(node_name).get_node_type()}_{node_name}.png"

            print(f"Saving plot for node: {node_name} under {img_path} ")
            figure = plt.gcf()
            figure.set_size_inches(19.2, 10.08)
            plt.savefig(img_path, dpi=100)

        else:
            plt.show()

        plt.close('all')
