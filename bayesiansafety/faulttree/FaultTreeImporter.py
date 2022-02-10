"""Class for importing Fault Trees from an OpenPSA model exchange file.
    see also https://open-psa.github.io/joomla1.5/index.php.html
"""
from typing import Dict, List, Optional, Tuple, Set

from bayesiansafety.utils.utils import xml_tree_path_generator
from bayesiansafety.faulttree.FaultTreeProbNode import FaultTreeProbNode
from bayesiansafety.faulttree.FaultTreeLogicNode import FaultTreeLogicNode
from bayesiansafety.faulttree.BayesianFaultTree import BayesianFaultTree


class ParsedEventParams:
    name = None
    prob_value = None
    is_time_dependent = None

    def __init__(self, name: str, prob_value: Optional[float] = None, is_time_dependent: Optional[bool] = None) -> None:
        self.name = name
        self.prob_value = prob_value
        self.is_time_dependent = is_time_dependent


class FaultTreeImporter:
    """Class for importing Fault Trees from an OpenPSA model exchange format file.
        see also https://open-psa.github.io/joomla1.5/index.php.html
    """

    __xml_file_path = None
    __fault_tree_name = None     # name of the Fault Tree
    # key = xml element uuid, value = dict of attributes for that element
    __xml_element_dict = None
    __default_tree_name = "FaultTree"

    def load(self, xml_file_path: str) -> BayesianFaultTree:
        """Parse a Fault Tree and map it to a BayesianNetwork as an instance of core.BayesianFaultTree.


        Args:
            xml_file_path (path): Path to the OpenPSA model exchange file.

        Returns:
            core.BayesianFaultTree: A BayesianFaultTree instance.
        """
        self.__xml_file_path = xml_file_path
        self.__xml_element_dict = {}

        probability_nodes, logic_nodes = self.parse_tree_elements()
        return BayesianFaultTree(self.__fault_tree_name, probability_nodes, logic_nodes)

    def parse_tree_elements(self) -> Tuple[List[FaultTreeProbNode], List[FaultTreeLogicNode]]:
        """Generator-based traversing the XML structure of the OpenPSA file.
            Each parsed element is treated as unique and given a UUID.
            Element attributes (e.g. name, values...) are collected in a dict.
            Paths leading to an element are collected as clear, human readeable path
            as well as an UUID-path containing the same elements.
            This allows easier mapping later on.
            Based on the element-type logic nodes (i.e. gates) and probability nodes (i.e. basic events)
            are instantiated and later on forwarded to the ctor of core.BayesianFaultTree.

        Returns:
            tuple(list<bayesianfaulttree.FaultTreeProbNode>, list<bayesianfaulttree.FaultTreeLogicNode>): Lists of Fault Tree elements.
        """
        basic_events = {}  # where key = id, value = tuple(name, pbf, boolean (is time dependent) )
        gate_inputs = {}  # where key  = id, value = set<input name>
        gates = {}  # where key = id, value = tuple<name,  type =or, and....)>
        parameters = {}  # where key = name, value = tuple(id, val)
        event_param_mapping = {}  # where key = event id, value = param name

        for path_tuple, id_element_tup in xml_tree_path_generator(self.__xml_file_path):
            node_identifier, elem_attrib = id_element_tup
            if node_identifier not in self.__xml_element_dict:
                self.__xml_element_dict[node_identifier] = elem_attrib
            tag_path = []
            element_path = []
            id_path = []
            for tag, node_id in path_tuple:
                tag_path.append(tag)
                id_path.append(node_id)
                element_path.append(self.__xml_element_dict[node_id])

            # reverse paths for intuitive access (convenience)
            rev_tag_path = tag_path[::-1]
            rev_element_path = element_path[::-1]
            rev_id_path = id_path[::-1]

            if any(key in rev_tag_path for key in ['define-fault-tree', 'model-data']):

                selector_type = rev_tag_path[0]

                # parsing static events and params
                if selector_type == 'float':
                    name = rev_element_path[1].get("name", None)
                    node_identifier = rev_id_path[1]
                    prob_value = rev_element_path[0].get("value", None)
                    prob_value = float(prob_value) if prob_value else 0.0

                    if rev_tag_path[1] == "define-basic-event":
                        # combo indicates a basic event with a static prob of failure
                        if node_identifier not in basic_events:
                            basic_events[node_identifier] = ParsedEventParams(
                                name, prob_value, False)

                    if rev_tag_path[1] == "define-parameter":
                        # we found a paramter -> currently we only support lambdas
                        if node_identifier not in parameters:
                            parameters[name] = (node_identifier, prob_value)

                # parsing time dependent events (exponential only!)
                if selector_type == 'parameter':
                    if rev_tag_path[1] == "exponential" and rev_tag_path[2] == "define-basic-event":
                        # combo indicates a basic event with a time dependent prob of failure
                        param_name = rev_element_path[0].get("name", None)
                        event_name = rev_element_path[2].get("name", None)
                        event_node_identifier = rev_id_path[2]

                        basic_events[event_node_identifier] = ParsedEventParams(
                            event_name, None, True)
                        event_param_mapping[event_node_identifier] = param_name

                # parsing gates
                if selector_type in ('basic-event','gate'):
                    # indicates inputs as well as the gate combining the inputs itself
                    input_name = rev_element_path[0].get("name", None)
                    scoped_gate_name = rev_element_path[2].get("name", None)
                    scoped_gate_id = rev_id_path[2]

                    scoped_gate_type = rev_tag_path[1]

                    if scoped_gate_id not in gate_inputs:
                        gate_inputs[scoped_gate_id] = set([input_name])
                    else:
                        gate_inputs[scoped_gate_id].update([input_name])

                    if scoped_gate_id not in gates:
                        gates[scoped_gate_id] = (
                            scoped_gate_name, scoped_gate_type)

                if selector_type == 'define-fault-tree':
                    self.__fault_tree_name = rev_element_path[0].get(
                        "name", None)
                    self.__fault_tree_name = self.__fault_tree_name if self.__fault_tree_name else self.__default_tree_name

        return self.__preprocess_parsed_information(basic_events, gate_inputs, gates, parameters, event_param_mapping)

    def __preprocess_parsed_information(self, basic_events: Dict[str, Tuple[str, float, bool]], gate_inputs: Dict[str, Set[str]], gates: Dict[str, Tuple[str, str]], parameters: Dict[str, Tuple[str, float]], event_param_mapping: Dict[str, str]):
        # basic_events = dict() # where key = id, value = tuple(name, pbf, boolean (is time dependent) )
        # gate_inputs = dict() # where key  = id, value = set<input name>
        # gates = dict() # where key = id, value = tuple<name,  type =or, and....)>
        # parameters = dict() # where key = name, value = tuple(id, val)
        # event_param_mapping = dict() # where key = event id, value = param name

        # map lambdas for the time dependent nodes
        for node_identifier, event_params in basic_events.items():
            if event_params.is_time_dependent:
                scoped_lambda_name = event_param_mapping[node_identifier]
                scoped_lambda_value = parameters[scoped_lambda_name][1]
                event_params.prob_value = scoped_lambda_value
                basic_events[node_identifier] = event_params

        probability_nodes = [FaultTreeProbNode(
            event_params.name, event_params.prob_value, event_params.is_time_dependent) for event_params in basic_events.values()]
        logic_nodes = []
        for node_identifier, inputs in gate_inputs.items():
            gate_name, gate_type = gates[node_identifier]
            logic_nodes.append(FaultTreeLogicNode(
                gate_name, list(inputs), gate_type))

        return probability_nodes, logic_nodes
