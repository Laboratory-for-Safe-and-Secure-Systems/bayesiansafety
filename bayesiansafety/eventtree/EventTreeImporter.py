"""Class for importing Event Trees from an OpenPSA model exchange file or construction from manual configuartion.
See also https://open-psa.github.io/joomla1.5/index.php.html
"""
from typing import Dict, List, Set, Tuple, Union
import networkx as nx
from networkx.classes.digraph import DiGraph

from bayesiansafety.utils.utils import xml_tree_path_generator, create_uuid, create_random_uuid
from bayesiansafety.eventtree.EventTreeObjects import InitiatingEvent, FunctionalEvent, Consequence, Path
from bayesiansafety.eventtree.BayesianEventTree import BayesianEventTree


class EventTreeImporter:

    """Class for importing Event Trees from an OpenPSA model exchange format file or construction from manual configuartion.
    See also https://open-psa.github.io/joomla1.5/index.php.html
    """

    def load(self, xml_file_path: str) -> BayesianEventTree:
        """Parse an Event Tree from a file.
            Parsed tree object nodes will be given a data attribute containing objects
            of type BayesianEventTree.EventTreeObjects describing branching elements (i.e. functional event)
            path elements (i.e. branching probabilities) and consequences (i.e. possible outcome events).
            Method returns an instance of bayesianeventtree.BayesianEventTree.

        Args:
            xml_file_path (path): Path to the OpenPSA model exchange file.

        Returns:
            BayesianEventTree: Parsed tree structure as directed, acyclic graph object encapsulated in a BayesianEventTree.
        """
        connections, et_object_dict, tree_name = self.parse_tree_elements(
            xml_file_path)
        parsed_tree_model = self.build_tree(
            node_connections=connections, et_objects=et_object_dict)
        return BayesianEventTree(name=tree_name, tree_obj=parsed_tree_model)

    def construct(self, name: str, paths: Union[List[Tuple[FunctionalEvent, str, float]], Tuple[Consequence, str]]) -> BayesianEventTree:
        """Create an Event Tree from manual configuration. Configuration specifies all potential paths between the initiating event and all consequences.
            Constructed tree object nodes will be given a data attribute containing objects
            of type BayesianEventTree.EventTreeObjects describing branching elements (i.e. functional event)
            path elements (i.e. branching probabilities) and consequences (i.e. possible outcome events).
            Method returns an instance of bayesianeventtree.BayesianEventTree.

        Args:
            name (str): Name of the Event Tree
            paths (list<tuple<FunctionalEvent, str, float> or tuple<Consequence, str>>): Paths/configuartion specifies all potential paths between the initiating event and all consequences.
                A path is a list of tuples consisting of an instance of Event followed by the branch name and branching probability. The last tuple of each path has to be an
                instance of Consequence followed by the name of the actual outcome.
        Returns:
            BayesianEventTree: Parsed tree structure as directed, acyclic graph object encapsulated in a BayesianEventTree.
        """
        connections, et_object_dict = self.convert_tree_elements(paths)
        parsed_tree_model = self.build_tree(node_connections=connections, et_objects=et_object_dict)
        return BayesianEventTree(name=name, tree_obj=parsed_tree_model)

    def convert_tree_elements(self, paths: Union[List[Tuple[FunctionalEvent, str, float]], Tuple[Consequence, str]]) -> Set[Tuple[Tuple[str, str], Dict[str, Union[InitiatingEvent, FunctionalEvent, Path, Consequence]]]]:
        """ Parsing of a user defined definition of an Event Tree.
            Each parsed element is treated as unique and given a UUID.
            Element attributes (e.g. name, values...) acting as node data are collected in a dictionary.
            Connections beteween elements define the topology of the tree.

        Args:
            paths (list<tuple<FunctionalEvent, str, float> or tuple<Consequence, str>>): Paths/configuartion specifies all potential paths between the initiating event and all consequences.
                A path is a list of tuples consisting of an instance of Event followed by the branch name and branching probability. The last tuple of each path has to be an
                instance of Consequence followed by the name of the actual outcome.
        Returns:
            set< tuple<src uuid, dest uuid>>, dict<uuid, EventTreeObjects>: Node connections of the tree and dictionary specifying data for each node.

        Raises:
            ValueError: Raised if given paths are invalid.
        """
        node_connections = set()  # set< tuple<src uuid, dest uuid>>
        # dict of parsed Event Tree objects  with key: object uuid, value = Event Tree object
        et_object_dict = {}

        init_event_uuid = create_uuid(paths)
        et_object_dict[init_event_uuid] = InitiatingEvent("Init")

        for path in paths:
            if not isinstance(path[0][0], FunctionalEvent):
                raise ValueError(f"First element of the path: {path} needs to be an instance of Event and not {type(path[0][0])}")

            if not isinstance(path[-1][0], Consequence):
                raise ValueError(f"Last element of the path: {path} needs to be an instance of Consequence and not {type(path[-1][0])}")

            previous_elem_uuid = init_event_uuid
            for tup in path:
                if len(tup) == 3:

                    uuid_func_ev = create_uuid(
                        str(path[:path.index(tup)+1]) + str(tup[0]))
                    uuid_path = create_uuid(
                        str(path[:path.index(tup)+1]) + str(tup[:2]))

                    et_object_dict[uuid_func_ev] = FunctionalEvent(tup[0].name)
                    et_object_dict[uuid_path] = Path(f"{tup[0].name}.{tup[1]}", tup[1], tup[2], tup[0].name)

                    node_connections.add((uuid_func_ev, uuid_path))

                    node_connections.add((previous_elem_uuid, uuid_func_ev))
                    previous_elem_uuid = uuid_path

                elif len(tup) == 2:
                    uuid_cnsq = create_random_uuid()
                    et_object_dict[uuid_cnsq] = Consequence(tup[1])
                    node_connections.add((previous_elem_uuid, uuid_cnsq))

                else:
                    raise ValueError(f"Invalid path segment given {tup}.")

        return node_connections, et_object_dict

    def parse_tree_elements(self, xml_file_path: str) -> List[Tuple[Tuple[str, str], Dict[str, Union[InitiatingEvent, FunctionalEvent, Path, Consequence]]]]:
        """Generator-based traversing the XML structure of a given OpenPSA file.
            Each parsed element is treated as unique and given a UUID.
            Element attributes (e.g. name, values...) are collected in a dictionary.
            Paths leading to an element are collected as clear, human readeable path
            as well as an UUID-path containing the same elements.
            This allows easier mapping later on.

        Raises:
            ValueError: Raised if a parsed and processed element should be re-evaluated.

        Args:
            xml_file_path (path): Description

        Returns:
            list< tuple<src uuid, dest uuid>>, dict<uuid, EventTreeObjects>, str: Node connections of the tree, dictionary specifying data
                for each node and parsed name of the Event Tree
        """

        node_connections = []       # list< tuple<src uuid, dest uuid>>
        # key = xml element uuid, value = dict of attributes for that element
        xml_element_dict = {}
        # dict of parsed Event Tree objects  with key: object uuid, value = Event Tree object
        et_object_dict = {}
        root_id = None              # uuid of the root for this Event Tree

        for path_tuple, id_element_tup in xml_tree_path_generator(xml_file_path):
            node_identifier, elem_attrib = id_element_tup
            if node_identifier not in xml_element_dict:
                xml_element_dict[node_identifier] = elem_attrib
            tag_path = []
            element_path = []
            id_path = []
            for tag, node_id in path_tuple:
                tag_path.append(tag)
                id_path.append(node_id)
                element_path.append(xml_element_dict[node_id])

            # reverse paths for intuitive access (convenience)
            rev_tag_path = tag_path[::-1]
            rev_element_path = element_path[::-1]
            rev_id_path = id_path[::-1]

            if 'define-event-tree' not in rev_tag_path:
                continue

            opsa_obj = None
            opsa_obj_tree_id = None

            selector_type = rev_tag_path[0]

            # create Event Tree objects
            if selector_type == 'sequence':
                name = rev_element_path[0].get("name", None)
                opsa_obj = Consequence(name=name)
                opsa_obj_tree_id = rev_id_path[0]

            if selector_type == 'fork':
                name = rev_element_path[0].get("functional-event", None)

                opsa_obj = FunctionalEvent(name=name)
                opsa_obj_tree_id = rev_id_path[0]

            if selector_type == 'float':
                sourcing_func_event = rev_element_path[3].get(
                    "functional-event", None)
                state = rev_element_path[2].get("state", None)
                probability = rev_element_path[0].get("value", None)
                probability = float(
                    probability) if probability is not None else None

                name = f"{sourcing_func_event}.{state}"
                opsa_obj = Path(name=name, state=state,
                                probability=probability, f_event_name=sourcing_func_event)
                opsa_obj_tree_id = rev_id_path[2]

            if selector_type == 'initial-state':
                name = rev_element_path[1].get("name", None)
                opsa_obj = InitiatingEvent(name=name)
                opsa_obj_tree_id = rev_id_path[1]
                root_id = opsa_obj_tree_id

            if opsa_obj:
                if opsa_obj_tree_id not in et_object_dict:
                    et_object_dict[opsa_obj_tree_id] = opsa_obj

                else:
                    raise ValueError(f"{opsa_obj.container_type} object with name {name} and id {opsa_obj_tree_id} already processed.")

            # create node_connections:
            if selector_type == 'sequence':
                end_idx = rev_tag_path.index('initial-state')
                for idx in range(1, end_idx):
                    src = rev_id_path[idx]
                    dest = rev_id_path[idx-1]

                    new_connection = (src, dest)
                    if new_connection not in node_connections:
                        node_connections.append(new_connection)

        tree_name = et_object_dict[root_id].name
        return node_connections, et_object_dict, tree_name

    def build_tree(self, node_connections: List[Tuple[str, str]], et_objects: Dict[str, Union[InitiatingEvent, FunctionalEvent, Path, Consequence]]) -> DiGraph:
        """Create a directed, acyclic graph from the tree specification of the given OpenPSA XML-structure.

        Returns:
            networkx.DiGraph: Parsed tree structure as directed, acyclic graph object. Additional metadata is
                               added for each node via a custom attribute "data".

        Raises:
            ValueError: Raised if multiple roots should be encountered during the tree creation

        Args:
            node_connections ( list< tuple<src uuid, dest uuid>>): Node connections of the tree object (UUIDs)
            et_objects (dict<uuid, EventTreeObjects): Dictionary of parsed Event Tree objects  with key: object uuid, value = Event Tree object
                specifying the data of a tree node.
        """
        model = nx.DiGraph()
        model.add_edges_from(node_connections)

        # We need to construct an oriented tree from of a breadth-first-search
        # to have the correct order of nodes in the graphs nodes-dictionary (insertion order is preserved in Python 3.7+)
        # this is important for correct traversal of functional events (causal/temporal order)
        roots = list((v for v, d in model.in_degree() if d == 0))
        if len(roots) > 1:
            raise ValueError(f"Parsed Event Tree might be invalid. Found multiple ({len(roots)} root nodes: { [et_objects[root_id].name for root_id in roots] } ")

        root = roots[0]
        model = nx.bfs_tree(model, source=root)

        # next add parsed EventTreeObjects as data/payload to the network
        node_attributes = {node_id: {"data": et_object}
                           for node_id, et_object in et_objects.items()}
        nx.set_node_attributes(model, node_attributes)
        return model
