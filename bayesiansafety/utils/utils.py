"""Collection of generic helper functions.
"""
import os
import uuid
import xml.etree.ElementTree as ET
from typing import Any, List,  Set, Tuple,  Union, Mapping
from networkx.classes.digraph import DiGraph

def create_dir(path: str) -> None:
    """Create a directory if it does not already exists.

    Args:
        path (path): path to the directory that should be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def check_file_exists(file_path: str) -> None:
    """Checks if file exists and raises an exception if not.

    Args:
        file_path (str): Path to the file under question.

    Raises:
        ValueError: Raised if file does not exist
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"Given file path: {file_path} does not lead to an existing file.")


def remove_duplicate_sets(list_of_sets: List[Set[str]]) -> List[Set[str]]:
    """Remove duplicate entries in a list of sets.

    Args:
        list_of_sets (list<set>): List of sets containing duplicates

    Returns:
        list<set>: List of unique sets.
    """
    individual = list(set([frozenset(elem) for elem in list_of_sets]))
    return [set(elem) for elem in individual]


def remove_duplicate_tuples(list_of_tuples: List[Tuple]) -> List[Tuple]:
    """Remove duplicate entries in a list of tuples.

    Args:
        list_of_tuples (list<tuple>): List of tuples containing duplicates

    Returns:
        list<tuple>: List of unique tuples.
    """
    # return list(set([elem for elem in list_of_tuples])) # this does not preserve the order of elements
    unqiues = []
    for tup in list_of_tuples:
        if tup not in unqiues:
            unqiues.append(tup)
    return unqiues


def flatten_list_of_lists(list_of_lists: List[List[Any]]) -> List[Any]:
    """Create a flat list from a list of lists.

    Args:
        list_of_lists (list<list<obj>>): The list containing other lists to flatten.

    Returns:
        list<obj>: Flat list of objects.
    """

    return [item for sublist in list_of_lists for item in sublist]


def pprint_map(map_obj: Union[List[Tuple[str]], Mapping], header: Tuple[str]) -> None:
    """Pretty print map objects. This is mainly used when presenting dataframe like data

    Args:
        map_obj (map, list<tuple<str>>): Map or list of tuples representing one row of data to print
        header (tuple<str>): Header for data to print (column description)

    Returns:
        None: Returns when no data is provided.

    Raises:
        TypeError: Error if passed data (map_obj) is not of type map or list
        ValueError: Raised if there is a mismatch between number of columns described by header and actual data.
    """
    # same functionality
    #import pandas as pd
    #df = pd.DataFrame(map_obj, columns = header)
    # print(df.to_string())

    free_space = 4
    if not isinstance(map_obj, (map, list)):
        raise TypeError("Given map object is not of type list or map.")

    if not isinstance(header, (tuple, list)):
        raise TypeError("Given header is not a tuple.")

    map_obj = list(map_obj)

    if len(map_obj) == 0:
        print(" ".join(header))
        print("No data was provided")
        return

    if len(map_obj[0]) != len(header):
        raise ValueError(
            "Header length does not match number of columns to print.")

    col_widths = dict.fromkeys(header, 0)
    str_map_obj = [[str(x) for x in tup] for tup in map_obj]

    for col, elem in enumerate(zip(*str_map_obj)):
        col_widths[header[col]] = len(max(elem, key=len))

    print(" ".join([elem.ljust(col_widths[elem] + free_space, " ")
                    for elem in header]))
    for data in map_obj:
        print(" ".join([str(elem).ljust(col_widths[header[col]] +
                                        free_space, " ") for col, elem in enumerate(data)]))


def create_uuid(obj: Any) -> str:
    """Create a unique identifier string based on a given object.

    Args:
        obj (Any): Object to create an identifier for.

    Returns:
        str: Unique identifier
    """
    return str(uuid.uuid3(uuid.NAMESPACE_OID, str(obj)))


def create_random_uuid() -> str:
    """Create a random unqique identifier string.

    Returns:
        str: Unique identifiere.
    """
    return str(uuid.uuid4())


def xml_tree_path_generator(xml_file_path: str) -> List[Tuple[str, str]]:
    """Generator functions providing all traversable element paths of a XML tree.

    Yields:
        list<tuple<element tag, uuid>>: List of paths (clear view element names, element uuid)
                                                                                  and the last path ElementTree.Elements attribute dictionary.
    """
    check_file_exists(xml_file_path)

    path = []
    id_element_tup = None
    iterator = ET.iterparse(xml_file_path, events=('start', 'end'))
    for event, element in iterator:
        if event == 'start':
            node_identifier = create_uuid(element)

            id_element_tup = (node_identifier, element.attrib)
            # if node_identifier not in self.__xml_element_dict.keys():
            #    self.__xml_element_dict[node_identifier] = element.attrib

            path.append((element.tag, node_identifier))
            yield path, id_element_tup
        else:
            path.pop()


def get_root_and_leaves(tree_obj: DiGraph) -> Tuple[str, List[str]]:
    """Returns the root name as well as the names of all leaves (i.e. outcomes, basis events) of a given
        Event Tree DiGraph-object.

    Args:
        tree_obj (DiGraph): DiGraph object (e.g. Event Tree or Fault Tree)

    Returns:
        Tuple[str, Tuple[str, ...]]: root, tuple of leave names

    Raises:
        TypeError: Raised if given tree object is not a DiGraph.
        ValueError: Raised if there is more than 1 root node.
    """

    if not isinstance(tree_obj, DiGraph):
        raise TypeError(f"Tree object must be a valid networkX DiGraph but is of type {type(tree_obj)}.")


    roots = list((v for v, d in tree_obj.in_degree() if d == 0))

    if len(roots) != 1:
        raise ValueError(f"Found {len(roots)}: {roots} for given DiGraph but expected 1 root only.")

    root = roots[0]
    leaves = list((v for v, d in tree_obj.out_degree() if d == 0))
    return root, leaves
