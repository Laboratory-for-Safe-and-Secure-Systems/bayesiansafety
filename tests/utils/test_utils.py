import os
from unittest.mock import patch
import pytest

from bayesiansafety.utils.utils import *

@patch("os.makedirs")
@pytest.mark.parametrize("dir_path, expect_call", [(r"C:\Users", False), ("1234RandomPaTh", True)]) 
def test_create_dir(mock_makedirs, dir_path, expect_call):

    create_dir(dir_path)
    if expect_call:
         mock_makedirs.assert_called()


def test_check_file_exists_true():
    check_file_exists(os.path.realpath(__file__))
    assert True


def test_check_file_exists_invalid_path_raises_exception():
    expected_exc_substring = "does not lead to an existing file."
    invalid_path = str(os.path.realpath(__file__)) + "abc123" 

    with pytest.raises(Exception) as e:
        assert check_file_exists(invalid_path)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("input, expected", [( [ [] ], [] ),  ( [[1], [2]], [1,2] ),
                                             ( [["A", "B"], ["C"]], ["A", "B", "C"] ),
                                             ( [[type, None], []], [type, None] ), ]) 
def test_flatten_list_of_lists(input, expected):
    assert flatten_list_of_lists(input) == expected


def test_remove_duplicate_tuples_returns_correct():
    duplicates = [("A", 1), ("B", 2), ("C",3), ("A", 1)]
    uniques = [("A", 1), ("B", 2), ("C",3)]

    candidates = remove_duplicate_tuples(duplicates)
    assert len(candidates) == len(uniques)
    assert all([tup in duplicates for tup in uniques] )


def test_remove_duplicate_sets_returns_correct():
    duplicates = [{"A", "B", "C"}, {"A", "C", "B"}, {"B", "A", "C"}]
    unique = {"A", "B", "C"}
   
    candidates = remove_duplicate_sets(duplicates)
    assert len(candidates) == 1
    assert len( candidates[0] ^ unique) == 0


@patch("builtins.print")
def test_pprint_map_wrong_map_input_type_raises_exception(mock_print):
    expected_exc_substring = "Given map object is not of type list or map."
    bad_map_obj = 123
    header = ("test")

    with pytest.raises(Exception) as e:
        assert pprint_map(bad_map_obj, header)

    assert expected_exc_substring in str(e.value)


@patch("builtins.print")
def test_pprint_map_bad_header_type_raises_exception(mock_print):
    expected_exc_substring = "Given header is not a tuple."
    bad_header = 123
    map_obj = [("t_1"), ("t_2")]

    with pytest.raises(Exception) as e:
        assert pprint_map(map_obj, bad_header)

    assert expected_exc_substring in str(e.value)


@patch("builtins.print")
@pytest.mark.parametrize("map_obj, header", [([("t_1"), ("t_2")], ["too_short"]), ([("too_short")], ["head_1", "head_2"])]) 
def test_pprint_map_mismatch_lengths_raises_exception(mock_print, map_obj, header):
    expected_exc_substring = "Header length does not match number of columns to print."

    with pytest.raises(Exception) as e:
        assert pprint_map(map_obj, header)

    assert expected_exc_substring in str(e.value)


@patch("builtins.print")
def test_pprint_map_empty_map_obj_prints_default_output(mock_print):
    expected_print_output = "No data was provided"
    header = ["test"]
    empty_map_obj = []
    pprint_map(empty_map_obj, header)

    mock_print.assert_called_with(expected_print_output)



def test_xml_tree_path_generator_returns_correct():
    cur_dir_path = os.path.dirname(os.path.realpath(__file__))
    test_file_path = os.path.abspath(os.path.join(cur_dir_path, os.pardir, "test_data", 'test_xml_traversal.xml'))

    all_paths = dict()
    nr_unique_nodes = 10

    traversable_elements = ['level-0', 'level-1', 'level-2-a', 'level-2-b', 'level-2-c', 'level-3', 'level-4-a', 'level-4-b', 'level-5-a', 'level-5-b']
    elements_no_attributes = ['level-0', 'level-2-c']
    elements_name_attribute =  [('level-1','le-1'), ('level-2-a','le-2-a'), ('level-2-b','le-2-b'), ('level-3','le-3'), ('level-4-a','le-4-a'), ('level-4-b','le-4-b')]
    elements_value_attribute = [('level-5-a', 1.0), ('level-5-b', 2.0)]

    for path_tuple, id_element_tup in xml_tree_path_generator(test_file_path):
        all_paths[path_tuple[-1][0]] = id_element_tup

    assert len(all_paths.keys()) == nr_unique_nodes
    assert set(all_paths.keys()) == set(traversable_elements)

    unique_ids = [uuid for uuid, params in all_paths.values()]
    assert len(set(unique_ids)) == nr_unique_nodes

    for element in elements_no_attributes:
        assert all_paths[element][1] == {}

    for element, expected_name in elements_name_attribute:
        assert all_paths[element][1]['name'] == expected_name

    for element, expected_value in elements_value_attribute:
        assert float(all_paths[element][1]['value']) == expected_value

