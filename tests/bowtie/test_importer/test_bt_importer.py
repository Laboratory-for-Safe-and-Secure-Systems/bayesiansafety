import os
import pytest

from bayesiansafety.bowtie import BowTieImporter
from bayesiansafety.bowtie import BayesianBowTie

cur_dir_path = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.abspath(os.path.join(cur_dir_path, os.pardir, os.pardir, "test_data"))


@pytest.mark.parametrize("file_name",
    [ "openpsa_bt_heat_exchanger_accident.xml", "openpsa_bt_simple_model.xml"])
def test_bt_importer_correct_loading_of_bt_instance(file_name):

    test_file_path = os.path.abspath(os.path.join(test_data_dir, file_name))
    bay_bt = BowTieImporter().load(test_file_path)

    assert bay_bt.name is not None
    assert isinstance(bay_bt, BayesianBowTie)


@pytest.mark.parametrize("file_name, expected_elements",
    [ ( "openpsa_bt_heat_exchanger_accident.xml", ["Vapor","HTPC","Vent_Sys","ATCS","MTCS","Fan","Duct","Vent","Belt","A_Valve","T_Ctrl_Sys","Sensors","P_Unit","M_Valve","T_Sys","Operator","Thermo","Ignition","Sprinkler","Alarm"]),
      ( "openpsa_bt_simple_model.xml"           , ["OR_TLE", "AND", "OR_B4_5", "OR_B1_2", "Basic_event_3", "OR_B4_5", "Basic_event_4", "Basic_event_5", "Basic_event_1", "Basic_event_2", "func_ev_1", "func_ev_2"] )
     ])
def test_bt_importer_all_elements_loaded(file_name, expected_elements):
    test_file_path = os.path.abspath(os.path.join(test_data_dir, file_name))
    bay_bt = BowTieImporter().load(test_file_path)

    assert set(expected_elements).issubset(set(bay_bt.model_elements.keys()))
    assert len(set(bay_bt.model_elements.keys())) == len(set(expected_elements)) + 1 # accounting for the consequence node
