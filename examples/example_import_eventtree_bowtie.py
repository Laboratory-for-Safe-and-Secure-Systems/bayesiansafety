"""Example showing the core capabilitites of the various importers and the Bow-Tie class
    That are:
        - Load a Fault Tree or an Event Tree from an OpenPSA file
        - Create an Event Tree by specifying all paths between inittiating event and all consequences
        - Load a Bow-Tie model from an OpenPSA file
        - Create a Bow-Tie model from a previously loaded Fault Tree and Event Tree
"""
import os
import sys
cur_dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(cur_dir_path, os.pardir)))

from bayesiansafety.core.inference.InferenceFactory import InferenceFactory
from bayesiansafety.bowtie import BayesianBowTie,  BowTieImporter
from bayesiansafety.eventtree import EventTreeImporter, FunctionalEvent, Consequence
from bayesiansafety.faulttree.FaultTreeImporter import FaultTreeImporter

xml_path = "../tests/test_data/openpsa_bt_heat_exchanger_accident.xml"


print(">>Load a Fault Tree from a file \n")
bay_ft = FaultTreeImporter().load(xml_path)
bay_ft.model.plot_graph()


print(">>Load an Event Tree from a file \n")
bay_et = EventTreeImporter().load(xml_path)
bay_et.model.plot_graph()
print(bay_et.get_consequence_likelihoods())


print(">>Construct an Event Tree manually \n")
FE_1 = FunctionalEvent(name = "FE1", options = ["e11", "e12"])
FE_2 = FunctionalEvent(name = "FE2", options = ["e21", "e22"])
CNSQ = Consequence(name = "CNSQ", options = ["c1", "c2"])

tree_paths = [  ( (FE_1, "e11", 0.7), (FE_2, "e21", 0.01), (CNSQ, "c1") ),
                ( (FE_1, "e11", 0.7), (FE_2, "e22", 0.99), (CNSQ, "c2") ),
                ( (FE_1, "e12", 0.3), (FE_2, "e21", 0.1), (CNSQ, "c1")  ),
                ( (FE_1, "e12", 0.3), (FE_2, "e22", 0.9), (CNSQ, "c2")  ) ]

bay_et_2 = EventTreeImporter().construct("bay_et_2", tree_paths)
bay_et_2.model.plot_graph()
print(bay_et_2.get_consequence_likelihoods())


print("-"*150)
print("\nBuild Bow-Tie from previously loaded Event Tree and Faul Tree - TLE as default pivot node \n")
bowtie_1 = BayesianBowTie("bowtie_1", bay_ft, bay_et)
bowtie_1.model.plot_graph()
print(bowtie_1.get_consequence_likelihoods())


print("Build Bow-Tie from previously loaded Event Tree and Faul Tree - TLE as explicitly pivot node \n")
bowtie_2 = BayesianBowTie("bowtie_2", bay_ft, bay_et, pivot_node="Vapor")
bowtie_2.model.plot_graph()
print(bowtie_2.get_consequence_likelihoods())


print("Build Bow-Tie from previously loaded Event Tree and Faul Tree - multiple functional events are affected by the pivot node (trigger) \n")
bowtie_3 = BayesianBowTie("bowtie_3", bay_ft, bay_et, causal_arc_et_nodes={"Alarm":"failing", "Sprinkler":"failing"})
bowtie_3.model.plot_graph()
print(bowtie_3.get_consequence_likelihoods())


print("Build Bow-Tie directly from file \n")
bowtie_4 = BowTieImporter().load(xml_path)
bowtie_4.model.plot_graph()
print(bowtie_4.get_consequence_likelihoods())
