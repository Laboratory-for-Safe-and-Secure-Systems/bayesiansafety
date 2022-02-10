"""
Class for importing a BowTie-model from an OpenPSA model exchange file.
    see also https://open-psa.github.io/joomla1.5/index.php.html
"""
from typing import Optional

from bayesiansafety.utils.utils import check_file_exists
from bayesiansafety.eventtree import EventTreeImporter
from bayesiansafety.faulttree import FaultTreeImporter
from bayesiansafety.bowtie import BayesianBowTie


class BowTieImporter:

    """
    Class for importing a Bow-Tie model from an OpenPSA model exchange file.
        see also https://open-psa.github.io/joomla1.5/index.php.html
    """

    def load(self, xml_file_path: str, bowtie_name: Optional[str] = None) -> BayesianBowTie:
        """Parse the Event Tree  and Fault Tree defined in the OpenPSA file.
            The top-level event of the Faul Tree is treated as pivot node.
            Both parsed trees are passed to bayesianbowtie.BayesianBowTie,
            and an instance thereof is returned.

        Returns:
            bayesianbowtie.BayesianBowTie: BowTie model with the top-level event of the Faul Tree serving as pivot node.

        Args:
            xml_file_path (path): Path to the OpenPSA model exchange file.
            bowtie_name (str, optional): Name of the Bow-Tie model. If no name is given a default name will be used.
        """

        check_file_exists(xml_file_path)

        bay_ft = FaultTreeImporter().load(xml_file_path)
        bay_et = EventTreeImporter().load(xml_file_path)
        bowtie_name = bowtie_name if bowtie_name else f"BowTie_{bay_ft.name}_{bay_et.name}"
        return BayesianBowTie(bowtie_name, bay_ft, bay_et)
