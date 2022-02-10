"""Meta objects used when parsing an OpenPSA file / user define Event Tree.
    These elements also serve as data containers in the tree-representation of the Event Tree.
see also https://open-psa.github.io/mef/mef/event_tree_layer.html
"""
from enum import Enum
from typing import List, Optional


class EtType(Enum):

    """Enum specifying constant attributes for Event Tree objects.

    Attributes:
        CONSEQUENCE (str): Specifies a Consequence node
        FUNCTIONAL_EVENT (str): Specifies a FunctionalEvent node
        INIT (str): Specifies a InitiatingEvent node
        PATH (str): Specifies a Path node
    """

    INIT = "ET_Initiating_Event"
    FUNCTIONAL_EVENT = "ET_Functional_Event"
    CONSEQUENCE = "ET_Consequence"
    PATH = "ET_Path"


class EtElement:

    """Basis class for Event Tree objects.

    Attributes:
        container_type (EtType): member that allows string acces to the elements type.
        name (str): Name of the element.
    """

    name = None
    container_type = None

    def __init__(self, name: str, container_type: EtType) -> None:
        """Ctor of this class.

        Args:
            name (str): Name of the element.
            container_type (EtType): member that allows string acces to the elements type.
        """
        self.name = name
        self.container_type = container_type


class InitiatingEvent(EtElement):

    """Ctor of this class.
    """

    def __init__(self, name: str) -> None:
        """Summary

        Args:
            name (str): Name of the element.
        """
        EtElement.__init__(self, name, EtType.INIT)


class FunctionalEvent(EtElement):

    """Event specifies a fork and it's potential outcomes in the Event Tree.

    Attributes:
        options (list<str>): List of valid options (i.e. outcomes of the functional event).
    """

    options = None

    def __init__(self, name: str, options: Optional[List[str]] = None) -> None:
        """Ctor of this class.

        Args:
            name (str): Name of the element.
            options (None, optional): Description
        """
        EtElement.__init__(self, name, EtType.FUNCTIONAL_EVENT)
        self.options = options


class Consequence(EtElement):

    """Consequence specifies a end of the Event Tree and it's potential final outcomes.

    Attributes:
        options (list<str>): List of valid options (i.e. final outcomes of the Event Tree).
    """

    options = None

    def __init__(self, name: str, options: Optional[List[str]] = None) -> None:
        """Ctor of this class.

        Args:
            name (str): Name of the element.
            options (list<str>): List of valid options (i.e. final outcomes of the Event Tree).
        """
        EtElement.__init__(self, name, EtType.CONSEQUENCE)
        self.options = options


class Path(EtElement):

    """Path specifies a connecting element between two functional events via
       the outcome of the source event (i.e. its outcome) and its probability.

    Attributes:
        probability (float): Probability for an outcome (i.e. event probabilities.)
        outcome (str): Outcome of a sourcing functional event.
    """

    state = None
    probability = None

    def __init__(self, name: str, state: str, probability: float, f_event_name: str) -> None:
        """Ctor of this class.

        Args:
            name (str): Name of the element.
            probability (float): Probability for an outcome (i.e. event probabilities.)
            state (str): Outcome of a sourcing functional event.
            f_event_name (sr): Name of the related (i.e. sourcing) functional event
        """
        EtElement.__init__(self, name, EtType.PATH)  # f"{consequence_name}.{outcomeCtor of this class.}"
        self.state = state
        self.probability = probability
        self.f_event_name = f_event_name
