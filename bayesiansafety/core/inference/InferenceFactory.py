"""
This class serves as factory to instantiate an inference engine
based on the selected backend.
Currently supported backends are pgmpy and pyAgrum.
See for more information on pgmpy: https://github.com/pgmpy/pgmpy
See for more information on pyagrum: https://agrum.gitlab.io/  and https://gitlab.com/agrumery/aGrUM
"""
from enum import Enum
from typing import Optional, Union

from bayesiansafety.core import BayesianNetwork
from bayesiansafety.core.inference.InferencePgmpy import PgmpyInference
from bayesiansafety.core.inference.InferencePyagrum import PyagrumInference


class Backend(Enum):

    """Enum class for a consistent use of a selected backend.

    Attributes:
        PGMPY (int): selected backend is pgmpy    https://github.com/pgmpy/pgmpy
        PYAGRUM (int): selected backend is pyAgrum  https://agrum.gitlab.io/  & https://gitlab.com/agrumery/aGrUM
    """
    PGMPY = 0
    PYAGRUM = 1


class InferenceFactory:

    """Class to manage which inference backend gets instantiated when a query is requested from a BN.
        This class acts factory-like.
        In the long run the backend and additional configurations like an inference algorithm are intended
        to be parameterizable from a config file. For now the used (default) backend is configered by
        setting the member __backend.

    Attributes:
        engine (core.inference.Inference*): Product of the factory. A BaySafety wrapper instance of the
                selected inference engine (i.e. third party library used for inference).
    """

    __backend = Backend.PGMPY  # DEFAULT BACKEND
    __model = None
    engine = None

    def __init__(self, model: BayesianNetwork) -> None:
        """Ctor of this class.

        Args:
            model (core.BayesianNetwork): BaySafety-BN for which an Inference-engine should be instantiated.
        """
        self.__model = model
        self.__backend = self.get_configured_backend()
        self.__create_engine()

    def get_engine(self, backend: Optional[Backend] = None) -> Union[PgmpyInference, PyagrumInference]:
        """Get the current instantiation of the inference engine for the model passed at construction.
            If a backend is specified, get the (new) inference engine compliant with the given backend.

        Args:
            backend (core.inference.InferenceFactory.Backend, optional): Backend (i.e. third party library for
                        inference) which should be used.

        Returns:
            core.inference.Inference*: Product of the factory. A BaySafety wrapper instance of the
                selected inference engine.
        """
        if backend is not None and backend is not self.__backend:
            self.__create_engine(backend=backend)

        return self.engine

    def __create_engine(self, backend: Optional[Backend] = None):
        """The main "factory" method of this class.
            Based on a given backend construct a suitable inference instance (i.e. "engine").
            This method manages the creation of the factories "product".

        Args:
            backend (core.inference.InferenceFactory.Backend, optional): Backend (i.e. third party library for
                        inference) which should be used.

        Raises:
            ValueError: Raised if the selected backend is invalid.
        """
        backend = backend if backend is not None else self.__backend

        if isinstance(backend, Backend):
            if backend is Backend.PGMPY:
                self.engine = self.__create_pgmpy_inference()

            if backend is Backend.PYAGRUM:
                self.engine = self.__create_pyagrum_inference()

            self.__backend = backend

        else:
            raise ValueError(f"Invalid backend selected: {str(backend)}.")

    def __create_pgmpy_inference(self) -> PgmpyInference:
        """Handler to produce an inference instance based on the backend "pgmpy".

        Returns:
            inference engine: Instance of core.inference.InferencePgmpy.py.
        """
        # optionally do some additional preperations
        return PgmpyInference(model=self.__model)

    def __create_pyagrum_inference(self) -> PyagrumInference:
        """Handler to produce an inference instance based on the backend "pyagrum".

        Returns:
            inference engine: Instance of core.inference.InferencePyagrum.py.
        """
        # optionally do some additional preperations
        return PyagrumInference(model=self.__model)

    def get_configured_backend(self) -> Backend:
        """Query the currently selected backend (i.e. third party library used for the factory product)
        """
        # default backend -> ´might be later parsed from config file
        return self.__backend
