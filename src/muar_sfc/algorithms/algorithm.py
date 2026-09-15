"""Contains an interface for new algorithms.

Author: Bruno Martins
"""

from abc import ABC, abstractmethod

__author__ = ["Bruno Martins"]


class Algorithm(ABC):
    """Abstract base class for algorithm implementation.

    This code defines a common interface for designing new algorithms.
    it's highly recommend that all methods described here are incorporated
    into any new class.
    """

    @abstractmethod
    def __init__(
        self,
        name: str,
        substrate_network: object,
        sfc: object,
        node_info: object,
        src_substrate_node: int,
        route_info: dict,
        latency: float,
    ) -> None:
        super().__init__()
        pass

    @abstractmethod
    def clear_all(self):
        """clear all instance variables for the simulation."""
        pass

    @abstractmethod
    def install_substrate_network(self, substrate_network: object) -> object:
        """_summary_

        Args:
            substrate_network (object): _description_

        Returns:
            object: _description_
        """
        pass

    @abstractmethod
    def install_SFC(self, sfc: object) -> object:
        """_summary_

        Args:
            sfc (object): _description_

        Returns:
            object: _description_
        """
        pass

    @abstractmethod
    def start_algorithm(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """
        pass

    @abstractmethod
    def get_latency(self) -> float:
        """Gets total latency for a given SFC path.

        Returns:
            float: Total latency from src to dst.
        """
        pass

    @abstractmethod
    def get_route_info(self) -> dict:
        """

        Returns:
            dict: Route information for each SF in the current SFC.
        """
        pass

    @abstractmethod
    def algorithm(self, substrate_network: object, sfc: object) -> bool:
        """Algorithm's main logic

        Args:
            substrate_network (object): _description_
            sfc (object): _description_

        Returns:
            bool: True if a path was found, False otherwise.
        """
        pass
