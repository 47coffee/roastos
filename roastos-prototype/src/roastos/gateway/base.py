from __future__ import annotations

from abc import ABC, abstractmethod

from roastos.types import Control
from roastos.gateway.schemas import RoastMeasurementFrame

"""This module defines the base class for the RoastOS gateway, which is responsible for interfacing with the 
physical roasting machine. The BaseRoasterGateway class is an abstract base class that specifies 
the methods for connecting to the machine, reading sensor data frames, and applying control inputs. 
Concrete implementations of this gateway would handle the specific communication protocols 
and hardware interactions required to operate the roasting machine and retrieve real-time data during the roast process."""

class BaseRoasterGateway(ABC):
    @abstractmethod
    def connect(self) -> None:
        ...

    @abstractmethod
    def read_frame(self) -> RoastMeasurementFrame:
        ...

    @abstractmethod
    def apply_control(self, control: Control) -> None:
        ...