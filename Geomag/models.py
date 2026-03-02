from dataclasses import dataclass
from typing import Any


@dataclass
class RunContext:
    num_runs: int
    window_size: int
    geomag_map: Any


@dataclass
class Particle:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    weight: float = 1.0


class PFState:
    # TODO: Implement PF state container and particle management logic.
    def __init__(self, init_pos, mag_map):
        pass

    # TODO: Return current estimated position.
    def get_pos(self):
        pass


# Backward-compatible alias to keep naming close to prior script.
PF_State = PFState
