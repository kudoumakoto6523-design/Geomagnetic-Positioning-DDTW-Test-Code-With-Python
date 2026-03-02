from Geomag.algorithms import get_map
from Geomag.models import RunContext


class Initializer:
    def __init__(self, num_runs=11, window_size=100):
        self.num_runs = num_runs
        self.window_size = window_size

    def create_context(self):
        # TODO: Extend bootstrap with full dataset/config loading for own CSV and UJI TXT.
        geomag_map = get_map()
        return RunContext(
            num_runs=self.num_runs,
            window_size=self.window_size,
            geomag_map=geomag_map,
        )
