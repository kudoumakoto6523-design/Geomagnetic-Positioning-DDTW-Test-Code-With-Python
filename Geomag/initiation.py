from Geomag.algorithms import get_map
from Geomag.models import RunContext


class Initializer:
    def __init__(
        self,
        num_runs=1,
        window_size=1000,
        route_source="uji",
        sensor_source="uji",
        data_root="data/raw",
        uji_test_file="tt01.txt",
        own_data_dir="data/Geomagnetic Navigation 2026-03-03 15-28-45",
    ):
        self.num_runs = num_runs
        self.window_size = window_size
        self.route_source = route_source
        self.sensor_source = sensor_source
        self.data_root = data_root
        self.uji_test_file = uji_test_file
        self.own_data_dir = own_data_dir

    def create_context(self):
        # Prefer building map; fallback to existing artifacts if optional deps are missing.
        try:
            geomag_map = get_map(source="uji", data_root=self.data_root)
        except Exception:
            geomag_map = {
                "source": "uji",
                "output_model_npz": "data/processed/uji_mag_model_kriging.npz",
                "output_preview_npz": "data/processed/uji_mag_grid_preview_kriging.npz",
            }
        return RunContext(
            num_runs=self.num_runs,
            window_size=self.window_size,
            geomag_map=geomag_map,
            route_source=self.route_source,
            sensor_source=self.sensor_source,
            data_root=self.data_root,
            uji_test_file=self.uji_test_file,
            own_data_dir=self.own_data_dir,
        )
