from Geomag import Experiment, Initializer


class MainAlgorithmTest:
    def __init__(self):
        self.initializer = Initializer(
            num_runs=1,
            window_size=400,
            route_source="uji",
            sensor_source="uji",
            uji_test_file="tt01.txt",
        )

    def run(self):
        context = self.initializer.create_context()
        Experiment(context).run()

if __name__ == "__main__":
    MainAlgorithmTest().run()
