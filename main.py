from Geomag import Experiment, Initializer


def main():
    context = Initializer(num_runs=11, window_size=100).create_context()
    Experiment(context).run()


if __name__ == "__main__":
    main()
