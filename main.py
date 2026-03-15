from Geomag import (
    Experiment,
    PFConfig,
    Initializer,
    PDRConfig,
)


class ExampleExperiment(Experiment):
    def __init__(self, context):
        pdr_config = PDRConfig(
            step_judge="peak_dynamic",
            step_judge_params={"peak_sigma": 0.4, "peak_prominence": 0.2, "min_samples_per_step": 4.5,},
            step_length="weinberg",
            step_length_params={"weinberg_k": 0.45},
            heading="gyro",
            heading_params={"dt": 0.01},
            mag="norm_mean",
        )
        pf_config = PFConfig(
            state_params={"num_particles": 5000, "min_particles": 2000, "max_particles": 10000000000000},
            motion="gaussian",
            motion_params={"heading_noise_std": 0.01, "step_noise_std": 0.01},
            weight="ddtw",
            weight_params={"sigma": 0.1, "max_hist": 100},
            particle_size="kld",
            particle_size_params={"epsilon": 0.10, "bin_size_xy": 0.5, "bin_size_theta": 0.35},
            resample_trigger="ess_or_target",
            resample_trigger_params={"ess_ratio_threshold": 0.40},
            resample="cso",
        )
        super().__init__(context, pdr_config=pdr_config, pf_config=pf_config)


if __name__ == "__main__":
    context = Initializer(
        num_runs=1,
        window_size=400,
        route_source="uji",
        sensor_source="uji",
        uji_test_file="tt02.txt",
    ).create_context()
    ExampleExperiment(context).run(show=True)
