from apps.covid_19.calibration.base import run_calibration_chain, get_priors_and_targets

country = "philippines"


PAR_PRIORS = [
    {
        'param_name': 'contact_rate',
        'distribution': 'uniform',
        'distri_params': [0.010, 0.05]
    },
    {
        'param_name': 'start_time',
        'distribution': 'uniform',
        'distri_params': [0., 40.]
    },
    # Add extra params for negative binomial likelihood
    {
        'param_name': 'infection_deathsXall_dispersion_param',
        'distribution': 'uniform',
        'distri_params': [.1, 5.]
    },
    {
        "param_name": "compartment_periods_calculated.incubation.total_period",
        "distribution": "gamma",
        "distri_mean": 5.,
        "distri_ci": [4.4, 5.6]
    },
    {
        "param_name": "compartment_periods_calculated.total_infectious.total_period",
        "distribution": "gamma",
        "distri_mean": 7.,
        "distri_ci": [4.5, 9.5]
    },
]

# Death counts:
death_times = [31, 33, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 152, 154, 155, 157, 158, 160]
death_values = [1, 1, 3, 3, 4, 5, 6, 6, 12, 9, 14, 15, 13, 23, 30, 24, 21, 35, 40, 32, 30, 39, 26, 28, 25, 18, 26, 21, 18, 15, 18, 18, 18, 21, 15, 16, 12, 19, 3, 9, 13, 10, 14, 10, 12, 17, 7, 8, 6, 16, 3, 7, 9, 4, 7, 9, 8, 5, 13, 5, 2, 15, 3, 7, 5, 9, 11, 9, 7, 2, 2, 13, 2, 6, 6, 6, 1, 2, 5, 3, 3, 3, 3, 1, 3, 2, 1, 1, 1]

TARGET_OUTPUTS = [
    {
        "output_key": "infection_deathsXall",
        "years": death_times,
        "values": death_values,
        "loglikelihood_distri": "negative_binomial",
    }
]


def run_phl_calibration_chain(max_seconds: int, run_id: int):
    # run_calibration_chain(max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode='grid_based',
    #                       _grid_info=par_grid, _run_extra_scenarios=False, _multipliers=MULTIPLIERS)
    run_calibration_chain(max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode='autumn_mcmc',
                          _run_extra_scenarios=False)


if __name__ == "__main__":
    run_phl_calibration_chain(15 * 60 * 60, 0)  # first argument only relevant for autumn_mcmc mode (time limit in seconds)
