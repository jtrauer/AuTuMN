import logging
import numpy as np

from autumn.calibration import Calibration
from autumn.tool_kit.params import load_params
from autumn.calibration.utils import ignore_calibration_target_before_date
from autumn.constants import Region

from .model import build_model

N_ITERS = 100000
N_BURNED = 0
N_CHAINS = 1

logger = logging.getLogger(__name__)


"""
Base parameters
"""

BASE_CALIBRATION_PARAMS = [
    # Arbitrary, but always required and this range should span the range of values that would be needed
    {
        "param_name": "contact_rate",
        "distribution": "uniform",
        "distri_params": [0.015, 0.07],
    },
    # Arbitrary, but useful to allow epidemic to take off from a flexible time
    {
        "param_name": "start_time",
        "distribution": "uniform",
        "distri_params": [0.0, 40.0],
    },
    # Rationale for the following two parameters described in parameters table of the methods Gdoc at:
    # https://docs.google.com/document/d/1Uhzqm1CbIlNXjowbpTlJpIphxOm34pbx8au2PeqpRXs/edit#
    {
        "param_name": "compartment_periods_calculated.exposed.total_period",
        "distribution": "trunc_normal",
        "distri_params": [5.5, 0.97],
        "trunc_range": [1.0, np.inf],
    },
    {
        "param_name": "compartment_periods_calculated.active.total_period",
        "distribution": "trunc_normal",
        "distri_params": [6.5, 0.77],
        "trunc_range": [1.0, np.inf],
    },
]


def provide_default_calibration_params(excluded_params=()):
    """
    Provide the standard default parameters as listed above, unless requested not to include any.

    :param excluded_params: tuple
        strings of the parameters that are not to be returned
    :return: list
        calibration parameters
    """

    return [
        BASE_CALIBRATION_PARAMS[param]
        for param in range(len(BASE_CALIBRATION_PARAMS))
        if BASE_CALIBRATION_PARAMS[param]["param_name"] not in excluded_params
    ]


"""
General calibration methods
"""


def run_calibration_chain(
    max_seconds: int,
    run_id: int,
    num_chains: int,
    region: str,
    par_priors,
    target_outputs,
    mode="autumn_mcmc",
):
    """
    Run a calibration chain for the covid model

    num_iters: Maximum number of iterations to run.
    available_time: Maximum time, in seconds, to run the calibration.
    mode is either 'lsm' or 'autumn_mcmc'
    """

    logger.info(f"Preparing to run covid model calibration for region {region}")
    params = load_params("covid_19", region)
    calib = Calibration(
        "covid_19",
        build_model,
        params,
        par_priors,
        target_outputs,
        run_id,
        num_chains,
        param_set_name=region,
        adaptive_proposal=False,
    )
    logger.info("Starting calibration.")
    calib.run_fitting_algorithm(
        run_mode=mode,
        n_iterations=N_ITERS,
        n_burned=N_BURNED,
        n_chains=N_CHAINS,
        available_time=max_seconds,
    )
    logger.info(f"Finished calibration for run {run_id}.")


def add_standard_dispersion_parameter(params, target_outputs, output_name):
    """
    Add standard dispersion parameter for negative binomial distribution

    :param params: list
        Parameter priors to be updated by this function
    :param target_outputs: list
        Target outputs, to see whether the quantity of interest is an output
    :param output_name: str
        Name of the output of interest
    :return: list
        Updated version of the parameter priors
    """

    if any([i_target["output_key"] == output_name for i_target in target_outputs]):
        params += [
            {
                "param_name": output_name + "_dispersion_param",
                "distribution": "uniform",
                "distri_params": [0.1, 5.0],
            },
        ]
    return params


"""
Application-specific methods

Philippines
"""


def add_standard_philippines_params(params):
    """
    Add standard set of parameters to vary case detection for the Philippines
    """

    return params + [
        {
            "param_name": "ifr_multiplier",
            "distribution": "uniform",
            "distri_params": [1.5, 2.28]
        },
        {
            "param_name": "testing_to_detection.assumed_cdr_parameter",
            "distribution": "uniform",
            "distri_params": [0.1, 0.4],
        },
        {
            "param_name": "start_time",
            "distribution": "uniform",
            "distri_params": [40.0, 60.0],
        },
        {
            "param_name": "microdistancing.parameters.max_effect",
            "distribution": "uniform",
            "distri_params": [0.25, 0.75],
        },
    ]


def add_standard_philippines_targets(targets):

    # Ignore notification values before day 100
    notifications = ignore_calibration_target_before_date(targets["notifications"], 100)

    return [
        {
            "output_key": "notifications",
            "years": notifications["times"],
            "values": notifications["values"],
            "loglikelihood_distri": "normal",
            "time_weights": assign_trailing_weights_to_halves(14, notifications["times"]),
        },
        {
            "output_key": "icu_occupancy",
            "years": [targets["icu_occupancy"]["times"][-1]],
            "values": [targets["icu_occupancy"]["values"][-1]],
            "loglikelihood_distri": "negative_binomial",
        },
        {
            "output_key": "accum_deaths",
            "years": [targets["infection_deaths"]["times"][-1]],
            "values": [targets["infection_deaths"]["values"][-1]],
            "loglikelihood_distri": "negative_binomial",
        },
    ]


def assign_trailing_weights_to_halves(end_weights, calibration_target):
    """
    Create a list of (float) halves and ones, of the length of the calibration target, with the last few values being
    ones and the earlier values being halves.

    :param end_weights: int
        How many values at the end should be ones
    :param calibration_target: list
        List of calibration targets to determine the length of the weights to be returned
    :return: list
        List of the weights as described above
    """

    time_weights = [0.5] * (len(calibration_target) - end_weights)
    time_weights += [1.0] * end_weights
    return time_weights


"""
Victoria
"""


def add_standard_victoria_params(params, region):

    return params + [
        {
            "param_name": "contact_rate",
            "distribution": "uniform",
            "distri_params": [0.006 if region in Region.VICTORIA_RURAL else 0.012, 0.07],
        },
        {
            "param_name": "ifr_multiplier",  # Less to constrain this, so just to propagate some uncertainty
            "distribution": "trunc_normal",
            "distri_params": [1.0, 0.3 if region in Region.VICTORIA_RURAL else 1.0],
            "trunc_range": [
                0.33 if region in Region.VICTORIA_RURAL else 0.25,
                3.0 if region in Region.VICTORIA_RURAL else 4.0,
            ],
        },
        {
            "param_name": "seasonal_force",
            "distribution": "uniform",
            "distri_params": [0.0, 0.4],
        },
        {
            "param_name": "testing_to_detection.assumed_cdr_parameter",
            "distribution": "uniform",
            "distri_params": [0.08, 0.3],
        },
        {
            "param_name": "symptomatic_props_multiplier",
            "distribution": "trunc_normal",
            "distri_params": [1.0, 0.1],
            "trunc_range": [0.5, np.inf],
        },
        {
            "param_name": "hospital_props_multiplier",
            "distribution": "trunc_normal",
            "distri_params": [1.0, 1.0],
            "trunc_range": [0.2, np.inf],
        },
        {
            "param_name": "compartment_periods.icu_early",
            "distribution": "trunc_normal",
            "distri_params": [12.7, 4.0],
            "trunc_range": [3.0, np.inf],
        },
        {
            "param_name": "compartment_periods.icu_late",
            "distribution": "trunc_normal",
            "distri_params": [10.8, 4.0],
            "trunc_range": [3.0, np.inf],
        },
        {
            "param_name": "movement_prop",
            "distribution": "uniform",
            "distri_params": [0.05, 0.4],
        },
        {
            "param_name": "microdistancing.parameters.max_effect",
            "distribution": "beta",
            "distri_mean": 0.8,
            "distri_ci": [0.4, 0.95],
        },
    ]


def add_standard_victoria_targets(target_outputs, targets, region):

    notifications_to_ignore = 2
    notification_times = targets["notifications"]["times"][:-notifications_to_ignore]
    notification_values = targets["notifications"]["values"][:-notifications_to_ignore]

    # Calibrate all Victoria sub-regions to notifications
    target_outputs += [
        {
            "output_key": targets["notifications"]["output_key"],
            "years": notification_times,
            "values": notification_values,
            "loglikelihood_distri": "normal",
        }
    ]

    # Also calibrate Victoria metro sub-regions to deaths, hospital admission and ICU admissions
    if region in Region.VICTORIA_METRO:

        deaths_to_ignore = 3
        total_infection_death_times = targets["infection_deaths"]["times"][:-deaths_to_ignore]
        total_infection_death_values = targets["infection_deaths"]["values"][:-deaths_to_ignore]

        target_outputs += [
            {
                "output_key": targets["infection_deaths"]["output_key"],
                "years": total_infection_death_times,
                "values": total_infection_death_values,
                "loglikelihood_distri": "normal",
            },
        ]

        new_hosp_to_ignore = 7
        hospital_admission_times = targets["hospital_admissions"]["times"][:-new_hosp_to_ignore]
        hospital_admission_values = targets["hospital_admissions"]["values"][:-new_hosp_to_ignore]

        target_outputs += [
            {
                "output_key": "new_hospital_admissions",
                "years": hospital_admission_times,
                "values": hospital_admission_values,
                "loglikelihood_distri": "normal",
            },
        ]

        new_icu_to_ignore = 7
        icu_admission_times = targets["icu_admissions"]["times"][:-new_icu_to_ignore]
        icu_admission_values = targets["icu_admissions"]["values"][:-new_icu_to_ignore]

        target_outputs += [
            {
                "output_key": "new_icu_admissions",
                "years": icu_admission_times,
                "values": icu_admission_values,
                "loglikelihood_distri": "normal",
            },
        ]

    return target_outputs


def get_trapezoidal_weights(target_times):
    return list(range(len(target_times), len(target_times) * 2))
