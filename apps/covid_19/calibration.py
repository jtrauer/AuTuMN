import logging
import numpy as np
from itertools import accumulate

from autumn.calibration import Calibration
from autumn.tool_kit.params import load_params
from autumn.calibration.utils import (
    ignore_calibration_target_before_date,
    add_dispersion_param_prior_for_gaussian,
)
from autumn.constants import Region

from apps.covid_19.mixing_optimisation.utils import get_prior_distributions_for_opti
from autumn.tool_kit.params import load_targets

from .model import build_model

N_ITERS = 100000
N_BURNED = 0
N_CHAINS = 1

logger = logging.getLogger(__name__)


"""
Base parameters
"""

BASE_CALIBRATION_PARAMS = [
    # Rationale for the following two parameters described in parameters table of the methods Gdoc at:
    # https://docs.google.com/document/d/1Uhzqm1CbIlNXjowbpTlJpIphxOm34pbx8au2PeqpRXs/edit#
    {
        "param_name": "sojourn.compartment_periods_calculated.exposed.total_period",
        "distribution": "trunc_normal",
        "distri_params": [5.5, 0.97],
        "trunc_range": [1.0, np.inf],
    },
    {
        "param_name": "sojourn.compartment_periods_calculated.active.total_period",
        "distribution": "trunc_normal",
        "distri_params": [6.5, 0.77],
        "trunc_range": [4.0, np.inf],
    },
]


def provide_default_calibration_params(excluded_params=(), override_params=[]):
    """
    Provide the standard default parameters as listed above, unless requested not to include any.

    :param excluded_params: tuple
        strings of the parameters that are not to be returned
    :param override_params: list of dictionaries
    :return: list
        calibration parameters
    """
    params_to_skip = [prior_dict["param_name"] for prior_dict in override_params]
    params_to_skip += list(excluded_params)
    priors = [
        BASE_CALIBRATION_PARAMS[param]
        for param in range(len(BASE_CALIBRATION_PARAMS))
        if BASE_CALIBRATION_PARAMS[param]["param_name"] not in params_to_skip
    ]
    priors += override_params
    return priors


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
    adaptive_proposal=True,
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
        region_name=region,
        adaptive_proposal=adaptive_proposal,
        initialisation_type=params["default"]["metropolis_initialisation_type"],
        metropolis_init_rel_step_size=params["default"]["metropolis_init_rel_step_size"],
    )
    logger.info("Starting calibration.")
    calib.run_fitting_algorithm(
        run_mode=mode,
        n_iterations=N_ITERS,
        n_burned=N_BURNED,
        n_chains=N_CHAINS,
        available_time=max_seconds,
        haario_scaling_factor=params["default"]["haario_scaling_factor"],
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
                "param_name": f"{output_name}_dispersion_param",
                "distribution": "uniform",
                "distri_params": [0.1, 5.0],
            },
        ]
    return params


def remove_early_points_to_prevent_crash(target_outputs, priors):
    """
    Trim the beginning of the time series when model start time is varied during the MCMC
    """
    idx = None
    for i, p in enumerate(priors):
        if p["param_name"] == "time.start":
            idx = i
            break

    if idx is not None:
        latest_start_time = priors[idx]["distri_params"][1]
        for target in target_outputs:
            first_idx_to_keep = next(
                x[0] for x in enumerate(target["years"]) if x[1] > latest_start_time
            )
            target["years"] = target["years"][first_idx_to_keep:]
            target["values"] = target["values"][first_idx_to_keep:]

    return target_outputs


def get_truncated_output(target_group, start, end):
    """
    Extract a specific output out of the relevant target
    """

    start_date_index = target_group["times"].index(start)
    final_date_index = target_group["times"].index(end)
    times = target_group["times"][start_date_index: final_date_index + 1]
    values = target_group["values"][start_date_index: final_date_index + 1]
    return times, values


"""
Application-specific methods

Philippines
"""


def accumulate_target(existing_targets, target_name, category=""):
    """
    Create a cumulative version of a target from the raw daily (annual) rates.
    """

    # Pull the times straight out
    new_target = \
        {f"accum_{target_name}{category}":
             {"times": existing_targets[f"{target_name}{category}"]["times"]}
         }

    # Accumulate the values
    new_target[f"accum_{target_name}{category}"]["values"] = \
        list(accumulate(existing_targets[f"{target_name}{category}"]["values"]))

    return new_target


def add_standard_philippines_params(params, region):
    """
    Add standard set of parameters to vary case detection for the Philippines
    """

    return params + [
        {
            "param_name": "contact_rate",
            "distribution": "uniform",
            "distri_params": [0.02, 0.04],
        },
        {
            "param_name": "testing_to_detection.assumed_cdr_parameter",
            "distribution": "uniform",
            "distri_params": [0.02, 0.11],
        },
        {
            "param_name": "mobility.microdistancing.behaviour.parameters.max_effect",
            "distribution": "uniform",
            "distri_params": [0.1, 0.6],
        },
        {
            "param_name": "infectious_seed",
            "distribution": "uniform",
            "distri_params": [10.0, 100.0],
        },
        {
            "param_name": "clinical_stratification.props.symptomatic.multiplier",
            "distribution": "trunc_normal",
            "distri_params": [1.0, 0.2],
            "trunc_range": [0.5, np.inf],
        },
        {
            "param_name": "clinical_stratification.props.hospital.multiplier",
            "distribution": "trunc_normal",
            "distri_params": [1.0, 0.2],
            "trunc_range": [0.5, np.inf],
        },
        {
            "param_name": "infection_fatality.multiplier",
            "distribution": "trunc_normal",
            "distri_params": [1.0, 0.2],
            "trunc_range": [0.5, np.inf],
        },
    ]


def add_standard_philippines_targets(targets):

    # Ignore notification values before day 100

    # Jamie's approach used for the paper
    # notifications = ignore_calibration_target_before_date(targets["notifications"], 100)
    notifications = targets["notifications"]

    # Alternative approach, didn't work particularly well
    # time_weights = list(np.linspace(0.4, 1.0, len(targets["notifications"]["times"])))

    # Let's try this:
    # Make the time weights proportional to the values being calibrated against, plus emphasise the last 14
    time_weights = notifications["values"][:-14] + [val * 5. for val in notifications["values"][-14:]]

    return [
        {
            "output_key": "notifications",
            "years": notifications["times"],
            "values": notifications["values"],
            "loglikelihood_distri": "negative_binomial",
            "time_weights": time_weights,
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


"""
Victoria
"""


def add_standard_victoria_params(params, region, include_micro=True):

    params_to_return = params + [
        {
            "param_name": "contact_rate",
            "distribution": "uniform",
            "distri_params": [0.008 if region in Region.VICTORIA_RURAL else 0.02, 0.07],
        },
        {
            "param_name": "seasonal_force",
            "distribution": "uniform",
            "distri_params": [0.0, 0.25],
        },
        {
            "param_name": "clinical_stratification.props.symptomatic.multiplier",
            "distribution": "trunc_normal",
            "distri_params": [1.0, 0.1],
            "trunc_range": [0.5, np.inf],
        },
        {
            "param_name": "infection_fatality.multiplier",
            "distribution": "trunc_normal",
            "distri_params": [1.5, 0.5],
            "trunc_range": [0.33, 3.0],
        },
        {
            "param_name": "testing_to_detection.assumed_cdr_parameter",
            "distribution": "uniform",
            "distri_params": [0.2, 0.5],
        },
        {
            "param_name": "importation.movement_prop",
            "distribution": "uniform",
            "distri_params": [0.05, 0.4],
        },
        {
            "param_name": "clinical_stratification.props.hospital.multiplier",
            "distribution": "trunc_normal",
            "distri_params": [1.0, 1.0],
            "trunc_range": [0.6, np.inf],
        },
        {
            "param_name": "sojourn.compartment_periods.icu_early",
            "distribution": "trunc_normal",
            "distri_params": [12.7, 4.0],
            "trunc_range": [3.0, np.inf],
        },
        {
            "param_name": "sojourn.compartment_periods.icu_late",
            "distribution": "trunc_normal",
            "distri_params": [10.8, 4.0],
            "trunc_range": [6.0, np.inf],
        },
    ]
    if include_micro:
        params_to_return.append(
            {
                "param_name": "mobility.microdistancing.behaviour.parameters.max_effect",
                "distribution": "beta",
                "distri_mean": 0.8,
                "distri_ci": [0.6, 0.9],
            },
        )
    return params_to_return


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


"""
European countries for the optimisation project
"""


def get_targets_and_priors_for_opti(country, likelihood_type="normal"):
    targets = load_targets("covid_19", country)

    hospital_targets = [t for t in list(targets.keys()) if "hospital" in t or "icu" in t]
    if len(hospital_targets) > 1:
        hospital_targets = [t for t in list(targets.keys()) if "new_" in t]

    notifications = targets["notifications"]
    deaths = targets["infection_deaths"]
    hospitalisations = targets[hospital_targets[0]]

    par_priors = get_prior_distributions_for_opti()

    target_outputs = [
        {
            "output_key": "notifications",
            "years": notifications["times"],
            "values": notifications["values"],
            "loglikelihood_distri": likelihood_type,
        },
        {
            "output_key": "infection_deaths",
            "years": deaths["times"],
            "values": deaths["values"],
            "loglikelihood_distri": likelihood_type,
        },
        {
            "output_key": hospital_targets[0],
            "years": hospitalisations["times"],
            "values": hospitalisations["values"],
            "loglikelihood_distri": likelihood_type,
        },
    ]

    # Add seroprevalence data except for Italy where the survey occurred a long time after the peak and
    # where there is a high risk of participation bias (individuals in isolation if had a positive antibody test).
    if country != "italy":
        prop_seropositive = targets["proportion_seropositive"]
        target_outputs.append(
            {
                "output_key": "proportion_seropositive",
                "years": prop_seropositive["times"],
                "values": prop_seropositive["values"],
                "loglikelihood_distri": "normal",
                "sd": 0.04,
            }
        )

    if likelihood_type == "normal":
        par_priors = add_dispersion_param_prior_for_gaussian(par_priors, target_outputs)
    else:
        for output_name in ["notifications", "infection_deaths", hospital_targets[0]]:
            par_priors = add_standard_dispersion_parameter(par_priors, target_outputs, output_name)
    target_outputs = remove_early_points_to_prevent_crash(target_outputs, par_priors)

    return target_outputs, par_priors


def truncate_targets_from_time(target, time):
    start_index = next(x[0] for x in enumerate(target["times"]) if x[1] > time)
    target["times"] = target["times"][start_index:]
    target["values"] = target["values"][start_index:]
    return target
