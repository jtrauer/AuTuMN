import numpy as np

from apps.covid_19.model.victorian_outputs import CLUSTERS
from apps.covid_19.calibration import (
    provide_default_calibration_params, get_truncated_output
)
from autumn.constants import Region
from autumn.tool_kit.params import load_targets
from autumn.calibration.utils import add_dispersion_param_prior_for_gaussian
from apps.covid_19 import calibration as base

CLUSTERS = [Region.to_filename(r) for r in Region.VICTORIA_SUBREGIONS]

# Just calibrate to June, July, August and September for now (but run for some lead in time at the start)
TARGETS_START_TIME = 153  # 1st June
TARGETS_END_TIME = 305  # 31st October


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    target_outputs = get_target_outputs(TARGETS_START_TIME, TARGETS_END_TIME)
    priors = get_priors(target_outputs)
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.VICTORIA,
        priors,
        target_outputs,
    )


def get_priors(target_outputs: list):
    priors = provide_default_calibration_params()
    priors += [
        {
            "param_name": "contact_rate",
            "distribution": "uniform",
            "distri_params": [0.015, 0.06],
        },
        {
            "param_name": "infectious_seed",
            "distribution": "uniform",
            "distri_params": [20., 60.],
        },
        {
            "param_name": "victorian_clusters.contact_rate_multiplier_regional",
            "distribution": "uniform",
            "distri_params": [0.3, 3.],
        },
        {
            "param_name": "victorian_clusters.contact_rate_multiplier_north_metro",
            "distribution": "uniform",
            "distri_params": [0.3, 3.],
        },
        {
            "param_name": "victorian_clusters.contact_rate_multiplier_west_metro",
            "distribution": "uniform",
            "distri_params": [0.3, 3.],
        },
        {
            "param_name": "victorian_clusters.contact_rate_multiplier_south_metro",
            "distribution": "uniform",
            "distri_params": [0.3, 3.],
        },
        {
            "param_name": "victorian_clusters.contact_rate_multiplier_south_east_metro",
            "distribution": "uniform",
            "distri_params": [0.3, 3.],
        },
        {
            "param_name": "victorian_clusters.contact_rate_multiplier_loddon_mallee",
            "distribution": "uniform",
            "distri_params": [0.3, 3.],
        },
        {
            "param_name": "victorian_clusters.contact_rate_multiplier_barwon_south_west",
            "distribution": "uniform",
            "distri_params": [0.3, 3.],
        },
        {
            "param_name": "victorian_clusters.contact_rate_multiplier_hume",
            "distribution": "uniform",
            "distri_params": [0.3, 3.],
        },
        {
            "param_name": "victorian_clusters.contact_rate_multiplier_gippsland",
            "distribution": "uniform",
            "distri_params": [0.3, 3.],
        },
        {
            "param_name": "victorian_clusters.contact_rate_multiplier_grampians",
            "distribution": "uniform",
            "distri_params": [0.3, 3.],
        },
        {
            "param_name": "seasonal_force",
            "distribution": "uniform",
            "distri_params": [0.0, 0.5],
        },
        {
            "param_name": "clinical_stratification.props.symptomatic.multiplier",
            "distribution": "trunc_normal",
            "distri_params": [1.0, 0.1],
            "trunc_range": [0.5, np.inf],
        },
        {
            "param_name": "infection_fatality.multiplier",
            "distribution": "uniform",
            "distri_params": [0.5, 4.],
        },
        {
            "param_name": "testing_to_detection.assumed_cdr_parameter",
            "distribution": "uniform",
            "distri_params": [0.2, 0.5],
        },
        {
            "param_name": "clinical_stratification.props.hospital.multiplier",
            "distribution": "uniform",
            "distri_params": [0.5, 3.0],
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
        {
            "param_name": "victorian_clusters.intercluster_mixing",
            "distribution": "uniform",
            "distri_params": [0.01, 0.03],
        },
        {
            "param_name": "victorian_clusters.metro.mobility.microdistancing.behaviour.parameters.upper_asymptote",
            "distribution": "uniform",
            "distri_params": [0.1, 0.5],
        },
        {
            "param_name": "victorian_clusters.metro.mobility.microdistancing.face_coverings.parameters.upper_asymptote",
            "distribution": "uniform",
            "distri_params": [0.0, 0.4],
        },
    ]

    priors = add_dispersion_param_prior_for_gaussian(priors, target_outputs)
    return priors


def get_target_outputs(start_date, end_date):
    targets = load_targets("covid_19", Region.VICTORIA)

    # Total Victorian notifications for each time point
    notification_times, notification_values = \
        get_truncated_output(targets["notifications"], start_date, end_date)
    target_outputs = [
        {
            "output_key": "notifications",
            "years": notification_times,
            "values": notification_values,
            "loglikelihood_distri": "normal",
        }
    ]

    # death_times, death_values = \
    #     get_truncated_output(targets["infection_deaths"], start_date, end_date)
    # target_outputs += [
    #     {
    #         "output_key": "infection_deaths",
    #         "years": death_times,
    #         "values": death_values,
    #         "loglikelihood_distri": "normal",
    #     }
    # ]

    print(notification_times)
    print(len(notification_times))
    print(notification_values)
    print(len(notification_values))
    # print(death_values)

    # # Accumulated notifications at the end date for all clusters
    # for cluster in CLUSTERS:
    #     output_key = f"notifications_for_cluster_{cluster}"
    #     max_value_index = notification_values.index(max(notification_values))
    #     target_outputs += [
    #         {
    #             "output_key": output_key,
    #             "years": targets[output_key]["times"][max_value_index - 2: max_value_index + 2],
    #             "values": targets[output_key]["values"][max_value_index - 2: max_value_index + 2],
    #             "loglikelihood_distri": "normal",
    #         }
    #     ]
    #     print("\n")
    #     print(cluster)
    #     print(targets[output_key]["values"][max_value_index - 2: max_value_index + 2])

        # # Accumulated other indicators at the end date for Metro clusters only
        # if cluster.replace("_", "-") in Region.VICTORIA_METRO:
        #     for indicator in ("hospital_admissions", "icu_admissions"):
        #         targets.update(base.accumulate_target(targets, indicator, category=f"_for_cluster_{cluster}"))
        #
        #         # To deal with "infection_deaths" needing to be changed to just "deaths" for the output
        #         indicator_name = "deaths" if "deaths" in indicator else indicator
        #         indicator_key = f"accum_{indicator_name}_for_cluster_{cluster}"
        #
        #         output_key = f"accum_{indicator}_for_cluster_{cluster}"
        #         target_outputs += [
        #             {
        #                 "output_key": indicator_key,
        #                 "years": [targets[output_key]["times"][-1]],
        #                 "values": [targets[output_key]["values"][-1]],
        #                 "loglikelihood_distri": "normal",
        #             }
        #         ]

    return target_outputs



# def add_vic_dispersion_param_priors(priors, target_outputs):
#     target_groups = {
#         "notifications": ["metro", "rural"],
#         "hospital_admissions": ["metro"],
#         "icu_admissions": ["metro"],
#         "accum_deaths": ["metro"],
#     }
#     clusters_by_group = {
#         "metro": Region.VICTORIA_METRO,
#         "rural": Region.VICTORIA_RURAL,
#     }
#
#     for output, cluster_types in target_groups.items():
#         for cluster_type in cluster_types:
#             # read max value among all relevant targets
#             max_val = -1e6
#             for t in target_outputs:
#                 region_name = t['output_key'].split("_for_cluster_")[1]
#                 if t['output_key'].startswith(output) and region_name.replace("_", "-") in clusters_by_group[cluster_type]:
#                     assert t["loglikelihood_distri"] == "normal", \
#                         "The dispersion parameter is designed for a Gaussian likelihood"
#                     max_val = max(
#                         max_val,
#                         max(t["values"])
#                     )
#
#             # sd_ that would make the 95% gaussian CI cover half of the max value (4*sd = 95% width)
#             sd_ = 0.25 * max_val / 4.0
#             lower_sd = sd_ / 2.0
#             upper_sd = 2.0 * sd_
#
#             priors.append(
#                 {
#                     "param_name": f"{output}_{cluster_type}_dispersion_param",
#                     "distribution": "uniform",
#                     "distri_params": [lower_sd, upper_sd],
#                 },
#             )
#
#     return priors
