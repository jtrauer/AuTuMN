import numpy as np

from apps.covid_19.model.victorian_outputs import CLUSTERS
from apps.covid_19.calibration import (
    provide_default_calibration_params,
)
from autumn.constants import Region
from autumn.tool_kit.params import load_targets
from autumn.calibration.utils import add_dispersion_param_prior_for_gaussian
from apps.covid_19 import calibration as base

CLUSTERS = [Region.to_filename(r) for r in Region.VICTORIA_SUBREGIONS]


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    target_outputs = get_target_outputs()
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
            "distri_params": [0.015, 0.07],
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
        {
            "param_name": "victorian_clusters.intercluster_mixing",
            "distribution": "uniform",
            "distri_params": [0.01, 0.03],
        },
        {
            "param_name": "victorian_clusters.regional.contact_rate_multiplier",
            "distribution": "trunc_normal",
            "distri_params": [1., 0.3],
            "trunc_range": [0.5, np.inf],
        },
        {
            "param_name": "victorian_clusters.metro.mobility.microdistancing.behaviour.upper_asymptote",
            "distribution": "uniform",
            "distri_params": [0.1, 0.4],
        },
        {
            "param_name": "victorian_clusters.metro.mobility.microdistancing.face_coverings.upper_asymptote",
            "distribution": "uniform",
            "distri_params": [0.0, 0.4],
        },
    ]

    priors = add_vic_dispersion_param_priors(priors, target_outputs)
    return priors


def add_vic_dispersion_param_priors(priors, target_outputs):
    target_groups = {
        "notifications": ["metro", "rural"],
        "hospital_admissions": ["metro"],
        "icu_admissions": ["metro"],
        "infection_deaths": ["metro"],
    }
    clusters_by_group = {
        "metro": Region.VICTORIA_METRO,
        "rural": Region.VICTORIA_RURAL,
    }

    for output, cluster_types in target_groups.items():
        for cluster_type in cluster_types:
            # read max value among all relevant targets
            max_val = -1e6
            for t in target_outputs:
                region_name = t['output_key'].split("_for_cluster_")[1]
                if t['output_key'].startswith(output) and region_name.replace("_", "-") in clusters_by_group[cluster_type]:
                    assert t["loglikelihood_distri"] == "normal", \
                        "The dispersion parameter is designed for a Gaussian likelihood"
                    max_val = max(
                        max_val,
                        max(t["values"])
                    )

            # sd_ that would make the 95% gaussian CI cover half of the max value (4*sd = 95% width)
            sd_ = 0.25 * max_val / 4.0
            lower_sd = sd_ / 2.0
            upper_sd = 2.0 * sd_

            priors.append(
                {
                    "param_name": f"{output}_{cluster_type}_dispersion_param",
                    "distribution": "uniform",
                    "distri_params": [lower_sd, upper_sd],
                },
            )

    return priors


def get_target_outputs():
    targets = load_targets("covid_19", Region.VICTORIA)
    target_outputs = []

    # Calibrate all Victoria sub-regions to notifications
    for cluster in CLUSTERS:
        output_key = f"notifications_for_cluster_{cluster}"

        # Currently set to end date anyway, but can reduce the amount of data calibrated to
        final_date = 303
        final_date_index = targets[output_key]["times"].index(final_date)
        notification_times = targets[output_key]["times"][:final_date_index + 1]
        notification_values = targets[output_key]["values"][:final_date_index + 1]

        target_outputs += [
            {
                "output_key": targets[output_key]["output_key"],
                "years": notification_times,
                "values": notification_values,
                "loglikelihood_distri": "normal",
            }
        ]

        if cluster.replace("_", "-") in Region.VICTORIA_METRO:
            output_key = f"hospital_admissions_for_cluster_{cluster}"
            final_date_index = targets[output_key]["times"].index(final_date)
            hospital_admission_times = targets[output_key]["times"][:final_date_index + 1]
            hospital_admission_values = targets[output_key]["values"][:final_date_index + 1]
            target_outputs += [
                {
                    "output_key": targets[output_key]["output_key"],
                    "years": hospital_admission_times,
                    "values": hospital_admission_values,
                    "loglikelihood_distri": "normal",
                }
            ]
            output_key = f"icu_admissions_for_cluster_{cluster}"
            final_date_index = targets[output_key]["times"].index(final_date)
            icu_admission_times = targets[output_key]["times"][:final_date_index + 1]
            icu_admission_values = targets[output_key]["values"][:final_date_index + 1]
            target_outputs += [
                {
                    "output_key": targets[output_key]["output_key"],
                    "years": icu_admission_times,
                    "values": icu_admission_values,
                    "loglikelihood_distri": "normal",
                }
            ]
            output_key = f"infection_deaths_for_cluster_{cluster}"
            final_date_index = targets[output_key]["times"].index(final_date)
            death_times = targets[output_key]["times"][:final_date_index + 1]
            death_values = targets[output_key]["values"][:final_date_index + 1]
            target_outputs += [
                {
                    "output_key": targets[output_key]["output_key"],
                    "years": death_times,
                    "values": death_values,
                    "loglikelihood_distri": "normal",
                }
            ]

    return target_outputs
