from autumn.constants import Region
from apps.covid_19.calibration import base
from apps.covid_19.calibration.base import provide_default_calibration_params, add_standard_dispersion_parameter


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.MANILA,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
    )


# notification data:
notification_times = [
44,
61,
68,
75,
82,
89,
96,
103,
110,
117,
124,
131,
138,
145,
152,
159,
166,
173,
180,
187,
]

notification_values = [
2,
4,
71,
333,
771,
1267,
942,
1134,
828,
879,
919,
1000,
974,
1987,
1303,
1582,
1533,
2617,
4327,
5457,
]

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": notification_times,
        "values": notification_values,
        "loglikelihood_distri": "negative_binomial",
    },
]


PAR_PRIORS = provide_default_calibration_params()
PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "notifications")

PAR_PRIORS += [
    {
        "param_name": "time_variant_detection.maximum_gradient",  # shape parameter
        "distribution": "uniform",
        "distri_params": [0.05, 0.1],
    },
    {
        "param_name": "time_variant_detection.max_change_time",
        "distribution": "uniform",
        "distri_params": [70.0, 110.0],
    },
    {
        "param_name": "time_variant_detection.end_value",
        "distribution": "uniform",
        "distri_params": [0.10, 0.90],
    },
    # parameters to derive age-specific IFRs
    {
        "param_name": "ifr_double_exp_model_params.k",
        "distribution": "uniform",
        "distri_params": [6., 14.],
    },
]
