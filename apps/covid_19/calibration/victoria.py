from autumn.constants import Region
from apps.covid_19.calibration import base
from apps.covid_19.calibration.base import provide_default_calibration_params, add_standard_dispersion_parameter


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.VICTORIA,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
    )


case_times = [
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    155,
    156,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    170,
    171,
    172,
    173,
    174,
    175,
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
    208,
]
case_counts = [
    1,
    1,
    1,
    3,
    1,
    0,
    4,
    8,
    7,
    9,
    11,
    11,
    22,
    17,
    15,
    14,
    21,
    34,
    29,
    31,
    46,
    34,
    36,
    39,
    11,
    20,
    21,
    16,
    13,
    10,
    8,
    13,
    2,
    0,
    4,
    2,
    4,
    4,
    5,
    8,
    1,
    3,
    0,
    1,
    4,
    3,
    2,
    1,
    3,
    2,
    5,
    1,
    9,
    10,
    19,
    13,
    8,
    15,
    13,
    11,
    5,
    1,
    17,
    5,
    6,
    12,
    6,
    2,
    4,
    6,
    5,
    5,
    7,
    6,
    3,
    0,
    2,
    6,
    7,
    3,
    6,
    5,
    3,
    9,
    0,
    2,
    1,
    1,
    1,
    1,
    0,
    4,
    6,
    3,
    2,
    7,
    11,
    6,
    5,
    10,
    11,
    25,
    19,
    16,
    17,
    20,
    33,
    30,
    41,
    49,
    75,
    64,
    73,
    77,
    66,
    108,
    74,
    127,
    191,
    134,
    165,
    288,
    216,
    273,
    177,
    270,
    238,
    317,
    428,
    217,
    363,
    275,
    374,
    484,
    403,
    300,
    357,
    459,
]

icu_times = [
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    155,
    156,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    170,
    171,
    172,
    173,
    174,
    175,
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
]
icu_counts = [
    6,
    7,
    7,
    10,
    11,
    11,
    13,
    12,
    13,
    13,
    15,
    14,
    14,
    15,
    18,
    17,
    13,
    12,
    10,
    11,
    12,
    13,
    10,
    11,
    11,
    10,
    11,
    11,
    9,
    9,
    7,
    7,
    7,
    6,
    6,
    6,
    6,
    6,
    6,
    5,
    5,
    4,
    6,
    6,
    7,
    7,
    7,
    5,
    5,
    5,
    5,
    5,
    3,
    3,
    3,
    3,
    4,
    3,
    2,
    2,
    2,
    1,
    2,
    2,
    2,
    1,
    3,
    2,
    2,
    1,
    2,
    1,
    1,
    1,
    2,
    3,
    2,
    2,
    2,
    2,
    3,
    2,
    2,
    3,
    2,
    2,
    1,
    1,
    1,
    1,
    1,
    2,
    4,
    6,
    3,
    3,
    6,
    10,
    8,
    10,
    13,
    16,
    17,
    18,
    27,
    28,
    30,
    32,
    26,
    29,
    33,
    38,
    42,
    41,
    44,
    46
]

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": case_times,
        "values": case_counts,
        "loglikelihood_distri": "negative_binomial",
        "time_weights": list(range(1, len(case_times) + 1)),
    }
]

PAR_PRIORS = provide_default_calibration_params(["start_time"])
PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "notifications")

PAR_PRIORS += [
    {
        "param_name": "seasonal_force",
        "distribution": "uniform",
        "distri_params": [0., 0.8],
    },
    # Programmatic parameters
    {
        "param_name": "time_variant_detection.end_value",
        "distribution": "beta",
        "distri_mean": 0.85,
        "distri_ci": [0.8, 0.9],
    },
]
