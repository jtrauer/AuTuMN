import os
import yaml
from autumn.calibration import Calibration

from .model import build_model

N_ITERS = 100000
N_BURNED = 0
N_CHAINS = 1
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(FILE_DIR, "params.yml")

with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)


def run_calibration_chain(max_seconds: int, run_id: int):
    """
    Run a calibration chain for the Mongolia TB model

    num_iters: Maximum number of iterations to run.
    available_time: Maximum time, in seconds, to run the calibration.
    """
    print(f"Preparing to run Mongolia TB model calibration for run {run_id}")
    calib = Calibration(
        "mongolia",
        build_model,
        params["default"],
        PAR_PRIORS,
        TARGET_OUTPUTS,
        run_id,
        total_nb_chains=1,
    )
    print("Starting calibration.")
    calib.run_fitting_algorithm(
        run_mode="autumn_mcmc",
        n_iterations=N_ITERS,
        n_burned=N_BURNED,
        n_chains=N_CHAINS,
        available_time=max_seconds,
    )
    print(f"Finished calibration for run {run_id}.")


PAR_PRIORS = [
    {
        "param_name": "contact_rate",
        "distribution": "uniform",
        "distri_params": [10.0, 20.0],
    },
    {
        "param_name": "adult_latency_adjustment",
        "distribution": "uniform",
        "distri_params": [2.0, 6.0],
    },
    {
        "param_name": "dr_amplification_prop_among_nonsuccess",
        "distribution": "uniform",
        "distri_params": [0.15, 0.25],
    },
    {
        "param_name": "self_recovery_rate",
        "distribution": "uniform",
        "distri_params": [0.18, 0.29],
    },
    {
        "param_name": "tb_mortality_rate",
        "distribution": "uniform",
        "distri_params": [0.33, 0.44],
    },
    {
        "param_name": "rr_transmission_recovered",
        "distribution": "uniform",
        "distri_params": [0.8, 1.2],
    },
    {
        "param_name": "cdr_multiplier",
        "distribution": "uniform",
        "distri_params": [0.66, 1.5],
    },
]

TARGET_OUTPUTS = [
    {
        "output_key": "prevXinfectiousXamong",
        "years": [2015.0],
        "values": [757.0],
        "cis": [(620.0, 894.0)],
    },
    {
        "output_key": "prevXlatentXamongXage_5",
        "years": [2016.0],
        "values": [960.0],
        "cis": [(902.0, 1018.0)],
    },
    {
        "output_key": "prevXinfectiousXstrain_mdrXamongXinfectious",
        "years": [2015.0],
        "values": [503.0],
        "cis": [(410.0, 670.0)],
    },
    # {"output_key": "notifications", "years": [2015.0], "values": [4685.0]},
]

# MULTIPLIERS = {
#     "prevXinfectiousXamong": 1.0e5,
#     "prevXlatentXamongXage_5": 1.0e4,
#     "prevXinfectiousXstrain_mdrXamongXinfectious": 1.0e4,
# }
