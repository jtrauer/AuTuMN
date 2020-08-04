from apps.covid_19.calibration import base
from autumn.constants import Region
from apps.covid_19.mixing_optimisation.utils import (
    get_prior_distributions_for_opti,
    get_target_outputs_for_opti,
    get_hospital_targets_for_opti,
    get_weekly_summed_targets,
    add_dispersion_param_prior_for_gaussian,
)
from autumn.tool_kit.utils import print_target_to_plots_from_calibration


country = Region.BELGIUM

PAR_PRIORS = get_prior_distributions_for_opti()
update_priors = {
    "contact_rate": [0.03, 0.05],
    "start_time": [10., 30.],
    "time_variant_detection.max_change_time": [75., 100.],
    "time_variant_detection.end_value": [.15, .30],
    "hospital_props_multiplier": [.6, 1.],
}

for i, par in enumerate(PAR_PRIORS):
    if par["param_name"] in update_priors:
        PAR_PRIORS[i]["distri_params"] = update_priors[par["param_name"]]

TARGET_OUTPUTS = get_target_outputs_for_opti(country, source='who', data_start_time=61, data_end_time=197)  # notifications and deaths

# Use weekly counts
for target in TARGET_OUTPUTS:
    target["years"], target["values"] = get_weekly_summed_targets(target["years"], target["values"])

MULTIPLIERS = {}

PAR_PRIORS = add_dispersion_param_prior_for_gaussian(PAR_PRIORS, TARGET_OUTPUTS, MULTIPLIERS)


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds, run_id, num_chains, country, PAR_PRIORS, TARGET_OUTPUTS, mode="autumn_mcmc",
    )


if __name__ == "__main__":
    print_target_to_plots_from_calibration(TARGET_OUTPUTS)
