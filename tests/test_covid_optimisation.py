from apps.covid_19.mixing_optimisation import mixing_opti as opti
from autumn.constants import Region
from summer.model import StratifiedModel
import pytest

AVAILABLE_MODES = [
    "by_age",
    "by_location",
]
AVAILABLE_CONFIGS = range(4)
DECISION_VARS = {
    "by_age": [1.0] * 16,
    "by_location": [1.0, 1.0, 1.0],
}


@pytest.mark.mixing_optimisation
@pytest.mark.github_only
def test_run_root_model_for_uk():
    root_model = opti.run_root_model(Region.UNITED_KINGDOM, {})
    assert type(root_model) is StratifiedModel


@pytest.mark.xfail
def test_build_params_for_phases_2_and_3():
    for mode in AVAILABLE_MODES:
        for config in AVAILABLE_CONFIGS:
            scenario_params = opti.build_params_for_phases_2_and_3(
                DECISION_VARS[mode], config=config, mode=mode
            )
            assert "mixing" in scenario_params and "end_time" in scenario_params


@pytest.mark.mixing_optimisation
@pytest.mark.github_only
def test_full_optimisation_iteration_for_uk():
    country = Region.UNITED_KINGDOM
    root_model = opti.run_root_model(country, {})
    for mode in AVAILABLE_MODES:
        for config in AVAILABLE_CONFIGS:
            h, d, yoll, p_immune, m = opti.objective_function(
                DECISION_VARS[mode], root_model, mode, country, config
            )
