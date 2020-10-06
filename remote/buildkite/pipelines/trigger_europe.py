from remote.buildkite.buildkite import (
    Pipeline,
    CommandStep,
    InputStep,
    TextInputField,
    SelectInputField,
)

from .calibrate import chains_field, branch_field, runtime_field, trigger_field, spot_field
from .full import burn_in_field

fields = [chains_field, branch_field, runtime_field, burn_in_field, trigger_field, spot_field]
input_step = InputStep(
    key="calibration-settings",
    run_condition=None,
    fields=fields,
)
trigger_step = CommandStep(key="run-triggers", command="./scripts/buildkite.sh trigger europe")
pipeline = Pipeline(key="trigger-europe", steps=[input_step, trigger_step])