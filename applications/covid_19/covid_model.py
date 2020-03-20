import os
import yaml
from summer_py.summer_model import (
    StratifiedModel,
)

from autumn import constants
from autumn.constants import Compartment
from autumn.tb_model import (
    list_all_strata_for_mortality,
)
from autumn.tool_kit.scenarios import get_model_times_from_inputs
from applications.covid_19.flows import \
    add_infection_flows, add_to_presympt_flows, add_recovery_flows, add_within_exposed_flows, \
    add_within_infectious_flows, replicate_compartment, multiply_flow_value_for_multiple_compartments,\
    add_infection_death_flows, add_within_presympt_flows, add_to_infectious_flows
from applications.covid_19.stratification import stratify_by_age, stratify_by_infectiousness
from applications.covid_19.covid_outputs import find_incidence_outputs
from autumn.demography.social_mixing import load_specific_prem_sheet
from autumn.demography.ageing import add_agegroup_breaks
from autumn.tool_kit.utils import repeat_list_elements
from applications.covid_19.covid_outputs import create_request_stratified_incidence_covid

# Database locations
file_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, 'inputs.db')
PARAMS_PATH = os.path.join(file_dir, 'params.yml')


def build_covid_model(update_params={}):
    """
    Build the master function to run the TB model for Covid-19

    :param update_params: dict
        Any parameters that need to be updated for the current run
    :return: StratifiedModel
        The final model with all parameters and stratifications
    """

    # Get user-requested parameters
    with open(PARAMS_PATH, 'r') as yaml_file:
        params = yaml.safe_load(yaml_file)
    model_parameters = params['default']

    # Update, not needed for baseline run
    model_parameters.update(update_params)

    # Australian population sizes
    total_pops = \
        [1464776, 1502644, 1397182, 1421612, 1566792, 1664609, 1703852, 1561686, 1583254, 1581460, 1523557,
         1454332, 1299406, 1188989, 887721, 652671 + 460555 + 486847]

    # Define single compartments that don't need to be replicated
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.RECOVERED,
    ]

    # Get progression rates from sojourn times
    for compartment in ['exposed', 'presympt', 'infectious']:
        model_parameters['within_' + compartment] = 1. / model_parameters[compartment + '_period']
    model_parameters['to_infectious'] = 1. / model_parameters['within_presympt']

    # Replicate compartments that need to be repeated
    compartments, _, _ = \
        replicate_compartment(
            model_parameters['n_compartment_repeats'],
            compartments,
            Compartment.EXPOSED,
            []
        )
    compartments, infectious_compartments, _ = \
        replicate_compartment(
            model_parameters['n_compartment_repeats'],
            compartments,
            'presympt',
            []
        )
    compartments, infectious_compartments, init_pop = \
        replicate_compartment(
            model_parameters['n_compartment_repeats'],
            compartments,
            Compartment.INFECTIOUS,
            infectious_compartments,
            infectious_seed=model_parameters['infectious_seed']
        )

    # Multiply the progression rate by the number of compartments to keep the average time in exposed the same
    model_parameters = \
        multiply_flow_value_for_multiple_compartments(
            model_parameters, Compartment.EXPOSED, 'within_exposed'
        )
    model_parameters = \
        multiply_flow_value_for_multiple_compartments(
            model_parameters, 'presympt', 'within_presympt'
        )
    model_parameters = \
        multiply_flow_value_for_multiple_compartments(
            model_parameters, Compartment.INFECTIOUS, 'within_infectious'
        )
    model_parameters['to_infectious'] = model_parameters['within_exposed']

    # Set integration times
    integration_times = \
        get_model_times_from_inputs(
            model_parameters['start_time'],
            model_parameters['end_time'],
            model_parameters['time_step']
        )

    # Sequentially add groups of flows to flows list
    flows = []
    flows = add_infection_flows(
        flows,
        model_parameters['n_compartment_repeats']
    )

    flows = add_within_exposed_flows(
        flows,
        model_parameters['n_compartment_repeats']
    )
    flows = add_within_presympt_flows(
        flows,
        model_parameters['n_compartment_repeats'],
        'presympt'
    )
    flows = add_within_infectious_flows(
        flows,
        model_parameters['n_compartment_repeats'],
        Compartment.INFECTIOUS
    )

    flows = add_to_presympt_flows(
        flows,
        model_parameters['n_compartment_repeats'],
        model_parameters['n_compartment_repeats'],
        'presympt',
        'within_exposed'
    )
    flows = add_to_infectious_flows(
        flows,
        model_parameters['n_compartment_repeats'],
        model_parameters['n_compartment_repeats'],
        Compartment.INFECTIOUS,
        'to_infectious'
    )

    flows = add_recovery_flows(
        flows,
        model_parameters['n_compartment_repeats'],
        Compartment.INFECTIOUS
    )
    flows = add_infection_death_flows(
        flows,
        model_parameters['n_compartment_repeats']
    )

    mixing_matrix = \
        load_specific_prem_sheet(
            'all_locations_1',
            model_parameters['country']
        )

    output_connections = find_incidence_outputs(model_parameters)

    # Define model
    _covid_model = StratifiedModel(
        integration_times,
        compartments,
        init_pop,
        model_parameters,
        flows,
        birth_approach='no_birth',
        starting_population=sum(total_pops),
        infectious_compartment=infectious_compartments
    )

    # Stratify model by age without demography
    if 'agegroup' in model_parameters['stratify_by']:
        model_parameters = add_agegroup_breaks(model_parameters)
        _covid_model = \
            stratify_by_age(
                _covid_model,
                model_parameters['all_stratifications']['agegroup'],
                mixing_matrix,
                total_pops,
                model_parameters
            )
        output_connections.update(
            create_request_stratified_incidence_covid(
                model_parameters['incidence_stratification'],
                model_parameters['all_stratifications'],
                model_parameters['n_compartment_repeats'],
                model_parameters['n_compartment_repeats']
            )
        )

    # Stratify infectious compartment as high or low infectiousness as requested
    if 'infectiousness' in model_parameters['stratify_by']:
        progression_props = repeat_list_elements(2, model_parameters['age_infect_progression'])

        # Repeat the age-specific CFRs for all but the top age bracket, and use the average of the last two for the last
        age_specific_cfrs = \
            repeat_list_elements(2, model_parameters['age_cfr'][: -1]) + \
            [(model_parameters['age_cfr'][-1] + model_parameters['age_cfr'][-2]) / 2.]
        _covid_model = \
            stratify_by_infectiousness(
                _covid_model,
                model_parameters,
                infectious_compartments,
                age_specific_cfrs,
                progression_props
            )

    _covid_model.output_connections = output_connections

    _covid_model.death_output_categories = \
        list_all_strata_for_mortality(_covid_model.compartment_names)

    return _covid_model