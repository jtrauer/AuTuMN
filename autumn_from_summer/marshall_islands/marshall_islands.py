from autumn_from_summer.tb_model import *
from autumn_from_summer.tool_kit import *
from time import time
from datetime import datetime
import os

now = datetime.now()

# location for output database
output_db_path = os.path.join(os.getcwd(), 'databases/outputs_' + now.strftime("%m_%d_%Y_%H_%M_%S") + '.db')


def build_rmi_timevariant_cdr():
    cdr = {1950.: 0., 1980.: .10, 1990.: .20, 2000.: .30, 2010.: .4, 2015: .5}
    return scale_up_function(cdr.keys(), cdr.values(), smoothness=0.2, method=5)


def build_rmi_timevariant_tsr():
    tsr = {1950.: 0., 1970.: .2, 1994.: .6, 2000.: .85, 2010.: .87, 2016: .87}
    return scale_up_function(tsr.keys(), tsr.values(), smoothness=0.2, method=5)

#def build_rmi_timevariant_bcg():
    #bcg = {1950.: 0., 1990.: .49, 1995.: .71, 2000.: .90, 2018.: .98}
    #return scale_up_function(tsr.keys(), tsr.values(), smoothness=0.2, method=5)

def build_rmi_model(update_params={}):

    stratify_by = ['age', 'location', 'diabetes']#'organ', 'diabetes'] #'strain']

    # some default parameter values
    external_params = {  # run configuration
                       'start_time': 1850.,
                       'end_time': 2035.,
                       'time_step': 1.,
                       'start_population': 10000,
                       # base model definition:
                       'contact_rate': 20.,
                       'rr_transmission_recovered': .63,
                       'rr_transmission_infected': 0.21,
                       'latency_adjustment': 2.,  # used to modify progression rates during calibration
                       'self_recovery_rate': 0.231,  # this is for smear-positive TB
                       'tb_mortality_rate': 0.389,  # this is for smear-positive TB
                       'prop_smearpos': .3,
                         # MDR-TB:
                       'dr_amplification_prop_among_nonsuccess': 0.15,
                       'prop_mdr_detected_as_mdr': 0.5,
                       'mdr_tsr': .6,
                        # diagnostic sensitivity by organ status:
                        'diagnostic_sensitivity_smearpos': 1.,
                        'diagnostic_sensitivity_smearneg': .7,
                        'diagnostic_sensitivity_extrapul': .5,
                         # adjustments by location and diabetes
                       'rr_transmission_ebeye': 0.9,  # reference: majuro
                       'rr_transmission_otherislands': 1.,  # reference: majuro
                       'rr_progression_has_diabetes': 3.11,  # reference: no_diabetes
                       # IPT
                       'ipt_age_0_ct_coverage': 0.,  # Children contact tracing coverage  .17
                       'ipt_age_5_ct_coverage': 0.,  # Children contact tracing coverage
                       'ipt_age_15_ct_coverage': 0.,  # Children contact tracing coverage
                       'ipt_age_60_ct_coverage': 0.,  # Children contact tracing coverage
                       'yield_contact_ct_tstpos_per_detected_tb': 2.,  # expected number of infections traced per index
                       'ipt_efficacy': .75,   # based on intention-to-treat
                       'ds_ipt_switch': 1.,  # used as a DS-specific multiplier to the coverage defined above
                       'mdr_ipt_switch': .0,  # used as an MDR-specific multiplier to the coverage defined above
                       # Treatment improvement (C-DOTS)
                       'reduction_negative_tx_outcome': 0.,
                       # ACF for intervention groups
                       'acf_coverage': 0.,
                       'acf_sensitivity': .9,
                       'acf_majuro_switch': 0.,
                       'acf_ebeye_switch': 0.,
                       'acf_otherislands_switch': 0.,
                        # LTBI ACF for intervention groups
                       'acf_ltbi_coverage': 0.,
                       'acf_ltbi_sensitivity': .9,
                       'acf_ltbi_efficacy': .85, # higher than ipt_efficacy as higher completion rate
                       'acf_ltbi_majuro_switch': 0.,
                       'acf_ltbi_ebeye_switch': 0.,
                       'acf_ltbi_otherislands_switch': 0.,
                       }
    # update external_params with new parameter values found in update_params
    external_params.update(update_params)

    model_parameters = \
        {"contact_rate": external_params['contact_rate'],
         "contact_rate_recovered": external_params['contact_rate'] * external_params['rr_transmission_recovered'],
         "contact_rate_infected": external_params['contact_rate'] * external_params['rr_transmission_infected'],
         "recovery": external_params['self_recovery_rate'],
         "infect_death": external_params['tb_mortality_rate'],
         "universal_death_rate": 1.0 / 70.0,
         "case_detection": 0.,
         "ipt_rate": 0.,
         "acf_rate": 0.,
         "acf_ltbi_rate": external_params['acf_ltbi_coverage'] * external_params['acf_ltbi_sensitivity'] * external_params['acf_ltbi_efficacy'],
         "dr_amplification": .0,  # high value for testing
         "crude_birth_rate": 35.0 / 1e3}

    input_db_path = os.path.join(os.getcwd(), 'databases/inputs.db')
    input_database = InputDB(database_name=input_db_path)
    n_iter = int(round((external_params['end_time'] - external_params['start_time']) / external_params['time_step'])) + 1
    integration_times = numpy.linspace(external_params['start_time'], external_params['end_time'], n_iter).tolist()

    model_parameters.update(change_parameter_unit(provide_aggregated_latency_parameters(), 365.251))

    # sequentially add groups of flows
    flows = add_standard_infection_flows([])
    flows = add_standard_latency_flows(flows)
    flows = add_standard_natural_history_flows(flows)

    # compartments
    compartments = ["susceptible", "early_latent", "late_latent", "infectious", "recovered"]

    # derived output definition
    out_connections = {
        "incidence_early": {"origin": "early_latent", "to": "infectious"},
        "incidence_late": {"origin": "late_latent", "to": "infectious"}
    }

    # define model     #replace_deaths  add_crude_birth_rate
    if len(stratify_by) > 0:
        _tb_model = StratifiedModel(
            integration_times, compartments, {"infectious": 1e-3}, model_parameters, flows, birth_approach="add_crude_birth_rate",
            starting_population=external_params['start_population'],
            output_connections=out_connections)
    else:
        _tb_model = EpiModel(
            integration_times, compartments, {"infectious": 1e-3}, model_parameters, flows, birth_approach="add_crude_birth_rate",
            starting_population=external_params['start_population'])

    # add crude birth rate from un estimates
    _tb_model = get_birth_rate_functions(_tb_model, input_database, 'MNG')

    # add case detection process to basic model
    _tb_model.add_transition_flow(
        {"type": "standard_flows", "parameter": "case_detection", "origin": "infectious", "to": "recovered"})

    # Add IPT as a customised flow
    def ipt_flow_func(model, n_flow):
        if not hasattr(model, 'strains') or len(model.strains) < 2:
            infectious_populations = model.infectious_populations['all_strains'][0]
        else:
            infectious_populations = \
                    model.infectious_populations[find_stratum_index_from_string(
                        model.transition_flows.at[n_flow, "parameter"], "strain")][0]

        n_early_latent_comps = len([model.compartment_names[i] for i in range(len(model.compartment_names)) if
                                   model.compartment_names[i][0:12] == 'early_latent'])

        return infectious_populations / float(n_early_latent_comps)

    _tb_model.add_transition_flow(
        {"type": "customised_flows", "parameter": "ipt_rate", "origin": "early_latent", "to": "recovered",
         "function": ipt_flow_func})

    # add ACF flow
    _tb_model.add_transition_flow(
        {"type": "standard_flows", "parameter": "acf_rate", "origin": "infectious", "to": "recovered"})

    # add LTBI ACF flow
    _tb_model.add_transition_flow(
        {"type": "standard_flows", "parameter": "acf_ltbi_rate", "origin": "early_latent", "to": "recovered"})

    # load time-variant case detection rate
    cdr_scaleup = build_rmi_timevariant_cdr()
    disease_duration = 3.
    prop_to_rate = convert_competing_proportion_to_rate(1.0 / disease_duration)
    detect_rate = return_function_of_function(cdr_scaleup, prop_to_rate)

    # load time-variant treatment success rate
    rmi_tsr = build_rmi_timevariant_tsr()

    # create a tb_control_recovery_rate function combining case detection and treatment success rates
    tb_control_recovery_rate = \
        lambda t: detect_rate(t) *\
                  (rmi_tsr(t) + external_params['reduction_negative_tx_outcome'] * (1. - rmi_tsr(t)))

    # initialise ipt_rate function assuming coverage of 1.0 before age stratification
    ipt_rate_function = lambda t: detect_rate(t) * 1.0 *\
                                  external_params['yield_contact_ct_tstpos_per_detected_tb'] * external_params['ipt_efficacy']

    # initialise acf_rate function
    acf_rate_function = lambda t: external_params['acf_coverage'] * external_params['acf_sensitivity'] *\
                                  (rmi_tsr(t) + external_params['reduction_negative_tx_outcome'] * (1. - rmi_tsr(t)))

    # assign newly created functions to model parameters
    if len(stratify_by) == 0:
        _tb_model.time_variants["case_detection"] = tb_control_recovery_rate
        _tb_model.time_variants["ipt_rate"] = ipt_rate_function
        _tb_model.time_variants["acf_rate"] = acf_rate_function

    else:
        _tb_model.adaptation_functions["case_detection"] = tb_control_recovery_rate
        _tb_model.parameters["case_detection"] = "case_detection"

        _tb_model.adaptation_functions["ipt_rate"] = ipt_rate_function
        _tb_model.parameters["ipt_rate"] = "ipt_rate"

        _tb_model.adaptation_functions["acf_rate"] = acf_rate_function
        _tb_model.parameters["acf_rate"] = "acf_rate"

    if "strain" in stratify_by:
        mdr_adjustment = external_params['prop_mdr_detected_as_mdr'] * external_params['mdr_tsr'] / .9  # /.9 for last DS TSR

        _tb_model.stratify("strain", ["ds", "mdr"], ["early_latent", "late_latent", "infectious"], verbose=False,
                           requested_proportions={"mdr": 0.},
                           adjustment_requests={
                               'contact_rate': {'ds': 1., 'mdr': 1.},
                               'case_detection': {"mdr": mdr_adjustment},
                               'ipt_rate': {"ds": 1., #external_params['ds_ipt_switch'],
                                            "mdr": external_params['mdr_ipt_switch']}
                           })

        _tb_model.add_transition_flow(
            {"type": "standard_flows", "parameter": "dr_amplification",
             "origin": "infectiousXstrain_ds", "to": "infectiousXstrain_mdr",
             "implement": len(_tb_model.all_stratifications)})

        dr_amplification_rate = \
            lambda t: detect_rate(t) * (1. - rmi_tsr(t)) *\
                      (1. - external_params['reduction_negative_tx_outcome']) *\
                      external_params['dr_amplification_prop_among_nonsuccess']

        _tb_model.adaptation_functions["dr_amplification"] = dr_amplification_rate
        _tb_model.parameters["dr_amplification"] = "dr_amplification"

    if 'organ' in stratify_by:
        props_smear = {"smearpos": external_params['prop_smearpos'],
                       "smearneg": 1. - (external_params['prop_smearpos'] + .42),
                       "extrapul": .42}
        mortality_adjustments = {"smearpos": 1., "smearneg": .64, "extrapul": .64}
        recovery_adjustments = {"smearpos": 1., "smearneg": .56, "extrapul": .56}
        diagnostic_sensitivity = {}
        for stratum in ["smearpos", "smearneg", "extrapul"]:
            diagnostic_sensitivity[stratum] = external_params["diagnostic_sensitivity_" + stratum]
        _tb_model.stratify("organ", ["smearpos", "smearneg", "extrapul"], ["infectious"],
                           infectiousness_adjustments={"smearpos": 1., "smearneg": 0.25, "extrapul": 0.},
                           verbose=False, requested_proportions=props_smear,
                           adjustment_requests={'recovery': recovery_adjustments,
                                                'infect_death': mortality_adjustments,
                                                'case_detection': diagnostic_sensitivity},
                           entry_proportions=props_smear)

    if "age" in stratify_by:
        age_breakpoints = [0, 5, 15, 60]
        age_infectiousness = get_parameter_dict_from_function(logistic_scaling_function(10.0), age_breakpoints)
        age_params = get_adapted_age_parameters(age_breakpoints)
        age_params.update(split_age_parameter(age_breakpoints, "contact_rate"))

        # adjustment of latency parameters
        for param in ['early_progression', 'late_progression']:
            for age_break in age_breakpoints:
                age_params[param][str(age_break) + 'W'] *= external_params['latency_adjustment']

        pop_morts = get_pop_mortality_functions(input_database, age_breakpoints, country_iso_code='MNG')
        age_params["universal_death_rate"] = {}
        for age_break in age_breakpoints:
            _tb_model.time_variants["universal_death_rateXage_" + str(age_break)] = pop_morts[age_break]
            _tb_model.parameters["universal_death_rateXage_" + str(age_break)] = "universal_death_rateXage_" + str(age_break)

            age_params["universal_death_rate"][str(age_break) + 'W'] = "universal_death_rateXage_" + str(age_break)
        _tb_model.parameters["universal_death_rateX"] = 0.

        # age-specific IPT
        ipt_by_age = {'ipt_rate': {}}
        for age_break in age_breakpoints:
            ipt_by_age['ipt_rate'][str(age_break)] = external_params['ipt_age_' + str(age_break) + '_ct_coverage']
        age_params.update(ipt_by_age)

        # add BCG effect without stratification assuming constant 100% coverage
        bcg_wane = create_sloping_step_function(15.0, 0.3, 30.0, 1.0)
        age_bcg_efficacy_dict = get_parameter_dict_from_function(lambda value: bcg_wane(value), age_breakpoints)
        age_params.update({'contact_rate': age_bcg_efficacy_dict})

        _tb_model.stratify("age", copy.deepcopy(age_breakpoints), [], {}, adjustment_requests=age_params,
                           infectiousness_adjustments=age_infectiousness, verbose=False)

        # patch for IPT to overwrite parameters when ds_ipt has been turned off while we still need some coverage at baseline
        if external_params['ds_ipt_switch'] == 0. and external_params['mdr_ipt_switch'] == 1.:
            _tb_model.parameters['ipt_rateXstrain_dsXage_0'] = 0.17
            for age_break in [5, 15, 60]:
                _tb_model.parameters['ipt_rateXstrain_dsXage_' + str(age_break)] = 0.

    if "location" in stratify_by:
        props_location = {'majuro': .523, 'ebeye': .2, 'otherislands': .277}
        raw_relative_risks_loc = {'majuro': 1.}
        for stratum in ['ebeye', 'otherislands']:
            raw_relative_risks_loc[stratum] = external_params['rr_transmission_' + stratum]
        scaled_relative_risks_loc = scale_relative_risks_for_equivalence(props_location, raw_relative_risks_loc)

        # dummy matrix for mixing by location
        location_mixing = numpy.array([1., .2, .1, .2, 1., .1, .1,	.1,	1.]).reshape((3, 3))

        location_adjustments = {}
        for beta_type in ['', '_infected', '_recovered']:
            location_adjustments['contact_rate' + beta_type] = scaled_relative_risks_loc

        location_adjustments['acf_rate'] = {}
        for stratum in ['majuro', 'ebeye', 'otherislands']:
            location_adjustments['acf_rate'][stratum] = external_params['acf_' + stratum + '_switch']

        _tb_model.stratify("location", ['majuro', 'ebeye', 'otherislands'], [],
                           requested_proportions=props_location, verbose=False, entry_proportions=props_location,
                           adjustment_requests=location_adjustments,
                           mixing_matrix=location_mixing
                           )

    if 'diabetes' in stratify_by:
        props_diabetes = {'has_diabetes': 0.2, 'no_diabetes': 0.8}
        progression_adjustments = {"has_diabetes": 3.11, "no_diabetes": 1.}

        _tb_model.stratify("diabetes", ["has_diabetes", "no_diabetes"], [],
                           verbose=False, requested_proportions=props_diabetes,
                           adjustment_requests={'early_progression': progression_adjustments,
                                                'late_progression': progression_adjustments},
                           entry_proportions=props_diabetes)

    # _tb_model.transition_flows.to_csv("transitions.csv")
    # _tb_model.death_flows.to_csv("deaths.csv")

    return _tb_model



if __name__ == "__main__":
    load_model = False

    scenario_params = {
            # 1: {'ipt_age_0_ct_coverage': .5},
            # 2: {'ipt_age_0_ct_coverage': .5, 'ipt_age_5_ct_coverage': .5, 'ipt_age_15_ct_coverage': .5,
            #     'ipt_age_60_ct_coverage': .5},
            # 3: {'ipt_age_0_ct_coverage': .5, 'ipt_age_5_ct_coverage': .5, 'ipt_age_15_ct_coverage': .5,
            #      'ipt_age_60_ct_coverage': .5, 'ds_ipt_switch': 0., 'mdr_ipt_switch': 1.},
            # 4: {'mdr_tsr': .8},
            # 5: {'reduction_negative_tx_outcome': 0.5},
            # 6: {'acf_coverage': .2, 'acf_urban_ger_switch': 1.},
            # 7: {'acf_coverage': .2, 'acf_mine_switch': 1.},
            # 8: {'diagnostic_sensitivity_smearneg': 1., 'prop_mdr_detected_as_mdr': .9}
        }
    scenario_list = [0]
    scenario_list.extend(list(scenario_params.keys()))

    if load_model:
        load_mcmc = True

        if load_mcmc:
            models = load_calibration_from_db('outputs_11_27_2019_14_07_54.db')
            scenario_list = range(len(models))
        else:
            models = []
            scenarios_to_load = scenario_list
            for sc in scenarios_to_load:
                print("Loading model for scenario " + str(sc))
                model_dict = load_model_scenario(str(sc), database_name='outputs_11_27_2019_13_12_43.db')
                models.append(DummyModel(model_dict))
    else:
        t0 = time()
        models = run_multi_scenario(scenario_params, 2020., build_rmi_model)
        store_run_models(models, scenarios=scenario_list, database_name=output_db_path)
        delta = time() - t0
        print("Running time: " + str(round(delta, 1)) + " seconds")

    req_outputs = ['prevXinfectiousXamong',
                   'prevXinfectiousXorgan_smearposXamongXinfectious',
                   'prevXinfectiousXorgan_smearnegXamongXinfectious',
                   'prevXinfectiousXorgan_extrapulXamongXinfectious',
                   # 'prevXlatentXamong',
                   # 'prevXinfectiousXamongXage_15Xage_60',
                   # 'prevXinfectiousXamongXage_15Xage_60Xhousing_ger',
                   # 'prevXinfectiousXamongXage_15Xage_60Xhousing_non-ger',
                   # 'prevXinfectiousXamongXage_15Xage_60Xlocation_rural',
                   # 'prevXinfectiousXamongXage_15Xage_60Xlocation_province',
                   # 'prevXinfectiousXamongXage_15Xage_60Xlocation_urban',
                   # 'prevXinfectiousXamongXhousing_gerXlocation_urban',
                   # 'prevXlatentXamongXhousing_gerXlocation_urban',
                   #
                   # 'prevXinfectiousXstrain_mdrXamong'
                   ]

    multipliers = {
        'prevXinfectiousXstrain_mdrXamongXinfectious': 100.,
        'prevXinfectiousXstrain_mdrXamong': 1.e5
    }

    targets_to_plot = {'prevXinfectiousXamongXage_15Xage_60': [[2015.], [560.]],
                       #'prevXlatentXamongXage_5': [[2016.], [9.6]],
                       #'prevXinfectiousXamongXage_15Xage_60Xhousing_ger': [[2015.], [613.]],
                       #'prevXinfectiousXamongXage_15Xage_60Xhousing_non-ger': [[2015.], [436.]],
                       #'prevXinfectiousXamongXage_15Xage_60Xlocation_rural': [[2015.], [529.]],
                       #'prevXinfectiousXamongXage_15Xage_60Xlocation_province': [[2015.], [513.]],
                       #'prevXinfectiousXamongXage_15Xage_60Xlocation_urban': [[2015.], [586.]],
                       #'prevXinfectiousXstrain_mdrXamongXinfectious': [[2016.], [5.3]]
                       }

    translations = {'prevXinfectiousXamong': 'TB prevalence (/100,000)',
                    'prevXinfectiousXamongXage_0': 'TB prevalence among 0-4 y.o. (/100,000)',
                    'prevXinfectiousXamongXage_5': 'TB prevalence among 5-14 y.o. (/100,000)',
                    'prevXinfectiousXamongXage_15': 'TB prevalence among 15-59 y.o. (/100,000)',
                    'prevXinfectiousXamongXage_60': 'TB prevalence among 60+ y.o. (/100,000)',
                    'prevXinfectiousXamongXhousing_ger': 'TB prev. among Ger population (/100,000)',
                    'prevXinfectiousXamongXhousing_non-ger': 'TB prev. among non-Ger population(/100,000)',
                    'prevXinfectiousXamongXlocation_rural': 'TB prev. among rural population (/100,000)',
                    'prevXinfectiousXamongXlocation_province': 'TB prev. among province population (/100,000)',
                    'prevXinfectiousXamongXlocation_urban': 'TB prev. among urban population (/100,000)',
                    'prevXlatentXamong': 'Latent TB infection prevalence (%)',
                    'prevXlatentXamongXage_5': 'Latent TB infection prevalence among 5-14 y.o. (%)',
                    'prevXlatentXamongXage_0': 'Latent TB infection prevalence among 0-4 y.o. (%)',
                    'prevXinfectiousXamongXage_15Xage_60': 'TB prev. among 15+ y.o. (/100,000)',
                    'prevXinfectiousXamongXage_15Xage_60Xhousing_ger': 'TB prev. among 15+ y.o. Ger population (/100,000)',
                    'prevXinfectiousXamongXage_15Xage_60Xhousing_non-ger': 'TB prev. among 15+ y.o. non-Ger population (/100,000)',
                    'prevXinfectiousXamongXage_15Xage_60Xlocation_rural': 'TB prev. among 15+ y.o. rural population (/100,000)',
                    'prevXinfectiousXamongXage_15Xage_60Xlocation_province': 'TB prev. among 15+ y.o. province population (/100,000)',
                    'prevXinfectiousXamongXage_15Xage_60Xlocation_urban': 'TB prev. among 15+ y.o. urban population (/100,000)',
                    'prevXinfectiousXstrain_mdrXamongXinfectious': 'Proportion of MDR-TB among TB (%)',
                    'prevXinfectiousXamongXhousing_gerXlocation_urban': 'TB prevalence in urban Ger population (/100,000)',
                    'age_0': 'age 0-4',
                    'age_5': 'age 5-14',
                    'age_15': 'age 15-59',
                    'age_60': 'age 60+',
                    'housing_ger': 'ger',
                    'housing_non-ger': 'non-ger',
                    'location_rural': 'rural',
                    'location_province': 'province',
                    'location_urban': 'urban',
                    'strain_ds': 'DS-TB',
                    'strain_mdr': 'MDR-TB',
                    'prevXinfectiousXstrain_mdrXamong': 'Prevalence of MDR-TB (/100,000)'
                    }

    create_multi_scenario_outputs(models, req_outputs=req_outputs, out_dir='test_25_11', targets_to_plot=targets_to_plot,
                                  req_multipliers=multipliers, translation_dictionary=translations,
                                  scenario_list=scenario_list)