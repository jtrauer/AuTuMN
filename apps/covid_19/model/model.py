from copy import deepcopy

import numpy as np

from summer import StratifiedModel
from autumn import inputs
from autumn.constants import Flow, BirthApproach
from autumn.curve import tanh_based_scaleup, scale_up_function
from autumn.environment.seasonality import get_seasonal_forcing
from autumn.tool_kit.scenarios import get_model_times_from_inputs
from apps.covid_19.model.susceptibility_heterogeneity import get_gamma_data, check_modelled_susc_cv
from autumn.tool_kit.utils import (
    normalise_sequence,
    repeat_list_elements,
    repeat_list_elements_average_last_two,
)

from apps.covid_19.constants import Compartment, ClinicalStratum
from apps.covid_19.model.importation import get_all_vic_notifications
from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS, Region

from . import outputs, preprocess
from .preprocess.testing import find_cdr_function_from_test_data
from .validate import validate_params


"""
Compartments
"""
# People who are infectious.
INFECTIOUS_COMPARTMENTS = [
    Compartment.LATE_EXPOSED,
    Compartment.EARLY_ACTIVE,
    Compartment.LATE_ACTIVE,
]
# People who are infected, but may or may not be infectuous.
DISEASE_COMPARTMENTS = [Compartment.EARLY_EXPOSED, *INFECTIOUS_COMPARTMENTS]
# All model compartments
COMPARTMENTS = [Compartment.SUSCEPTIBLE, Compartment.RECOVERED, *DISEASE_COMPARTMENTS]


def build_model(params: dict) -> StratifiedModel:
    """
    Build the compartmental model from the provided parameters.
    """
    validate_params(params)

    """
    Integration times
    """
    start_time, end_time, time_step = params["start_time"], params["end_time"], params["time_step"]
    times = get_model_times_from_inputs(round(start_time), end_time, time_step,)

    """
    Basic intercompartmental flows
    """
    # Time periods calculated from periods (or "sojourn times")
    base_periods = params["compartment_periods"]
    periods_calc = params["compartment_periods_calculated"]
    compartment_periods = preprocess.compartments.calc_compartment_periods(
        base_periods, periods_calc
    )

    # Inter-compartmental transition flows
    flows = deepcopy(preprocess.flows.DEFAULT_FLOWS)
    flow_params = {
        "contact_rate": params["contact_rate"],
        "infect_death": 0,  # Placeholder to be overwritten in clinical stratification
    }

    # Add parameters for the during-disease progression flows
    for comp_name, comp_period in compartment_periods.items():
        flow_params[f"within_{comp_name}"] = 1.0 / comp_period

    # Waning immunity (if requested)
    if not params["full_immunity"]:
        flow_params["immunity_loss_rate"] = 1.0 / params["immunity_duration"]
        flows.append(
            {
                "type": Flow.STANDARD,
                "origin": Compartment.RECOVERED,
                "to": Compartment.SUSCEPTIBLE,
                "parameter": "immunity_loss_rate",
            }
        )

    # Just set the importation flow without specifying its value, which is done later (if requested)
    # Entry compartment is set to LATE_ACTIVE in the model creation process, because the default would be susceptible
    implement_importation = params["implement_importation"]
    if implement_importation:
        flows.append({"type": Flow.IMPORT, "parameter": "importation_rate"})

    """
    Population creation and distribution
    """

    # Set age groups
    agegroup_max, agegroup_step = params["agegroup_breaks"]
    agegroup_strata = [str(s) for s in range(0, agegroup_max, agegroup_step)]

    # Get country population by age-group
    country_iso3 = params["iso3"]
    mobility_region = params["mobility_region"]
    pop_region = params["pop_region"]
    total_pops = inputs.get_population_by_agegroup(
        agegroup_strata, country_iso3, pop_region, year=params["pop_year"]
    )

    # Distribute infectious seed across infectious split sub-compartments
    infectious_seed = params["infectious_seed"]
    total_disease_time = sum([compartment_periods[c] for c in DISEASE_COMPARTMENTS])
    init_pop = {
        c: infectious_seed * compartment_periods[c] / total_disease_time
        for c in DISEASE_COMPARTMENTS
    }

    # Assign the remainder starting population to the S compartment
    # (must be specified because entry_compartment is late_infectious)
    init_pop[Compartment.SUSCEPTIBLE] = sum(total_pops) - sum(init_pop.values())

    """
    Model instantiation
    """

    model = StratifiedModel(
        times,
        COMPARTMENTS,
        init_pop,
        flow_params,
        flows,
        birth_approach=BirthApproach.NO_BIRTH,
        entry_compartment=Compartment.LATE_ACTIVE,
        starting_population=sum(total_pops),
        infectious_compartments=INFECTIOUS_COMPARTMENTS,
    )

    """
    Seasonal forcing
    """

    if params["seasonal_force"]:
        seasonal_func = get_seasonal_forcing(
            365.0, 173.0, params["seasonal_force"], params["contact_rate"]
        )
        model.time_variants["contact_rate"] = seasonal_func

    """
    Dynamic heterogeneous mixing by age
    """

    static_mixing_matrix = preprocess.mixing_matrix.build_static(country_iso3)
    dynamic_location_mixing_params = params["mixing"]
    dynamic_age_mixing_params = params["mixing_age_adjust"]
    microdistancing = params["microdistancing"]
    smooth_google_data = params["smooth_google_data"]
    npi_effectiveness_params = params["npi_effectiveness"]
    google_mobility_locations = params["google_mobility_locations"]
    microdistancing_locations = params["microdistancing_locations"]
    dynamic_mixing_matrix = preprocess.mixing_matrix.build_dynamic(
        country_iso3,
        mobility_region,
        dynamic_location_mixing_params,
        dynamic_age_mixing_params,
        npi_effectiveness_params,
        google_mobility_locations,
        microdistancing,
        smooth_google_data,
        microdistancing_locations,
    )
    model.set_dynamic_mixing_matrix(dynamic_mixing_matrix)

    """
    Age stratification
    """

    # Set distribution of starting population
    comp_split_props = {
        agegroup: prop for agegroup, prop in zip(agegroup_strata, normalise_sequence(total_pops))
    }

    if params["importation_props_by_age"]:
        importation_props_by_age = params["importation_props_by_age"]
    else:
        importation_props_by_age = {s: 1.0 / len(agegroup_strata) for s in agegroup_strata}

    flow_adjustments = {
        "contact_rate": params["age_based_susceptibility"],
        "importation_rate": importation_props_by_age,
    }

    # We use "agegroup" instead of "age" for this model, to avoid triggering automatic demography features
    # (which also works on the assumption that the time unit is years, so would be totally wrong)
    model.stratify(
        "agegroup",
        agegroup_strata,
        compartments_to_stratify=COMPARTMENTS,
        comp_split_props=comp_split_props,
        flow_adjustments=flow_adjustments,
        mixing_matrix=static_mixing_matrix,
    )

    """
    Case detection
    """

    # More empiric approach based on per capita testing rates
    if params["testing_to_detection"]:
        assumed_tests_parameter = params["testing_to_detection"]["assumed_tests_parameter"]
        assumed_cdr_parameter = params["testing_to_detection"][
            "assumed_cdr_parameter"
        ]  # Typically a calibration parameter

        # Use state denominator for testing rates for the Victorian health cluster models
        testing_region = "Victoria" if country_iso3 == "AUS" else pop_region
        testing_year = 2020 if country_iso3 == "AUS" else params["pop_year"]

        testing_pops = inputs.get_population_by_agegroup(
            agegroup_strata, country_iso3, testing_region, year=testing_year
        )

        detected_proportion = find_cdr_function_from_test_data(
            assumed_tests_parameter, assumed_cdr_parameter, country_iso3, testing_pops,
        )

    # Approach based on a hyperbolic tan function
    else:
        detect_prop_params = params["time_variant_detection"]

        def detected_proportion(t):
            return tanh_based_scaleup(
                detect_prop_params["maximum_gradient"],
                detect_prop_params["max_change_time"],
                detect_prop_params["start_value"],
                detect_prop_params["end_value"],
            )(t)

    """
    Importation 
    """
    # Determine how many importations there are, including the undetected and asymptomatic importations
    # This is defined 8x10 year bands, 0-70+, which we transform into 16x5 year bands 0-75+
    symptomatic_props = repeat_list_elements(2, params["symptomatic_props"])
    import_symptomatic_prop = sum(
        [
            import_prop * sympt_prop
            for import_prop, sympt_prop in zip(importation_props_by_age.values(), symptomatic_props)
        ]
    )

    def modelled_abs_detection_proportion_imported(t):
        return import_symptomatic_prop * detected_proportion(t)

    if implement_importation:
        if (
            pop_region
            and pop_region.lower().replace("__", "_").replace("_", "-")
            in Region.VICTORIA_SUBREGIONS
        ):
            import_times, importation_data = get_all_vic_notifications(
                excluded_regions=(pop_region,)
            )
            movement_prop = params["movement_prop"]
            movement_to_region = sum(total_pops) / sum(testing_pops) * movement_prop
            import_cases = [i_cases * movement_to_region for i_cases in importation_data]

        else:
            import_times = params["data"]["times_imported_cases"]
            import_cases = params["data"]["n_imported_cases"]

        import_rate_func = preprocess.importation.get_importation_rate_func_as_birth_rates(
            import_times, import_cases, modelled_abs_detection_proportion_imported
        )
        model.time_variants["importation_rate"] = import_rate_func

    """
    Stratify the model by clinical status
    Stratify the infectious compartments of the covid model (not including the pre-symptomatic compartments, which are
    actually infectious)

    - notifications are derived from progress from early to late for some strata
    - the proportion of people moving from presympt to early infectious, conditioned on age group
    - rate of which people flow through these compartments (reciprocal of time, using within_* which is a rate of ppl / day)
    - infectiousness levels adjusted by early/late and for clinical strata
    - we start with an age stratified infection fatality rate
        - 50% of deaths for each age bracket die in ICU
        - the other deaths go to hospital, assume no-one else can die from COVID
        - should we ditch this?

    """
    agegroup_strata = model.stratifications[0].strata
    clinical_strata = [
        ClinicalStratum.NON_SYMPT,
        ClinicalStratum.SYMPT_NON_HOSPITAL,
        ClinicalStratum.SYMPT_ISOLATE,
        ClinicalStratum.HOSPITAL_NON_ICU,
        ClinicalStratum.ICU,
    ]
    non_hospital_strata = [
        ClinicalStratum.NON_SYMPT,
        ClinicalStratum.SYMPT_NON_HOSPITAL,
        ClinicalStratum.SYMPT_ISOLATE,
    ]
    hospital_strata = [
        ClinicalStratum.HOSPITAL_NON_ICU,
        ClinicalStratum.ICU,
    ]

    """
    Infectiousness adjustments
    """
    strata_infectiousness = {}
    for stratum in clinical_strata:
        key = f"{stratum}_infect_multiplier"
        if key in params:
            strata_infectiousness[stratum] = params[key]

    """
    Make an infectiousness adjustment for isolation/quarantine
    Adjust infectiousness for pre-symptomatics so that they are less infectious.
    """
    clinical_inf_overwrites = [
        {
            "comp_name": Compartment.LATE_EXPOSED,
            "comp_strata": {},
            "value": params[f"{Compartment.LATE_EXPOSED}_infect_multiplier"],
        }
    ]

    for stratum in clinical_strata:
        if stratum in params["late_infect_multiplier"]:
            adjustment = {
                "comp_name": Compartment.LATE_ACTIVE,
                "comp_strata": {"clinical": stratum},
                "value": params["late_infect_multiplier"][stratum],
            }
            clinical_inf_overwrites.append(adjustment)

    # FIXME: This is not a desirable API, it's not really clear what is happening.
    model.individual_infectiousness_adjustments = clinical_inf_overwrites

    """
    Adjust infection death rates for hospital patients (ICU and non-ICU)
    """

    # Proportion of people in age group who die, given the number infected: dead / total infected.
    infection_fatality_props = get_infection_fatality_proportions(
        infection_fatality_props_10_year=params["infection_fatality_props"],
        infection_rate_multiplier=params["ifr_multiplier"],
        iso3=params["iso3"],
        pop_region=params["pop_region"],
        pop_year=params["pop_year"],
    )

    # Get the proportion of people in each clinical stratum, relative to total people in compartment.
    abs_props = get_absolute_strata_proportions(
        symptomatic_props=symptomatic_props,
        icu_props=params["icu_prop"],
        hospital_props=params["hospital_props"],
        symptomatic_props_multiplier=params["symptomatic_props_multiplier"],
        hospital_props_multiplier=params["hospital_props_multiplier"],
    )

    # Get the proportion of people who die for each strata/agegroup, relative to total infected.
    abs_death_props = get_absolute_death_proportions(
        abs_props=abs_props,
        infection_fatality_props=infection_fatality_props,
        icu_mortality_prop=params["icu_mortality_prop"],
    )

    # Calculate relative death proportions for each strata / agegroup.
    # This is the number of people in strata / agegroup who die, given the total num people in that strata / agegroup.
    relative_death_props = {
        stratum: np.array(abs_death_props[stratum]) / np.array(abs_props[stratum])
        for stratum in (
            ClinicalStratum.HOSPITAL_NON_ICU,
            ClinicalStratum.ICU,
            ClinicalStratum.NON_SYMPT,
        )
    }

    # Now we want to convert these death proprotions into flow rates.
    # These flow rates are the death rates for hospitalised patients in ICU and non-ICU.
    # We assume everyone who dies does so at the end of their time in the "late active" compartment.
    # We split the flow rate out of "late active" into a death or recovery flow, based on the relative death proportion.
    hospital_death_rates = (
        relative_death_props[ClinicalStratum.HOSPITAL_NON_ICU]
        * model.parameters[f"within_hospital_late"]
    )
    icu_death_rates = (
        relative_death_props[ClinicalStratum.ICU] * model.parameters[f"within_icu_late"]
    )

    # Apply adjusted infection death rates for hospital patients (ICU and non-ICU)
    # Death and non-death progression between infectious compartments towards the recovered compartment
    death_adjs = {}
    for idx, agegroup in enumerate(agegroup_strata):
        age_key = f"agegroup_{agegroup}"
        death_adjs[f"infect_deathX{age_key}"] = {
            "hospital_non_icuW": hospital_death_rates[idx],
            "icuW": icu_death_rates[idx],
        }

    """
    Adjust early exposed sojourn times.
    """

    # Progression rates into the infectious compartment(s)
    # Define progression rates into non-symptomatic compartments using parameter adjustment.
    early_exposed_adjs = {}
    for age_idx, age in enumerate(agegroup_strata):
        key = f"within_{Compartment.EARLY_EXPOSED}Xagegroup_{age}"
        early_exposed_adjs[key] = {
            ClinicalStratum.NON_SYMPT: abs_props[ClinicalStratum.NON_SYMPT][age_idx],
            ClinicalStratum.ICU: abs_props[ClinicalStratum.ICU][age_idx],
            ClinicalStratum.HOSPITAL_NON_ICU: abs_props[ClinicalStratum.HOSPITAL_NON_ICU][age_idx],
            ClinicalStratum.SYMPT_NON_HOSPITAL: f"prop_{ClinicalStratum.SYMPT_NON_HOSPITAL}_{agegroup}",
            ClinicalStratum.SYMPT_ISOLATE: f"prop_{ClinicalStratum.SYMPT_ISOLATE}_{agegroup}",
        }
        get_abs_prop_isolated = get_abs_prop_isolated_factory(
            age_idx, abs_props, detected_proportion
        )
        get_abs_prop_sympt_non_hospital = get_abs_prop_sympt_non_hospital_factory(
            age_idx, abs_props, get_abs_prop_isolated
        )
        model.time_variants[f"prop_{ClinicalStratum.SYMPT_ISOLATE}_{age}"] = get_abs_prop_isolated
        model.time_variants[
            f"prop_{ClinicalStratum.SYMPT_NON_HOSPITAL}_{age}"
        ] = get_abs_prop_sympt_non_hospital

    """
    Adjust early active sojourn times.
    """

    # Over-write rate of progression for early compartments for hospital and ICU
    within_hospital_early = model.parameters["within_hospital_early"]
    within_icu_early = model.parameters["within_icu_early"]
    early_active_adjs = {
        f"within_{Compartment.EARLY_ACTIVE}Xagegroup_{agegroup}": {
            f"{ClinicalStratum.HOSPITAL_NON_ICU}W": within_hospital_early,
            f"{ClinicalStratum.ICU}W": within_icu_early,
        }
        for agegroup in agegroup_strata
    }

    """
    Adjust late active sojourn times.
    """

    hospital_survival_props = 1 - relative_death_props[ClinicalStratum.HOSPITAL_NON_ICU]
    icu_survival_props = 1 - relative_death_props[ClinicalStratum.ICU]

    hospital_survival_rates = hospital_survival_props * model.parameters[f"within_hospital_late"]
    icu_survival_rates = icu_survival_props * model.parameters[f"within_icu_late"]

    late_active_adjs = {}
    for idx, agegroup in enumerate(agegroup_strata):
        age_key = f"agegroup_{agegroup}"
        late_active_adjs[f"within_{Compartment.LATE_ACTIVE}X{age_key}"] = {
            "hospital_non_icuW": hospital_survival_rates[idx],
            "icuW": icu_survival_rates[idx],
        }

    """
    Clinical proportions for imported cases.
    """

    # Work out time-variant clinical proportions for imported cases accounting for quarantine
    import_adjs = {}
    implement_importation = params["implement_importation"]
    import_quarantine = params["import_quarantine"]
    if implement_importation:
        tvs = model.time_variants
        importation_props_by_clinical = {}

        # Create scale-up function for quarantine
        quarantine_func = scale_up_function(
            import_quarantine["times"], import_quarantine["values"], method=4
        )

        # Loop through age groups and set the appropriate clinical proportions
        for agegroup in agegroup_strata:
            param_key = f"within_{Compartment.EARLY_EXPOSED}Xagegroup_{agegroup}"

            # Proportion entering non-symptomatic stratum reduced by the quarantined (and so isolated) proportion
            model.time_variants[
                f"tv_prop_importedX{agegroup}X{ClinicalStratum.NON_SYMPT}"
            ] = lambda t: early_exposed_adjs[param_key][ClinicalStratum.NON_SYMPT] * (
                1.0 - quarantine_func(t)
            )

            # Proportion ambulatory also reduced by quarantined proportion due to isolation
            model.time_variants[
                f"tv_prop_importedX{agegroup}X{ClinicalStratum.SYMPT_NON_HOSPITAL}"
            ] = lambda t: tvs[early_exposed_adjs[param_key][ClinicalStratum.SYMPT_NON_HOSPITAL]](
                t
            ) * (
                1.0 - quarantine_func(t)
            )

            # Proportion isolated includes those that would have been detected anyway and the ones above quarantined
            model.time_variants[
                f"tv_prop_importedX{agegroup}X{ClinicalStratum.SYMPT_ISOLATE}"
            ] = lambda t: quarantine_func(t) * (
                tvs[early_exposed_adjs[param_key][ClinicalStratum.SYMPT_NON_HOSPITAL]](t)
                + early_exposed_adjs[param_key][ClinicalStratum.NON_SYMPT]
            ) + tvs[
                early_exposed_adjs[param_key][ClinicalStratum.SYMPT_ISOLATE]
            ](
                t
            )

            # Create request with correct syntax for SUMMER for hospitalised and then non-hospitalised
            importation_props_by_clinical[agegroup] = {
                stratum: float(early_exposed_adjs[param_key][stratum])
                for stratum in hospital_strata
            }
            importation_props_by_clinical[agegroup].update(
                {
                    stratum: f"tv_prop_importedX{agegroup}X{stratum}"
                    for stratum in non_hospital_strata
                }
            )
            import_adjs[f"importation_rateXagegroup_{agegroup}"] = importation_props_by_clinical[
                agegroup
            ]

    # Only stratify infected compartments
    compartments_to_stratify = [
        Compartment.LATE_EXPOSED,
        Compartment.EARLY_ACTIVE,
        Compartment.LATE_ACTIVE,
    ]

    flow_adjustments = {
        **death_adjs,
        **early_exposed_adjs,
        **early_active_adjs,
        **late_active_adjs,
        **import_adjs,
    }

    # Stratify the model using the SUMMER stratification function
    model.stratify(
        "clinical",
        clinical_strata,
        compartments_to_stratify,
        infectiousness_adjustments=strata_infectiousness,
        flow_adjustments=flow_adjustments,
    )

    """
    Susceptibility stratification
    """

    susceptibility_heterogeneity = params["susceptibility_heterogeneity"]

    if susceptibility_heterogeneity:
        tail_cut = susceptibility_heterogeneity["tail_cut"]
        bins = susceptibility_heterogeneity["bins"]
        coeff_var = susceptibility_heterogeneity["coeff_var"]

        # Interpret data requests
        _, _, susc_values, susc_pop_props, _ = get_gamma_data(tail_cut, bins, coeff_var)
        check_modelled_susc_cv(susc_values, susc_pop_props, coeff_var)

        # Define strata names
        susc_strata_names = [f"suscept_{i_susc}" for i_susc in range(bins)]

        # Assign susceptibility values
        susc_adjustments = {
            susc_name: susc_value for susc_name, susc_value in zip(susc_strata_names, susc_values)
        }

        # Assign proportions of the population
        sus_pop_splits = {
            susc_name: susc_prop for susc_name, susc_prop in zip(susc_strata_names, susc_pop_props)
        }

        # Apply to all age groups individually (given current SUMMER API)
        susceptibility_adjustments = {
            f"contact_rateXagegroup_{str(i_agegroup)}": susc_adjustments
            for i_agegroup in agegroup_strata
        }

        # Stratify
        model.stratify(
            "suscept",
            list(susc_adjustments.keys()),
            [Compartment.SUSCEPTIBLE],
            flow_adjustments=susceptibility_adjustments,
            comp_split_props=sus_pop_splits,
        )

    """
    Set up and track derived output functions
    """

    # Set up derived outputs
    incidence_connections = outputs.get_incidence_connections(model.compartment_names)
    progress_connections = outputs.get_progress_connections(model.compartment_names)
    death_output_connections = outputs.get_infection_death_connections(model.compartment_names)
    model.add_flow_derived_outputs(incidence_connections)
    model.add_flow_derived_outputs(progress_connections)
    model.add_flow_derived_outputs(death_output_connections)

    # Build notification derived output function
    notification_func = outputs.get_calc_notifications_covid(
        implement_importation, modelled_abs_detection_proportion_imported,
    )
    local_notification_func = outputs.get_calc_notifications_covid(
        False, modelled_abs_detection_proportion_imported
    )

    # Build life expectancy derived output function
    life_expectancy = inputs.get_life_expectancy_by_agegroup(agegroup_strata, country_iso3)[0]
    life_expectancy_latest = [life_expectancy[agegroup][-1] for agegroup in life_expectancy]
    life_lost_func = outputs.get_calculate_years_of_life_lost(life_expectancy_latest)

    # Build hospital occupancy func
    compartment_periods = params["compartment_periods"]
    icu_early_period = compartment_periods["icu_early"]
    hospital_early_period = compartment_periods["hospital_early"]
    calculate_hospital_occupancy = outputs.get_calculate_hospital_occupancy(
        icu_early_period, hospital_early_period
    )

    func_outputs = {
        # Case-related
        "notifications": notification_func,
        "local_notifications": local_notification_func,
        "notifications_at_sympt_onset": outputs.get_notifications_at_sympt_onset,
        # Death-related
        "years_of_life_lost": life_lost_func,
        "accum_deaths": outputs.calculate_cum_deaths,
        # Health care-related
        "hospital_occupancy": calculate_hospital_occupancy,
        "icu_occupancy": outputs.calculate_icu_occupancy,
        "new_hospital_admissions": outputs.calculate_new_hospital_admissions_covid,
        "new_icu_admissions": outputs.calculate_new_icu_admissions_covid,
        # Other
        "proportion_seropositive": outputs.calculate_proportion_seropositive,
    }

    # Derived outputs for the optimization project.
    if mobility_region in OPTI_REGIONS:
        func_outputs["accum_years_of_life_lost"] = outputs.calculate_cum_years_of_life_lost

    model.add_function_derived_outputs(func_outputs)
    return model


def get_abs_prop_isolated_factory(age_idx, abs_props, prop_detect_among_sympt_func):
    def get_abs_prop_isolated(t):
        """
        Returns the absolute proportion of infected becoming isolated at home.
        Isolated people are those who are detected but not sent to hospital.
        """
        abs_prop_detected = abs_props["sympt"][age_idx] * prop_detect_among_sympt_func(t)
        abs_prop_isolated = abs_prop_detected - abs_props["hospital"][age_idx]
        if abs_prop_isolated < 0:
            # If more people go to hospital than are detected, ignore detection
            # proportion, and assume no one is being isolated.
            abs_prop_isolated = 0

        return abs_prop_isolated

    return get_abs_prop_isolated


def get_abs_prop_sympt_non_hospital_factory(age_idx, abs_props, get_abs_prop_isolated_func):
    def get_abs_prop_sympt_non_hospital(t):
        """
        Returns the absolute proportion of infected not entering the hospital.
        This also does not count people who are isolated.
        This is only people who are not detected.
        """
        return (
            abs_props["sympt"][age_idx]
            - abs_props["hospital"][age_idx]
            - get_abs_prop_isolated_func(t)
        )

    return get_abs_prop_sympt_non_hospital


def get_absolute_strata_proportions(
    symptomatic_props,
    icu_props,
    hospital_props,
    symptomatic_props_multiplier,
    hospital_props_multiplier,
):
    """
    Returns the proportion of people in each clinical stratum.
    ie: Given all the people people who are infected, what proportion are in each strata?
    Each of these are stratified into 16 age groups 0-75+
    """
    # Apply multiplier to proportions
    hospital_props = [min([p * hospital_props_multiplier, 1.0]) for p in hospital_props]
    symptomatic_props = [min([p * symptomatic_props_multiplier, 1.0]) for p in symptomatic_props]

    # Find the absolute progression proportions.
    symptomatic_props_arr = np.array(symptomatic_props)
    hospital_props_arr = np.array(hospital_props)
    # Determine the absolute proportion of early exposed who become sympt vs non-sympt
    sympt, non_sympt = subdivide_props(1, symptomatic_props_arr)
    # Determine the absolute proportion of sympt who become hospitalized vs non-hospitalized.
    sympt_hospital, sympt_non_hospital = subdivide_props(sympt, hospital_props_arr)
    # Determine the absolute proportion of hospitalized who become icu vs non-icu.
    sympt_hospital_icu, sympt_hospital_non_icu = subdivide_props(sympt_hospital, icu_props)

    return {
        "sympt": sympt,
        "non_sympt": non_sympt,
        "hospital": sympt_hospital,
        "sympt_non_hospital": sympt_non_hospital,  # Over-ridden by a by time-varying proportion later
        ClinicalStratum.ICU: sympt_hospital_icu,
        ClinicalStratum.HOSPITAL_NON_ICU: sympt_hospital_non_icu,
    }


def get_absolute_death_proportions(abs_props, infection_fatality_props, icu_mortality_prop):
    """
    Calculate death proportions: find where the absolute number of deaths accrue
    Represents the number of people in a strata who die given the total number of people infected.
    """
    NUM_AGE_STRATA = 16
    abs_death_props = {
        ClinicalStratum.NON_SYMPT: np.zeros(NUM_AGE_STRATA),
        ClinicalStratum.ICU: np.zeros(NUM_AGE_STRATA),
        ClinicalStratum.HOSPITAL_NON_ICU: np.zeros(NUM_AGE_STRATA),
    }
    for age_idx in range(NUM_AGE_STRATA):
        age_ifr_props = infection_fatality_props[age_idx]

        # Make sure there are enough asymptomatic and hospitalised proportions to fill the IFR
        thing = (
            abs_props["non_sympt"][age_idx]
            + abs_props[ClinicalStratum.HOSPITAL_NON_ICU][age_idx]
            + abs_props[ClinicalStratum.ICU][age_idx] * icu_mortality_prop
        )
        age_ifr_props = min(thing, age_ifr_props)

        # Absolute proportion of all patients dying in ICU
        # Maximum ICU mortality allowed
        thing = abs_props[ClinicalStratum.ICU][age_idx] * icu_mortality_prop
        abs_death_props[ClinicalStratum.ICU][age_idx] = min(thing, age_ifr_props)

        # Absolute proportion of all patients dying in hospital, excluding ICU
        thing = max(
            age_ifr_props
            - abs_death_props[ClinicalStratum.ICU][
                age_idx
            ],  # If left over mortality from ICU for hospitalised
            0.0,  # Otherwise zero
        )
        abs_death_props[ClinicalStratum.HOSPITAL_NON_ICU][age_idx] = min(
            thing,
            # Otherwise fill up hospitalised
            abs_props[ClinicalStratum.HOSPITAL_NON_ICU][age_idx],
        )

        # Absolute proportion of all patients dying out of hospital
        thing = (
            age_ifr_props
            - abs_death_props[ClinicalStratum.ICU][age_idx]
            - abs_death_props[ClinicalStratum.HOSPITAL_NON_ICU][age_idx]
        )  # If left over mortality from hospitalised
        abs_death_props[ClinicalStratum.NON_SYMPT][age_idx] = max(0.0, thing)  # Otherwise zero

        # Check everything sums up properly
        allowed_rounding_error = 6
        assert round(
            abs_death_props[ClinicalStratum.ICU][age_idx]
            + abs_death_props[ClinicalStratum.HOSPITAL_NON_ICU][age_idx]
            + abs_death_props[ClinicalStratum.NON_SYMPT][age_idx],
            allowed_rounding_error,
        ) == round(age_ifr_props, allowed_rounding_error)
        # Check everything sums up properly
        allowed_rounding_error = 6
        assert round(
            abs_death_props[ClinicalStratum.ICU][age_idx]
            + abs_death_props[ClinicalStratum.HOSPITAL_NON_ICU][age_idx]
            + abs_death_props[ClinicalStratum.NON_SYMPT][age_idx],
            allowed_rounding_error,
        ) == round(age_ifr_props, allowed_rounding_error)

    return abs_death_props


def get_infection_fatality_proportions(
    infection_fatality_props_10_year, infection_rate_multiplier, iso3, pop_region, pop_year,
):
    """
    Returns the Proportion of people in age group who die, given the total number of people in that compartment.
    ie: dead / total infected
    """
    if_props_10_year = infection_rate_multiplier * np.array(infection_fatality_props_10_year)
    # Calculate the proportion of 80+ years old among the 75+ population
    elderly_populations = inputs.get_population_by_agegroup(
        [0, 75, 80], iso3, pop_region, year=pop_year
    )
    prop_over_80 = elderly_populations[2] / sum(elderly_populations[1:])
    # Infection fatality rate by age group.
    # Data in props used 10 year bands 0-80+, but we want 5 year bands from 0-75+
    # Calculate 75+ age bracket as weighted average between 75-79 and half 80+
    return repeat_list_elements_average_last_two(if_props_10_year, prop_over_80)


def subdivide_props(base_props: np.ndarray, split_props: np.ndarray):
    """
    Split an array (base_array) of proportions into two arrays (split_arr, complement_arr)
    according to the split proportions provided (split_prop).
    """
    split_arr = base_props * split_props
    complement_arr = base_props * (1 - split_props)
    return split_arr, complement_arr
