"""
FIXME: These all need tests.
"""
from summer.model import StratifiedModel
from apps.covid_19.constants import Compartment as CompartmentType, ClinicalStratum
from typing import List, Dict
from datetime import date

import numpy as np

from summer.model import StratifiedModel
from summer.compartment import Compartment
from summer.model.derived_outputs import (
    InfectionDeathFlowOutput,
    TransitionFlowOutput,
)


NOTIFICATION_STRATUM = [
    ClinicalStratum.SYMPT_ISOLATE,
    ClinicalStratum.HOSPITAL_NON_ICU,
    ClinicalStratum.ICU,
]

HOSPITAL_STRATUM = [
    ClinicalStratum.HOSPITAL_NON_ICU,
    ClinicalStratum.ICU,
]


def get_calc_notifications_covid(
        include_importation,
        prop_detected_func,
        request_stratum=""
):
    def calculate_notifications_covid(
            time_idx: int,
            model: StratifiedModel,
            compartment_values: np.ndarray,
            derived_outputs: Dict[str, np.ndarray],
    ):
        """
        Returns the number of notifications for a given time.
        The fully stratified incidence outputs must be available before calling this function
        """
        notifications_count = 0.0
        for output_name, output_values in derived_outputs.items():
            is_progress = "progressX" in output_name
            is_notify_stratum = any([stratum in output_name for stratum in NOTIFICATION_STRATUM])
            is_request_stratum = request_stratum in output_name
            if is_progress and is_notify_stratum and is_request_stratum:
                notifications_count += output_values[time_idx]

        if include_importation:
            time = model.times[time_idx]
            notifications_count += prop_detected_func(time) * model.get_parameter_value(
                "importation_rate", time
            )

        return notifications_count

    return calculate_notifications_covid


def calculate_new_hospital_admissions_covid(
    time_idx: int,
    model: StratifiedModel,
    compartment_values: np.ndarray,
    derived_outputs: Dict[str, np.ndarray],
):
    hosp_admissions = 0.0
    for output_name, output_values in derived_outputs.items():
        if "progressX" in output_name and any(
            [stratum in output_name for stratum in HOSPITAL_STRATUM]
        ):
            hosp_admissions += output_values[time_idx]

    return hosp_admissions


def calculate_new_icu_admissions_covid(
    time_idx: int,
    model: StratifiedModel,
    compartment_values: np.ndarray,
    derived_outputs: Dict[str, np.ndarray],
):
    icu_admissions = 0.0
    for output_name, output_values in derived_outputs.items():
        if "progressX" in output_name and f"clinical_{ClinicalStratum.ICU}" in output_name:
            icu_admissions += output_values[time_idx]

    return icu_admissions


def get_calculate_hospital_occupancy(
    model: StratifiedModel, icu_early_period, hospital_early_period
):
    period_icu_patients_in_hospital = max(
        icu_early_period - hospital_early_period,
        0.0,
    )
    proportion_icu_patients_in_hospital = period_icu_patients_in_hospital / icu_early_period
    late_active_idxs = []
    early_active_idxs = []
    for i, comp in enumerate(model.compartment_names):
        is_late_active = comp.has_name(CompartmentType.LATE_ACTIVE)
        is_early_active = comp.has_name(CompartmentType.EARLY_ACTIVE)
        is_icu = comp.has_stratum("clinical", ClinicalStratum.ICU)
        is_hospital_non_icu = comp.has_stratum("clinical", ClinicalStratum.HOSPITAL_NON_ICU)
        if is_late_active and (is_icu or is_hospital_non_icu):
            # Both ICU and hospital late active compartments
            late_active_idxs.append(i)
        elif is_early_active and is_icu:
            # A proportion of the early active ICU compartment
            early_active_idxs.append(i)

    def calculate_hospital_occupancy(
        time_idx: int,
        _: StratifiedModel,
        compartment_values: np.ndarray,
        derived_outputs: Dict[str, np.ndarray],
    ):
        hospital_prev = 0.0
        hospital_prev += np.sum(compartment_values[late_active_idxs])
        hospital_prev += (
            np.sum(compartment_values[early_active_idxs]) * proportion_icu_patients_in_hospital
        )
        return hospital_prev

    return calculate_hospital_occupancy


def calculate_icu_occupancy(
    time_idx: int,
    model: StratifiedModel,
    compartment_values: np.ndarray,
    derived_outputs: Dict[str, np.ndarray],
):
    icu_prev = 0
    for i, comp in enumerate(model.compartment_names):
        is_late_active = comp.has_name(CompartmentType.LATE_ACTIVE)
        is_icu = comp.has_stratum("clinical", ClinicalStratum.ICU)
        if is_late_active and is_icu:
            icu_prev += compartment_values[i]

    return icu_prev


def calculate_proportion_seropositive(
    time_idx: int,
    model: StratifiedModel,
    compartment_values: np.ndarray,
    derived_outputs: Dict[str, np.ndarray],
):
    n_seropositive = 0
    for i, comp in enumerate(model.compartment_names):
        if comp.has_name(CompartmentType.RECOVERED):
            n_seropositive += compartment_values[i]

    return n_seropositive / compartment_values.sum()


def make_age_specific_seroprevalence_output(agegroup):
    def calculate_proportion_seropositive_by_age(
        time_idx: int,
        model: StratifiedModel,
        compartment_values: np.ndarray,
        derived_outputs: Dict[str, np.ndarray],
    ):
        n_seropositive = 0
        agegroup_population = 0
        for i, comp in enumerate(model.compartment_names):
            if comp.has_stratum("agegroup", agegroup):
                agegroup_population += compartment_values[i]
                if comp.has_name(CompartmentType.RECOVERED):
                    n_seropositive += compartment_values[i]
        return n_seropositive / agegroup_population

    return calculate_proportion_seropositive_by_age


def get_calculate_years_of_life_lost(life_expectancy_by_agegroup):
    def calculate_years_of_life_lost(
        time_idx: int,
        model: StratifiedModel,
        compartment_values: np.ndarray,
        derived_outputs: Dict[str, np.ndarray],
    ):
        total_yoll = 0.0
        age_strata = model.stratifications[0].strata
        for i, agegroup in enumerate(age_strata):
            for output_name, output_values in derived_outputs.items():
                if f"infection_deathsXagegroup_{agegroup}" in output_name:
                    total_yoll += output_values[time_idx] * life_expectancy_by_agegroup[i]

        return total_yoll

    return calculate_years_of_life_lost


def get_notifications_at_sympt_onset(
    time_idx: int,
    model: StratifiedModel,
    compartment_values: np.ndarray,
    derived_outputs: Dict[str, np.ndarray],
):
    notifications_sympt_onset = 0.0
    for output_name, output_values in derived_outputs.items():
        if "incidence" in output_name and (
            f"clinical_{ClinicalStratum.SYMPT_ISOLATE}" in output_name
            or f"clinical_{ClinicalStratum.HOSPITAL_NON_ICU}" in output_name
            or f"clinical_{ClinicalStratum.ICU}" in output_name
        ):
            notifications_sympt_onset += output_values[time_idx]
    return notifications_sympt_onset


def calculate_cum_deaths(
    time_idx: int,
    model: StratifiedModel,
    compartment_values: np.ndarray,
    derived_outputs: Dict[str, np.ndarray],
):
    """
    Cumulative deaths, used for minimize deaths optimization.
    """
    return derived_outputs["infection_deaths"][: (time_idx + 1)].sum()


def make_agespecific_cum_deaths_func(agegroup):
    def calculate_cum_deaths_by_age(
        time_idx: int,
        model: StratifiedModel,
        compartment_values: np.ndarray,
        derived_outputs: Dict[str, np.ndarray],
    ):
        starts_with = f"infection_deathsXagegroup_{agegroup}X"
        deaths_by_age = 0
        for derived_output_name in list(derived_outputs.keys()):
            if derived_output_name.startswith(starts_with):
                deaths_by_age += derived_outputs[derived_output_name][: (time_idx + 1)].sum()

        return deaths_by_age

    return calculate_cum_deaths_by_age


def calculate_cum_years_of_life_lost(
    time_idx: int,
    model: StratifiedModel,
    compartment_values: np.ndarray,
    derived_outputs: Dict[str, np.ndarray],
):
    """
    Cumulative years of life lost, used for minimize YOLL optimization.
    """
    return derived_outputs["years_of_life_lost"][: time_idx + 1].sum()


def get_progress_connections(comps: List[Compartment]):
    """
    Track "progress": flow from early infectious cases to late infectious cases.
    """
    return _get_transition_flow_connections(
        output_name="progress",
        source=CompartmentType.EARLY_ACTIVE,
        dest=CompartmentType.LATE_ACTIVE,
        comps=comps,
    )


def get_incidence_connections(comps: List[Compartment]):
    """
    Track "incidence": flow from presymptomatic cases to infectious cases.
    """
    return _get_transition_flow_connections(
        output_name="incidence",
        source=CompartmentType.LATE_EXPOSED,
        dest=CompartmentType.EARLY_ACTIVE,
        comps=comps,
    )


def _get_transition_flow_connections(
    output_name: str, source: str, dest: str, comps: List[Compartment]
):
    connections = {}
    connections[output_name] = TransitionFlowOutput(
        source=source,
        dest=dest,
        source_strata={},
        dest_strata={},
    )
    for comp in comps:
        if not comp.has_name(dest):
            continue

        strata = comp.get_strata()
        for i in range(1, len(strata) + 1):
            strata_used = strata[0:i]
            output_key = "X".join([output_name, *strata_used])
            strat_vals_used = {
                k: v for k, v in comp._strat_values.items() if f"{k}_{v}" in strata_used
            }
            connections[output_key] = TransitionFlowOutput(
                source=source,
                dest=dest,
                source_strata={},
                dest_strata=strat_vals_used,
            )

    return connections


def get_infection_death_connections(compartments: List[Compartment]):
    """
    Assumes only late infectious can die from the infection.
    """
    connections = {}
    connections["infection_deaths"] = InfectionDeathFlowOutput(
        source=CompartmentType.LATE_ACTIVE, source_strata={}
    )
    for comp in compartments:
        if not comp.has_name(CompartmentType.LATE_ACTIVE):
            continue

        output_key = "X".join(["infection_deaths", *comp.get_strata()])
        connections[output_key] = InfectionDeathFlowOutput(
            source=CompartmentType.LATE_ACTIVE,
            source_strata=comp._strat_values,
        )

    return connections
