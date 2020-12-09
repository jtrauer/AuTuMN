"""
Type definition for model parameters
"""
from datetime import date
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, validator, root_validator
from pydantic.dataclasses import dataclass

from apps.covid_19.constants import BASE_DATE, BASE_DATETIME


class Time(BaseModel):
    """
    Model time period
    Running time-related - for COVID-19 model
    All times are assumed to be in days and reference time is 31st Dec 2019
    """

    start: float
    end: float
    step: float


class TimeSeries(BaseModel):
    """A set of values with associated time points"""

    times: List[float]
    values: List[float]

    @root_validator(pre=True, allow_reuse=True)
    def check_lengths(cls, values):
        vs, ts = values.get("values"), values.get("times")
        assert len(ts) == len(vs), f"TimeSeries length mismatch."
        return values

    @validator("times", pre=True, allow_reuse=True)
    def parse_dates_to_days(dates):
        return [(d - BASE_DATE).days if type(d) is date else d for d in dates]


class Country(BaseModel):
    """The country that the model is based in."""

    iso3: str


class Population(BaseModel):
    """Model population parameters"""

    region: Optional[str]  # None/null means default to parent country
    year: int


class Sojourn(BaseModel):
    """Parameters for determining how long a person stays in a given compartment."""

    class CalcPeriod(BaseModel):
        total_period: float
        proportions: Dict[str, float]

    # Mean time in days spent in each compartment
    compartment_periods: Dict[str, float]
    # Mean time spent in each compartment, defined via proportions
    compartment_periods_calculated: Dict[str, CalcPeriod]


class MixingLocation(BaseModel):
    # Whether to append or overwrite times / values
    append: bool
    # Times for dynamic mixing func.
    times: List[int]
    # Values for dynamic mixing func.
    values: List[Any]

    @root_validator(pre=True, allow_reuse=True)
    def check_lengths(cls, values):
        vs, ts = values.get("values"), values.get("times")
        assert len(ts) == len(vs), f"Mixing series length mismatch."
        return values

    @validator("times", pre=True, allow_reuse=True)
    def parse_dates_to_days(dates):
        return [(d - BASE_DATE).days if type(d) is date else d for d in dates]


class EmpiricMicrodistancingParams(BaseModel):
    max_effect: float
    times: List[float]
    values: List[float]


class TanhMicrodistancingParams(BaseModel):
    b: float
    c: float
    sigma: float
    upper_asymptote: float


class MicroDistancingFunc(BaseModel):
    function_type: str
    parameters: Union[EmpiricMicrodistancingParams, TanhMicrodistancingParams]


class Mobility(BaseModel):
    """Google mobility params"""

    region: Optional[str]  # None/null means default to parent country
    mixing: Dict[str, MixingLocation]
    age_mixing: Optional[Dict[str, TimeSeries]]
    microdistancing: Dict[str, MicroDistancingFunc]
    microdistancing_locations: List[str]
    smooth_google_data: bool
    square_mobility_effect: bool
    npi_effectiveness: Dict[str, float]
    google_mobility_locations: Dict[str, List[str]]


class AgeStratification(BaseModel):
    """Parameters used in age based stratification"""

    max_age: int  # Maximum age used for age strata.
    age_step_size: int  # Step size used for age strata.
    # Susceptibility by age
    susceptibility: Dict[str, float]


class StrataProps(BaseModel):
    props: List[float]
    multiplier: float


class ClinicalProportions(BaseModel):
    hospital: StrataProps
    symptomatic: StrataProps


class ClinicalStratification(BaseModel):
    """Parameters used in clinical status based stratification"""

    props: ClinicalProportions
    icu_prop: float  # Proportion of those hospitalised that are admitted to ICU
    icu_mortality_prop: float  # Death proportion ceiling for ICU mortality
    late_infect_multiplier: Dict[str, float]
    late_exposed_infect_multiplier: float
    non_sympt_infect_multiplier: float


class InfectionFatality(BaseModel):
    """Parameters relating to death from infection"""

    # Calibrated multiplier for props.
    multiplier: float
    # Proportion of people dying / total infected by age.
    props: List[float]


class CaseDetection(BaseModel):
    """Time variant detection of cases"""

    maximum_gradient: float  # The shape parameter to the tanh-based curve
    max_change_time: float  # Point at which curve inflects
    start_value: float  # Starting value - lower asymptote for increasing function
    end_value: float  # End value - upper asymptote for increasing function


class TestingToDetection(BaseModel):
    """
    More empiric approach based on per capita testing rates
    An alternative to CaseDetection.
    """

    assumed_tests_parameter: float
    assumed_cdr_parameter: float
    smoothing_period: int


class SusceptibilityHeterogeneity(BaseModel):
    """Specifies heterogeneity in susceptibility"""

    bins: int
    tail_cut: float
    coeff_var: float


class Importation(BaseModel):
    case_timeseries: TimeSeries
    quarantine_timeseries: TimeSeries
    props_by_age: Optional[Dict[str, float]]
    movement_prop: Optional[float]


class MetroClusterStratification(BaseModel):
    mobility: Mobility


class RegionalClusterStratification(BaseModel):
    mobility: Mobility


class VictorianClusterStratification(BaseModel):
    intercluster_mixing: float
    contact_rate_multiplier_regional: float
    contact_rate_multiplier_north_metro: float
    contact_rate_multiplier_west_metro: float
    contact_rate_multiplier_south_metro: float
    contact_rate_multiplier_south_east_metro: float
    contact_rate_multiplier_loddon_mallee: float
    contact_rate_multiplier_barwon_south_west: float
    contact_rate_multiplier_hume: float
    contact_rate_multiplier_gippsland: float
    contact_rate_multiplier_grampians: float
    metro: MetroClusterStratification
    regional: RegionalClusterStratification


class ParamConfig:
    """Config for parameter models"""

    anystr_strip_whitespace = True  # Strip whitespace
    allow_mutation = False  # Params should be immutable


@dataclass(config=ParamConfig)
class Parameters:
    contact_rate: float
    infect_death: float
    universal_death_rate: float
    infectious_seed: float
    seasonal_force: Optional[float]  # Seasonal forcing factor
    elderly_mixing_reduction: Optional[dict]
    waning_immunity_duration: Optional[float]
    stratify_by_infection_history: bool
    rel_prop_symptomatic_experienced: Optional[float]
    haario_scaling_factor: float
    metropolis_initialisation_type: str
    # Modular parameters.
    time: Time
    country: Country
    population: Population
    sojourn: Sojourn
    mobility: Mobility
    infection_fatality: InfectionFatality
    age_stratification: AgeStratification
    clinical_stratification: ClinicalStratification
    case_detection: CaseDetection
    testing_to_detection: Optional[TestingToDetection]
    importation: Optional[Importation]
    victorian_clusters: Optional[VictorianClusterStratification]
    # Dummy parameters - not used
    notifications_dispersion_param: float
    icu_occupancy_dispersion_param: float
    proportion_seropositive_dispersion_param: float
    hospital_occupancy_dispersion_param: float
    new_hospital_admissions_dispersion_param: float
    new_icu_admissions_dispersion_param: float
    infection_deaths_dispersion_param: float
    accum_deaths_dispersion_param: float

    notifications_for_cluster_barwon_south_west_dispersion_param: float
    notifications_for_cluster_gippsland_dispersion_param: float
    notifications_for_cluster_hume_dispersion_param: float
    notifications_for_cluster_loddon_mallee_dispersion_param: float
    notifications_for_cluster_grampians_dispersion_param: float
    notifications_for_cluster_north_metro_dispersion_param: float
    notifications_for_cluster_south_east_metro_dispersion_param: float
    notifications_for_cluster_south_metro_dispersion_param: float
    notifications_for_cluster_west_metro_dispersion_param: float

    accum_hospital_admissions_for_cluster_north_metro_dispersion_param: float
    accum_hospital_admissions_for_cluster_south_east_metro_dispersion_param: float
    accum_hospital_admissions_for_cluster_south_metro_dispersion_param: float
    accum_hospital_admissions_for_cluster_west_metro_dispersion_param: float

    accum_icu_admissions_for_cluster_north_metro_dispersion_param: float
    accum_icu_admissions_for_cluster_south_east_metro_dispersion_param: float
    accum_icu_admissions_for_cluster_south_metro_dispersion_param: float
    accum_icu_admissions_for_cluster_west_metro_dispersion_param: float

    accum_deaths_for_cluster_north_metro_dispersion_param: float
    accum_deaths_for_cluster_south_east_metro_dispersion_param: float
    accum_deaths_for_cluster_south_metro_dispersion_param: float
    accum_deaths_for_cluster_west_metro_dispersion_param: float

    notifications_metro_dispersion_param: float
    notifications_rural_dispersion_param: float
    hospital_admissions_metro_dispersion_param: float
    icu_admissions_metro_dispersion_param: float
    accum_deaths_metro_dispersion_param: float
