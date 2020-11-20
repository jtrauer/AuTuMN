import matplotlib.ticker as ticker
from matplotlib import colors
import datetime
from typing import List


PLOT_TEXT_DICT = {
    "contact_rate": "infection risk per contact",
    "hospital_props_multiplier": "hospital risk multiplier",
    "compartment_periods.icu_early": "pre-ICU period",
    "testing_to_detection.assumed_cdr_parameter": "CDR at base testing rate",
    "microdistancing.parameters.max_effect": "max effect microdistancing",
    "icu_occupancy": "ICU occupancy",
    # TB model parameters
    "start_population_size": "initial population size",
    "late_reactivation_multiplier": "late reactivation multiplier",
    "time_variant_tb_screening_rate.maximum_gradient": "screening profile (shape)",
    "time_variant_tb_screening_rate.max_change_time": "screening profile (inflection)",
    "time_variant_tb_screening_rate.end_value": "screening profile (final rate)",
    "user_defined_stratifications.location.adjustments.detection_rate.ebeye": "rel. screening rate (Ebeye)",
    "user_defined_stratifications.location.adjustments.detection_rate.other": "rel. screening rate (Other Isl.)",
    "extra_params.rr_progression_diabetes": "rel. progression rate (diabetes)",
    "rr_infection_recovered": "RR infection (recovered)",
    "pt_efficacy": "PT efficacy",
    "infect_death_rate_dict.smear_positive": "TB mortality (smear-pos)",
    "infect_death_rate_dict.smear_negative": "TB mortality (smear-neg)",
    "self_recovery_rate_dict.smear_positive": "Self cure rate (smear-pos)",
    "self_recovery_rate_dict.smear_negative": "Self cure rate (smear-neg)",
    "proportion_seropositive": "seropositive percentage",
    "infection_deaths": "deaths per day",
    "notifications": "notifications per day",
    "incidence": "incident episodes per day",
    "accum_deaths": "cumulative deaths",
    "new_hospital_admissions": "new hospitalisations per day",
    "new_icu_admissions": "new ICU admissions per day",
    "hospital_occupancy": "hospital occupancy",
    "sojourn.compartment_periods_calculated.exposed.total_period": "incubation period",
    "sojourn.compartment_periods_calculated.active.total_period": "duration active",
    "seasonal_force": "seasonal forcing",
    "mobility.microdistancing.behaviour.parameters.max_effect": "max effect microdistancing",
    "mobility.microdistancing.behaviour_adjuster.parameters.sigma": "microdist max wane",
    "mobility.microdistancing.behaviour_adjuster.parameters.c": "microdist wane time",
    "clinical_stratification.props.hospital.multiplier": "hospitalisation multiplier",
    "clinical_stratification.icu_prop": "ICU proportion",
    "sojourn.compartment_periods.icu_early": "ICU duration",
    "other_locations": "other locations",
    "clinical_stratification.props.symptomatic.multiplier": "symptomatic proportion multiplier",
    "manila": "national capital region",
}

ALPHAS = (1.0, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08)
# https://matplotlib.org/3.1.0/gallery/color/named_colors.html
COLORS = (
    # Blues
    ["lightsteelblue", "cornflowerblue", "royalblue", "navy"],
    # Purples
    ["plum", "mediumorchid", "darkviolet", "rebeccapurple"],
    # Greens
    ["palegreen", "mediumspringgreen", "mediumseagreen", "darkgreen"],
    # Yellows
    ["lightgoldenrodyellow", "palegoldenrod", "gold", "darkgoldenrod"],
    # Orangey-browns
    ["papayawhip", "navajowhite", "burlywood", "saddlebrown"],
    # Cyans
    ["lightcyan", "paleturquoise", "darkcyan", "darkslategrey"],
    # Greys
    ["lightgrey", "darkgrey", "dimgrey", "black"],
    # Reds
    ["lightsalmon", "darksalmon", "tomato", "darkred"],
)
REF_DATE = datetime.date(2019, 12, 31)


def get_plot_text_dict(param_string, capitalise_first_letter=False, remove_underscore=True, remove_dot=True):
    """
    Get standard text for use in plotting as title, y-label, etc.
    """

    text = PLOT_TEXT_DICT[param_string] if param_string in PLOT_TEXT_DICT else param_string
    if capitalise_first_letter:
        text = text[0].upper() + text[1:]
    if remove_underscore:
        text = text.replace("_", " ")
    if remove_dot:
        text = text.replace(".", " ")
    return text


def change_xaxis_to_date(axis, ref_date, date_str_format="%#d-%b", rotation=30):
    """
    Change the format of a numerically formatted x-axis to date.
    """

    def to_date(x_value, pos):
        date = ref_date + datetime.timedelta(days=int(x_value))
        return date.strftime(date_str_format)

    date_format = ticker.FuncFormatter(to_date)
    axis.xaxis.set_major_formatter(date_format)
    axis.xaxis.set_tick_params(rotation=rotation)


def _apply_transparency(color_list: List[str], alphas: List[str]):
    """Make a list of colours transparent, based on a list of alphas"""
    for i in range(len(color_list)):
        for j in range(len(color_list[i])):
            rgb_color = list(colors.colorConverter.to_rgb(color_list[i][j]))
            color_list[i][j] = rgb_color + [alphas[i]]
    return color_list


def _plot_targets_to_axis(axis, values: List[float], times: List[int], on_uncertainty_plot=False):
    """
    Plot output value calibration targets as points on the axis.
    # TODO: add back ability to plot confidence interval
    x_vals = [time, time]
    axis.plot(x_vals, values[1:], "m", linewidth=1, color="red")
    axis.scatter(time, values[0], marker="o", color="red", s=30)
    axis.scatter(time, values[0], marker="o", color="white", s=10)
    """
    assert len(times) == len(values), "Targets have inconsistent length"
    # Plot a single point estimate
    if on_uncertainty_plot:
        axis.scatter(times, values, marker="o", color="black", s=10)
    else:
        axis.scatter(times, values, marker="o", color="red", s=30, zorder=999)
        axis.scatter(times, values, marker="o", color="white", s=10, zorder=999)
