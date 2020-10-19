from autumn.tool_kit.params import load_targets
from autumn import db
from autumn.plots.calibration.plots import get_posterior
from autumn.curve import tanh_based_scaleup
from autumn.plots.uncertainty.plots import plot_timeseries_with_uncertainty

from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS, COUNTRY_TITLES

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from numpy import mean, quantile
import numpy as np
import os


# --------------  Load outputs form databases
def get_calibration_outputs(country):
    calib_dirpath = f"../../pbi_databases/calibration/{country}/"
    mcmc_tables = db.load.load_mcmc_tables(calib_dirpath)
    mcmc_params = db.load.load_mcmc_params_tables(calib_dirpath)
    uncertainty_df = db.load.load_uncertainty_table(calib_dirpath)

    return mcmc_tables, mcmc_params, uncertainty_df


def get_all_countries_calibration_outputs():
    calibration_outputs = {}
    for country in OPTI_REGIONS:
        mcmc_tables, mcmc_params, uncertainty_df = get_calibration_outputs(country)
        calibration_outputs[country] = {
            "mcmc_tables": mcmc_tables, "mcmc_params": mcmc_params, "uncertainty_df": uncertainty_df
        }

    return calibration_outputs


def get_targets(region_name):
    return load_targets("covid_19", region_name)


def get_parameter_values(calibration_outputs):
    param_values = {}
    for country, outputs in calibration_outputs.items():
        if len(outputs["mcmc_params"]) == 0:
            continue
        param_values[country] = {}
        all_param_names = [p for p in outputs["mcmc_params"][0]["name"].unique().tolist() if 'dispersion_param' not in p]
        for param_name in all_param_names:
            param_values[country][param_name] = get_posterior(
                outputs["mcmc_params"], outputs["mcmc_tables"], param_name, 0
            )[param_name].tolist()
    return param_values

# --------------  Make figure with posterior estimates

param_info = {
    'contact_rate': {'name': 'transmission prob.', 'range': [0.02, 0.06]},
    'time.start': {'name': 'model start time', 'range': [0., 40.]},
    'sojourn.compartment_periods_calculated.exposed.total_period': {'name': 'incubation time', 'range': [3., 7.]},
    'sojourn.compartment_periods_calculated.active.total_period': {'name': 'time infectious', 'range': [4., 10.]},
    'infection_fatality.multiplier': {'name': 'IFR multiplier', 'range': [.8, 1.2]},
    'case_detection.maximum_gradient': {'name': 'detection (shape)', 'range': [.05, .1]},
    'case_detection.max_change_time': {'name': 'detection (inflection)', 'range': [100., 250.]},
    'case_detection.end_value': {'name': 'detection (prop_final)', 'range': [.10, .90]},
    'case_detection.start_value': {'name': 'detection (prop_start)', 'range': [0., .10]},
    'icu_prop': {'name': 'prop ICU among hosp.', 'range': [.15, .20]},
    'compartment_periods.hospital_late': {'name': 'hopital duration', 'range': [17.7, 20.4]},
    'compartment_periods.icu_late': {'name': 'time in ICU', 'range': [9., 13.]},
    'clinical_stratification.props.hospital.multiplier': {'name': 'hosp. prop. multiplier', 'range': [.75, 1.25]},
    'symptomatic_props_multiplier': {'name': 'sympt. prop. multiplier', 'range': [.6, 1.4]},
    'mobility.microdistancing.behaviour.parameters.c': {'name': 'microdist. (inflection)', 'range': [80, 130]},
    'mobility.microdistancing.behaviour.parameters.upper_asymptote': {'name': 'microdist. (final)', 'range': [.25, .6]},
    'mobility.microdistancing.behaviour_adjuster.parameters.c': {'name': 'microdist. wane (inflection)', 'range': [150, 250]},
    'mobility.microdistancing.behaviour_adjuster.parameters.sigma': {'name': 'microdist. wane (final)', 'range': [.5, 1]},
}


def make_posterior_ranges_figure(param_values):
    n_panels = len(param_values['belgium'])
    country_list = OPTI_REGIONS[::-1]
    n_col = 4
    n_row = int(n_panels // n_col)
    if n_col * n_row < n_panels:
        n_row += 1

    fig, axs = plt.subplots(n_row, n_col, figsize=(11, 12))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.4)

    i_col = -1
    i_row = 0
    for param_name in list(param_values['belgium'].keys()):
        i_col += 1
        if i_col >= n_col:
            i_col = 0
            i_row += 1
        h = 0
        for country in country_list:
            h += 1
            # find mean and CI
            values = param_values[country][param_name]
            point_estimate = mean(values)
            low_95, low_50, med, up_50, up_95 = quantile(values, q=(.025, .25, .5, .75, .975))

            axs[i_row, i_col].plot([low_95, up_95], [h, h], linewidth=1, color='black')
            axs[i_row, i_col].plot([low_50, up_50], [h, h], linewidth=3, color='steelblue')
            axs[i_row, i_col].plot([point_estimate], [h], marker='o', color='crimson', markersize=5)

        axs[i_row, i_col].plot([0], [0])
        axs[i_row, i_col].set_ylim((0.5, 6.5))

        axs[i_row, i_col].set_title(param_info[param_name]['name'], fontsize=10.5)
        axs[i_row, i_col].set_xlim(param_info[param_name]['range'])

        # Format x-ticks if requested
        if "xticks" in param_info[param_name]:
            axs[i_row, i_col].set_xticks(param_info[param_name]['xticks'])
            axs[i_row, i_col].set_xticklabels(param_info[param_name]['xlabels'])

        if "multiplier" in param_info[param_name]:
            axs[i_row, i_col].set_xlabel(param_info[param_name]["multiplier"], labelpad=-7.5)

        # Set y-ticks and yticks-labels
        if i_col == 0:
            axs[i_row, i_col].set_yticks([1, 2, 3, 4, 5, 6])
            axs[i_row, i_col].set_yticklabels([c.title().replace("United-Kingdom", "UK") for c in country_list])
        else:
            axs[i_row, i_col].set_yticks([])

    # Leave blank axis for remaining panels
    for i_col_blank in range(i_col + 1, n_col):
        axs[i_row, i_col_blank].axis("off")

    plt.tight_layout()
    plt.savefig("figures/param_posteriors.pdf")


# --------------  Make figure with posterior time-variant detection
def get_all_posterior_detection_percentiles(param_values):
    for country in OPTI_REGIONS:
        print(country)
        country_perc = get_country_posterior_detection_percentiles(param_values[country])
        file_path_ = os.path.join('dumped_files', 'dumped_detection_percentiles_' + country + '.npy')
        with open(file_path_, "wb") as f:
            np.save(f, country_perc)


def get_country_posterior_detection_percentiles(country_param_values):

    calculated_times = list(range(300))[30:]
    store_matrix = np.zeros((len(calculated_times), len(country_param_values['time.start'])))

    for i in range(len(country_param_values['time.start'])):
        if 'case_detection.start_value' in country_param_values:
            sigma = country_param_values['case_detection.start_value'][i]
        else:
            sigma = 0.
        my_func = tanh_based_scaleup(country_param_values['case_detection.maximum_gradient'][i],
                                     country_param_values['case_detection.max_change_time'][i],
                                     sigma,
                                     country_param_values['case_detection.end_value'][i]
                                     )
        detect_vals = [my_func(float(t)) for t in calculated_times]
        store_matrix[:, i] = detect_vals

    perc = np.percentile(store_matrix, [2.5, 25, 50, 75, 97.5], axis=1)
    calculated_times = np.array([calculated_times])
    perc = np.concatenate((calculated_times, perc))

    return perc


def plot_posterior_detection():

    # load percentiles previously dumped
    percentiles = {}

    x_ticks = [61, 122, 183, 245, 306]
    x_ticks_labels = ["Mar 1", "May 1", "Jul 1", "Sep 1", "Nov 1"]

    for country in OPTI_REGIONS:
        file_path_ = os.path.join('dumped_files', 'dumped_detection_percentiles_' + country + '.npy')
        with open(file_path_, "rb") as f:
            percentiles[country] = np.load(f)

    n_panels = len(OPTI_REGIONS)

    n_col = 3
    n_row = int(n_panels // n_col)
    if n_col * n_row < n_panels:
        n_row += 1

    fig, axs = plt.subplots(n_row, n_col, figsize=(13, 7))
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=.5)

    i_col = -1
    i_row = 0
    for country in OPTI_REGIONS:
        i_col += 1
        if i_col >= n_col:
            i_col = 0
            i_row += 1

        times = percentiles[country][0, :]
        median = percentiles[country][3, :]
        low_95 = percentiles[country][1, :]
        up_95 = percentiles[country][5, :]
        low_50 = percentiles[country][2, :]
        up_50 = percentiles[country][4, :]

        axs[i_row, i_col].fill_between(times, low_95, up_95, facecolor="lightsteelblue")
        axs[i_row, i_col].fill_between(times, low_50, up_50, facecolor="cornflowerblue")
        axs[i_row, i_col].plot(times, median, color="navy")

        axs[i_row, i_col].set_ylim((0., 1.))

        c_title = country.title() if country != "united-kingdom" else "United Kingdom"

        axs[i_row, i_col].set_title(c_title)
        axs[i_row, i_col].set_xticks(x_ticks)
        axs[i_row, i_col].set_xticklabels(x_ticks_labels)
        axs[i_row, i_col].set_xlim((45, 300))

        if i_col == 0:
            axs[i_row, i_col].set_ylabel("proportion of symptomatic\ndetected")

    plt.tight_layout()
    plt.savefig("figures/detection_posteriors.pdf")


# --------------  Make figure with fits to data
SEROSURVEYS = {
    'belgium': [
        {'time_window': [90., 96], 'value': .029, 'ci': [.023, .034]},  # 30 mar  5 Apr
        {'time_window': [111, 117], 'value': .06, 'ci': [.051, .071]},  # 20 -26 Apr
        {'time_window': [139, 146], 'value': .069, 'ci': [.059, .080]},  # 18 -25 May
        {'time_window': [160, 165], 'value': .055, 'ci': [.047, .065]},  # 8 - 13 jun
        {'time_window': [181, 185], 'value': .045, 'ci': [.037, .054]},  # 29 Jun -3 Jul
    ],
    'france': [
        {'time_window': [85., 98], 'value': .0271},
    ],
    'italy': [
        {'time_window': [146., 197], 'value': .0250},  # 25 may 15 jul
    ],
    'spain': [
        {'time_window': [118, 132], 'value': .05, 'ci': [0.047, 0.054]},
        {'time_window': [139, 153], 'value': .052, 'ci': [0.049, 0.055]},
        {'time_window': [160, 174], 'value': .052, 'ci': [0.049, 0.055]},
    ],
    'sweden': [
        {'time_window': [122, 152], 'value': .108, 'ci': [0.079, 0.137]},
    ],
    'united-kingdom': [
        {'time_window': [118, 124], 'value': .0710},
        {'time_window': [172, 195], 'value': .06, 'ci': [.058, .061]},
    ]
}


def make_calibration_fits_figure(calibration_outputs, seroprevalence=False):

    if not seroprevalence:
        countries_per_row = 2
        n_target_outputs = 3
        show_title = True
        lab_fontsize = 13
        target_outputs = {}
        for country in OPTI_REGIONS:
            targets = get_targets(country)
            target_outputs[country] = ["notifications", "infection_deaths"]
            hospital_target = [t for t in list(targets.keys()) if 'hospital' in t or 'icu' in t][0]
            target_outputs[country].append(hospital_target)
    else:
        show_title = False
        lab_fontsize = 15
        countries_per_row = 3
        n_target_outputs = 1
        target_outputs = {}
        for country in OPTI_REGIONS:
            target_outputs[country] = ["proportion_seropositive"]

    n_countries_per_col = int(6 / countries_per_row)

    width = n_target_outputs * countries_per_row * 5
    height = n_countries_per_col * 5
    fig = plt.figure(constrained_layout=True, figsize=(width, height))  # (w, h)
    widths = [5] * (countries_per_row * n_target_outputs)
    heights = [1, 6] * n_countries_per_col
    spec = fig.add_gridspec(ncols=countries_per_row * n_target_outputs, nrows=len(heights), width_ratios=widths,
                            height_ratios=heights)

    text_size = 23
    i_row = 1
    i_col = 0
    for country in OPTI_REGIONS:
        # write country name
        ax = fig.add_subplot(spec[i_row-1, i_col: i_col + n_target_outputs])
        ax.text(0.5, 0.2, COUNTRY_TITLES[country], fontsize=text_size, horizontalalignment='center',
                verticalalignment='center')
        ax.axis("off")
        for output in target_outputs[country]:
            show_ylab = (i_col == 0) and seroprevalence

            ax = fig.add_subplot(spec[i_row, i_col])
            country_targets = get_targets(country)
            targets = {k: v for k, v in country_targets.items() if v["output_key"] == output}
            plot_timeseries_with_uncertainty(
                None, calibration_outputs[country]["uncertainty_df"], output, [0], targets,
                False, 40, 276, ax, n_xticks=None, title_font_size=16, label_font_size=lab_fontsize,
                requested_x_ticks=[61, 122, 183, 245], show_title=show_title, show_ylab=show_ylab,
                add_targets=(not seroprevalence)
            )
            for s in SEROSURVEYS[country]:
                if "ci" in s:
                    rect = patches.Rectangle((s['time_window'][0], s['ci'][0]),
                                             s['time_window'][1] - s['time_window'][0],
                                             s['ci'][1] - s['ci'][0],
                                             linewidth=0, facecolor='gold', alpha=.4)
                    rect.set_zorder(1)
                    ax.add_patch(rect)
                ax.plot(s['time_window'], [s['value']] * 2, linewidth=1.5, color='black')



            i_col += 1
            if i_col == countries_per_row * n_target_outputs:
                i_col = 0
                i_row += 2

    # Add vertical separator line
    if not seroprevalence:
        line = plt.Line2D((.5, .5), (-.1, 1.9), color="grey", linewidth=1)
        fig.add_artist(line)

    filename = "figures/model_fits"
    if seroprevalence:
        filename += "_seroprevalence"
    plt.savefig(filename + ".pdf")
