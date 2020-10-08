"""
Calibration plots
"""
import os
import copy
import logging
from math import log
from typing import List, Tuple, Callable
from random import choices
import streamlit as st

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot
from scipy import stats
import plotly.express as px

from autumn import db
from autumn.calibration.utils import calculate_prior, raise_error_unsupported_prior
from autumn.plots.plotter import Plotter, COLOR_THEME
from autumn import inputs
from apps.covid_19.model.preprocess.testing import find_cdr_function_from_test_data
from autumn.tool_kit.scenarios import get_model_times_from_inputs

logger = logging.getLogger(__name__)

PLOT_TEXT_DICT = {
    "contact_rate": "infection risk per contact",
    "compartment_periods_calculated.exposed.total_period": "incubation period",
    "compartment_periods_calculated.active.total_period": "duration active",
    "hospital_props_multiplier": "hospital risk multiplier",
    "compartment_periods.icu_early": "pre-ICU period",
    "icu_prop": "ICU proportion",
    "testing_to_detection.assumed_cdr_parameter": "CDR at base testing rate",
    "microdistancing.parameters.max_effect": "max effect microdistancing",
    "infection_deaths": "mortality",
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
}


def get_plot_text_dict(param_string, capitalise_first_letter=False):
    text = PLOT_TEXT_DICT[param_string] if param_string in PLOT_TEXT_DICT else param_string
    if capitalise_first_letter:
        text = text[0].upper() + text[1:]
    return text


# def find_max_burn_in(mcmc_params):
#     chain_length = 0
#     for i_chain in range(len(mcmc_params)):
#         parameters = mcmc_params[i_chain]["name"].unique().tolist()
#         chain_length = max(int(len(mcmc_params[0]) / len(parameters)), chain_length)
#     return chain_length


def find_min_chain_length_from_mcmc_tables(mcmc_tables):
    chain_lengths = [mcmc_tables[i_chain]["run"].tolist()[-1] for i_chain in range(len(mcmc_tables))]
    return min(chain_lengths)


"""
Parameter diagnostics
"""


def plot_acceptance_ratio(
        plotter: Plotter,
        mcmc_tables: List[pd.DataFrame],
        burn_in: int,
        label_font_size=6,
        dpi_request=300
):
    """
    Plot the progressive acceptance ratio over iterations.
    """
    fig, axis, _, _, _, _ = plotter.get_figure()
    mcmc_df = db.process.append_tables(mcmc_tables)
    chains = mcmc_df["chain"].unique().tolist()
    for chain in chains:
        df = mcmc_df[mcmc_df["chain"] == chain]

        # Collate acceptances and ratio
        count, total, ratios = 0, 0, []
        for accept in df["accept"]:
            total += 1
            if accept:
                count += 1
            ratios.append(count / total)

        # Plot
        axis.plot(ratios, alpha=0.8, linewidth=0.7)

        # Add vertical line for burn-in point
        if burn_in > 0:
            axis.axvline(x=burn_in, color=COLOR_THEME[1], linestyle="dotted")

    axis.set_ylabel("Acceptance ratio", fontsize=label_font_size)
    axis.set_ylim(bottom=0.)
    axis.set_xlabel("Iterations", fontsize=label_font_size)
    pyplot.setp(axis.get_yticklabels(), fontsize=label_font_size)
    pyplot.setp(axis.get_xticklabels(), fontsize=label_font_size)
    plotter.save_figure(
        fig, filename=f"acceptance_ratio", title_text=f"Acceptance Ratio", dpi_request=dpi_request
    )


def plot_prior(i: int, prior_dict: dict, path: str):
    if prior_dict["distribution"] == "lognormal":
        logger.error("Cannot plot prior distributions for lognormal.")
        return

    fig, ax = pyplot.subplots()
    x_range = workout_plot_x_range(prior_dict)
    x_values = np.linspace(x_range[0], x_range[1], num=1000)
    y_values = [calculate_prior(prior_dict, x, log=False) for x in x_values]
    zeros = [0.0 for i in x_values]
    pyplot.fill_between(x_values, y_values, zeros, color="cornflowerblue")

    if "distri_mean" in prior_dict:
        pyplot.axvline(
            x=prior_dict["distri_mean"], ymin=0, ymax=100 * max(y_values), linewidth=1, color="red",
        )
    if "distri_ci" in prior_dict:
        pyplot.axvline(
            x=prior_dict["distri_ci"][0],
            ymin=0,
            ymax=100 * max(y_values),
            linewidth=0.7,
            color="red",
        )
        pyplot.axvline(
            x=prior_dict["distri_ci"][1],
            ymin=0,
            ymax=100 * max(y_values),
            linewidth=0.7,
            color="red",
        )

    pyplot.xlabel(prior_dict["param_name"])
    pyplot.ylabel("prior PDF")

    # place a text box in upper left corner to indicate the prior details
    props = dict(boxstyle="round", facecolor="dimgray", alpha=0.5)
    textstr = (
        prior_dict["distribution"]
        + "\n("
        + str(round(float(prior_dict["distri_params"][0]), 3))
        + ", "
        + str(round(float(prior_dict["distri_params"][1]), 3))
        + ")"
    )
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    pyplot.tight_layout()
    filename = os.path.join(path, prior_dict["param_name"] + ".png")
    pyplot.savefig(filename)


def workout_plot_x_range(prior_dict):
    if prior_dict["distribution"] == "uniform":
        x_range = prior_dict["distri_params"]
    elif prior_dict["distribution"] == "beta":
        a = prior_dict["distri_params"][0]
        b = prior_dict["distri_params"][1]
        x_range = stats.beta.ppf([0.005, 0.995], a, b)
    elif prior_dict["distribution"] == "gamma":
        shape = prior_dict["distri_params"][0]
        scale = prior_dict["distri_params"][1]
        x_range = stats.gamma.ppf([0.005, 0.995], shape, 0.0, scale)
    elif prior_dict["distribution"] == "trunc_normal":
        lower = prior_dict["trunc_range"][0]
        upper = prior_dict["trunc_range"][1]
        x_range = [lower - 0.05 * (upper - lower), upper + 0.05 * (upper - lower)]

        mean = prior_dict["distri_params"][0]
        sd = prior_dict["distri_params"][1]
        x_range = stats.norm.ppf([0.005, 0.995], mean, sd)

        if lower != -np.inf:
            x_range[0] = lower - 0.05 * (x_range[1] - lower)
        if upper != np.inf:
            x_range[1] = upper + 0.05 * (upper - x_range[0])

    else:
        raise_error_unsupported_prior(prior_dict["distribution"])

    return x_range


def plot_mcmc_parameter_trace(
        plotter: Plotter,
        mcmc_params: List[pd.DataFrame],
        burn_in: int,
        param_name: str
):
    """
    Plot the prameter traces for each MCMC run.
    """
    fig, axis, _, _, _, _ = plotter.get_figure()
    for idx, table_df in enumerate(mcmc_params):
        param_mask = table_df["name"] == param_name
        param_df = table_df[param_mask]
        axis.plot(param_df["run"], param_df["value"], alpha=0.8, linewidth=0.7)
        if burn_in > 0:
            axis.axvline(x=burn_in, color=COLOR_THEME[1], linestyle="dotted")

    axis.set_ylabel(param_name)
    axis.set_xlabel("MCMC iterations")
    plotter.save_figure(fig, filename=f"{param_name}-traces", title_text=f"{param_name}-traces")


def plot_multiple_param_traces(
        plotter: Plotter,
        mcmc_params: List[pd.DataFrame],
        burn_in: int,
        title_font_size: int,
        label_font_size: int,
        capitalise_first_letter: bool,
        dpi_request: int
):

    # Except not the dispersion parameters - only the epidemiological ones
    parameters = \
        [param for param in mcmc_params[0].loc[:, "name"].unique().tolist() if
         "dispersion_param" not in param]

    fig, axes, _, n_rows, n_cols, indices = \
        plotter.get_figure(len(parameters), share_xaxis=True, share_yaxis=False)

    for i in range(n_rows * n_cols):
        axis = axes[indices[i][0], indices[i][1]]

        if i < len(parameters):
            param_name = parameters[i]

            vals_df = None
            for table_df in mcmc_params:
                param_mask = table_df["name"] == param_name
                table_vals = table_df[param_mask].value
                if vals_df is not None:
                    vals_df = vals_df.append(table_vals)
                else:
                    vals_df = table_vals

            for idx, table_df in enumerate(mcmc_params):
                param_mask = table_df["name"] == param_name
                param_df = table_df[param_mask]
                axis.plot(param_df["run"], param_df["value"], alpha=0.8, linewidth=0.7)

            axis.set_title(
                get_plot_text_dict(
                    param_name, capitalise_first_letter=capitalise_first_letter
                ), fontsize=title_font_size
            )

            if indices[i][0] == n_rows - 1:
                x_label = "Iterations" if capitalise_first_letter else "iterations"
                axis.set_xlabel(x_label, fontsize=label_font_size)
            pyplot.setp(axis.get_yticklabels(), fontsize=label_font_size)
            pyplot.setp(axis.get_xticklabels(), fontsize=label_font_size)

            if burn_in > 0:
                axis.axvline(x=burn_in, color=COLOR_THEME[1], linestyle="dotted")

        else:
            axis.axis("off")

    fig.tight_layout()
    plotter.save_figure(fig, filename=f"all_posteriors", dpi_request=dpi_request)


def plot_loglikelihood_trace(plotter: Plotter, mcmc_tables: List[pd.DataFrame], burn_in=0):
    """
    Plot the loglikelihood traces for each MCMC run.
    """
    fig, axis, _, _, _, _ = plotter.get_figure()

    for idx, table_df in enumerate(mcmc_tables):
        accept_mask = table_df["accept"] == 1
        table_df[accept_mask].loglikelihood.plot.line(ax=axis, alpha=0.8, linewidth=0.7)

    axis.set_ylabel("Loglikelihood")
    axis.set_xlabel("MCMC iterations")

    if burn_in:
        axis.axvline(x=burn_in, color=COLOR_THEME[1], linestyle="dotted")
        y_min = min(table_df.loglikelihood[burn_in:])
        y_max = max(table_df.loglikelihood[burn_in:])
        axis.set_ylim((y_min - 0.2 * (y_max - y_min), y_max + 0.2 * (y_max - y_min)))

    plotter.save_figure(fig, filename="loglikelihood-traces", title_text="loglikelihood-traces")


def plot_burn_in(plotter: Plotter, num_iters: int, burn_in: int):
    """
    Plot the trade off been num iters and burn-in for MCMC runs.
    """
    fig, axis, _, _, _, _ = plotter.get_figure()

    def floor(n):
        val = num_iters - n
        return val if val > 0 else 0

    values = [floor(i) for i in range(num_iters)]

    fig, axis, _, _, _, _ = plotter.get_figure()

    axis.plot(values, color=COLOR_THEME[0])
    axis.set_ylabel("Number iters after burn-in")
    axis.set_xlabel("Burn-in size")
    axis.set_ylim(bottom=-5, top=num_iters)
    axis.set_xlim(left=0, right=num_iters)
    axis.axvline(x=burn_in, color=COLOR_THEME[1], linestyle="dotted")
    axis.axhline(y=num_iters - burn_in, color=COLOR_THEME[1], linestyle="dotted")
    plotter.save_figure(fig, filename="burn-in", title_text="burn-in")


def get_posterior(mcmc_params, param_name, burn_in=0):
    vals_df = None
    for table_df in mcmc_params:
        param_mask = (table_df["name"] == param_name) & (table_df["run"] > burn_in)
        table_vals = table_df[param_mask].value
        if vals_df is not None:
            vals_df = vals_df.append(table_vals)
        else:
            vals_df = table_vals
    return vals_df


def plot_posterior(
        plotter: Plotter,
        mcmc_params: List[pd.DataFrame],
        burn_in: int,
        param_name: str,
        num_bins: int
):
    """
    Plots the posterior distribution of a given parameter in a histogram.
    """
    vals_df = get_posterior(mcmc_params, param_name, burn_in)
    fig, axis, _, _, _, _ = plotter.get_figure()
    vals_df.hist(bins=num_bins, ax=axis)
    plotter.save_figure(
        fig, filename=f"{param_name}-posterior", title_text=f"{param_name} posterior"
    )


def plot_multiple_posteriors(
        plotter: Plotter,
        mcmc_params: List[pd.DataFrame],
        burn_in: int,
        num_bins: int,
        title_font_size: int,
        label_font_size: int,
        capitalise_first_letter: bool,
        dpi_request: int
):
    """
    Plots the posterior distribution of a given parameter in a histogram.
    """

    # Except not the dispersion parameters - only the epidemiological ones
    parameters = \
        [param for param in mcmc_params[0].loc[:, "name"].unique().tolist() if
         "dispersion_param" not in param]
    fig, axes, _, n_rows, n_cols, indices = plotter.get_figure(len(parameters))

    for i in range(n_rows * n_cols):
        axis = axes[indices[i][0], indices[i][1]]

        if i < len(parameters):
            param_name = parameters[i]
            vals_df = get_posterior(mcmc_params, param_name, burn_in)

            # Plot histograms
            vals_df.hist(bins=num_bins, ax=axis)
            axis.set_title(
                get_plot_text_dict(
                    param_name, capitalise_first_letter=capitalise_first_letter
                ), fontsize=title_font_size
            )
            pyplot.setp(axis.get_yticklabels(), fontsize=label_font_size)
            pyplot.setp(axis.get_xticklabels(), fontsize=label_font_size)

        else:
            axis.axis("off")

    fig.tight_layout()
    plotter.save_figure(fig, filename=f"all_posteriors", dpi_request=dpi_request)


def plot_param_vs_loglike(mcmc_tables, mcmc_params, param_name, burn_in, axis):
    for mcmc_df, param_df in zip(mcmc_tables, mcmc_params):
        df = param_df.merge(mcmc_df, on=["run", "chain"])
        mask = (df["accept"] == 1) & (df["name"] == param_name) & (df["run"] > burn_in)
        df = df[mask]
        param_values = df["value"]
        loglikelihood_values = [-log(-v) for v in df["loglikelihood"]]
        axis.plot(param_values, loglikelihood_values, ".")


def plot_parallel_coordinates(
        plotter: Plotter,
        mcmc_tables: List[pd.DataFrame],
        mcmc_params: List[pd.DataFrame],
):
    parameters = \
        [param for param in mcmc_params[0].loc[:, "name"].unique().tolist() if
         "dispersion_param" not in param]

    target_n_lines = 500.
    n_samples = int(target_n_lines / len(mcmc_tables))
    combined_mcmc_df = merge_and_pivot_mcmc_parameters_loglike(
        mcmc_tables,
        mcmc_params,
        parameters,
        n_samples_per_chain=n_samples
    )
    w = len(parameters) * 200
    h = 800
    labels = {}
    for param in parameters:
        labels[param] = PLOT_TEXT_DICT[param] if param in PLOT_TEXT_DICT else param
    figure = px.parallel_coordinates(
        combined_mcmc_df,
        color='fitness',
        dimensions=parameters,
        labels=labels,
        color_continuous_scale=px.colors.diverging.Tealrose,
        height=h,
        width=w
    )
    figure.show()


def merge_and_pivot_mcmc_parameters_loglike(mcmc_tables, mcmc_params, parameters, n_samples_per_chain=None):

    combined_mcmc_df = None
    for mcmc_df, param_df in zip(mcmc_tables, mcmc_params):
        mask = mcmc_df["accept"] == 1
        mcmc_df = mcmc_df[mask]
        n_iter = len(mcmc_df.index) if n_samples_per_chain is None else min(n_samples_per_chain, len(mcmc_df.index))
        mcmc_df = mcmc_df.iloc[-n_iter:]
        for param in parameters:
            param_vals = []
            for c, r in zip(mcmc_df["chain"], mcmc_df["run"]):
                mask1 = param_df["chain"] == c
                mask2 = param_df['name'] == param
                mask3 = param_df["run"] == r
                param_vals.append(
                   param_df[mask1][mask2][mask3]["value"].iloc[0]
                )
            mcmc_df[param] = param_vals

        if combined_mcmc_df is None:
            combined_mcmc_df = mcmc_df
        else:
            combined_mcmc_df = combined_mcmc_df.append(mcmc_df)

    combined_mcmc_df['fitness'] = [-log(-v) for v in combined_mcmc_df["loglikelihood"]]
    return combined_mcmc_df


def plot_loglikelihood_surface(
        plotter: Plotter,
        mcmc_tables: List[pd.DataFrame],
        mcmc_params: List[pd.DataFrame],
        param_1,
        param_2,
):
    combined_mcmc_df = merge_and_pivot_mcmc_parameters_loglike(
        mcmc_tables,
        mcmc_params,
        [param_1, param_2]
    )

    fig = px.scatter_3d(combined_mcmc_df, x=param_1, y=param_2, z='fitness',
              color='chain')
    fig.show()



def plot_single_param_loglike(
        plotter: Plotter,
        mcmc_tables: List[pd.DataFrame],
        mcmc_params: List[pd.DataFrame],
        burn_in: int,
        param_name: str,
):
    """
    Plots the loglikelihood against parameter values.
    """
    fig, axis, _, _, _, _ = plotter.get_figure()
    plot_param_vs_loglike(mcmc_tables, mcmc_params, param_name, burn_in, axis)
    axis.set_xlabel(param_name)
    axis.set_ylabel("-log(-loglikelihood)")
    plotter.save_figure(
        fig,
        filename=f"likelihood-against-{param_name}",
        title_text=f"likelihood against {param_name}",
    )


def plot_param_vs_param_by_chain(
        plotter: Plotter,
        mcmc_params: List[pd.DataFrame],
        parameters: list,
        label_font_size: int,
        label_chars: int,
        dpi_request: int
):
    """
    Plot the parameter traces for each MCMC chain with separate colouring.
    """

    fig, axes, _, _, _, _ = plotter.get_figure(n_panels=len(parameters) ** 2)

    for chain in range(len(mcmc_params)):
        for x_idx, x_param_name in enumerate(parameters):
            x_param_mask = mcmc_params[chain]["name"] == x_param_name
            for y_idx, y_param_name in enumerate(parameters):
                y_param_mask = mcmc_params[chain]["name"] == y_param_name

                # Get axis and turn off ticks
                axis = axes[x_idx, y_idx]
                axis.xaxis.set_ticks([])
                axis.yaxis.set_ticks([])

                # Plot
                if x_idx > y_idx:
                    axis.scatter(
                        mcmc_params[chain][x_param_mask]["value"].to_list(),
                        mcmc_params[chain][y_param_mask]["value"].to_list(),
                        alpha=0.5,
                        s=0.1
                    )
                elif x_idx == y_idx:
                    axis.hist(
                        mcmc_params[chain][x_param_mask]["value"].to_list()
                    )
                else:
                    axis.axis("off")

                # Set labels
                if y_idx == 0:
                    axis.set_ylabel(x_param_name[:label_chars], rotation=0, fontsize=label_font_size)
                if x_idx == len(parameters) - 1:
                    axis.set_xlabel(y_param_name[:label_chars], fontsize=label_font_size)

    # Save
    plotter.save_figure(fig, filename="parameter_correlation_matrix", dpi_request=dpi_request)


def plot_param_vs_param(
        plotter: Plotter,
        mcmc_params: List[pd.DataFrame],
        parameters: list,
        burn_in: int,
        style: str,
        bins: int,
        label_font_size: int,
        label_chars: int,
        dpi_request: int
):
    """
    Plot the parameter correlation matrices for each parameter combination.
    """

    # Prelims
    fig, axes, _, _, _, _ = plotter.get_figure(n_panels=len(parameters) ** 2)
    x_data, y_data = {}, {}

    # Get x and y data separately and collate up over the chains
    for x_idx, x_param_name in enumerate(parameters):
        x_data[x_param_name] = []
        for chain in range(len(mcmc_params)):
            x_param_mask = \
                (mcmc_params[chain]["name"] == x_param_name) & (mcmc_params[chain]["run"] > burn_in)
            x_data[x_param_name] += \
                mcmc_params[chain][x_param_mask]["value"].to_list()
    for y_idx, y_param_name in enumerate(parameters):
        y_data[y_param_name] = []
        for chain in range(len(mcmc_params)):
            y_param_mask = \
                (mcmc_params[chain]["name"] == y_param_name) & (mcmc_params[chain]["run"] > burn_in)
            y_data[y_param_name] += \
                mcmc_params[chain][y_param_mask]["value"].to_list()

    # Loop over parameter combinations
    for x_idx, x_param_name in enumerate(parameters):
        for y_idx, y_param_name in enumerate(parameters):
            axis = axes[x_idx, y_idx]
            axis.xaxis.set_ticks([])
            axis.yaxis.set_ticks([])

            # Plot
            if x_idx > y_idx:
                if style == "Scatter":
                    axis.scatter(
                        x_data[x_param_name], y_data[y_param_name], alpha=0.5, s=0.1, color="k"
                    )
                elif style == "KDE":
                    sns.kdeplot(x_data[x_param_name], y_data[y_param_name], ax=axis, shade=True, levels=5, lw=1.)
                else:
                    axis.hist2d(
                        x_data[x_param_name], y_data[y_param_name], bins=bins
                    )
            elif x_idx == y_idx:
                axis.hist(
                    x_data[x_param_name], color=[0.2, 0.2, 0.6] if style == "Shade" else "k", bins=bins
                )
            else:
                axis.axis("off")

            # Axis labels
            if y_idx == 0:
                axis.set_ylabel(x_param_name[:label_chars], rotation=0, fontsize=label_font_size, labelpad=10)
            if x_idx == len(parameters) - 1:
                axis.set_xlabel(y_param_name[:label_chars], fontsize=label_font_size, labelpad=3)

    # Save
    plotter.save_figure(fig, filename="parameter_correlation_matrix", dpi_request=dpi_request)


def plot_all_params_vs_loglike(
        plotter: Plotter,
        mcmc_tables: List[pd.DataFrame],
        mcmc_params: List[pd.DataFrame],
        burn_in: int,
        title_font_size: int,
        label_font_size: int,
        capitalise_first_letter: bool,
        dpi_request: int,
):

    # Except not the dispersion parameters - only the epidemiological ones
    parameters = \
        [param for param in mcmc_params[0].loc[:, "name"].unique().tolist() if
         "dispersion_param" not in param]

    fig, axes, _, n_rows, n_cols, indices = \
        plotter.get_figure(len(parameters), share_xaxis=False, share_yaxis=True)

    for i in range(n_rows * n_cols):
        axis = axes[indices[i][0], indices[i][1]]

        if i < len(parameters):
            param_name = parameters[i]
            plot_param_vs_loglike(mcmc_tables, mcmc_params, param_name, burn_in, axis)
            axis.set_title(
                get_plot_text_dict(
                    param_name, capitalise_first_letter=capitalise_first_letter
                ), fontsize=title_font_size
            )
            if indices[i][0] == n_rows - 1:
                x_label = "Iterations" if capitalise_first_letter else "iterations"
                axis.set_xlabel(x_label, fontsize=label_font_size)
            pyplot.setp(axis.get_yticklabels(), fontsize=label_font_size)
            pyplot.setp(axis.get_xticklabels(), fontsize=label_font_size)

        else:
            axis.axis("off")

    fig.tight_layout()
    plotter.save_figure(fig, filename=f"all_posteriors", dpi_request=dpi_request)


def sample_outputs_for_calibration_fit(
        output_name: str,
        mcmc_tables: List[pd.DataFrame],
        do_tables: List[pd.DataFrame],
        burn_in: int,
):
    assert len(mcmc_tables) == len(do_tables)
    mcmc_df = db.process.append_tables(mcmc_tables)
    do_df = db.process.append_tables(do_tables)

    # Determine max chain length, throw away first half of that if no burn-in request
    discard_point = mcmc_df["run"].max() // 2 if \
        burn_in == 0 else \
        burn_in
    mcmc_df = mcmc_df[mcmc_df["run"] >= discard_point]

    # Choose runs with probability proportional to their weights.
    weights = mcmc_df["weight"].tolist()
    run_choices = list(zip(mcmc_df["chain"].tolist(), mcmc_df["run"].tolist()))
    num_chosen = 20 * len(mcmc_tables)
    chosen_runs = choices(run_choices, weights=weights, k=num_chosen)

    outputs = []
    for chain, run in chosen_runs:
        mask = (do_df["run"] == run) & (do_df["chain"] == chain)
        times = do_df[mask]["times"]
        values = do_df[mask][output_name]
        outputs.append([times, values])

    # Find MLE run
    mle_df = db.process.find_mle_run(mcmc_df)
    run = mle_df["run"].iloc[0]
    chain = mle_df["chain"].iloc[0]
    # Automatically use the MLE run as the last chosen run
    mask = (do_df["run"] == run) & (do_df["chain"] == chain)
    times = do_df[mask]["times"]
    values = do_df[mask][output_name]
    outputs.append([times, values])

    return outputs


def plot_calibration(axis, output, outputs, targets, is_logscale):
    # Track the maximum value being plotted
    max_value = 0.0

    for times, values in outputs:
        axis.plot(times, values)
        max_value = max(values) if max(values) > max_value else max_value

    # Mark the MLE run with a dotted line
    axis.plot(outputs[-1][0], outputs[-1][1], linestyle=(0, (1, 3)), color="black", linewidth=3)

    # Add plot targets
    output_config = {"output_key": output, "values": [], "times": []}
    for t in targets.values():
        if t["output_key"] == output:
            output_config = t

    values = output_config["values"]
    times = output_config["times"]
    _plot_targets_to_axis(axis, values, times, on_uncertainty_plot=False)

    # Find upper limit for y-axis
    if values:
        upper_buffer = 2.0
        max_target = max(values)
        upper_ylim = (
            max_value if max_value < max_target * upper_buffer else max_target * upper_buffer
        )
    else:
        upper_ylim = max_value

    if is_logscale:
        axis.set_yscale("log")
    else:
        axis.set_ylim([0.0, upper_ylim])

    return axis


def plot_calibration_fit(
    plotter: Plotter, output_name: str, outputs: list, targets, is_logscale=False,
):
    fig, axis, _, _, _, _ = plotter.get_figure()
    plot_calibration(
        axis, output_name, outputs, targets, is_logscale
    )
    if is_logscale:
        filename = f"calibration-fit-{output_name}-logscale"
        title_text = f"Calibration fit for {output_name} (logscale)"
    else:
        filename = f"calibration-fit-{output_name}"
        title_text = f"Calibration fit for {output_name}"
    plotter.save_figure(fig, filename=filename, title_text=title_text)


def plot_multi_fit(
        plotter: Plotter, output_names: list, outputs: dict, targets, is_logscale=False,
        title_font_size=8, label_font_size=8, dpi_request=300, capitalise_first_letter=False,
):

    fig, axes, _, n_rows, n_cols, indices = \
        plotter.get_figure(len(output_names), share_xaxis=True)

    for i_output, output in enumerate(output_names):
        axis = plot_calibration(
            axes[indices[i_output][0], indices[i_output][1]], output, outputs[output], targets, is_logscale
        )
        axis.set_title(
            get_plot_text_dict(
                output, capitalise_first_letter=capitalise_first_letter
            ), fontsize=title_font_size
        )
        pyplot.setp(axis.get_yticklabels(), fontsize=label_font_size)
        pyplot.setp(axis.get_xticklabels(), fontsize=label_font_size)
        filename = f"calibration-fit-{output}"

    fig.tight_layout()
    plotter.save_figure(fig, filename=filename, dpi_request=dpi_request)


def plot_cdr_curves(
        plotter: Plotter,
        mcmc_tables: List[pd.DataFrame],
        mcmc_params,
        burn_in: int,
):
    fig, axis, _, _, _, _ = plotter.get_figure()

    # Manually input some parameters - need to change this
    assumed_tests_parameter = 1.0e-4
    iso3 = "PHL"
    testing_year = 2020
    times = get_model_times_from_inputs(round(40.), 365., 1.)
    agegroup_strata = [str(s) for s in range(0, 80, 5)]

    # Collate parameters
    testing_to_detection_values = []
    for i_chain in range(len(mcmc_params)):
        param_mask = \
            mcmc_params[i_chain]["name"] == "testing_to_detection.assumed_cdr_parameter"
        testing_to_detection_values += \
            mcmc_params[i_chain]["value"][param_mask].tolist()

    # Get CDR function
    testing_pops = inputs.get_population_by_agegroup(
        agegroup_strata, iso3, None, year=testing_year
    )
    for assumed_cdr_parameter in testing_to_detection_values:
        detected_proportion = find_cdr_function_from_test_data(
            assumed_tests_parameter,
            assumed_cdr_parameter,
            iso3,
            testing_pops,
        )

        # Plot outputs
        axis.plot(times, [detected_proportion(i_time) for i_time in times], alpha=0.3, color="k", linewidth=2.)
        axis.set_xlabel("time")
        axis.set_ylabel("case detection rate (proportion)")
        axis.set_ylim([0., 1.])
    plotter.save_figure(fig, filename=f"cdr_curves")


def _overwrite_non_accepted_mcmc_runs(mcmc_tables: List[pd.DataFrame], column_name: str):
    """
    Count non-accepted rows in a MCMC trace as the last accepted row.
    Modifies mcmc_tables in-place.
    """
    for table_df in mcmc_tables:
        prev_val = None
        for idx in range(len(table_df)):
            if table_df.at[idx, "accept"] == 1:
                prev_val = table_df.at[idx, column_name]
            else:
                table_df.at[idx, column_name] = prev_val


def _plot_targets_to_axis(axis, values: List[float], times: List[int], on_uncertainty_plot=False):
    """
    Plot output value calibration targets as points on the axis.
    """
    assert len(times) == len(values), "Targets have inconsistent length"
    # Plot a single point estimate
    if on_uncertainty_plot:
        axis.scatter(times, values, marker="o", color="black", s=10, zorder=999)
    else:
        axis.scatter(times, values, marker="o", color="red", s=30, zorder=999)
        axis.scatter(times, values, marker="o", color="white", s=10, zorder=999)
