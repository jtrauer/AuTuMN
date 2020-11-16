"""
Plotting projection uncertainty.
"""
from typing import List
import logging
import datetime

import pandas as pd
from math import ceil

from autumn.plots.plotter import Plotter
from autumn.plots.utils import _plot_targets_to_axis
from matplotlib import pyplot
import matplotlib.ticker as mtick
from autumn.plots.utils import change_xaxis_to_date
from numpy import mean
from autumn.plots.utils import get_plot_text_dict, _apply_transparency, COLORS, ALPHAS, REF_DATE

logger = logging.getLogger(__name__)


def plot_timeseries_with_uncertainty(
    plotter: Plotter,
    uncertainty_df: pd.DataFrame,
    output_name: str,
    scenario_idxs: List[int],
    targets: dict,
    is_logscale=False,
    x_low=0.0,
    x_up=1e6,
    axis=None,
    n_xticks=None,
    ref_date=REF_DATE,
    add_targets=True,
    overlay_uncertainty=True,
    title_font_size=12,
    label_font_size=10,
    dpi_request=300,
    capitalise_first_letter=False,
    legend=False,
    requested_x_ticks=None,
    show_title=True,
    show_ylab=True,
):
    """
    Plots the uncertainty timeseries for one or more scenarios.
    Also plots any calibration targets that are provided.
    """

    single_panel = axis is None
    if single_panel:
        fig, axis, _, _, _, _ = plotter.get_figure()

    n_scenarios_to_plot = min([len(scenario_idxs), len(COLORS)])
    colors = _apply_transparency(COLORS[:n_scenarios_to_plot], ALPHAS[:n_scenarios_to_plot])

    # Plot each scenario on a single axis.
    for scenario_idx in scenario_idxs[:n_scenarios_to_plot]:
        if scenario_idx < len(colors):
            scenario_colors = colors[scenario_idx]
        else:
            scenario_colors = colors[-1]
        _plot_uncertainty(
            axis,
            uncertainty_df,
            output_name,
            scenario_idx,
            x_up,
            x_low,
            scenario_colors,
            overlay_uncertainty=overlay_uncertainty,
            start_quantile=0,
        )

    # Add plot targets
    if add_targets:
        values, times = _get_target_values(targets, output_name)
        trunc_values = [v for (v, t) in zip(values, times) if x_low <= t <= x_up]
        trunc_times = [t for (v, t) in zip(values, times) if x_low <= t <= x_up]
        _plot_targets_to_axis(axis, trunc_values, trunc_times, on_uncertainty_plot=True)

    # Sort out x-axis
    change_xaxis_to_date(axis, ref_date, rotation=0)
    axis.tick_params(axis="x", labelsize=label_font_size)
    axis.tick_params(axis="y", labelsize=label_font_size)

    if output_name == "proportion_seropositive":
        axis.yaxis.set_major_formatter(mtick.PercentFormatter(1, symbol=""))
    if show_title:
        axis.set_title(get_plot_text_dict(output_name), fontsize=title_font_size)

    if requested_x_ticks is not None:
        pyplot.xticks(requested_x_ticks)
    elif n_xticks is not None:
        pyplot.locator_params(axis="x", nbins=n_xticks)

    if is_logscale:
        axis.set_yscale("log")
    elif not (output_name.startswith("rel_diff") or output_name.startswith("abs_diff")):
        axis.set_ylim(ymin=0)

    if show_ylab:
        axis.set_ylabel(get_plot_text_dict(output_name), fontsize=label_font_size)

    if legend:
        pyplot.legend(labels=scenario_idxs)

    if single_panel:
        idx_str = "-".join(map(str, scenario_idxs))
        filename = f"uncertainty-{output_name}-{idx_str}"
        plotter.save_figure(fig, filename=filename, dpi_request=dpi_request)


def _plot_uncertainty(
    axis,
    uncertainty_df: pd.DataFrame,
    output_name: str,
    scenario_idx: int,
    x_up: float,
    x_low: float,
    colors: List[str],
    overlay_uncertainty=True,
    start_quantile=0,
):
    """Plots the uncertainty values in the provided dataframe to an axis"""
    mask = (
        (uncertainty_df["type"] == output_name)
        & (uncertainty_df["scenario"] == scenario_idx)
        & (uncertainty_df["time"] <= x_up)
        & (uncertainty_df["time"] >= x_low)
    )

    df = uncertainty_df[mask]
    times = df.time.unique()
    quantiles = {}
    quantile_vals = df["quantile"].unique().tolist()
    for q in quantile_vals:
        mask = df["quantile"] == q
        quantiles[q] = df[mask]["value"].tolist()

    #import streamlit as st

    #st.write(times)
    q_keys = sorted([float(k) for k in quantiles.keys()])
    num_quantiles = len(q_keys)
    half_length = num_quantiles // 2
    # if overlay_uncertainty:
    #     for i in range(start_quantile, half_length):
    #         color = colors[i]
    #         start_key = q_keys[i]
    #         end_key = q_keys[-(i + 1)]
    #         axis.fill_between(times, quantiles[start_key], quantiles[end_key], facecolor=color)

    if overlay_uncertainty:
        for i in range(start_quantile, half_length):
            color = colors[i]
            start_key = q_keys[i]
            end_key = q_keys[-(i + 1)]
            axis.plot(times, quantiles[start_key], quantiles[end_key])
            pyplot.legend(q_keys, loc="best")

    # df.index = df.time
    # axis.plot(df.index, df.value)
    # pyplot.legend(df.columns, loc="best")
    # change_xaxis_to_date(axes, REF_DATE)

    if num_quantiles % 2:
        q_key = q_keys[half_length]
        axis.plot(times, quantiles[q_key], color=colors[3])


def plot_multi_output_timeseries_with_uncertainty(
    plotter: Plotter,
    uncertainty_df: pd.DataFrame,
    output_names: str,
    scenarios: list,
    all_targets: dict,
    is_logscale=False,
    x_low=0.0,
    x_up=2000.0,
    n_xticks=None,
):
    if len(output_names) * len(scenarios) == 0:
        return
    # pyplot.rcParams.update({'font.size': 15})

    max_n_col = 2
    n_panels = len(output_names)
    n_cols = min(max_n_col, n_panels)
    n_rows = ceil(n_panels / max_n_col)

    fig = pyplot.figure(constrained_layout=True, figsize=(n_cols * 7, n_rows * 5))  # (w, h)
    spec = fig.add_gridspec(ncols=n_cols, nrows=n_rows)

    i_col = 0
    i_row = 0

    for output_name in output_names:
        targets = {k: v for k, v in all_targets.items() if v["output_key"] == output_name}
        ax = fig.add_subplot(spec[i_row, i_col])
        plot_timeseries_with_uncertainty(
            plotter,
            uncertainty_df,
            output_name,
            scenarios,
            targets,
            is_logscale,
            x_low,
            x_up,
            ax,
            n_xticks,
        )
        i_col += 1
        if i_col == max_n_col:
            i_col = 0
            i_row += 1

    plotter.save_figure(fig, filename="multi_uncertainty", subdir="outputs", title_text="")

    # out_dir = "apps/tuberculosis/regions/marshall_islands/figures/calibration_targets/"
    # filename = out_dir + "targets"
    # pyplot.savefig(filename + ".pdf")


def plot_seroprevalence_by_age(
    plotter: Plotter,
    uncertainty_df: pd.DataFrame,
    scenario_id: int,
    time: float,
    ref_date=REF_DATE,
    axis=None,
    name="",
):
    single_panel = axis is None
    if single_panel:
        fig, axis, _, _, _, _ = plotter.get_figure()
    mask = (uncertainty_df["scenario"] == scenario_id) & (uncertainty_df["time"] == time)
    df = uncertainty_df[mask]
    quantile_vals = df["quantile"].unique().tolist()
    seroprevalence_by_age = {}
    sero_outputs = [
        output
        for output in df["type"].unique().tolist()
        if "proportion_seropositiveXagegroup_" in output
    ]

    max_value = -10.0
    if len(sero_outputs) == 0:
        axis.text(0.0, 0.5, "Age-specific seroprevalence outputs are not available for this run")
    else:
        for output in sero_outputs:
            output_mask = df["type"] == output
            age = output.split("proportion_seropositiveXagegroup_")[1]
            seroprevalence_by_age[age] = {}
            for q in quantile_vals:
                q_mask = df["quantile"] == q
                seroprevalence_by_age[age][q] = [
                    100.0 * v for v in df[output_mask][q_mask]["value"].tolist()
                ]

        q_keys = sorted(quantile_vals)
        num_quantiles = len(q_keys)
        half_length = num_quantiles // 2

        for i, age in enumerate(list(seroprevalence_by_age.keys())):
            x_pos = 2.5 + float(age)
            axis.plot(
                [x_pos, x_pos],
                [seroprevalence_by_age[age][q_keys[0]], seroprevalence_by_age[age][q_keys[-1]]],
                "-",
                color="black",
                lw=1.0,
            )
            max_value = max(max_value, seroprevalence_by_age[age][q_keys[-1]][0])

            if num_quantiles % 2:
                q_key = q_keys[half_length]
                label = None if i > 0 else "model"
                axis.plot(
                    x_pos,
                    seroprevalence_by_age[age][q_key],
                    "o",
                    color="black",
                    markersize=4,
                    label=label,
                )

        axis.set_xlabel("age (years)", fontsize=13)
        axis.set_ylabel("% seropositive", fontsize=13)

        _date = ref_date + datetime.timedelta(days=time)

        axis.set_title(f'{name} {_date.strftime("%d/%m/%Y")}', fontsize=15)

    if single_panel:
        plotter.save_figure(fig, filename="sero_by_age", subdir="outputs", title_text="")

    return max_value


def plot_seroprevalence_by_age_against_targets(
    plotter, uncertainty_df, selected_scenario, serosurvey_data, n_columns
):
    n_surveys = len(serosurvey_data)
    n_rows = ceil(n_surveys / n_columns)

    with pyplot.style.context("ggplot"):
        fig = pyplot.figure(constrained_layout=True, figsize=(n_columns * 7, n_rows * 5))  # (w, h)
        spec = fig.add_gridspec(ncols=n_columns, nrows=n_rows)

        i_row = 0
        i_col = 0
        for survey in serosurvey_data:
            # plot model outputs
            midpoint_time = int(mean(survey["time_range"]))
            ax = fig.add_subplot(spec[i_row, i_col])
            s_name = "" if not "survey_name" in survey else survey["survey_name"]
            max_value = plot_seroprevalence_by_age(
                plotter, uncertainty_df, selected_scenario, time=midpoint_time, axis=ax, name=s_name
            )

            # add data
            shift = 0.5
            max_prev = -10.0
            for i, measure in enumerate(survey["measures"]):
                mid_age = mean(measure["age_range"])
                ax.plot(
                    [mid_age + shift, mid_age + shift],
                    [measure["ci"][0], measure["ci"][1]],
                    "-",
                    color="red",
                    lw=1.0,
                )
                label = None if i > 0 else "data"
                ax.plot(mid_age + shift, measure["central"], "o", color="red", ms=4, label=label)
                # ax.axvline(x=measure["age_range"][0], linestyle="--", color='grey', lw=.5)
                max_prev = max(max_prev, measure["ci"][1])

            if i_col + i_row == 0:
                ax.legend(loc="upper right", fontsize=12, facecolor="white")

            ax.set_ylim((0.0, max(max_prev * 1.3, max_value * 1.3)))

            i_col += 1
            if i_col == n_columns:
                i_row += 1
                i_col = 0

        if plotter is not None:
            plotter.save_figure(fig, filename="multi_sero_by_age", subdir="outputs", title_text="")
        else:
            return fig


def _get_target_values(targets: dict, output_name: str):
    """Pulls out values for a given target"""
    output_config = {"values": [], "times": []}
    for t in targets.values():
        if t["output_key"] == output_name:
            output_config = t

    values = output_config["values"]
    times = output_config["times"]
    return values, times
