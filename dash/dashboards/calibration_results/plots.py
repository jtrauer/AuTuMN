import os
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import random
import yaml

from autumn.tool_kit.params import load_params
from autumn.plots.plotter import StreamlitPlotter
from autumn.plots.calibration.plots import find_shortest_chain_length, get_posterior, get_epi_params
from autumn import db, plots, inputs
from apps.covid_19.model.preprocess.testing import find_cdr_function_from_test_data
from apps.covid_19.mixing_optimisation.serosurvey_by_age.survey_data import get_serosurvey_data
from autumn.tool_kit.scenarios import get_model_times_from_inputs

from dash.utils import create_downloadable_csv, round_sig_fig
from dash import selectors

PLOT_FUNCS = {}


def write_mcmc_centiles(
    mcmc_params,
    mcmc_tables,
    burn_in,
    sig_figs,
    centiles,
):
    """
    Write a table of parameter centiles from the MCMC chain outputs.
    """

    # Get parameter names
    parameters = get_epi_params(mcmc_params)

    # Create empty dataframe
    params_df = pd.DataFrame(index=parameters, columns=centiles)

    # Populate with data
    for param_name in parameters:
        param_values = get_posterior(mcmc_params, mcmc_tables, param_name, burn_in)
        centile_values = np.percentile(param_values, centiles)
        rounded_centile_values = [round_sig_fig(i_value, sig_figs) for i_value in centile_values]
        params_df.loc[param_name] = rounded_centile_values

    # Display
    create_downloadable_csv(params_df, "posterior_centiles")
    st.write(params_df)


def get_uncertainty_df(calib_dir_path, mcmc_tables, targets):

    try:  # if PBI processing has been performed already
        uncertainty_df = db.load.load_uncertainty_table(calib_dir_path)
    except:  # calculates percentiles
        derived_output_tables = db.load.load_derived_output_tables(calib_dir_path)
        mcmc_all_df = db.load.append_tables(mcmc_tables)
        do_all_df = db.load.append_tables(derived_output_tables)

        # Determine max chain length, throw away first half of that
        max_run = mcmc_all_df["run"].max()
        half_max = max_run // 2
        mcmc_all_df = mcmc_all_df[mcmc_all_df["run"] >= half_max]
        uncertainty_df = db.uncertainty.calculate_mcmc_uncertainty(mcmc_all_df, do_all_df, targets)
    return uncertainty_df


def get_uncertainty_data(calib_dir_path, mcmc_tables, output, burn_in):
    derived_output_tables = db.load.load_derived_output_tables(calib_dir_path, column=output)
    return plots.calibration.plots.sample_outputs_for_calibration_fit(
        output, mcmc_tables, derived_output_tables, burn_in=burn_in
    )


def print_mle_parameters(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    df = db.load.append_tables(mcmc_tables)
    param_df = db.load.append_tables(mcmc_params)
    params = db.process.find_mle_params(df, param_df)
    create_downloadable_csv(pd.Series(params), "mle_parameters")
    st.write(params)


PLOT_FUNCS["Print MLE parameters"] = print_mle_parameters


def plot_acceptance_ratio(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    label_font_size = st.sidebar.slider("Label font size", 1, 15, 10)
    burn_in = st.sidebar.slider("Burn in", 0, find_shortest_chain_length(mcmc_tables), 0)
    dpi_request = st.sidebar.slider("DPI", 50, 2000, 300)
    plots.calibration.plots.plot_acceptance_ratio(
        plotter, mcmc_tables, burn_in, label_font_size=label_font_size, dpi_request=dpi_request
    )


PLOT_FUNCS["Acceptance ratio"] = plot_acceptance_ratio


def get_cdr_constants(default_params):
    iso3 = default_params["country"]["iso3"]
    testing_year = default_params["population"]["year"]
    assumed_tests_parameter = default_params["testing_to_detection"]["assumed_tests_parameter"]
    smoothing_period = default_params["testing_to_detection"]["smoothing_period"]
    agegroup_params = default_params["age_stratification"]
    time_params = default_params["time"]

    # Derive times and age group breaks in the same way as the model does
    times = get_model_times_from_inputs(
        time_params["start"], time_params["end"], time_params["step"]
    )
    agegroup_strata = [
        str(s) for s in range(0, agegroup_params["max_age"], agegroup_params["age_step_size"])
    ]
    return (
        iso3,
        testing_year,
        assumed_tests_parameter,
        smoothing_period,
        agegroup_params,
        time_params,
        times,
        agegroup_strata,
    )


def plot_cdr_curves(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    param_name = "testing_to_detection.assumed_cdr_parameter"
    region_name = region.replace("-", "_")
    end_date = st.sidebar.slider("End date", 1, 365, 275)
    samples = st.sidebar.slider("Samples", 1, 200, 10)
    label_rotation = st.sidebar.slider("Label rotation", 0, 90, 0)

    # Extract parameters relevant to this function
    params = load_params(app_name, region_name)
    (
        iso3,
        testing_year,
        assumed_tests_parameter,
        smoothing_period,
        agegroup_params,
        time_params,
        times,
        agegroup_strata,
    ) = get_cdr_constants(params["default"])

    # Collate parameters into one structure
    testing_to_detection_values = []
    for i_chain in range(len(mcmc_params)):
        param_mask = mcmc_params[i_chain]["name"] == param_name
        testing_to_detection_values += mcmc_params[i_chain]["value"][param_mask].tolist()

    # Sample testing values from all the ones available, to avoid plotting too many curves
    if samples > len(testing_to_detection_values):
        st.write("Warning: Requested samples greater than detection values estimated")
        samples = len(testing_to_detection_values)
    sampled_test_to_detect_vals = random.sample(testing_to_detection_values, samples)

    # Get CDR function - needs to be done outside of autumn, because it is importing from the apps
    testing_pops = inputs.get_population_by_agegroup(agegroup_strata, iso3, None, year=testing_year)
    detected_proportion = []
    for assumed_cdr_parameter in sampled_test_to_detect_vals:
        detected_proportion.append(
            find_cdr_function_from_test_data(
                assumed_tests_parameter,
                assumed_cdr_parameter,
                smoothing_period,
                iso3,
                testing_pops,
            )
        )

    plots.calibration.plots.plot_cdr_curves(
        plotter, times, detected_proportion, end_date, label_rotation
    )


PLOT_FUNCS["CDR curves"] = plot_cdr_curves


def plot_timeseries_with_uncertainty(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    available_outputs = [o["output_key"] for o in targets.values()]
    chosen_output = st.sidebar.selectbox("Select calibration target", available_outputs)
    targets = {k: v for k, v in targets.items() if v["output_key"] == chosen_output}

    uncertainty_df = get_uncertainty_df(calib_dir_path, mcmc_tables, targets)

    x_min = round(min(uncertainty_df["time"]))
    x_max = round(max(uncertainty_df["time"]))
    x_low, x_up = selectors.create_xrange_selector(x_min, x_max)

    available_scenarios = uncertainty_df["scenario"].unique()
    selected_scenarios = st.multiselect("Select scenarios", available_scenarios)

    (
        title_font_size,
        label_font_size,
        dpi_request,
        capitalise_first_letter,
    ) = selectors.create_standard_plotting_sidebar()
    is_logscale = st.sidebar.checkbox("Log scale")
    is_targets = st.sidebar.checkbox("Show targets")
    is_overlay_unceratinty = st.sidebar.checkbox("Overlay uncertainty")
    is_legend = st.sidebar.checkbox("Add legend")
    plots.uncertainty.plots.plot_timeseries_with_uncertainty(
        plotter,
        uncertainty_df,
        chosen_output,
        selected_scenarios,
        targets,
        is_logscale,
        x_low,
        x_up,
        add_targets=is_targets,
        overlay_uncertainty=is_overlay_unceratinty,
        title_font_size=title_font_size,
        label_font_size=label_font_size,
        dpi_request=dpi_request,
        capitalise_first_letter=capitalise_first_letter,
        legend=is_legend,
    )


PLOT_FUNCS["Output uncertainty"] = plot_timeseries_with_uncertainty


def plot_multiple_timeseries_with_uncertainty(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    available_outputs = [o["output_key"] for o in targets.values()]
    chosen_outputs = st.multiselect("Select outputs", available_outputs)

    uncertainty_df = get_uncertainty_df(calib_dir_path, mcmc_tables, targets)

    x_min = round(min(uncertainty_df["time"]))
    x_max = round(max(uncertainty_df["time"]))
    x_low, x_up = selectors.create_xrange_selector(x_min, x_max)

    available_scenarios = uncertainty_df["scenario"].unique()
    selected_scenarios = st.multiselect("Select scenarios", available_scenarios)
    is_logscale = st.sidebar.checkbox("Log scale")
    n_xticks = st.sidebar.slider("Number of x ticks", 1, 10, 6)
    title_font_size = st.sidebar.slider("Title font size", 1, 30, 12)
    label_font_size = st.sidebar.slider("Label font size", 1, 30, 10)
    plots.uncertainty.plots.plot_multi_output_timeseries_with_uncertainty(
        plotter,
        uncertainty_df,
        chosen_outputs,
        selected_scenarios,
        targets,
        is_logscale,
        x_low,
        x_up,
        n_xticks,
        title_font_size=title_font_size,
        label_font_size=label_font_size,
    )


PLOT_FUNCS["Multi-output uncertainty"] = plot_multiple_timeseries_with_uncertainty


def plot_calibration_fit(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    # Set up interface
    available_outputs = [o["output_key"] for o in targets.values()]
    chain_length = find_shortest_chain_length(mcmc_tables)
    burn_in = st.sidebar.slider(
        "Burn in (select 0 for default behaviour of discarding first half)", 0, chain_length, 0
    )
    chosen_output = st.sidebar.selectbox("Select calibration target", available_outputs)
    is_logscale = st.sidebar.checkbox("Log scale")

    # Get data for plotting
    outputs = get_uncertainty_data(calib_dir_path, mcmc_tables, chosen_output, burn_in)

    # Call main plotting function
    plots.calibration.plots.plot_calibration_fit(
        plotter, chosen_output, outputs, targets, is_logscale
    )


PLOT_FUNCS["Output calibration fit"] = plot_calibration_fit


def plot_multi_output_fit(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    # Set up interface
    available_outputs = [o["output_key"] for o in targets.values()]
    (
        title_font_size,
        label_font_size,
        dpi_request,
        capitalise_first_letter,
    ) = selectors.create_standard_plotting_sidebar()
    is_logscale = st.sidebar.checkbox("Log scale")
    chain_length = find_shortest_chain_length(mcmc_tables)
    burn_in = st.sidebar.slider(
        "Burn in (select 0 for default behaviour of discarding first half)", 0, chain_length, 0
    )

    # Get data for plotting
    outputs = {
        output: get_uncertainty_data(calib_dir_path, mcmc_tables, output, burn_in)
        for output in available_outputs
    }

    # Call main plotting function
    plots.calibration.plots.plot_multi_fit(
        plotter,
        available_outputs,
        outputs,
        targets,
        is_logscale,
        title_font_size,
        label_font_size,
        dpi_request,
        capitalise_first_letter,
    )


PLOT_FUNCS["Multi-output fit"] = plot_multi_output_fit


def plot_mcmc_parameter_trace(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    chosen_param = selectors.parameter(mcmc_params[0])
    chain_length = find_shortest_chain_length(mcmc_tables)
    burn_in = st.sidebar.slider("Burn in", 0, chain_length, 0)
    plots.calibration.plots.plot_mcmc_parameter_trace(plotter, mcmc_params, burn_in, chosen_param)


PLOT_FUNCS["Parameter trace"] = plot_mcmc_parameter_trace


def plot_all_param_traces(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    (
        title_font_size,
        label_font_size,
        dpi_request,
        capitalise_first_letter,
    ) = selectors.create_standard_plotting_sidebar()
    chain_length = find_shortest_chain_length(mcmc_tables)
    burn_in = st.sidebar.slider("Burn in", 0, chain_length, 0)
    plots.calibration.plots.plot_multiple_param_traces(
        plotter,
        mcmc_params,
        burn_in,
        title_font_size,
        label_font_size,
        capitalise_first_letter,
        dpi_request,
    )


PLOT_FUNCS["All param traces"] = plot_all_param_traces


def plot_loglikelihood_vs_parameter(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    chosen_param = selectors.parameter(mcmc_params[0])
    chain_length = find_shortest_chain_length(mcmc_tables)
    burn_in = st.sidebar.slider("Burn in", 0, chain_length, 0)
    plots.calibration.plots.plot_single_param_loglike(
        plotter, mcmc_tables, mcmc_params, burn_in, chosen_param
    )


PLOT_FUNCS["Loglikelihood vs param"] = plot_loglikelihood_vs_parameter


def plot_loglike_vs_all_params(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    (
        title_font_size,
        label_font_size,
        dpi_request,
        capitalise_first_letter,
    ) = selectors.create_standard_plotting_sidebar()
    chain_length = find_shortest_chain_length(mcmc_tables)
    burn_in = st.sidebar.slider("Burn in", 0, chain_length, 0)
    plots.calibration.plots.plot_all_params_vs_loglike(
        plotter,
        mcmc_tables,
        mcmc_params,
        burn_in,
        title_font_size,
        label_font_size,
        capitalise_first_letter,
        dpi_request,
    )


PLOT_FUNCS["All loglike vs params"] = plot_loglike_vs_all_params


def plot_posterior(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    chosen_param = selectors.parameter(mcmc_params[0])
    chain_length = find_shortest_chain_length(mcmc_tables)
    burn_in = st.sidebar.slider("Burn in", 0, chain_length, 0)
    num_bins = st.sidebar.slider("Number of bins", 1, 50, 16)

    prior = None
    priors = []
    try:
        priors_path = os.path.join(calib_dir_path, "priors-1.yml")
        with open(priors_path) as file:
            priors = yaml.load(file, Loader=yaml.FullLoader)
    except:
        st.write("Check if priors-1.yml exists in the output folder")

    for i, prior in enumerate(priors):
        if prior["param_name"] == chosen_param:
            prior = priors[i]
            break

    plots.calibration.plots.plot_posterior(
        plotter, mcmc_params, mcmc_tables, burn_in, chosen_param, num_bins, prior
    )


PLOT_FUNCS["Posterior distributions"] = plot_posterior


def plot_all_posteriors(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    priors = []
    try:
        priors_path = os.path.join(calib_dir_path, "priors-1.yml")
        with open(priors_path) as file:
            priors = yaml.load(file, Loader=yaml.FullLoader)
    except:
        st.write("Check if priors-1.yml exists in the output folder")

    (
        title_font_size,
        label_font_size,
        dpi_request,
        capitalise_first_letter,
    ) = selectors.create_standard_plotting_sidebar()
    chain_length = find_shortest_chain_length(mcmc_tables)
    burn_in = st.sidebar.slider("Burn in", 0, chain_length, 0)
    num_bins = st.sidebar.slider("Number of bins", 1, 50, 16)
    sig_figs = st.sidebar.slider("Significant figures", 0, 6, 3)
    plots.calibration.plots.plot_multiple_posteriors(
        plotter,
        mcmc_params,
        mcmc_tables,
        burn_in,
        num_bins,
        title_font_size,
        label_font_size,
        capitalise_first_letter,
        dpi_request,
        priors,
    )

    write_mcmc_centiles(mcmc_params, mcmc_tables, burn_in, sig_figs, [2.5, 50, 97.5])


PLOT_FUNCS["All posteriors"] = plot_all_posteriors


def plot_loglikelihood_trace(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    burn_in = selectors.burn_in(mcmc_tables)
    plots.calibration.plots.plot_loglikelihood_trace(plotter, mcmc_tables, burn_in)


PLOT_FUNCS["Loglikelihood trace"] = plot_loglikelihood_trace


def compare_loglikelihood_between_chains(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    plots.calibration.plots.plot_loglikelihood_boxplots(plotter, mcmc_tables)


PLOT_FUNCS["Compare loglikelihood between chains"] = compare_loglikelihood_between_chains


def plot_param_matrix_by_chain(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    """
    Now unused because I prefer the version that isn't by chain.
    """
    label_font_size = st.sidebar.slider("Label font size", 1, 15, 8)
    label_chars = st.sidebar.slider("Label characters", 1, 10, 2)
    dpi_request = st.sidebar.slider("DPI", 50, 2000, 300)
    parameters = mcmc_params[0]["name"].unique().tolist()
    st.write(parameters)
    plots.calibration.plots.plot_param_vs_param_by_chain(
        plotter, mcmc_params, parameters, label_font_size, label_chars, dpi_request
    )


def plot_param_matrix(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    parameters = mcmc_params[0]["name"].unique().tolist()
    chain_length = find_shortest_chain_length(mcmc_tables)
    burn_in = st.sidebar.slider("Burn in", 0, chain_length, 0)
    label_font_size = st.sidebar.slider("Label font size", 1, 15, 8)
    label_chars = st.sidebar.slider("Label characters", 1, 10, 2)
    bins = st.sidebar.slider("Bins", 4, 50, 20)
    style = st.sidebar.selectbox("Style", ["Shade", "Scatter", "KDE"])
    dpi_request = st.sidebar.slider("DPI", 50, 2000, 300)
    plots.calibration.plots.plot_param_vs_param(
        plotter,
        mcmc_params,
        parameters,
        burn_in,
        style,
        bins,
        label_font_size,
        label_chars,
        dpi_request,
    )
    st.write(parameters)


PLOT_FUNCS["Param versus param"] = plot_param_matrix


def plot_parallel_coordinates(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    plots.calibration.plots.plot_parallel_coordinates(
        plotter,
        mcmc_tables,
        mcmc_params,
    )


PLOT_FUNCS["Parallel Coordinates"] = plot_parallel_coordinates


def plot_loglikelihood_surface(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    options = mcmc_params[0]["name"].unique().tolist()
    param_1 = st.sidebar.selectbox("Select parameter 1", options)
    param_2 = st.sidebar.selectbox("Select parameter 2", options)

    plots.calibration.plots.plot_loglikelihood_surface(
        plotter,
        mcmc_tables,
        mcmc_params,
        param_1,
        param_2,
    )


PLOT_FUNCS["Loglikelihood 3d scatter"] = plot_loglikelihood_surface


def plot_seroprevalence_by_age(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    try:  # if PBI processing has been performed already
        uncertainty_df = db.load.load_uncertainty_table(calib_dir_path)
    except:  # calculates percentiles
        derived_output_tables = db.load.load_derived_output_tables(calib_dir_path)
        mcmc_all_df = db.load.append_tables(mcmc_tables)
        do_all_df = db.load.append_tables(derived_output_tables)

        # Determine max chain length, throw away first half of that
        max_run = mcmc_all_df["run"].max()
        half_max = max_run // 2
        mcmc_all_df = mcmc_all_df[mcmc_all_df["run"] >= half_max]
        uncertainty_df = db.uncertainty.calculate_mcmc_uncertainty(mcmc_all_df, do_all_df, targets)

    available_scenarios = uncertainty_df["scenario"].unique()
    selected_scenario = st.sidebar.selectbox("Select scenario", available_scenarios)
    min_time = int(min(uncertainty_df["time"]))
    max_time = int(max(uncertainty_df["time"]))
    time = st.sidebar.slider("time", min_time, max_time, max_time)

    sero_data = get_serosurvey_data()
    region = "belgium"  # FIXME
    if region in sero_data:
        fetch_targets = st.sidebar.checkbox("for all available targets", value=False)
        n_columns = st.sidebar.slider("Number of columns for multi-panel", 1, 5, 3)
    else:
        fetch_targets = False

    if fetch_targets:
        plots.uncertainty.plots.plot_seroprevalence_by_age_against_targets(
            plotter, uncertainty_df, selected_scenario, sero_data[region], n_columns
        )
    else:
        plots.uncertainty.plots.plot_seroprevalence_by_age(
            plotter, uncertainty_df, selected_scenario, time
        )


PLOT_FUNCS["Seroprevalence by age"] = plot_seroprevalence_by_age
