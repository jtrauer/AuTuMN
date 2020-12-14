import streamlit as st
from autumn import plots
from dash.dashboards.calibration_results.plots import get_uncertainty_data, get_uncertainty_df
from dash import selectors
from math import ceil
import matplotlib.pyplot as pyplot


from apps.covid_19.model.preprocess.mixing_matrix.microdistancing import get_microdistancing_funcs
from apps.covid_19.model.parameters import MicroDistancingFunc, EmpiricMicrodistancingParams

PLOT_FUNCS = {}


def multi_country_uncertainty(
    plotter, calib_dir_path, mcmc_tables, mcmc_params, targets, app_name, region_names
):
    """
    Code taken directly from the fit calibration file at this stage.
    """

    is_logscale = st.sidebar.checkbox("Log scale")
    n_xticks = st.sidebar.slider("Number of x ticks", 1, 10, 6)
    title_font_size = st.sidebar.slider("Title font size", 1, 30, 12)
    label_font_size = st.sidebar.slider("Label font size", 1, 30, 10)
    uncertainty_df = []

    # Hard coded for PHL and sub regions
    microdistancing_funcs = get_microdistancing_funcs(
        {
            "behaviour": MicroDistancingFunc(
                function_type="empiric",
                parameters=EmpiricMicrodistancingParams(
                    max_effect=0.28, times=[75.0, 314.0], values=[0.0, 1.0]
                ),
            )
        },
        ["other_locations", "school", "work", "home"],
        True,
    )

    for i_region in range(len(mcmc_tables)):
        uncertainty_df.append(
            get_uncertainty_df(calib_dir_path[i_region], mcmc_tables[i_region], targets[i_region])
        )

        if i_region == 0:
            available_outputs = [o["output_key"] for o in targets[i_region].values()]
            chosen_output = st.sidebar.selectbox("Select calibration target", available_outputs)
            if chosen_output == "notifications":
                chosen_micro = st.sidebar.checkbox("Overlay micro distance")
            x_min = round(min(uncertainty_df[0]["time"]))
            x_max = round(max(uncertainty_df[0]["time"]))
            x_low, x_up = selectors.create_xrange_selector(x_min, x_max)
            available_scenarios = uncertainty_df[0]["scenario"].unique()
            selected_scenarios = st.multiselect("Select scenarios", available_scenarios)



    plots.uncertainty.plots.plot_multicountry_timeseries_with_uncertainty(
        plotter,
        uncertainty_df,
        chosen_output,
        selected_scenarios,
        targets,
        region_names,
        is_logscale,
        x_low,
        x_up,
        n_xticks,
        microdistancing_funcs,
        title_font_size=title_font_size,
        label_font_size=label_font_size,
    )


PLOT_FUNCS["Multi-country uncertainty"] = multi_country_uncertainty

