import streamlit as st

from autumn.plots.plotter import StreamlitPlotter
from autumn.tool_kit.model_register import AppRegion
from autumn.tool_kit.scenarios import Scenario, get_model_times_from_inputs
from autumn.plots import plots
from autumn.tool_kit.params import load_params

from apps.covid_19.model.preprocess.mixing_matrix.adjust_location import (
    LocationMixingAdjustment,
    LOCATIONS,
    MICRODISTANCING_LOCATIONS,
)


PLOT_FUNCS = {}


def plot_flow_params(
    plotter: StreamlitPlotter, app: AppRegion,
):
    # Assume a COVID model
    model = app.build_model(app.params["default"])

    param_names = sorted(list({f.param_name for f in model.flows}))
    param_name = st.sidebar.selectbox("Select parameter", param_names)
    flows = [f for f in model.flows if f.param_name == param_name]
    is_logscale = st.sidebar.checkbox("Log scale")
    flow_funcs = [f.get_weight_value for f in flows]
    plots.plot_time_varying_input(
        plotter, f"flow-params-{param_name}", flow_funcs, model.times, is_logscale
    )

    init_dict = {}
    for f in flows:
        f_name = ""
        src = getattr(f, "source", None)
        dest = getattr(f, "dest", None)
        if src:
            f_name += f"from {src}"
        if dest:
            f_name += f" to {dest}"

        f_name = f_name.strip()
        init_dict[f_name] = f.get_weight_value(0)

    st.write("Values at start time:")
    st.write(init_dict)


PLOT_FUNCS["Flow weights"] = plot_flow_params


def plot_dynamic_inputs(
    plotter: StreamlitPlotter, app: AppRegion,
):
    # Assume a COVID model
    model = app.build_model(app.params["default"])
    tvs = model.time_variants
    tv_options = sorted(list(tvs.keys()))
    tv_key = st.sidebar.selectbox("Select function", tv_options)
    is_logscale = st.sidebar.checkbox("Log scale")
    tv_func = tvs[tv_key]
    plots.plot_time_varying_input(plotter, tv_key, tv_func, model.times, is_logscale)


PLOT_FUNCS["Time variant functions"] = plot_dynamic_inputs


def plot_location_mixing(plotter: StreamlitPlotter, app: AppRegion):
    if not app.app_name == "covid_19":
        # Assume a COVID model
        st.write("This only works for COVID-19 models :(")
        return

    params = app.params["default"]
    mixing = params.get("mixing")
    if not mixing:
        st.write("This model does not have location based mixing")
        return

    start_time = params["start_time"]
    end_time = params["end_time"]
    time_step = params["time_step"]
    times = get_model_times_from_inputs(round(start_time), end_time, time_step,)

    loc_key = st.sidebar.selectbox("Select location", LOCATIONS)
    is_logscale = st.sidebar.checkbox("Log scale")

    country_iso3 = params["iso3"]
    region = params["mobility_region"]
    microdistancing = params["microdistancing"]
    npi_effectiveness_params = params["npi_effectiveness"]
    google_mobility_locations = params["google_mobility_locations"]
    smooth_google_data = params.get("smooth_google_data")
    adjust = LocationMixingAdjustment(
        country_iso3,
        region,
        mixing,
        npi_effectiveness_params,
        google_mobility_locations,
        microdistancing,
        smooth_google_data,
    )
    if adjust.microdistancing_function and loc_key in MICRODISTANCING_LOCATIONS:
        loc_func = lambda t: adjust.microdistancing_function(t) * adjust.loc_adj_funcs[loc_key](t)
    elif loc_key in adjust.loc_adj_funcs:
        loc_func = lambda t: adjust.loc_adj_funcs[loc_key](t)
    else:
        loc_func = lambda t: 1

    plots.plot_time_varying_input(plotter, loc_key, loc_func, times, is_logscale)


PLOT_FUNCS["Dynamic location mixing"] = plot_location_mixing


def plot_model_targets(
    plotter: StreamlitPlotter, app: AppRegion,
):
    # Assume a COVID model
    scenario = Scenario(app.build_model, idx=0, params=app.params)
    with st.spinner("Running model..."):
        scenario.run()

    target_name_lookup = {t["title"]: t for t in app.targets.values()}
    title_options = sorted(list(target_name_lookup.keys()))
    title = st.sidebar.selectbox("Select a target", title_options)
    target = target_name_lookup[title]
    is_logscale = st.sidebar.checkbox("Log scale")
    plots.plot_outputs_single(plotter, scenario, target, is_logscale)


PLOT_FUNCS["Calibration targets"] = plot_model_targets