"""
This module contains the main disease modelling class.
"""
import logging
import copy
from typing import Tuple, List, Dict, Callable, Optional
from functools import lru_cache

import networkx
import numpy as np
from scipy.interpolate import interp1d

import summer2.flows as flows
from summer2.compartment import Compartment
from summer2.stratification import Stratification
from summer2.solver import solve_ode, SolverType
from summer2.adjust import FlowParam

logger = logging.getLogger()

FlowRateFunction = Callable[[List[Compartment], np.ndarray, np.ndarray, float], np.ndarray]


class CompartmentalModel:
    """
    A compartmental disease model.

    This model defines a set of compartments which each contain a population.
    Disease dynamics are defined by a set of flows which link the compartments together.
    The model is run over a period of time, starting from some initial conditions to predict the future state of a disease.

    Args:
        times: The start and end times.
        compartments: The compartments to simulate.
        infectious_compartments: The compartments which are counted as infectious.
        time_step (optional): The timesteps to return results for. Does not affect the ODE solver. Defaults to ``1``.

    Attributes:
        times (np.ndarray): The times that the model will simulate.
        compartments (List[Compartment]): The model's compartments.
        initial_population (np.ndarray): The model's starting population. The indices of this
            array will match up with ``compartments``. This is zero by default and can be set with ``set_initial_population``.
        outputs (np.ndarray): The values of each compartment for each requested timestep. For ``C`` compartments and
            ``T`` timesteps this will be a ``CxT`` matrix. The column indices of this array will match up with ``compartments`` and the row indices will match up with ``times``.
        derived_outputs (Dict[str, np.ndarray]): Additional results that are caculated from ``outputs`` for each timestep.


    """

    _DEFAULT_DISEASE_STRAIN = "default"
    _DEFAULT_MIXING_MATRIX = np.array([[1]])

    def __init__(
        self,
        times: Tuple[int, int],
        compartments: List[str],
        infectious_compartments: List[str],
        timestep: float = 1.0,
    ):
        start_t, end_t = times
        assert start_t >= 0, "Start time must be >= 0"
        assert end_t > start_t, "End time must be greater than start time"
        start_t = round(start_t)
        end_t = round(end_t)
        time_period = end_t - start_t + 1
        num_steps = time_period / timestep
        assert num_steps >= 1, "Time step should be less than time period."
        assert num_steps % 1 == 0, "Time step should be a factor of time period"
        self.times = np.linspace(start_t, end_t, num=int(num_steps))

        msg = "Infectious compartments must be a subset of compartments"
        assert all(n in compartments for n in infectious_compartments), msg
        self.compartments = [Compartment(n) for n in compartments]
        self._infectious_compartments = [Compartment(n) for n in infectious_compartments]
        self.initial_population = np.zeros_like(self.compartments, dtype=np.float)

        # Keeps track of original, pre-stratified compartment names.
        self._original_compartment_names = [Compartment.deserialize(n) for n in compartments]
        # Tracks total deaths per timestep for death-replacement birth flows
        self._total_deaths = None
        # Keeps track of Stratifications that have been applied.
        self._stratifications = []
        # Flows to be applied to the model compartments
        self._flows = []
        # No outputs until the model has been run.
        self.outputs = None
        self.derived_outputs = None
        # Track derived output requests in a dictionary.
        self._derived_output_requests = {}
        # Track derived output request dependencies in a directed acylic graph (DAG).
        self._derived_output_graph = networkx.DiGraph()

        # Mixing matrices a list of NxN arrays used to calculate force of infection.
        self._mixing_matrices = []

        # A list of dicts that has the strata required to match a row in the mixing matrix.
        self._mixing_categories = [{}]
        # A list of the different sub-categories ('strains') of the diease that we are modelling.
        self._disease_strains = [self._DEFAULT_DISEASE_STRAIN]

        self._update_compartment_indices()

    def _update_compartment_indices(self):
        """
        Update the mapping of compartment name to idx for quicker lookups.
        """
        compartment_idx_lookup = {}
        # Update the mapping of compartment name to idx for quicker lookups.
        for idx, c in enumerate(self.compartments):
            c.idx = idx
            compartment_idx_lookup[c] = idx

        for flow in self._flows:
            flow.update_compartment_indices(compartment_idx_lookup)

    def set_initial_population(self, distribution: Dict[str, float]):
        """
        Sets the initial population of the model, which is zero by default.

        Args:
            distribution: A map of populations to be assigned to compartments.

        """
        error_msg = "Cannot set initial population after the model has been stratified"
        assert not self._stratifications, error_msg
        for idx, comp in enumerate(self.compartments):
            pop = distribution.get(comp.name, 0)
            assert pop >= 0, f"Population for {comp.name} cannot be negative: {pop}"
            self.initial_population[idx] = pop

    """
    Adding flows
    """

    def add_crude_birth_flow(
        self,
        name: str,
        birth_rate: FlowParam,
        dest: str,
        dest_strata: Optional[Dict[str, str]] = None,
        expected_flow_count: Optional[int] = None,
    ):
        """
        Adds a crude birth rate flow to the model.
        The number of births will be determined by the product of the birth rate and total population.

        Args:
            name: The name of the new flow.
            birth_rate: The fractional crude birth rate per timestep.
            dest: The name of the destination compartment.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are created.

        """
        self._validate_param(name, birth_rate)
        self._add_entry_flow(
            flows.CrudeBirthFlow,
            name,
            birth_rate,
            dest,
            dest_strata,
            expected_flow_count,
        )

    def add_replacement_birth_flow(
        self,
        name: str,
        dest: str,
        dest_strata: Optional[Dict[str, str]] = None,
        expected_flow_count: Optional[int] = None,
    ):
        """
        Adds a death-replacing birth flow to the model.
        The number of births will replace the total number of deaths each year

        Args:
            name: The name of the new flow.
            dest: The name of the destination compartment.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are created.

        """
        # Only allow a single replacement flow to be added to the model.
        is_already_used = any([type(f) is flows.ReplacementBirthFlow for f in self._flows])
        if is_already_used:
            msg = "There is already a replacement birth flow in this model, cannot add a second."
            raise ValueError(msg)

        self._add_entry_flow(
            flows.ReplacementBirthFlow,
            name,
            self._get_total_deaths,
            dest,
            dest_strata,
            expected_flow_count,
        )

    def add_importation_flow(
        self,
        name: str,
        num_imported: FlowParam,
        dest: str,
        dest_strata: Optional[Dict[str, str]] = None,
        expected_flow_count: Optional[int] = None,
    ):
        """
        Adds an importation flow to the model, where people enter the destination compartment from outside the system.
        The number of people imported per timestep is be completely determined by ``num_imported``.

        Args:
            name: The name of the new flow.
            num_imported: The number of people imported per timestep.
            dest: The name of the destination compartment.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are created.

        """
        self._validate_param(name, num_imported)
        self._add_entry_flow(
            flows.ImportFlow, name, num_imported, dest, dest_strata, expected_flow_count
        )

    def _add_entry_flow(
        self,
        flow_cls,
        name: str,
        param: FlowParam,
        dest: str,
        dest_strata: Optional[Dict[str, str]],
        expected_flow_count: Optional[int],
    ):
        dest_strata = dest_strata or {}
        dest_comps = [c for c in self.compartments if c.is_match(dest, dest_strata)]
        new_flows = []
        for dest_comp in dest_comps:
            flow = flow_cls(name, dest_comp, param)
            new_flows.append(flow)

        self._validate_expected_flow_count(expected_flow_count, new_flows)
        self._flows += new_flows
        self._update_compartment_indices()

    def add_death_flow(
        self,
        name: str,
        death_rate: FlowParam,
        source: str,
        source_strata: Optional[Dict[str, str]] = None,
        expected_flow_count: Optional[int] = None,
    ):
        """
        Adds a flow where people die and leave the compartment, reducing the total population.

        Args:
            name: The name of the new flow.
            death_rate: The fractional death rate per timestep.
            source: The name of the source compartment.
            source_strata (optional): A whitelist of strata to filter the source compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are created.

        """
        self._add_exit_flow(
            flows.DeathFlow,
            name,
            death_rate,
            source,
            source_strata,
            expected_flow_count,
        )

    def add_universal_death_flows(self, base_name: str, death_rate: FlowParam):
        """
        Adds a universal death rate flow to every compartment in the model.
        The number of deaths per compartment will be determined by the product of
        the death rate and the compartment population.

        The base name will be used to create the name of each flow. For example a
        base name of "universal_death" applied to the "S" comparement will result in a flow called
        "universal_death_for_S".

        Args:
            base_name: The base name for each new flow.
            death_rate: The fractional death rate per timestep.

        Returns:
            List[str]: The names of the flows added.

        """
        # Only allow a single universal death flow with a given name to be added to the model.
        is_already_used = any([f.name.startswith(base_name) for f in self._flows])
        if is_already_used:
            msg = f"There is already a universal death flow called '{base_name}' in this model, cannot add a second."
            raise ValueError(msg)

        flow_names = []
        for comp_name in self._original_compartment_names:
            flow_name = f"{base_name}_for_{comp_name}"
            flow_names.append(flow_name)
            self._add_exit_flow(
                flows.DeathFlow,
                flow_name,
                death_rate,
                comp_name,
                source_strata={},
                expected_flow_count=None,
            )

        return flow_names

    def _add_exit_flow(
        self,
        flow_cls,
        name: str,
        param: FlowParam,
        source: str,
        source_strata: Optional[Dict[str, str]],
        expected_flow_count: Optional[int],
    ):
        source_strata = source_strata or {}
        self._validate_param(name, param)
        source_comps = [c for c in self.compartments if c.is_match(source, source_strata)]
        new_flows = []
        for source_comp in source_comps:
            flow = flow_cls(name, source_comp, param)
            new_flows.append(flow)

        self._validate_expected_flow_count(expected_flow_count, new_flows)
        self._flows += new_flows
        self._update_compartment_indices()

    def add_sojourn_flow(
        self,
        name: str,
        sojourn_time: FlowParam,
        source: str,
        dest: str,
        source_strata: Optional[Dict[str, str]] = None,
        dest_strata: Optional[Dict[str, str]] = None,
        expected_flow_count: Optional[int] = None,
    ):
        """
        Adds a flow where that models a "sojourn" through a compartment, where the flow rate
        is proportional to the inverse of the sojourn time. For example if there is a sojourn time of 10
        days, then the flow rate will be 10% of the occupants per day.

        Args:
            name: The name of the new flow.
            sojourn_time: The mean time sojourn time for a person in the compartment.
            source: The name of the source compartment.
            dest: The name of the destination compartment.
            source_strata (optional): A whitelist of strata to filter the source compartments.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are created.

        """
        self._add_transition_flow(
            flows.SojournFlow,
            name,
            sojourn_time,
            source,
            dest,
            source_strata,
            dest_strata,
            expected_flow_count,
        )

    def add_infection_frequency_flow(
        self,
        name: str,
        contact_rate: FlowParam,
        source: str,
        dest: str,
        source_strata: Optional[Dict[str, str]] = None,
        dest_strata: Optional[Dict[str, str]] = None,
        expected_flow_count: Optional[int] = None,
    ):
        """
        Adds a flow that infects people using an "infection frequency" contact rate, which is
        when the infectious multiplier is determined by the proportion of infectious people to the total population.

        Args:
            name: The name of the new flow.
            contact_rate: The contact rate.
            source: The name of the source compartment.
            dest: The name of the destination compartment.
            source_strata (optional): A whitelist of strata to filter the source compartments.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are created.

        """
        self._add_transition_flow(
            flows.InfectionFrequencyFlow,
            name,
            contact_rate,
            source,
            dest,
            source_strata,
            dest_strata,
            expected_flow_count,
            find_infectious_multiplier=self._get_infection_frequency_multiplier,
        )

    def add_infection_density_flow(
        self,
        name: str,
        contact_rate: FlowParam,
        source: str,
        dest: str,
        source_strata: Optional[Dict[str, str]] = None,
        dest_strata: Optional[Dict[str, str]] = None,
        expected_flow_count: Optional[int] = None,
    ):
        """
        Adds a flow that infects people using an "infection density" contact rate, which is
        when the infectious multiplier is determined by the number of infectious people.

        Args:
            name: The name of the new flow.
            contact_rate: The contact rate.
            source: The name of the source compartment.
            dest: The name of the destination compartment.
            source_strata (optional): A whitelist of strata to filter the source compartments.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are created.

        """
        self._add_transition_flow(
            flows.InfectionDensityFlow,
            name,
            contact_rate,
            source,
            dest,
            source_strata,
            dest_strata,
            expected_flow_count,
            find_infectious_multiplier=self._get_infection_density_multiplier,
        )

    def add_fractional_flow(
        self,
        name: str,
        fractional_rate: FlowParam,
        source: str,
        dest: str,
        source_strata: Optional[Dict[str, str]] = None,
        dest_strata: Optional[Dict[str, str]] = None,
        expected_flow_count: Optional[int] = None,
    ):
        """
        Adds a flow transfers people from a source to a destination based on the population of the source
        compartment and the fractional flow rate.

        Args:
            name: The name of the new flow.
            fractional_rate: The fraction of people that transfer per timestep.
            source: The name of the source compartment.
            dest: The name of the destination compartment.
            source_strata (optional): A whitelist of strata to filter the source compartments.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are created.

        """
        self._add_transition_flow(
            flows.FractionalFlow,
            name,
            fractional_rate,
            source,
            dest,
            source_strata,
            dest_strata,
            expected_flow_count,
        )

    def add_function_flow(
        self,
        name: str,
        flow_rate_func: FlowRateFunction,
        source: str,
        dest: str,
        source_strata: Optional[Dict[str, str]] = None,
        dest_strata: Optional[Dict[str, str]] = None,
        expected_flow_count: Optional[int] = None,
    ):
        """
        A flow that transfers people from a source to a destination based on a user-defined function.
        This can be used to define more complex flows if required. See `flows.FunctionFlow` for more details on the arguments to the function.

        Args:
            name: The name of the new flow.
            flow_rate_func:  A function that returns the flow rate, before adjustments.
            source: The name of the source compartment.
            dest: The name of the destination compartment.
            source_strata (optional): A whitelist of strata to filter the source compartments.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are created.

        """
        self._add_transition_flow(
            flows.FunctionFlow,
            name,
            flow_rate_func,
            source,
            dest,
            source_strata,
            dest_strata,
            expected_flow_count,
        )

    def _add_transition_flow(
        self,
        flow_cls,
        name: str,
        param: FlowParam,
        source: str,
        dest: str,
        source_strata: Optional[Dict[str, str]],
        dest_strata: Optional[Dict[str, str]],
        expected_flow_count: Optional[int],
        find_infectious_multiplier=None,
    ):
        source_strata = source_strata or {}
        dest_strata = dest_strata or {}
        if flow_cls is not flows.FunctionFlow:
            # Non standard param for FunctionFlow so cannot use this validation method.
            self._validate_param(name, param)

        dest_comps = [c for c in self.compartments if c.is_match(dest, dest_strata)]
        source_comps = [c for c in self.compartments if c.is_match(source, source_strata)]
        num_dest = len(dest_comps)
        num_source = len(dest_comps)
        msg = f"Expected equal number of source and dest compartments, but got {num_source} source and {num_dest} dest."
        assert num_dest == num_source, msg
        new_flows = []
        for source_comp, dest_comp in zip(source_comps, dest_comps):
            if find_infectious_multiplier:
                flow = flow_cls(
                    name,
                    source_comp,
                    dest_comp,
                    param,
                    find_infectious_multiplier=find_infectious_multiplier,
                )
            else:
                flow = flow_cls(name, source_comp, dest_comp, param)

            new_flows.append(flow)

        self._validate_expected_flow_count(expected_flow_count, new_flows)
        self._flows += new_flows
        self._update_compartment_indices()

    def _validate_param(self, flow_name: str, param: FlowParam):
        """
        Ensure that the supplied parameter produces sensible results for all timesteps.
        """
        is_all_positive = (
            all(map(lambda t: param(t) >= 0, self.times)) if callable(param) else param >= 0
        )
        error_msg = f"Parameter for {flow_name} must be >= 0 for all timesteps: {param}"
        assert is_all_positive, error_msg

    def _validate_expected_flow_count(
        self, expected_count: Optional[int], new_flows: List[flows.BaseFlow]
    ):
        """
        Ensure the number of new flows created is the expected amount
        """
        if expected_count is not None:
            # Check that we added the expected number of flows.
            actual_count = len(new_flows)
            msg = f"Expected to add {expected_count} flows but added {actual_count}"
            assert actual_count == expected_count, msg

    """
    Stratifying the model
    """

    def stratify_with(self, strat: Stratification):
        """
        Apply the stratification to the model's flows and compartments.

        Args:
            strat: The stratification to apply.

        """
        # Validate flow adjustments
        flow_names = [f.name for f in self._flows]
        for n in strat.flow_adjustments.keys():
            msg = f"Flow adjustment for '{n}' refers to a flow that is not present in the model."
            assert n in flow_names, msg

        # Validate infectiousness adjustments.
        msg = "All stratification infectiousness adjustments must refer to a compartment that is present in model."
        assert all(
            [c in self._original_compartment_names for c in strat.infectiousness_adjustments.keys()]
        ), msg

        if strat.mixing_matrix is not None:
            # Add this stratification's mixing matrix if a new one is provided.
            assert not strat.is_strain(), "Strains cannot have a mixing matrix."
            # Only allow mixing matrix to be supplied if there is a complete stratification.
            msg = "Mixing matrices only allowed for full stratification."
            assert strat.compartments == self._original_compartment_names, msg
            self._mixing_matrices.append(strat.mixing_matrix)
            # Update mixing categories for force of infection calculation.
            old_mixing_categories = self._mixing_categories
            self._mixing_categories = []
            for mc in old_mixing_categories:
                for stratum in strat.strata:
                    self._mixing_categories.append({**mc, strat.name: stratum})

        if strat.is_strain():
            # Track disease strain names, overriding default values.
            msg = "A disease strain stratification has already been applied, cannot use more than one."
            assert not any([s.is_strain() for s in self._stratifications]), msg
            self._disease_strains = strat.strata

        # Stratify compartments, split according to split_proportions

        prev_compartment_names = copy.copy(self.compartments)
        self.compartments = strat._stratify_compartments(self.compartments)
        self.initial_population = strat._stratify_compartment_values(
            prev_compartment_names, self.initial_population
        )

        # Stratify flows
        prev_flows = self._flows
        self._flows = []
        for flow in prev_flows:
            self._flows += flow.stratify(strat)

        # Update the mapping of compartment name to idx for quicker lookups.
        self._update_compartment_indices()

        if strat.is_ageing():
            # Create inter-compartmental flows for ageing from one stratum to the next.
            # The ageing rate is proportional to the width of the age bracket.
            # It's assumed that both ages and model timesteps are in years.
            ages = list(sorted(map(int, strat.strata)))
            for age_idx in range(len(ages) - 1):
                start_age = int(ages[age_idx])
                end_age = int(ages[age_idx + 1])
                for comp in prev_compartment_names:
                    if not comp.has_name_in_list(strat.compartments):
                        # Don't include unstratified compartments
                        continue

                    source = comp.stratify(strat.name, str(start_age))
                    dest = comp.stratify(strat.name, str(end_age))
                    self.add_sojourn_flow(
                        name=f"ageing_{source}_to_{dest}",
                        sojourn_time=end_age - start_age,
                        source=source.name,
                        dest=dest.name,
                        source_strata=source.strata,
                        dest_strata=dest.strata,
                        expected_flow_count=1,
                    )

        self._stratifications.append(strat)

    """
    Running the model
    """

    def run(
        self,
        solver: str = SolverType.SOLVE_IVP,
        solver_args: Optional[dict] = None,
        outputs_to_calculate: Optional[List[str]] = None,
    ):
        """
        Runs the model over the provided time span, calculating the outputs and the derived outputs.
        The model calculates the outputs by solving an ODE which is defined by the initial population and the inter-compartmental flows.

        **Note**: The default ODE solver used to produce the model's outputs does not necessarily evaluate every requested timestep. This adaptive
        solver can skip over times, or double back when trying to characterize the ODE. The final results are produced by interpolating the
        solution produced by the ODE solver. This means that model dynamics that only occur in short time periods may not be reflected in the outputs.

        Args:
            solver (optional): The ODE solver to use, defaults to SciPy's IVP solver.
            solver_args (optional): Extra arguments to supplied to the solver, see ``summer.solver`` for details.
            outputs_to_calculate (optional): Whitelist of derived outputs to calculate - any other outputs will be ignored.

        """
        solver_args = solver_args or {}
        # Do some setup.
        self._prepare_to_run()
        # Calculate the outputs (compartment sizes) by solving the ODE defined by _get_flows().
        self.outputs = solve_ode(
            solver,
            self._get_flow_rates,
            self.initial_population,
            self.times,
            solver_args,
        )
        # Calculate any requested derived outputs, based on the calculated compartment sizes.
        self.derived_outputs = self._calculate_derived_outputs(outputs_to_calculate)

    def _prepare_to_run(self):
        """
        Pre-run setup.
        Here we do any calculations/preparation are possible to do before the model runs.
        """
        # Functions will often be called multiple times per timestep, so we cache any time-varying functions.
        # First we find all time varying functions.
        funcs = set()
        for flow in self._flows:
            if isinstance(flow, flows.FunctionFlow):
                # Don't cache these, the input arguments cannot be stored in a dict ("non hashable")
                continue
            elif callable(flow.param):
                funcs.add(flow.param)
            for adj in flow.adjustments:
                if adj and callable(adj.param):
                    funcs.add(adj.param)

        # Cache return values to prevent re-computation. This will a little leak memory, which is fine.
        funcs_cached = {}
        for func in funcs:
            # Floating point return type is 8 bytes, meaning 2**17 values is ~1MB of memory.
            funcs_cached[func] = lru_cache(maxsize=2 ** 17)(func)

        # Finally, replace original functions with cached ones
        for flow in self._flows:
            if flow.param in funcs_cached:
                flow.param = funcs_cached[flow.param]
            for adj in flow.adjustments:
                if adj and adj.param in funcs_cached:
                    adj.param = funcs_cached[adj.param]

        # Optimize flow adjustments
        for f in self._flows:
            f.optimize_adjustments()

        # Split flows up into 3 groups: entry, exit and transition.
        self._entry_flows = [f for f in self._flows if issubclass(f.__class__, flows.BaseEntryFlow)]
        self._exit_flows = [f for f in self._flows if issubclass(f.__class__, flows.BaseExitFlow)]
        self._function_flows = [f for f in self._flows if isinstance(f, flows.FunctionFlow)]
        self._transition_flows = [
            f
            for f in self._flows
            if issubclass(f.__class__, flows.BaseTransitionFlow)
            and not isinstance(f, flows.FunctionFlow)
        ]

        # Check we didn't miss any flows
        num_split_flows = sum(
            len(fs)
            for fs in [
                self._entry_flows,
                self._exit_flows,
                self._transition_flows,
                self._function_flows,
            ]
        )
        assert len(self._flows) == num_split_flows

        # Prepare to track flow rates for derived outputs
        # We track the times and values of the request flows while integrating, and then
        # interpolate the results to get our flow-based derived outputs.

        # An ordered list of the times that have been tracked.
        self._flow_tracker_times = []
        # An ordered list of values for each flow, these will contain an array of flow rates for each timestep.
        self._transition_flow_values = []
        self._function_flow_values = []
        self._exit_flow_values = []
        self._entry_flow_values = []

        """
        Pre-run calculations to help determine force of infection multiplier at runtime.

        We start with a set of "mixing categories". These categories describe groups of compartments.
        For example, we might have the stratifications age {child, adult} and location {work, home}.
        In this case, the mixing categories would be {child x home, child x work, adult x home, adult x work}.
        Mixing categories are only created when a mixing matrix is supplied during stratification.

        There is a mapping from every compartment to a mixing category.
        This is only true if mixing matrices are supplied only for complete stratifications.
        There are `num_cats` categories and `num_comps` compartments.
        The category matrix is a (num_cats x num_comps) matrix of 0s and 1s, with a 1 when the compartment is in a given category.
        We expect only one category per compartment, but there may be many compartments per category.

        We can multiply the category matrix by the vector of compartment sizes to get the total number of people
        in each mixing category.

        We also create a vector of values in [0, inf) which describes how infectious each compartment is: compartment_infectiousness
        We can use this vector plus the compartment sizes to get the 'effective' number of infectious people per compartment.

        We can use the 'effective infectious' compartment sizes, plus the mixing category matrix
        to get the infected population per mixing category.

        Now that we know:
            - the total population per category
            - the infected population per category
            - the inter-category mixing coefficients (mixing matrix)

        We can calculate the infection density or frequency per category.
        Finally, at runtime, we can lookup which category a given compartment is in and look up its infectious multiplier (density or frequency).
        """
        # Figure out which compartments should be infectious
        self._infectious_mask = [
            c.has_name_in_list(self._infectious_compartments) for c in self.compartments
        ]

        # Find out the relative infectiousness of each compartment, for each strain.
        # If no strains have been created, we assume a default strain name.
        self._compartment_infectiousness = {
            strain_name: self._get_compartment_infectiousness_for_strain(strain_name)
            for strain_name in self._disease_strains
        }

        # Create a matrix that tracks which categories each compartment is in.
        # A matrix with size (num_cats x num_comps).
        num_comps = len(self.compartments)
        num_categories = len(self._mixing_categories)
        self._category_lookup = {}  # Map compartments to categories.
        self._category_matrix = np.zeros((num_categories, num_comps))
        for i, category in enumerate(self._mixing_categories):
            for j, comp in enumerate(self.compartments):
                if all(comp.has_stratum(k, v) for k, v in category.items()):
                    self._category_matrix[i][j] = 1
                    self._category_lookup[j] = i

    def _get_compartment_infectiousness_for_strain(self, strain: str):
        """
        Returns a vector of floats, each representing the relative infectiousness of each compartment.
        If a strain name is provided, find the infectiousness factor *only for that strain*.
        """
        # Figure out which compartments should be infectious
        infectious_mask = np.array(
            [c.has_name_in_list(self._infectious_compartments) for c in self.compartments]
        )
        # Find the infectiousness multipliers for each compartment being implemented in the model.
        # Start from assumption that each compartment is not infectious.
        compartment_infectiousness = np.zeros(self.initial_population.shape)
        # Set all infectious compartments to be equally infectious.
        compartment_infectiousness[infectious_mask] = 1

        # Apply infectiousness adjustments.
        for idx, comp in enumerate(self.compartments):
            inf_value = compartment_infectiousness[idx]
            for strat in self._stratifications:
                for comp_name, adjustments in strat.infectiousness_adjustments.items():
                    for stratum, adjustment in adjustments.items():
                        should_apply_adjustment = adjustment and comp.is_match(
                            comp_name, {strat.name: stratum}
                        )
                        if should_apply_adjustment:
                            # Cannot use time-varying funtions for infectiousness adjustments,
                            # because this is calculated before the model starts running.
                            inf_value = adjustment.get_new_value(inf_value, None)

            compartment_infectiousness[idx] = inf_value

        if strain != self._DEFAULT_DISEASE_STRAIN:
            # Filter out all values that are not in the given strain.
            strain_mask = np.zeros(self.initial_population.shape)
            for idx, compartment in enumerate(self.compartments):
                if compartment.has_stratum("strain", strain):
                    strain_mask[idx] = 1

            compartment_infectiousness *= strain_mask

        return compartment_infectiousness

    def _prepare_time_step(self, time: float, compartment_values: np.ndarray):
        """
        Pre-timestep setup. This should be run before `_get_flow_rates`.
        Here we set up any stateful updates that need to happen before we get the flow rates.
        """
        # Prepare total deaths for tracking deaths.
        self._total_deaths = 0

        # Prepare derived output flow tracking for this timestep.
        self._flow_tracker_times.append(time)
        self._transition_flow_values.append(np.zeros(len(self._transition_flows)))
        self._function_flow_values.append(np.zeros(len(self._function_flows)))
        self._exit_flow_values.append(np.zeros(len(self._exit_flows)))
        self._entry_flow_values.append(np.zeros(len(self._entry_flows)))

        # Find the effective infectious population for the force of infection calculations.
        mixing_matrix = self._get_mixing_matrix(time)

        # Calculate total number of people per category.
        # A vector with size (num_cats x 1).
        comp_values_transposed = compartment_values.reshape((compartment_values.shape[0], 1))
        self._category_populations = np.matmul(self._category_matrix, comp_values_transposed)

        # Calculate infectious populations for each strain.
        # Infection density / frequency is the infectious multiplier for each mixing category, calculated for each strain.
        self._infection_density = {}
        self._infection_frequency = {}
        for strain in self._disease_strains:
            strain_compartment_infectiousness = self._compartment_infectiousness[strain]

            # Calculate total infected people per category, including adjustment factors.
            # Returns a vector with size (num_cats x 1).
            infected_values = compartment_values * strain_compartment_infectiousness
            infected_values_transposed = infected_values.reshape((infected_values.shape[0], 1))
            infectious_populations = np.matmul(self._category_matrix, infected_values_transposed)
            self._infection_density[strain] = np.matmul(mixing_matrix, infectious_populations)

            # Calculate total infected person frequency per category, including adjustment factors.
            # A vector with size (num_cats x 1).
            category_frequency = infectious_populations / self._category_populations
            self._infection_frequency[strain] = np.matmul(mixing_matrix, category_frequency)

    def _get_force_idx(self, source: Compartment):
        """
        Returns the index of the source compartment in the infection multiplier vector.
        """
        return self._category_lookup[source.idx]

    def _get_total_deaths(self, *args, **kwargs) -> float:
        assert self._total_deaths is not None, "Total deaths has not been set."
        return self._total_deaths

    def _get_infection_frequency_multiplier(self, source: Compartment, dest: Compartment) -> float:
        """
        Get force of infection multiplier for a given compartment,
        using the 'infection frequency' calculation.
        """
        idx = self._get_force_idx(source)
        strain = dest.strata.get("strain", self._DEFAULT_DISEASE_STRAIN)
        return self._infection_frequency[strain][idx][0]

    def _get_infection_density_multiplier(self, source: Compartment, dest: Compartment):
        """
        Get force of infection multiplier for a given compartment,
        using the 'infection density' calculation.
        """
        idx = self._get_force_idx(source)
        strain = dest.strata.get("strain", self._DEFAULT_DISEASE_STRAIN)
        return self._infection_density[strain][idx][0]

    def _get_flow_rates(self, compartment_values: np.ndarray, time: float):
        """
        Get net flows into, out of, and between all compartments.
        This function is passed to solve_ode func and defines the dynamics of the model.
        """
        # Zero out compartment sizes to prevent negative values from messing up calcs.
        comp_vals = compartment_values.copy()
        zero_mask = comp_vals < 0
        comp_vals[zero_mask] = 0
        flow_rates = np.zeros(comp_vals.shape)
        self._prepare_time_step(time, comp_vals)
        flow_tracker_idx = len(self._flow_tracker_times) - 1

        # Apply inter-compartmental transition flows.
        for flow_idx, flow in enumerate(self._transition_flows):
            net_flow = flow.get_net_flow(compartment_values, time)
            flow_rates[flow.source.idx] -= net_flow
            flow_rates[flow.dest.idx] += net_flow
            self._transition_flow_values[flow_tracker_idx][flow_idx] = net_flow

        # Apply exit flows: deaths, exports, etc.
        # We apply deaths before births so that we can use 'total deaths' to calculate the birth rate, if required.
        for flow_idx, flow in enumerate(self._exit_flows):
            net_flow = flow.get_net_flow(compartment_values, time)
            flow_rates[flow.source.idx] -= net_flow
            if type(flow) is flows.DeathFlow:
                self._total_deaths += net_flow

            self._exit_flow_values[flow_tracker_idx][flow_idx] = net_flow

        # Apply entry flows: births, imports, etc.
        for flow_idx, flow in enumerate(self._entry_flows):
            net_flow = flow.get_net_flow(compartment_values, time)
            flow_rates[flow.dest.idx] += net_flow
            self._entry_flow_values[flow_tracker_idx][flow_idx] = net_flow

        # Apply inter-compartmental function flows.
        # These take the other flow rates as an input so they are executed last.
        for flow_idx, flow in enumerate(self._function_flows):
            net_flow = flow.get_net_flow(self.compartments, compartment_values, flow_rates, time)
            flow_rates[flow.source.idx] -= net_flow
            flow_rates[flow.dest.idx] += net_flow
            self._function_flow_values[flow_tracker_idx][flow_idx] = net_flow

        return flow_rates

    def _get_mixing_matrix(self, time: float) -> np.ndarray:
        """
        Returns the final mixing matrix for a given time.
        """
        mixing_matrix = self._DEFAULT_MIXING_MATRIX
        for mm_func in self._mixing_matrices:
            # Assume each mixing matrix is either an np.ndarray or a function of time that returns one.
            mm = mm_func(time) if callable(mm_func) else mm_func
            # Get Kronecker product of old and new mixing matrices.
            mixing_matrix = np.kron(mixing_matrix, mm)

        return mixing_matrix

    """
    Requesting and calculating derived outputs
    """
    _FLOW_REQUEST = "flow"
    _COMPARTMENT_REQUEST = "comp"
    _AGGREGATE_REQUEST = "agg"
    _CUMULATIVE_REQUEST = "cum"
    _FUNCTION_REQUEST = "func"

    def _calculate_derived_outputs(self, outputs_to_calculate: Optional[List[str]] = None):
        """
        Calculates all requested derived outputs from the calculated compartment sizes.
        """
        assert self.outputs is not None, "Cannot calculate derived outputs: model has not been run."
        error_msg = "Cannot calculate derived outputs: dependency graph has cycles."
        assert networkx.is_directed_acyclic_graph(self._derived_output_graph), error_msg
        graph = self._derived_output_graph.copy()

        if outputs_to_calculate is not None:
            # Only calculate the required outputs and their dependencies, ignore everything else.
            required_nodes = set()
            for name in outputs_to_calculate:
                # Find a list of the output required and its dependencies.
                output_dependencies = networkx.dfs_tree(graph.reverse(), source=name).reverse()
                required_nodes = required_nodes.union(output_dependencies.nodes)

            # Remove any nodes that aren't required from the graph.
            for node in graph.nodes:
                if not node in required_nodes:
                    graph.remove_node(node)

        derived_outputs = {}
        outputs_to_delete_after = []
        flow_types = [
            self._transition_flows,
            self._function_flows,
            self._exit_flows,
            self._entry_flows,
        ]
        flow_tracker_values = [
            self._transition_flow_values,
            self._function_flow_values,
            self._exit_flow_values,
            self._entry_flow_values,
        ]
        # Convert tracked flow values into a matrix where the 1st dimension is flow type, 2nd is time
        for i in range(len(flow_tracker_values)):
            flow_tracker_values[i] = np.array(flow_tracker_values[i]).T

        # Calculate all the outputs in the correct order so that each output has its dependencies fulfilled.
        for name in networkx.topological_sort(graph):
            request = self._derived_output_requests[name]
            request_type = request["request_type"]
            output = np.zeros(self.times.shape)

            if not request["save_results"]:
                # Delete the results of this output once the calcs are done.
                outputs_to_delete_after.append(name)

            if request_type == self._FLOW_REQUEST:
                # User wants to track a set of flow rates over time.
                times = self._flow_tracker_times
                values = np.zeros(len(times))
                for flows, flow_values in zip(flow_types, flow_tracker_values):
                    for flow_idx, flow in enumerate(flows):
                        is_matching_flow = (
                            flow.name == request["flow_name"]
                            and (
                                (not flow.source)
                                or flow.source.has_strata(request["source_strata"])
                            )
                            and ((not flow.dest) or flow.dest.has_strata(request["dest_strata"]))
                        )
                        if is_matching_flow:
                            values += flow_values[flow_idx]

                # Build a function to produce interpolated results
                solved_func = interp1d(times, values)
                # Populate output array with interpolated results
                num_times = len(self.times)
                interpolated_output = np.zeros(self.times.shape)
                for time_idx in range(num_times):
                    interpolated_output[time_idx] = solved_func(self.times[time_idx])

                use_raw_results = request["raw_results"]
                if use_raw_results:
                    # Use interpolated flow rates wiuth no post-processing.
                    output = interpolated_output
                else:
                    # Set the "flow rate" at time `t` to be an estimate of the flow rate
                    # that is calculated at time `t-1`. By convention, flows are zero at t=0.
                    # This is done so that we can estimate the number of people moving between compartments
                    # using tracked flow rates.
                    ignore_first_timestep_output = np.zeros(self.times.shape)
                    ignore_first_timestep_output[1:] = interpolated_output[1:]
                    offset_output = np.zeros(self.times.shape)
                    offset_output[1:] = interpolated_output[:-1]
                    output = (offset_output + ignore_first_timestep_output) / 2

            elif request_type == self._COMPARTMENT_REQUEST:
                # User wants to track a set of compartment sizes over time.
                compartments = request["compartments"]
                strata = request["strata"]
                comps = (
                    (i, c)
                    for i, c in enumerate(self.compartments)
                    if c.has_name_in_list(compartments)
                )
                idxs = [i for i, c in comps if c.is_match(c.name, strata)]
                output = self.outputs[:, idxs].sum(axis=1)

            elif request_type == self._AGGREGATE_REQUEST:
                # User wants to track the sum of a set of outputs over time.
                source_names = request["sources"]
                output = sum([derived_outputs[s] for s in source_names])

            elif request_type == self._CUMULATIVE_REQUEST:
                # User wants to track cumulative value of an output over time.
                source_name = request["source"]
                start_time = request["start_time"]
                max_time = self.times.max()
                if start_time and start_time > max_time:
                    # Handle case where the derived output starts accumulating after the last model timestep.
                    msg = f"Cumulative output '{name}' start time {start_time} is greater than max model time {max_time}, defaulting to {max_time}"
                    logger.warn(msg)
                    start_time = max_time

                if start_time is None:
                    output = np.cumsum(derived_outputs[source_name])
                else:
                    assert (
                        start_time in self.times
                    ), f"Start time {start_time} not in times for '{name}'"
                    start_idx = np.where(self.times == start_time)[0][0]
                    output[start_idx:] = np.cumsum(derived_outputs[source_name][start_idx:])

            elif request_type == self._FUNCTION_REQUEST:
                # User wants to track the results of a function of other outputs over time.
                func = request["func"]
                source_names = request["sources"]
                inputs = [derived_outputs[s] for s in source_names]
                output = func(*inputs)

            derived_outputs[name] = output

        # Delete any intermediate outputs that we don't want to save.
        for name in outputs_to_delete_after:
            del derived_outputs[name]

        return derived_outputs

    def request_output_for_flow(
        self,
        name: str,
        flow_name: str,
        source_strata: Optional[Dict[str, str]] = None,
        dest_strata: Optional[Dict[str, str]] = None,
        save_results: bool = True,
        raw_results: bool = False,
    ):
        """
        Adds a derived output to the model's results. The output
        will be the value of the requested flow at the at each timestep.

        Args:
            name: The name of the derived output.
            flow_name: The name of the flow to track.
            source_strata (optional): A whitelist of strata to filter the source compartments.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            save_results (optional): Whether to save or discard the results. Defaults to ``True``.
            raw_results (optional): Whether to use raw interpolated flow rates, or post-process them so that they're more
                represenative of the changes in compartment sizes. Defaults to ``False``.
        """
        source_strata = source_strata or {}
        dest_strata = dest_strata or {}
        msg = f"A derived output named {name} already exists."
        assert name not in self._derived_output_requests, msg
        is_flow_exists = any(
            [f.is_match(flow_name, source_strata, dest_strata) for f in self._flows]
        )
        assert is_flow_exists, f"No flow matches: {flow_name} {source_strata} {dest_strata}"
        self._derived_output_graph.add_node(name)
        self._derived_output_requests[name] = {
            "request_type": self._FLOW_REQUEST,
            "flow_name": flow_name,
            "source_strata": source_strata,
            "dest_strata": dest_strata,
            "raw_results": raw_results,
            "save_results": save_results,
        }

    def request_output_for_compartments(
        self,
        name: str,
        compartments: List[str],
        strata: Optional[Dict[str, str]] = None,
        save_results: bool = True,
    ):
        """
        Adds a derived output to the model's results. The output
        will be the aggregate population of the requested compartments at the at each timestep.

        Args:
            name: The name of the derived output.
            compartments: The name of the compartments to track.
            strata (optional): A whitelist of strata to filter the compartments.
            save_results (optional): Whether to save or discard the results.
        """
        strata = strata or {}
        msg = f"A derived output named {name} already exists."
        assert name not in self._derived_output_requests, msg
        is_match_exists = any(
            [any([c.is_match(name, strata) for name in compartments]) for c in self.compartments]
        )
        assert is_match_exists, f"No compartment matches: {compartments} {strata}"
        self._derived_output_graph.add_node(name)
        self._derived_output_requests[name] = {
            "request_type": self._COMPARTMENT_REQUEST,
            "compartments": compartments,
            "strata": strata,
            "save_results": save_results,
        }

    def request_aggregate_output(
        self,
        name: str,
        sources: List[str],
        save_results: bool = True,
    ):
        """
        Adds a derived output to the model's results. The output will be the aggregate of other derived outputs.

        Args:
            name: The name of the derived output.
            sources: The names of the derived outputs to aggregate.
            save_results (optional): Whether to save or discard the results.

        """
        msg = f"A derived output named {name} already exists."
        assert name not in self._derived_output_requests, msg
        for source in sources:
            assert (
                source in self._derived_output_requests
            ), f"Source {source} has not been requested."
            self._derived_output_graph.add_edge(source, name)

        self._derived_output_graph.add_node(name)
        self._derived_output_requests[name] = {
            "request_type": self._AGGREGATE_REQUEST,
            "sources": sources,
            "save_results": save_results,
        }

    def request_cumulative_output(
        self,
        name: str,
        source: str,
        start_time: int = None,
        save_results: bool = True,
    ):
        """
        Adds a derived output to the model's results. The output will be the cumulative value
        of another derived outputs over the model's time period.

        Args:
            name: The name of the derived output.
            source: The name of the derived outputs to accumulate.
            start_time (optional): The time to start accumulating from, defaults to model start time.
            save_results (optional): Whether to save or discard the results.

        """
        msg = f"A derived output named {name} already exists."
        assert name not in self._derived_output_requests, msg
        assert source in self._derived_output_requests, f"Source {source} has not been requested."
        self._derived_output_graph.add_node(name)
        self._derived_output_graph.add_edge(source, name)
        self._derived_output_requests[name] = {
            "request_type": self._CUMULATIVE_REQUEST,
            "source": source,
            "start_time": start_time,
            "save_results": save_results,
        }

    def request_function_output(
        self,
        name: str,
        func: Callable[[np.ndarray], np.ndarray],
        sources: List[str],
        save_results: bool = True,
    ):
        """
        Adds a derived output to the model's results. The output will be the result of a function
        which takes a list of sources as an input.

        Args:
            name: The name of the derived output.
            func: A function used to calculate the derived ouput.
            sources: The derived ouputs to input into the function.
            save_results (optional): Whether to save or discard the results.

        Example:
            Request a function-based derived ouput::

                model.request_output_for_compartments(
                    compartments=["S", "E", "I", "R"],
                    name="total_population",
                    save_results=False
                )
                model.request_output_for_compartments(
                    compartments=["R"],
                    name="recovered_population",
                     save_results=False
                )

                def calculate_proportion_seropositive(recovered_pop, total_pop):
                    return recovered_pop / total_pop

                model.request_function_output(
                    name="proportion_seropositive",
                    func=calculate_proportion_seropositive,
                    sources=["recovered_population", "total_population"],
                )

        """
        msg = f"A derived output named {name} already exists."
        assert name not in self._derived_output_requests, msg
        for source in sources:
            assert (
                source in self._derived_output_requests
            ), f"Source {source} has not been requested."
            self._derived_output_graph.add_edge(source, name)

        self._derived_output_graph.add_node(name)
        self._derived_output_requests[name] = {
            "request_type": self._FUNCTION_REQUEST,
            "func": func,
            "sources": sources,
            "save_results": save_results,
        }
