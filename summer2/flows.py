"""
This module contains the classes which are used to calculate inter-compartmental flow rates.
As a user of the framework you should not have to use these classes directly.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Callable

import numpy as np
from numba import jit

from summer2.adjust import BaseAdjustment, FlowParam, Multiply, Overwrite
from summer2.stratification import Stratification
from summer2.compartment import Compartment


class BaseFlow(ABC):
    """
    :meta private:
    Abstract base class for all flows.
    A flow represents the movement of people from one compartment to another.
    """

    name = None
    source = None
    dest = None
    param = None
    adjustments = None

    def _is_equal(self, flow):
        """For testing"""
        return (
            type(self) is type(flow)
            and self.name == flow.name
            and self.source == flow.source
            and self.dest == flow.dest
            and self.param == flow.param
            and len(self.adjustments) == len(flow.adjustments)
            and all(
                [
                    self.adjustments[i]._is_equal(flow.adjustments[i])
                    for i in range(len(self.adjustments))
                ]
            )
        )

    def is_match(self, name: str, source_strata: dict, dest_strata: dict) -> bool:
        """
        Returns True if the flow matches the given name and strata.
        """
        return (
            self.name == name
            and ((not source_strata) or (not self.source) or self.source.has_strata(source_strata))
            and ((not dest_strata) or (not self.dest) or self.dest.has_strata(dest_strata))
        )

    def get_weight_value(self, time: float):
        """
        Returns the flow's weight at a given time.
        Applies any stratification adjustments to the base parameter.
        """
        flow_rate = self.param(time) if callable(self.param) else self.param
        for adjustment in self.adjustments:
            flow_rate = adjustment.get_new_value(flow_rate, time)

        return flow_rate

    def update_compartment_indices(self, mapping: Dict[str, float]):
        """
        Update index which maps flow compartments to compartment value array.
        """
        if self.source:
            self.source.idx = mapping[self.source]
        if self.dest:
            self.dest.idx = mapping[self.dest]

    def optimize_adjustments(self):
        """
        Rearrange adjustments so that they produce the same result but run faster.
        This does not seem to actually impact runtime much.
        """
        # Start from the last Overwrite, no point calculating anything before that.
        last_overwrite_idx = None
        for idx in reversed(range(len(self.adjustments))):
            adj = self.adjustments[idx]
            if type(adj) is Overwrite:
                last_overwrite_idx = idx
                break

        if last_overwrite_idx:
            new_adjustments = self.adjustments[last_overwrite_idx:]
        else:
            new_adjustments = self.adjustments

        # Combine all constant multiplications into one constant.
        overwrites = [a for a in new_adjustments if type(a) is Overwrite]
        consts = [a for a in new_adjustments if type(a) is Multiply and not callable(a.param)]
        funcs = [a for a in new_adjustments if type(a) is Multiply and callable(a.param)]

        if consts:
            # Reduce multiple constants into one.
            const = np.prod([a.param for a in consts])
            consts = [Multiply(const)]

        self.adjustments = overwrites + consts + funcs

    @abstractmethod
    def get_net_flow(
        self,
        time: float,
        compartments: List[Compartment],
        compartment_values: np.ndarray,
        flows: list,
        flow_rates: np.ndarray,
    ) -> float:
        """
        Returns the net flow value at a given time.
        """
        pass

    @abstractmethod
    def stratify(self, strat: Stratification) -> list:
        """
        Returns a list of new, stratified flows to replace the current flow.
        """
        pass

    def copy(self, **kwargs):
        """
        Creates a modified copy of the flow for stratification.
        """
        return self.__class__(**kwargs)

    @abstractmethod
    def __repr__(self):
        """
        Returns a text representation of the flow.
        """
        pass


class BaseEntryFlow(BaseFlow):
    """
    :meta private:
    A flow where people enter the destination compartment, but there is no source.
    Eg. births, importation.
    """

    _is_birth_flow = False

    def __init__(
        self,
        name: str,
        dest: Compartment,
        param: FlowParam,
        adjustments: List[BaseAdjustment] = None,
    ):
        assert type(dest) is Compartment
        self.name = name
        self.adjustments = [a for a in (adjustments or []) if a and a.param is not None]
        self.dest = dest
        self.param = param

    def stratify(self, strat: Stratification) -> List[BaseFlow]:
        """
        Returns a list of new, stratified entry flows to replace the current flow.
        """
        if not self.dest.has_name_in_list(strat.compartments):
            # Flow destination is not stratified, do not stratify this flow.
            return [self]

        new_flows = []
        flow_adjustments = strat.get_flow_adjustment(self)
        is_birth_into_agegroup_flow = self._is_birth_flow and strat.is_ageing()

        msg = "Cannot adjust birth flows into age stratifications."
        assert not (is_birth_into_agegroup_flow and flow_adjustments), msg

        msg = f"Flow {self.name} has missing adjustments for {strat.name} strat."
        assert not (flow_adjustments and set(flow_adjustments.keys()) != set(strat.strata)), msg

        for stratum in strat.strata:
            new_adjustments = [*self.adjustments]
            if is_birth_into_agegroup_flow:
                # Use special rules for birth into age groups.
                if stratum != "0":
                    # Babies get born at age 0, and not at any other age. Skip!
                    continue
            else:
                # Not an ageing stratification, check for user-specified flow adjustments.
                if flow_adjustments:
                    strata_flow_adjustment = flow_adjustments.get(stratum)
                    if strata_flow_adjustment:
                        new_adjustments.append(strata_flow_adjustment)
                else:
                    # No adjustments specified for this flow.
                    # Default to equally dividing entry population between all strata.
                    entry_fraction = 1.0 / len(strat.strata)
                    new_adjustments.append(Multiply(entry_fraction))

            new_dest = self.dest.stratify(strat.name, stratum)
            new_flow = self.copy(
                name=self.name,
                dest=new_dest,
                param=self.param,
                adjustments=new_adjustments,
            )
            new_flows.append(new_flow)

        return new_flows

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.name}' to {self.dest}>"


class BaseExitFlow(BaseFlow):
    """
    :meta private:
    A flow where people exit the source compartment, but there is no destination
    Eg. deaths, emigration
    """

    def __init__(
        self,
        name: str,
        source: Compartment,
        param: FlowParam,
        adjustments: List[BaseAdjustment] = None,
    ):
        assert type(source) is Compartment
        self.name = name
        self.adjustments = [a for a in (adjustments or []) if a and a.param is not None]
        self.source = source
        self.param = param

    def stratify(self, strat: Stratification) -> List[BaseFlow]:
        """
        Returns a list of new, stratified exit flows to replace the current flow.
        """
        if not self.source.has_name_in_list(strat.compartments):
            # Flow source is not stratified, do not stratify this flow.
            return [self]

        flow_adjustments = strat.get_flow_adjustment(self)

        msg = f"Flow {self.name} has missing adjustments for {strat.name} strat."
        assert not (flow_adjustments and set(flow_adjustments.keys()) != set(strat.strata)), msg

        new_flows = []
        for stratum in strat.strata:
            new_adjustments = [*self.adjustments]
            # Check for user-specified flow adjustments.
            if flow_adjustments:
                strata_flow_adjustment = flow_adjustments.get(stratum)
                if strata_flow_adjustment:
                    new_adjustments.append(strata_flow_adjustment)

            new_source = self.source.stratify(strat.name, stratum)
            new_flow = self.copy(
                name=self.name,
                source=new_source,
                param=self.param,
                adjustments=new_adjustments,
            )
            new_flows.append(new_flow)

        return new_flows

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.name}' from {self.source}>"


class BaseTransitionFlow(BaseFlow):
    """
    :meta private:
    A flow where people move from the source compartment, to the destination.
    Eg. infection, recovery, progress of disease.
    """

    def __init__(
        self,
        name: str,
        source: Compartment,
        dest: Compartment,
        param: FlowParam,
        adjustments: List[BaseAdjustment] = None,
    ):
        assert type(source) is Compartment
        assert type(dest) is Compartment
        self.name = name
        self.adjustments = [a for a in (adjustments or []) if a and a.param is not None]
        self.source = source
        self.dest = dest
        self.param = param

    def stratify(self, strat: Stratification) -> List[BaseFlow]:
        """
        Returns a list of new, stratified flows to replace the current flow.
        """
        is_source_compartment_stratified = self.source.has_name_in_list(strat.compartments)
        is_dest_compartment_stratified = self.dest.has_name_in_list(strat.compartments)
        if not (is_dest_compartment_stratified or is_source_compartment_stratified):
            # Flow is not stratified, do not stratify this flow.
            return [self]

        new_flows = []
        flow_adjustments = strat.get_flow_adjustment(self)

        msg = f"Flow {self.name} has missing adjustments for {strat.name} strat."
        assert not (flow_adjustments and set(flow_adjustments.keys()) != set(strat.strata)), msg

        for stratum in strat.strata:
            # Find new compartments
            if is_source_compartment_stratified:
                new_source = self.source.stratify(strat.name, stratum)
            else:
                new_source = self.source

            if is_dest_compartment_stratified:
                new_dest = self.dest.stratify(strat.name, stratum)
            else:
                new_dest = self.dest

            # Find flow adjustments to apply to the new stratified flows.

            # There are three scenarios to consider here:
            # - Both the source and destination are stratified.
            # - The flow source has the required stratifications and the destination does not.
            #   For example - people recovering from I -> R with multiple I strata, all with different recovery rates.
            # - The destination has the required stratifications and the source does not.
            #   For example - people recovering from I -> R with multiple R strata, with different recovery proportions.
            new_adjustments = [*self.adjustments]

            # Should we apply an adjustment to conserve the flow rate?
            should_apply_conservation_split = (
                # If the destination has been stratified by the source hasn't been.
                (is_dest_compartment_stratified and not is_source_compartment_stratified)
                # Don't conserve flow rates for disease strain stratifications.
                and (not strat.is_strain())
                # Don't do this if the user has specified adjustments.
                and (not flow_adjustments)
            )
            if should_apply_conservation_split:
                # If the source is stratified but not the destination, then we need to account
                # for the resulting fan-out of flows by reducing the flow rate.
                # We don't do this for strains because this effect is already
                # captured by the infectiousness multiplier.
                entry_fraction = 1.0 / len(strat.strata)
                new_adjustments.append(Multiply(entry_fraction))
            elif flow_adjustments:
                # Use user-specified flow adjustments.
                strata_flow_adjustment = flow_adjustments.get(stratum)
                new_adjustments.append(strata_flow_adjustment)

            new_flow = self.copy(
                name=self.name,
                source=new_source,
                dest=new_dest,
                param=self.param,
                adjustments=new_adjustments,
            )
            new_flows.append(new_flow)

        return new_flows

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.name}' from {self.source} to {self.dest}>"


class CrudeBirthFlow(BaseEntryFlow):
    """
    A flow that calculates births using a 'crude birth rate' method.
    The number of births will be determined by the product of the birth rate and total population.

    Args:
        name: The flow name.
        dest: The destination compartment.
        param: The fraction of the population to be born per timestep.
        adjustments: Adjustments to the flow rate.

    """

    _is_birth_flow = True

    def get_net_flow(self, compartment_values, time):
        parameter_value = self.get_weight_value(time)
        total_population = _find_sum(compartment_values)
        return parameter_value * total_population


class ReplacementBirthFlow(BaseEntryFlow):
    """
    A flow that calculates births by replacing total deaths.
    The total number of deaths will be supplied by the parameter function.

    Args:
        name: The flow name.
        dest: The destination compartment.
        param: The total number of deaths per timestep.
        adjustments: Adjustments to the flow rate.

    """

    _is_birth_flow = True

    def get_net_flow(
        self,
        time: float,
        compartments: List[Compartment],
        compartment_values: np.ndarray,
        flows: List[BaseFlow],
        flow_rates: np.ndarray,
    ) -> float:
        return self.get_weight_value(time)


# Use Numba to speed up the calculation of the population.
@jit(nopython=True)
def _find_sum(compartment_values: np.ndarray) -> float:
    return compartment_values.sum()


class ImportFlow(BaseEntryFlow):
    """
    Calculates importation, where people enter the destination compartment from outside the system.
    The number of people imported per timestep is independent of the population.

    Args:
        name: The flow name.
        dest: The destination compartment.
        param: The number of people to be imported per timestep.
        adjustments: Adjustments to the flow rate.

    """

    _is_birth_flow = False

    def get_net_flow(
        self,
        time: float,
        compartments: List[Compartment],
        compartment_values: np.ndarray,
        flows: List[BaseFlow],
        flow_rates: np.ndarray,
    ) -> float:
        return self.get_weight_value(time)


class DeathFlow(BaseExitFlow):
    """
    A flow representing deaths. calculated from a fractional death rate.

    Args:
        name: The flow name.
        dest: The destination compartment.
        param: The fraction of the population who die per timestep.
        adjustments: Adjustments to the flow rate.

    """

    def get_net_flow(
        self,
        time: float,
        compartments: List[Compartment],
        compartment_values: np.ndarray,
        flows: List[BaseFlow],
        flow_rates: np.ndarray,
    ) -> float:
        parameter_value = self.get_weight_value(time)
        population = compartment_values[self.source.idx]
        flow_rate = parameter_value * population
        return flow_rate


class FractionalFlow(BaseTransitionFlow):
    """
    A flow that transfers people from a source to a destination based on
    the population of the source compartment and the fractional flow rate.

    Args:
        name: The flow name.
        source: The source compartment.
        dest: The destination compartment.
        param: The fraction of the source compartment to transfer per timestep.
        adjustments: Adjustments to the flow rate.

    """

    def get_net_flow(
        self,
        time: float,
        compartments: List[Compartment],
        compartment_values: np.ndarray,
        flows: List[BaseFlow],
        flow_rates: np.ndarray,
    ) -> float:
        parameter_value = self.get_weight_value(time)
        population = compartment_values[self.source.idx]
        return parameter_value * population


class SojournFlow(BaseTransitionFlow):
    """
    A flow that transfers people from a source to a destination based on
    the population of the source compartment and the sojourn time.

    Args:
        name: The flow name.
        source: The source compartment.
        dest: The destination compartment.
        param: The mean time spent in the compartment by n occupant.
        adjustments: Adjustments to the flow rate.

    """

    def get_net_flow(
        self,
        time: float,
        compartments: List[Compartment],
        compartment_values: np.ndarray,
        flows: List[BaseFlow],
        flow_rates: np.ndarray,
    ) -> float:
        parameter_value = self.get_weight_value(time)
        population = compartment_values[self.source.idx]
        return population / parameter_value


class FunctionFlow(BaseTransitionFlow):
    """
    A flow that transfers people from a source to a destination based on a user-defined function.
    This can be used to define more complex flows if requird.

    Args:
        name: The flow name.
        source: The source compartment.
        dest: The destination compartment.
        param: A function that returns the flow rate, before adjustments. See `get_net_flow` for this function's arguments.
        adjustments: Adjustments to the flow rate.

    """

    def get_net_flow(
        self,
        time: float,
        compartments: List[Compartment],
        compartment_values: np.ndarray,
        flows: List[BaseFlow],
        flow_rates: np.ndarray,
    ) -> float:
        flow_rate = self.param(time, compartments, compartment_values, flows, flow_rates)
        for adjustment in self.adjustments:
            flow_rate = adjustment.get_new_value(flow_rate, time)

        return flow_rate


class BaseInfectionFlow(BaseTransitionFlow):
    def __init__(
        self,
        name: str,
        source: Compartment,
        dest: Compartment,
        param: FlowParam,
        find_infectious_multiplier: Callable[[Compartment, Compartment], float],
        adjustments: List[BaseAdjustment] = None,
    ):
        assert type(source) is Compartment
        assert type(dest) is Compartment
        self.name = name
        self.adjustments = adjustments or []
        self.source = source
        self.dest = dest
        self.param = param
        self.find_infectious_multiplier = find_infectious_multiplier

    def get_net_flow(
        self,
        time: float,
        compartments: List[Compartment],
        compartment_values: np.ndarray,
        flows: List[BaseFlow],
        flow_rates: np.ndarray,
    ) -> float:
        multiplier = self.find_infectious_multiplier(self.source, self.dest)
        parameter_value = self.get_weight_value(time)
        population = compartment_values[self.source.idx]
        return parameter_value * population * multiplier

    def copy(self, **kwargs):
        """
        Creates a modified copy of the flow for stratification.
        """
        return self.__class__(
            **kwargs,
            find_infectious_multiplier=self.find_infectious_multiplier,
        )


class InfectionDensityFlow(BaseInfectionFlow):
    """
    An infection flow that should use the density of infectious people to calculate the force of infection factor.
    """


class InfectionFrequencyFlow(BaseInfectionFlow):
    """
    An infection flow that should use the frequency of infectious people to calculate the force of infection factor.
    """
