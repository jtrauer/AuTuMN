import numpy as np

from autumn.curve import scale_up_function

from apps.covid_19.model.parameters import Mobility, MixingType

from .adjust_base import BaseMixingAdjustment


class AgeMixingAdjustment(BaseMixingAdjustment):
    def __init__(self, mobility: Mobility):
        """Build the time variant age adjustment functions"""
        assert mobility.mixing_type == MixingType.age
        self.adjustment_funcs = {}
        mixing = mobility.mixing
        for age_idx, timeseries in mixing.items():
            func = scale_up_function(timeseries["times"], timeseries["values"], method=4)
            self.adjustment_funcs[str(age_idx)] = func

    def get_adjustment(self, time: float, mixing_matrix: np.ndarray) -> np.ndarray:
        """
        Apply time-varying age adjustments.
        Returns a new mixing matrix, modified to adjust for dynamic mixing changes for a given point in time.
        """
        # Iterate over matrix rows and columns
        new_mm = mixing_matrix.copy()
        rows, cols = new_mm.shape
        for i in range(rows):
            row_agegroup = AGE_GROUPS[i]
            row_adjust_func = self.adjustment_funcs.get(row_agegroup)
            row_multiplier = row_adjust_func(time) if row_adjust_func else 1
            for j in range(cols):
                col_agegroup = AGE_GROUPS[j]
                col_adjust_func = self.adjustment_funcs.get(col_agegroup)
                col_multiplier = col_adjust_func(time) if col_adjust_func else 1
                new_mm[i, j] *= row_multiplier * col_multiplier

        return new_mm


AGE_GROUPS = [
    "0",
    "5",
    "10",
    "15",
    "20",
    "25",
    "30",
    "35",
    "40",
    "45",
    "50",
    "55",
    "60",
    "65",
    "70",
    "75",
]
