# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing class for recalibrating blended probabilities.

.. See the documentation for a more detailed discussion of this plugin.
.. include:: extended_documentation/calibration/beta_recalibration/
   beta_recalibration.rst
"""

from copy import deepcopy
from typing import Any, Dict

import cf_units
import iris
import numpy as np
from numpy import ndarray
from iris.exceptions import CoordinateNotFoundError
from scipy.stats import beta

from improver import PostProcessingPlugin
from improver.metadata.probabilistic import (
    is_probability,
    get_threshold_coord_name_from_probability_name,
)


class BetaRecalibrate(PostProcessingPlugin):
    """Recalibrate probabilities using the cumulative distribution function
    of the beta distribution.
    """

    def __init__(self, recalibration_dict: Dict[str, Any]):
        """
        Args:
            recalibration_dict:
                Dictionary from which to calculate alpha and beta parameters for
                recalibrating blended output using the beta distribution. Dictionary
                format is as specified below. The same set of thresholds must be specified
                for each forecast period. Weights are interpolated from the
                forecast periods and thresholds specified in the cube to the forecast periods
                and thresholds of the cube. Weights are first be interpolated separately
                for each specified forecast period over the threhsold coordinate, then interpolated
                over the forecast period coordinate.

        Recalibration dictionary format::
            {
                "forecast_period": {
                    "24": {
                        "threshold": {
                            "0": {"alpha": 1.14, "beta": 1.23},
                            "0.1": {"alpha": 1.14, "beta": 1.23},
                            "0.5": {"alpha": 1.14, "beta": 1.23},
                        }
                    },
                    "48": {
                        "threshold": {
                            "0": {"alpha": 1.14, "beta": 1.23},
                            "0.1": {"alpha": 1.14, "beta": 1.23},
                            "0.5": {"alpha": 1.14, "beta": 1.23},
                        }
                    },
                },
                "forecast_period_units": "hours",
                "threshold_units": "mm",
            }

        The "forecast_period_units" and "threshold_units" keys are optional.
        If they are omitted, it is assumed that the units are the same as those
        used in forecast_period and threshold coordinates of the input cube.
        """
        self.recalibration_dict = recalibration_dict

    def process(self, cube):
        """Recalibrate cube using the beta distribution with the alpha
        and beta parameters specified in self.recalibration_dict.

        Args:
            cube:
                A cube containing a forecast_period coordinate.

        Returns:
            A cube having the same dimensions as the input, with data
            transformed by the beta distribution cdf.

        Raises:
            CoordinateNotFoundError: if cube does not contain probability data
            CoordinateNotFoundError: if cube does not contain forecast_period coordinate
            RuntimeError: if any interpolated values of alpha or beta are <= 0
        """

        if not (is_probability(cube)):
            raise CoordinateNotFoundError(
                "Input cube must be a probability forecast "
                "and contain a threshold coordinate."
            )
        if len(cube.coords("forecast_period")) == 0:
            raise CoordinateNotFoundError(
                "Recalibration input must contain forecast_period coordinate."
            )
        fps_string = list(self.recalibration_dict["forecast_period"].keys())
        curr_thresholds = None
        thresholds = sorted(
            list(self.recalibration_dict["forecast_period"][fp[0]]["thresholds"].keys())
        )
        for fp_string in fps_string[1:]:
            curr_thresholds = sorted(
                list(
                    self.recalibration_dict["forecast_period"][fp[0]][
                        "thresholds"
                    ].keys()
                )
            )
            if curr_thresholds != thresholds:
                raise ValueError(
                    "Thresholds must be the same for all forecast periods."
                )

        fps = [int(fp_string) for fp_string in fps_string]
        threshold_coord_name = get_threshold_coord_name_from_probability_name(
            cube.name()
        )
        cube_thresholds = cube.coord(threshold_coord_name).points
        alpha_arr = []
        beta_arr = []
        for fp_string in fps_string:
            # interpolate thresholds to cube thresholds
            recal_thresholds = []
            curr_alpha_arr = []
            curr_beta_arr = []
            for t_string in recalibration_dict[fp_string]:
                t = float(t_string)
                if "threshold_units" in self.recalibration_dict.keys():
                    # convert threshold points to cube units
                    units = cf_units.Unit(self.recalibration_dict["threshold_units"])
                    t = units.convert(t, cube.coord(threshold_coord_name).units)
                recal_thresholds.append(t)
                curr_alpha_arr.append(
                    self.recalibration_dict[fp_string][t_string]["alpha"]
                )
                curr_beta_arr.append(
                    self.recalibration_dict[fp_string][t_string]["beta"]
                )

            # sort alpha and beta by threshold and interpolate to cube thresholds
            sort_ind = np.argsort(recal_thresholds)
            alpha_arr.append(
                np.interp(cube_thresholds, thresholds, curr_alpha_arr[sort_ind])
            )
            beta_arr.append(
                np.interp(cube_thresholds, thresholds, curr_beta_arr[sort_ind])
            )

        # stack alpha and beta arrays in order of forecast period
        if "forecast_period_units" in self.recalibration_dict.keys():
            # convert forecast period to cube units
            units = cf_units.Unit(self.recalibration_dict["forecast_period_units"])
            fps = [units.convert(fp, cube.coord("forecast_period").units) for fp in fps]
        sort_ind = np.argsort(fps)
        fps = fps[sort_ind]
        alpha_arr = np.vstack(alpha_arr[sort_ind])
        beta_arr = np.vstack(beta_arr[sort_ind])

        # interpolate to cube forecast periods
        alpha_arr = np.hstack(
            [
                np.interp(cube_forecast_period, fps, alpha_arr[:, [i]])
                for i in range(alpha_arr.shape[1])
            ]
        )
        beta_arr = np.hstack(
            [
                np.interp(cube_forecast_period, fps, beta_arr[:, [i]])
                for i in range(beta_arr.shape[1])
            ]
        )
        if np.any(alpha_arr <= 0) or np.any(beta_arr <= 0):
            raise RuntimeError("interpolated alpha and beta parameters must be > 0")

        # nested slicing so we guarantee the order of slices
        cubelist = iris.cube.CubeList([])
        for i, fp_slice in enumerate(cube.slices_over("forecast_period")):
            for j, t_slice in fp_slice.slices_over(threshold_coord_name):
                t_slice.data = (
                    beta(alpha_arr[i][j], beta_arr[i][j])
                    .cdf(t_slice.data)
                    .astype(np.float32)
                )
                cubelist.append(t_slice)

        return cubelist.merge_cube()
