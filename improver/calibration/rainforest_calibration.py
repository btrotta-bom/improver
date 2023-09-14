# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""RainForests calibration Plugins.

.. Further information is available in:
.. include:: extended_documentation/calibration/rainforests_calibration/
   rainforests_calibration.rst

"""


from pathlib import Path
from typing import Dict, List, Tuple

import iris
import numpy as np
from cf_units import Unit
from iris.analysis import MEAN
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from numpy import ndarray

from improver import PostProcessingPlugin
from improver.constants import MINUTES_IN_HOUR, SECONDS_IN_MINUTE
from improver.ensemble_copula_coupling.utilities import (
    get_bounds_of_distribution,
    interpolate_multiple_rows_same_x,
)
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_manipulation import add_coordinate_to_cube, compare_coords


class ApplyRainForestsCalibration(PostProcessingPlugin):
    """Generic class to calibrate input forecast via RainForests.

    The choice of tree-model library is determined from package availability, and whether
    all required models files are available. Treelite is the preferred option, defaulting
    to lightGBM if requirements are missing.
    """

    def __new__(cls, model_config_dict: dict, threads: int = 1, bin_data: bool = False):
        """Initialise class object based on package and model file availability.

        Args:
            model_config_dict:
                Dictionary containing Rainforests model configuration variables.
            threads:
                Number of threads to use during prediction with tree-model objects.
            bin_data:
                Bin data according to splits used in models. This speeds up prediction
                if there are many data points which fall into the same bins for all models.

        Dictionary is of format::

        {
        "24": {
            "0.000010": {
                "lightgbm_model": "<path_to_lightgbm_model_object>",
                "treelite_model": "<path_to_treelite_model_object>"
            },
            "0.000050": {
                "lightgbm_model": "<path_to_lightgbm_model_object>",
                "treelite_model": "<path_to_treelite_model_object>"
            },
            "0.000100": {
                "lightgbm_model": "<path_to_lightgbm_model_object>",
                "treelite_model": "<path_to_treelite_model_object>"
            },
        }

        The keys specify the lead times and model threshold values, while the
        associated values are the path to the corresponding tree-model objects
        for that lead time and threshold.

        Treelite predictors are used if treelite_runtime is an installed dependency
        and an associated path has been provided for all thresholds, otherwise lightgbm
        Boosters are used as the default tree model type.
        """
        try:
            # Use treelite class, unless subsequent conditions fail.
            cls = ApplyRainForestsCalibrationTreelite
            # Try and initialise the treelite_runtime library to test if the package
            # is available.
            import treelite_runtime  # noqa: F401

            # Check that all required files have been specified.
            treelite_model_filenames = []
            for lead_time in model_config_dict.keys():
                threshold_keys = [
                    t
                    for t in model_config_dict[lead_time]
                    if t != "combined_feature_splits"
                ]
                for threshold in threshold_keys:
                    treelite_model_filenames.append(
                        model_config_dict[lead_time][threshold].get("treelite_model")
                    )
            if None in treelite_model_filenames:
                raise ValueError(
                    "Path to treelite model missing for one or more model thresholds "
                    "in model_config_dict, defaulting to using lightGBM models."
                )
        except (ModuleNotFoundError, ValueError):
            # Default to lightGBM.
            cls = ApplyRainForestsCalibrationLightGBM
            # Ensure all required files have been specified.
            lightgbm_model_filenames = []
            for lead_time in model_config_dict.keys():
                for threshold in model_config_dict[lead_time].keys():
                    lightgbm_model_filenames.append(
                        model_config_dict[lead_time][threshold].get("lightgbm_model")
                    )
            if None in lightgbm_model_filenames:
                raise ValueError(
                    "Path to lightgbm model missing for one or more model thresholds "
                    "in model_config_dict."
                )
        return super(ApplyRainForestsCalibration, cls).__new__(cls)

    def process(self) -> None:
        """Subclasses should override this function."""
        raise NotImplementedError(
            "process function must be called via subclass method."
        )

    def _get_num_features(self) -> int:
        """Subclasses should override this function."""
        raise NotImplementedError(
            "_get_num_features function must be called via subclass method."
        )

    def _check_num_features(self, features: CubeList) -> None:
        """Check that the correct number of features has been passed into the model.
        Args:
            features:
                Cubelist containing feature variables.
        """
        expected_num_features = self._get_num_features()
        if expected_num_features != len(features):
            raise ValueError(
                "Number of expected features does not match number of feature cubes."
            )

    def _get_feature_splits(self, model_config_dict) -> Dict[int, List[ndarray]]:
        """Get the combined feature splits (over all thresholds) for each lead time.
        
        Args:
            model_config_dict: dictionary of the same format expected by __init__

        Returns:
            dict where keys are the lead times and the values are lists of lists.
            The outer list has length equal to the number of model features, and it contains 
            the lists of feature splits for each feature. Each feature's list of splits is ordered.
        """
        split_feature_string = "split_feature="
        threshold_string = "threshold="
        combined_feature_splits = {}
        for lead_time in self.lead_times:
            all_splits = [set() for i in range(self._get_num_features())]
            for threshold_str in model_config_dict[str(lead_time)].keys():
                lgb_model_filename = Path(
                    model_config_dict[str(lead_time)][threshold_str].get(
                        "lightgbm_model"
                    )
                ).expanduser()
                with open(lgb_model_filename, "r") as f:
                    for line in f:
                        if line.startswith(split_feature_string):
                            line = line[len(split_feature_string) : -1]
                            if len(line) == 0:
                                continue
                            features = [int(x) for x in line.split(" ")]
                        elif line.startswith(threshold_string):
                            line = line[len(threshold_string) : -1]
                            if len(line) == 0:
                                continue
                            splits = [float(x) for x in line.split(" ")]
                            for feature_ind, threshold in zip(features, splits):
                                all_splits[feature_ind].add(threshold)
            combined_feature_splits[lead_time] = [np.sort(list(x)) for x in all_splits]
        return combined_feature_splits


class ApplyRainForestsCalibrationLightGBM(ApplyRainForestsCalibration):
    """Class to calibrate input forecast given via RainForests approach using light-GBM
    tree models"""

    def __new__(cls, model_config_dict: dict, threads: int = 1, bin_data: bool = False):
        """Check all model files are available before initialising."""
        lightgbm_model_filenames = []
        for lead_time in model_config_dict.keys():
            threshold_keys = [
                t
                for t in model_config_dict[lead_time]
                if t != "combined_feature_splits"
            ]
            for threshold in threshold_keys:
                lightgbm_model_filenames.append(
                    model_config_dict[lead_time][threshold].get("lightgbm_model")
                )
        if None in lightgbm_model_filenames:
            raise ValueError(
                "Path to lightgbm model missing for one or more model thresholds "
                "in model_config_dict."
            )
        return super(ApplyRainForestsCalibration, cls).__new__(cls)

    def __init__(
        self, model_config_dict: dict, threads: int = 1, bin_data: bool = False
    ):
        """Initialise the tree model variables used in the application of RainForests
        Calibration. LightGBM Boosters are used for tree model predictors.

        Args:
            model_config_dict:
                Dictionary containing Rainforests model configuration variables.
            threads:
                Number of threads to use during prediction with tree-model objects.
            bin_data:
                Bin data according to splits used in models. This speeds up prediction
                if there are many data points which fall into the same bins for all models.

        Dictionary is of format::

            {
                "24": {
                    "0.000010": {
                        "lightgbm_model": "<path_to_lightgbm_model_object>",
                        "treelite_model": "<path_to_treelite_model_object>"
                    },
                    "0.000050": {
                        "lightgbm_model": "<path_to_lightgbm_model_object>",
                        "treelite_model": "<path_to_treelite_model_object>"
                    },
                    "0.000100": {
                        "lightgbm_model": "<path_to_lightgbm_model_object>",
                        "treelite_model": "<path_to_treelite_model_object>"
                    },
                }
            }

        The keys specify the lead times and model threshold values, while the
        associated values are the path to the corresponding tree-model objects
        for that lead time and threshold.
        """
        from lightgbm import Booster

        self.model_input_converter = np.array

        # Model config is a nested dictionary. Keys of outer level are lead times, and
        # keys of inner level are thresholds. Convert these to int and float.
        lead_times = []
        self.model_thresholds = None
        self.tree_models = {}
        for lead_time_str in model_config_dict.keys():
            lead_time = int(lead_time_str)
            lead_times.append(lead_time)
            thresholds = []
            for threshold_str in model_config_dict[lead_time_str].keys():
                threshold = np.float32(threshold_str)
                thresholds.append(threshold)
                model_filename = Path(
                    model_config_dict[lead_time_str][threshold_str].get(
                        "lightgbm_model"
                    )
                ).expanduser()
                self.tree_models[lead_time, threshold] = Booster(
                    model_file=str(model_filename)
                ).reset_parameter({"num_threads": threads})
            if self.model_thresholds is None:
                self.model_thresholds = np.sort(thresholds)
        self.lead_times = np.sort(lead_times)

        self.bin_data = bin_data
        if self.bin_data:
            self.combined_feature_splits = self._get_feature_splits(model_config_dict)

    def _get_num_features(self) -> int:
        return next(iter(self.tree_models.values())).num_feature()


    def _align_feature_variables(
        self, feature_cubes: CubeList, forecast_cube: Cube
    ) -> Tuple[CubeList, Cube]:
        """Ensure that feature cubes have consistent dimension coordinates. If realization
        dimension present in any cube, all cubes lacking this dimension will have realization
        dimension added and broadcast along this new dimension.

        This situation occurs when derived fields (such as accumulated solar radiation)
        are used as predictors. As these fields do not contain a realization dimension,
        they must be broadcast to match the NWP fields that do contain realization, so that
        all features have consistent shape.

        In the case of deterministic models (those without a realization dimension), a
        realization dimension is added to allow consistent behaviour between ensemble and
        deterministic models.

        Args:
            feature_cubes:
                Cubelist containing feature variables to align.
            forecast_cube:
                Cube containing the forecast variable to align.

        Returns:
            - feature_cubes with realization coordinate added to each cube if absent
            - forecast_cube with realization coordinate added if absent

        Raises:
            ValueError:
                if feature/forecast variables have inconsistent dimension coordinates
                (excluding realization dimension), or if feature/forecast variables have
                different length realization coordinate over cubes containing a realization
                coordinate.
        """
        combined_cubes = CubeList(list([*feature_cubes, forecast_cube]))

        # Compare feature cube coordinates, raise error if dim-coords don't match
        compare_feature_coords = compare_coords(
            combined_cubes, ignored_coords=["realization"]
        )
        for misaligned_coords in compare_feature_coords:
            for coord_info in misaligned_coords.values():
                if coord_info["data_dims"] is not None:
                    raise ValueError(
                        "Mismatch between non-realization dimension coords."
                    )

        # Compare realization coordinates across cubes where present;
        # raise error if realization coordinates don't match, otherwise set
        # common_realization_coord to broadcast over.
        realization_coords = {
            variable.name(): variable.coord("realization")
            for variable in combined_cubes
            if variable.coords("realization")
        }
        if not realization_coords:
            # Case I: realization_coords is empty. Add single realization dim to all cubes.
            common_realization_coord = DimCoord(
                [0], standard_name="realization", units=1
            )
        else:
            # Case II: realization_coords is not empty.
            # Note: In future, another option here could be to filter to common realization
            # values using filter_realizations() in utilities.cube_manipulation.
            variables_with_realization = list(realization_coords.keys())
            sample_realization = realization_coords[variables_with_realization[0]]
            for feature in variables_with_realization[1:]:
                if realization_coords[feature] != sample_realization:
                    raise ValueError("Mismatch between realization dimension coords.")
            common_realization_coord = sample_realization

        # Add realization coord to cubes where absent by broadcasting along this dimension
        aligned_cubes = CubeList()
        for cube in combined_cubes:
            if not cube.coords("realization"):
                expanded_cube = add_coordinate_to_cube(
                    cube, new_coord=common_realization_coord
                )
                aligned_cubes.append(expanded_cube)
            else:
                aligned_cubes.append(cube)

        # Make data contiguous (required for numba interpolation)
        for cube in aligned_cubes:
            if not cube.data.flags["C_CONTIGUOUS"]:
                cube.data = np.ascontiguousarray(cube.data, dtype=cube.data.dtype)

        return aligned_cubes[:-1], aligned_cubes[-1]

    def _prepare_threshold_probability_cube(self, forecast_cube):
        """Initialise a cube with the same dimensions as the input forecast_cube,
        with an additional threshold dimension added as the leading dimension.

        Args:
            forecast_cube:
                Cube containing the forecast to be calibrated.

        Returns:
            An empty probability cube.
        """
        # Create a template for CDF, with threshold the leading dimension.
        forecast_variable = forecast_cube.name()

        probability_cube = create_new_diagnostic_cube(
            name=f"probability_of_{forecast_variable}_above_threshold",
            units="1",
            template_cube=forecast_cube,
            mandatory_attributes=generate_mandatory_attributes([forecast_cube]),
        )
        threshold_coord = DimCoord(
            self.model_thresholds,
            standard_name=forecast_variable,
            var_name="threshold",
            units=forecast_cube.units,
            attributes={"spp__relative_to_threshold": "above"},
        )
        probability_cube = add_coordinate_to_cube(
            probability_cube, new_coord=threshold_coord,
        )

        return probability_cube

    def _prepare_features_array(self, feature_cubes: CubeList) -> ndarray:
        """Convert gridded feature cubes into a numpy array, with feature variables
        sorted alphabetically.

        Note: It is expected that feature_cubes has been aligned using
        _align_feature_variables prior to calling this function.

        Args:
            feature_cubes:
                Cubelist containing the independent feature variables for prediction.

        Returns:
            Array containing flattened feature variables,

        Raises:
            ValueError:
                If flattened cubes have differing length.
        """
        # Get the names of features and sort alphabetically
        feature_variables = [cube.name() for cube in feature_cubes]
        feature_variables.sort()

        # Unpack the cube-data into an array to feed into the tree-models.
        features_list = []
        for feature in feature_variables:
            cube = feature_cubes.extract_cube(feature)
            features_list.append(cube.data.ravel()[:, np.newaxis])
        features_arr = np.concatenate(features_list, axis=1)

        return features_arr

    def _make_decreasing(self, probability_data: ndarray) -> ndarray:
        """Enforce monotonicity on the CDF data, where threshold dimension
        is assumed to be the leading dimension.

        This is achieved by identifying the minimum value progressively along
        the leading dimension by comparing to all preceding probability values along
        this dimension. The same is done for maximum values, comparing to all
        succeeding values along the leading dimension. Averaging these resulting
        arrays results in an array decreasing monotonically in the threshold dimension.

        Args:
            probability_data:
                The probability data as exceedence probabilities.

        Returns:
            The probability data, enforced to be monotonically decreasing along
            the leading dimension.
        """
        lower = np.minimum.accumulate(probability_data, axis=0)
        upper = np.flip(
            np.maximum.accumulate(np.flip(probability_data, axis=0), axis=0), axis=0
        )
        return 0.5 * (upper + lower)

    def _evaluate_probabilities(
        self, input_data: ndarray, lead_time_hours: int, output_data: ndarray,
    ) -> None:
        """Evaluate probability that forecast exceeds thresholds.

        Args:
            input_data:
                2-d array of data for the feature variables of the model
            lead_time_hours:
                lead time in hours
            output_data:
                array to populate with output; will be modified in place
        """

        if int(lead_time_hours) in self.lead_times:
            model_lead_time = lead_time_hours
        else:
            # find closest model lead time
            best_ind = np.argmin(np.abs(self.lead_times - lead_time_hours))
            model_lead_time = self.lead_times[best_ind]

        if self.bin_data:
            # bin by feature splits
            feature_splits = self.combined_feature_splits[model_lead_time]
            binned_data = np.empty(input_data.shape, dtype=np.int32)
            for i in range(len(feature_splits)):
                binned_data[:, i] = np.digitize(
                    input_data[:, i], bins=feature_splits[i]
                )
            # sort so rows in the same bins are grouped
            sort_ind = np.lexsort(
                tuple([binned_data[:, i] for i in range(input_data.shape[1])])
            )
            sorted_data = binned_data[sort_ind]
            reverse_sort_ind = np.argsort(sort_ind)
            # we only need to predict for rows which are different from the previous row
            diff = np.any(np.diff(sorted_data, axis=0) != 0, axis=1)
            predict_rows = np.concatenate([[0], np.nonzero(diff)[0] + 1])
            data_for_prediction = input_data[sort_ind][predict_rows]
            full_prediction = np.empty((input_data.shape[0],))
            # forward fill code inspired by this: https://stackoverflow.com/a/41191127
            fill_inds = np.zeros(len(full_prediction), dtype=np.int32)
            fill_inds[predict_rows] = predict_rows
            fill_inds = np.maximum.accumulate(fill_inds)
            dataset_for_prediction = self.model_input_converter(data_for_prediction)

            for threshold_index, threshold in enumerate(self.model_thresholds):
                model = self.tree_models[model_lead_time, threshold]
                prediction = model.predict(dataset_for_prediction)
                prediction = np.maximum(np.minimum(1, prediction), 0)
                full_prediction[predict_rows] = prediction
                full_prediction = full_prediction[fill_inds]
                # restore original order
                full_prediction = full_prediction[reverse_sort_ind]
                output_data[threshold_index, :] = np.reshape(
                    full_prediction, output_data.shape[1:]
                )
        else:
            dataset_for_prediction = self.model_input_converter(input_data)
            for threshold_index, threshold in enumerate(self.model_thresholds):
                model = self.tree_models[model_lead_time, threshold]
                prediction = model.predict(dataset_for_prediction)
                prediction = np.maximum(np.minimum(1, prediction), 0)
                output_data[threshold_index, :] = np.reshape(
                    prediction, output_data.shape[1:]
                )

    def _calculate_threshold_probabilities(
        self, forecast_cube: Cube, feature_cubes: CubeList,
    ) -> Cube:
        """Evaluate the threshold exceedence probabilities for each ensemble member in
        forecast_cube using the tree_models, with the associated feature_cubes taken as
        inputs to the tree_model predictors.

        Args:
            forecast_cube:
                Cube containing the variable to be calibrated.
            feature_cubes:
                Cubelist containing the independent feature variables for prediction.

        Returns:
            A cube containing threshold exceedence probabilities.

        Raises:
            ValueError:
                If an unsupported model object is passed. Expects lightgbm Booster, or
                treelite_runtime Predictor (if treelite dependency is available).
        """

        threshold_probability_cube = self._prepare_threshold_probability_cube(
            forecast_cube
        )

        input_dataset = self._prepare_features_array(feature_cubes)

        lead_time_hours = forecast_cube.coord("forecast_period").points[0] / (
            SECONDS_IN_MINUTE * MINUTES_IN_HOUR
        )

        self._evaluate_probabilities(
            input_dataset, lead_time_hours, threshold_probability_cube.data,
        )

        # Enforcing monotonicity
        threshold_probability_cube.data = self._make_decreasing(
            threshold_probability_cube.data
        )

        return threshold_probability_cube

    def _get_ensemble_distributions(
        self, probability_CDF: Cube, forecast: Cube, output_thresholds: ndarray
    ) -> Cube:
        """
        Interpolate probilities calculated at model thresholds to extract probabilities
        at output thresholds for all realizations.

        Args:
            probability_CDF:
                Cube containing the CDF of probabilities for each ensemble member at model
                thresholds.
            forecast:
                Cube containing NWP ensemble forecast.
            output_thresholds:
                Sorted array of thresholds at which to calculate the output probabilities.

        Returns:
            Cube containing probabilities at output thresholds for all realizations. Dimensions
            are same as forecast cube with additional threshold dimension first.
        """

        input_probabilties = probability_CDF.data
        output_thresholds = np.array(output_thresholds, dtype=np.float32)
        bounds_data = get_bounds_of_distribution(forecast.name(), forecast.units)
        lower_bound = bounds_data[0].astype(np.float32)
        if (len(self.model_thresholds) == len(output_thresholds)) and np.allclose(
            self.model_thresholds, output_thresholds
        ):
            output_probabilities = np.copy(input_probabilties.data)
        else:
            # add lower bound with probability 1
            input_probabilties = np.concatenate(
                [
                    np.ones((1,) + input_probabilties.shape[1:], dtype=np.float32),
                    input_probabilties,
                ],
                axis=0,
            )
            input_thresholds = np.concatenate([[lower_bound], self.model_thresholds])
            # reshape to 2 dimensions
            input_probabilties_2d = np.reshape(
                input_probabilties, (input_probabilties.shape[0], -1)
            )
            output_probabilities_2d = interpolate_multiple_rows_same_x(
                output_thresholds, input_thresholds, input_probabilties_2d.transpose()
            )
            output_probabilities = np.reshape(
                output_probabilities_2d.transpose(),
                (len(output_thresholds),) + input_probabilties.shape[1:],
            )
            # force interpolated probabilties to be monotone (sometimes they
            # are not due to small floating-point errors)
            output_probabilities = self._make_decreasing(output_probabilities)

        # set probability for lower bound to 1
        if np.isclose(output_thresholds[0], lower_bound):
            output_probabilities[0, :] = 1

        # Make output cube
        aux_coords_and_dims = []
        for coord in getattr(forecast, "aux_coords"):
            coord_dims = forecast.coord_dims(coord)
            if len(coord_dims) == 0:
                aux_coords_and_dims.append((coord.copy(), []))
            else:
                aux_coords_and_dims.append(
                    (coord.copy(), forecast.coord_dims(coord)[0] + 1)
                )
        forecast_variable = forecast.name()
        threshold_dim = iris.coords.DimCoord(
            output_thresholds.astype(np.float32),
            standard_name=forecast_variable,
            units=forecast.units,
            var_name="threshold",
            attributes={"spp__relative_to_threshold": "greater_than_or_equal_to"},
        )
        dim_coords_and_dims = [(threshold_dim, 0)] + [
            (coord.copy(), forecast.coord_dims(coord)[0] + 1)
            for coord in forecast.coords(dim_coords=True)
        ]
        probability_cube = iris.cube.Cube(
            output_probabilities.astype(np.float32),
            long_name=f"probability_of_{forecast_variable}_above_threshold",
            units=1,
            attributes=forecast.attributes,
            dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims,
        )
        return probability_cube

    def process(
        self,
        forecast_cube: Cube,
        feature_cubes: CubeList,
        output_thresholds: List,
        threshold_units: str = None,
    ) -> Cube:
        """Apply rainforests calibration to forecast cube.

        Ensemble forecasts must be in realization representation. Deterministic forecasts
        can be processed to produce a pseudo-ensemble; a realization dimension will be added
        to deterministic forecast cubes if one is not already present.

        The calibration is done in a situation dependent fashion using a series of
        decision-tree models to construct representative distributions which are
        then used to map each input ensemble member onto a series of realisable values.

        These distributions are formed in a two-step process:

        1. Evaluate CDF defined over the specified model thresholds for each ensemble member.
        Each exceedence probability is evaluated using the corresponding decision-tree model.

        2. Interpolate each ensemble member distribution to the output thresholds, then average
        over ensemble members

        Args:
            forecast_cube:
                Cube containing the forecast to be calibrated; must be as realizations.
            feature_cubes:
                Cubelist containing the feature variables (physical parameters) used as inputs
                to the tree-models for the generation of the associated probability distributions.
                Feature cubes are expected to have the same dimensions as forecast_cube, with
                the exception of the realization dimension. Where the feature_cube contains a
                realization dimension this is expected to be consistent, otherwise the cube will
                be broadcast along the realization dimension.
            output_thresholds:
                Set of output thresholds.
            threshold_units:
                Units in which output_thresholds are specified. If None, assumed to be the same as
                forecast_cube.

        Returns:
            The calibrated forecast cube.

        Raises:
            RuntimeError:
                If the number of tree models is inconsistent with the number of model
                thresholds.
        """
        # Check that the correct number of feature variables has been supplied.
        self._check_num_features(feature_cubes)

        # Align forecast and feature datasets
        aligned_features, aligned_forecast = self._align_feature_variables(
            feature_cubes, forecast_cube
        )

        # Evaluate the CDF using tree models.
        probability_CDF = self._calculate_threshold_probabilities(
            aligned_forecast, aligned_features
        )

        # convert units of output thresholds
        if threshold_units:
            original_threshold_unit = Unit(threshold_units)
            forecast_unit = forecast_cube.units
            output_thresholds_in_forecast_units = np.array(
                [
                    original_threshold_unit.convert(x, forecast_unit)
                    for x in output_thresholds
                ]
            )
        else:
            output_thresholds_in_forecast_units = np.array(output_thresholds)

        # Calculate probabilities at output thresholds
        probabilities_by_realization = self._get_ensemble_distributions(
            probability_CDF, aligned_forecast, output_thresholds_in_forecast_units
        )

        # Average over realizations
        output_cube = probabilities_by_realization.collapsed("realization", MEAN)
        output_cube.remove_coord("realization")

        return output_cube


class ApplyRainForestsCalibrationTreelite(ApplyRainForestsCalibrationLightGBM):
    """Class to calibrate input forecast given via RainForests approach using treelite
    compiled tree models"""

    def __new__(cls, model_config_dict: dict, threads: int = 1, bin_data: bool = False):
        """Check required dependency and all model files are available before initialising."""
        # Try and initialise the treelite_runtime library to test if the package
        # is available.
        import treelite_runtime  # noqa: F401

        # Check that all required files have been specified.
        treelite_model_filenames = []
        for lead_time in model_config_dict.keys():
            threshold_keys = [
                t
                for t in model_config_dict[lead_time]
                if t != "combined_feature_splits"
            ]
            for threshold in threshold_keys:
                treelite_model_filenames.append(
                    model_config_dict[lead_time][threshold].get("treelite_model")
                )
        if None in treelite_model_filenames:
            raise ValueError(
                "Path to treelite model missing for one or more model thresholds "
                "in model_config_dict, defaulting to using lightGBM models."
            )
        return super(ApplyRainForestsCalibration, cls).__new__(cls)

    def __init__(
        self, model_config_dict: dict, threads: int = 1, bin_data: bool = False
    ):
        """Initialise the tree model variables used in the application of RainForests
        Calibration. Treelite Predictors are used for tree model predictors.

        Args:
            model_config_dict:
                Dictionary containing Rainforests model configuration variables.
            threads:
                Number of threads to use during prediction with tree-model objects.
            bin_data:
                Bin data according to splits used in models. This speeds up prediction
                if there are many data points which fall into the same bins for all models.

        Dictionary is of format::

            {
                "24": {
                    "0.000010": {
                        "lightgbm_model": "<path_to_lightgbm_model_object>",
                        "treelite_model": "<path_to_treelite_model_object>"
                    },
                    "0.000050": {
                        "lightgbm_model": "<path_to_lightgbm_model_object>",
                        "treelite_model": "<path_to_treelite_model_object>"
                    },
                    "0.000100": {
                        "lightgbm_model": "<path_to_lightgbm_model_object>",
                        "treelite_model": "<path_to_treelite_model_object>"
                    },
                }
            }

        The keys specify the model threshold value, while the associated values
        are the path to the corresponding tree-model objects for that threshold.
        """
        from treelite_runtime import DMatrix, Predictor

        self.model_input_converter = DMatrix

        # Model config is a nested dictionary. Keys of outer level are lead times, and
        # keys of inner level are thresholds. Convert these to int and float.
        lead_times = []
        self.model_thresholds = None
        self.tree_models = {}
        for lead_time_str in model_config_dict.keys():
            lead_time = int(lead_time_str)
            lead_times.append(lead_time)
            thresholds = []
            for threshold_str in model_config_dict[lead_time_str].keys():
                threshold = np.float32(threshold_str)
                thresholds.append(threshold)
                model_filename = Path(
                    model_config_dict[lead_time_str][threshold_str].get(
                        "treelite_model"
                    )
                ).expanduser()
                self.tree_models[lead_time, threshold] = Predictor(
                    libpath=str(model_filename), verbose=False, nthread=threads
                )
            if self.model_thresholds is None:
                self.model_thresholds = np.sort(thresholds)
        self.lead_times = np.sort(lead_times)

        self.bin_data = bin_data
        if self.bin_data:
            self.combined_feature_splits = self._get_feature_splits(model_config_dict)

    def _get_num_features(self) -> int:
        return next(iter(self.tree_models.values())).num_feature