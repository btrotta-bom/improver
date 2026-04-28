# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from datetime import datetime

import iris
import numpy as np
import pytest
from iris.exceptions import CoordinateNotFoundError
from scipy.stats import beta

from improver.calibration.beta_recalibration import BetaRecalibrate
from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_probability_cube,
)
from improver.utilities.cube_manipulation import MergeCubes


@pytest.fixture
def forecast_grid():
    thresholds = [283, 288]
    forecast_data = np.arange(9, dtype=np.float32).reshape(3, 3) / 8.0
    forecast_data_stack = np.stack([forecast_data, forecast_data])
    forecast_1 = set_up_probability_cube(
        forecast_data_stack,
        thresholds,
        time=datetime(2017, 11, 11, 2, 0),
        frt=datetime(2017, 11, 11, 0, 0),
    )
    forecast_2 = set_up_probability_cube(
        forecast_data_stack,
        thresholds,
        time=datetime(2017, 11, 11, 6, 0),
        frt=datetime(2017, 11, 11, 0, 0),
    )
    forecasts_grid = MergeCubes()([forecast_1, forecast_2])
    return forecasts_grid


@pytest.fixture
def percentile_forecast_grid():
    percentiles = [50, 90]
    np.random.seed(0)
    forecast_data = (
        np.random.uniform(low=283, high=288, size=9).astype(np.float32).reshape(3, 3)
    )
    forecast_data_stack = np.stack([forecast_data, forecast_data + 5])
    forecast = set_up_percentile_cube(
        forecast_data_stack,
        percentiles,
        time=datetime(2017, 11, 11, 4, 0),
        frt=datetime(2017, 11, 11, 0, 0),
    )
    return forecast


@pytest.fixture
def recalibration_dict():
    recal_dict = {
        "forecast_period": {
            "0": {"threshold": {"273.0": {"alpha": 1, "beta": 1}}},
            "4": {"threshold": {"273.0": {"alpha": 1, "beta": 2}}},
        },
        "forecast_period_units": "hours",
        "threshold_units": "K",
    }
    return recal_dict


def test_no_threshold_coord(percentile_forecast_grid, recalibration_dict):
    # test that error is raised when the input is not a probability cube
    msg = "must be a probability forecast"
    plugin = BetaRecalibrate(recalibration_dict)
    with pytest.raises(CoordinateNotFoundError, match=msg):
        plugin.process(percentile_forecast_grid.copy())


def test_no_forecast_period_coord(forecast_grid, recalibration_dict):
    # test that error is raised when the input does not contain
    # a forecast_period coordinate
    forecast_grid.remove_coord("forecast_period")
    msg = "must contain forecast_period coordinate"
    plugin = BetaRecalibrate(recalibration_dict)
    with pytest.raises(CoordinateNotFoundError, match=msg):
        plugin.process(forecast_grid.copy())


def test_invalid_params(forecast_grid):
    # test that error is raised when alpha or beta parameter is invalid
    msg = "alpha and beta parameters must be > 0"
    recal_dict = {
        "forecast_period": {
            "0": {"threshold": {"273.0": {"alpha": 2, "beta": 0}}},
            "1": {"threshold": {"273.0": {"alpha": -1, "beta": 1}}},
        },
        "forecast_period_units": "hours",
        "threshold_units": "K",
    }
    plugin = BetaRecalibrate(recal_dict)
    with pytest.raises(RuntimeError, match=msg):
        plugin.process(forecast_grid.copy())


def test_different_thresholds(forecast_grid):
    # test that error is raised when different forecast periods have different thresholds
    msg = "Thresholds must be the same for all forecast periods"
    recal_dict = {
        "forecast_period": {
            "0": {"threshold": {"273.0": {"alpha": 2, "beta": 0.5}}},
            "1": {"threshold": {"274.0": {"alpha": 2, "beta": 1}}},
        },
        "forecast_period_units": "hours",
        "threshold_units": "K",
    }
    plugin = BetaRecalibrate(recal_dict)
    with pytest.raises(ValueError, match=msg):
        plugin.process(forecast_grid.copy())


def test_recalibrate(forecast_grid, recalibration_dict):
    # test recalibration with interpolation outside specified forecast period and threshold range
    plugin = BetaRecalibrate(recalibration_dict)
    result = plugin.process(forecast_grid.copy())
    f1 = forecast_grid.extract(iris.Constraint(time=lambda cell: cell.point.hour == 2))
    f2 = forecast_grid.extract(iris.Constraint(time=lambda cell: cell.point.hour == 6))
    f1.data = beta.cdf(f1.data, 1, 1.5)
    f2.data = beta.cdf(f2.data, 1, 2)
    expected = MergeCubes()([f1, f2])
    assert result.coords() == expected.coords()
    assert result.attributes == expected.attributes
    np.testing.assert_almost_equal(expected.data, result.data)


def test_interpolation(forecast_grid):
    # test with interpolation between specified forecast period and thresholds are interpolated
    recal_dict1 = {
        "forecast_period": {
            "0": {
                "threshold": {
                    "273.0": {"alpha": 2, "beta": 1},
                    "275.0": {"alpha": 1, "beta": 1},
                }
            },
            "2": {
                "threshold": {
                    "273.0": {"alpha": 2, "beta": 1},
                    "275.0": {"alpha": 2, "beta": 1},
                }
            },
        },
        "forecast_units": "hours",
        "threshold_units": "K",
    }
    recal_dict2 = {
        "forecast_period": {
            "0": {
                "threshold": {
                    "273.0": {"alpha": 2, "beta": 1},
                    "274.0": {"alpha": 1.5, "beta": 1},
                    "275.0": {"alpha": 1, "beta": 1},
                }
            },
            "1": {
                "threshold": {
                    "273.0": {"alpha": 2, "beta": 1},
                    "274.0": {"alpha": 1.75, "beta": 1},
                    "275.0": {"alpha": 1.5, "beta": 1},
                }
            },
            "2": {
                "threshold": {
                    "273.0": {"alpha": 2, "beta": 1},
                    "274.0": {"alpha": 2, "beta": 1},
                    "275.0": {"alpha": 2, "beta": 1},
                }
            },
        },
        "forecast_period_units": "hours",
        "threshold_units": "K",
    }
    plugin = BetaRecalibrate(recal_dict1)
    result1 = plugin.process(forecast_grid.copy())
    plugin = BetaRecalibrate(recal_dict2)
    result2 = plugin.process(forecast_grid.copy())
    assert result1.coords() == result2.coords()
    assert result1.attributes == result2.attributes
    np.testing.assert_almost_equal(result1.data, result2.data)


def test_params_equal_1(forecast_grid):
    # check that recalibration has no effect when alpha = beta = 1
    recalibration_dict = {
        "forecast_period": {"0": {"threshold": {"273.0": {"alpha": 1, "beta": 1}}}},
        "forecast_period_units": "hours",
        "threshold_units": "K",
    }
    plugin = BetaRecalibrate(recalibration_dict)
    result = plugin.process(forecast_grid.copy())

    assert result.coords() == forecast_grid.coords()
    assert result.attributes == forecast_grid.attributes
    np.testing.assert_almost_equal(result.data, forecast_grid.data)
