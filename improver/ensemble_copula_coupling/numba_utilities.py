# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""
This module defines the optional numba utilities for Ensemble Copula Coupling
plugins.

"""

import os

import numpy as np
from numba import config, njit, prange, set_num_threads

config.THREADING_LAYER = "omp"
if "OMP_NUM_THREADS" in os.environ:
    set_num_threads(int(os.environ["OMP_NUM_THREADS"]))


@njit(parallel=True)
def fast_interp_same_y(x: np.ndarray, xp: np.ndarray, fp: np.ndarray):
    """For each row i of xp, do the equivalent of np.interp(x, xp[i], fp).
    Args:
        x: 1-d array
        xp: n * m array, each row must be in non-decreasing order
        fp: 1-d array with length m
    Returns:
        n * len(x) array where each row i is equal to np.interp(x, xp[i], fp)
    """
    # check whether x is non-decreasing
    x_ordered = True
    for i in range(1, len(x)):
        if x[i] < x[i - 1]:
            x_ordered = False
            break
    max_ind = xp.shape[1]
    min_val = fp[0]
    max_val = fp[max_ind - 1]
    result = np.empty((xp.shape[0], len(x)))
    for i in prange(xp.shape[0]):
        ind = 0
        intercept = 0
        slope = 0
        x_lower = 0
        for j in range(len(x)):
            recalculate = False
            curr_x = x[j]
            # Find the indices of xp[i] to interpolate between. We need the
            # smallest index ind of xp[i] for which xp[i, ind] >= curr_x.
            if x_ordered:
                # Since x and x[i] are non-decreasing, ind for current j must be
                # greater than equal to ind for previous j.
                while (ind < max_ind) and (xp[i, ind] < curr_x):
                    ind = ind + 1
                    recalculate = True
            else:
                ind = np.searchsorted(xp[i], curr_x)
            # linear interpolation
            if ind == 0:
                result[i, j] = min_val
            elif ind == max_ind:
                result[i, j] = max_val
            else:
                if recalculate or not (x_ordered):
                    intercept = fp[ind - 1]
                    x_lower = xp[i, ind - 1]
                    h_diff = xp[i, ind] - x_lower
                    if h_diff < 1e-15:
                        # avoid division by very small values for numerical stability
                        slope = 0
                    else:
                        slope = (fp[ind] - intercept) / h_diff
                result[i, j] = intercept + (curr_x - x_lower) * slope
    return result
