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
"""Tests for the aggregate-dimensions CLI"""


import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_broadcast_mean(tmp_path):
    """
    Test mean aggregation on realization with broadcasting.
    """

    kgo_dir = acc.kgo_root() / "aggregate-dimensions"
    kgo_path = kgo_dir / "kgo_broadcast_mean.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--dimensions",
        "realization",
        "--aggregation",
        "mean",
        "--broadcast",
        "--new-name",
        "ensemble_mean_of_air_temperature",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_broadcast_std_dev(tmp_path):
    """
    Test std_dev aggregation on realization with broadcasting.
    """

    kgo_dir = acc.kgo_root() / "aggregate-dimensions"
    kgo_path = kgo_dir / "kgo_broadcast_stddev.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--dimensions",
        "realization",
        "--aggregation",
        "std_dev",
        "--broadcast",
        "--new-name",
        "ensemble_standard_deviation_of_air_temperature",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_no_broadcast_mean(tmp_path):
    """
    Test mean aggregation on realization without broadcasting.
    """

    kgo_dir = acc.kgo_root() / "aggregate-dimensions"
    kgo_path = kgo_dir / "kgo_no_broadcast.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--dimensions",
        "realization",
        "--aggregation",
        "mean",
        "--new-name",
        "ensemble_mean_of_air_temperature",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)