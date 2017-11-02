#!/usr/bin/env bats
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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

@test "nbhood -h" {
  run improver nbhood-iterate-with-mask -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-nbhood-iterate-with-mask [-h]
                                         [--radius RADIUS | --radii-by-lead-time RADII_BY_LEAD_TIME LEAD_TIME_IN_HOURS]
                                         [--ens_factor ENS_FACTOR]
                                         [--sum_or_fraction {sum,fraction}]
                                         [--re_mask]
                                         COORD_FOR_MASKING INPUT_FILE
                                         INPUT_MASK_FILE OUTPUT_FILE

Apply the requested neighbourhood method via the
ApplyNeighbourhoodProcessingWithAMask plugin to a file with one cube using a
mask. The masking coordinate is iterated over, so that different masks as
defined by the masking coordinate can be applied to the cube that is being
neighbourhood processed.

positional arguments:
  COORD_FOR_MASKING     Coordinate to iterate over when applying a mask to the
                        neighbourhood processing.
  INPUT_FILE            A path to an input NetCDF file to be processed.
  INPUT_MASK_FILE       A path to an input mask NetCDF file to be used to mask
                        the input file.
  OUTPUT_FILE           The output path for the processed NetCDF.

optional arguments:
  -h, --help            show this help message and exit
  --radius RADIUS       The radius (in m) for neighbourhood processing.
  --radii-by-lead-time RADII_BY_LEAD_TIME LEAD_TIME_IN_HOURS
                        The radii for neighbourhood processing and the
                        associated lead times at which the radii are valid.
                        The radii are in metres whilst the lead time has units
                        of hours. The radii and lead times are expected as
                        individual comma-separated lists with the list of
                        radii given first followed by a list of lead times to
                        indicate at what lead time each radii should be used.
                        For example: 10000,12000,14000 1,2,3 where a lead time
                        of 1 hour uses a radius of 10000m, a lead time of 2
                        hours uses a radius of 12000m, etc.
  --ens_factor ENS_FACTOR
                        The factor with which to adjust the neighbourhood size
                        for more than one ensemble member. If ens_factor = 1.0
                        this essentially conserves ensemble members if every
                        grid square is considered to be the equivalent of an
                        ensemble member.Optional, defaults to 1.0.
  --sum_or_fraction {sum,fraction}
                        The neighbourhood output can either be in the form of
                        a sum of the neighbourhood, or a fraction calculated
                        by dividing the sum of the neighbourhood by the
                        neighbourhood area. "fraction" is the default option.
  --re_mask             If re_mask is set (i.e. True), the original un-
                        neighbourhood processed mask is applied to mask out
                        the neighbourhood processed cube. If not set, re_mask
                        defaults to False and the original un-neighbourhood
                        processed mask is not applied. Therefore, the
                        neighbourhood processing may result in values being
                        present in areas that were originally masked.
__HELP__
  [[ "$output" == "$expected" ]]
}
