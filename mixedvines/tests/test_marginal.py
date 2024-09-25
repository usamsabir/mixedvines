# -*- coding: utf-8 -*-
# Copyright (C) 2017-2019, 2021 Arno Onken
#
# This file is part of the mixedvines package.
#
# The mixedvines package is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# The mixedvines package is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
"""This module implements tests for the marginal module."""
import numpy as np
from numpy.testing import assert_allclose
from mixedvines.marginal import Marginal
from scipy.stats import norm



def test_marginal_fit():
    """Tests the fit method."""
    samples = np.linspace(-2, 2, 3)
    # Normal distribution
    marginal = Marginal.fit(samples, True)
    # Comparison values
    r_logpdf = np.array([-2.15935316, -1.40935316, -2.15935316])
    p_logpdf = marginal.logpdf(samples)
    assert_allclose(p_logpdf, r_logpdf)

def test_sample_labels():
    #this test checks if the regex expression is correct or not. it gives a list of test cases
    #than it evaluates
    test_cases = [
        # Valid cases
        "AB12-Jan21-abc_Sample_123.45_pH7.2",
        "CD34-Aug22-xyz_Experiment_0.01_pH6.8",
        "EF56-Dec23-mno_Analysis_789_pH8.5",

        # Invalid cases
        "GH78-Feb24-pqr_duplicate_45.67_pH5.9",  # Contains 'duplicate'
        "IJ90-May25-stu_test_789.0_pH7.0",  # Contains 'test'
        "KL12-Abc26-vwx_Sample_-10.0_pH7.2",  # Invalid month
        "MN34-Jan27-yz_sample_67.89_pH6.8",  # Lowercase word after underscore
        "OP56-Dec28-abc_Experiment_-0.01_pH8.5",  # Negative number
        "QR78-Feb29-def_Analysis_0.00_pH5.9",  # Zero value
        "ST90-May30-ghi_Test_789.0_pHInvalid",  # Invalid pH format
        "UV12-Jan31-jkl_Sample_123.45_pH10.5",  # Valid pH > 10
        "ab34-Aug32-mno_Experiment_67.89_pH6.8",  # Lowercase start
        "WX56-Dec33-PQR_Analysis_789_pH8.5",  # Uppercase three letters after second hyphen
        "YZ78-Feb34-stu_Sample_45.67",  # Missing pH
        "AC90-May35-vwx_Test_789.0_pH7.0_extra",  # Extra part after pH
    ]

    expected_results = [
        True, True, True,
        False, False, False, False, False, False, False, True, False, False, False, False
    ]

    assert Marginal.match_sample_labels('', test_cases) == expected_results

def test_standardize():
    # Create a normal distribution with mean 10 and standard deviation 2
    rv_mixed = norm(loc=10, scale=2)
    marginal = Marginal(rv_mixed)

    # Create sample data
    samples = np.array([8, 10, 12, 14])

    # Call the standardize function
    standardized_samples = marginal.standardize(samples)

    # Expected standardized values
    expected = np.array([-1, 0, 1, 2])

    # Check if the standardized samples are close to the expected values
    np.testing.assert_array_almost_equal(standardized_samples, expected, decimal=6)
