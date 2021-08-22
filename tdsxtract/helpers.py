#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from collections import OrderedDict

import numpy as npo
import scipy.signal as signal
from scipy.constants import c


def pos2time(position):
    """Converts stage position to time shift.

    Parameters
    ----------
    position : array
        The position of the stage (in m).

    Returns
    -------
    array
        Corresponding time shift (in s).

    Notes
    -----
    2*position since 2 paths of the laser are shifted on the delay stage

    """
    time = 2 * position / c
    return time


def smooth(data, cutoff=0.05, order=3):
    """Data smoothing.

    Parameters
    ----------
    data : array
        The data to smooth.
    cutoff : float
        Filter cutoff frequency (the default is 0.05).
    order : int
        Filter order (the default is 3).

    Returns
    -------
    smooth_data: array
        Smoothed data.

    """
    # Buterworth filter
    B, A = signal.butter(order, cutoff, output="ba")
    smooth_data = signal.filtfilt(B, A, data)
    return smooth_data


def get_epsilon_estimate(voltage_reference, voltage_sample, time, sample_thickness):
    """Get a rough estimate of the permittivity value.

    Parameters
    ----------
    voltage_reference : array
        Reference voltage.
    voltage_sample : array
        Sample voltage.
    time : array
        Time steps (in s).
    sample_thickness : float
        Thickness of the sample (in m).

    Returns
    -------
    epsilon_estimate: float
        Permittivity estimate.

    """

    # Find the peak of the TD signals to calculate time delay
    # then the bulk refractive index
    ref_max = npo.argmax(voltage_reference)
    samp_max = npo.argmax(voltage_sample)
    t0 = time[ref_max]
    t1 = time[samp_max]
    # Good guess for bulk refractive index here
    epsilon_estimate = ((t1 - t0) * c / (sample_thickness) + 1) ** 2
    return epsilon_estimate


def fft(time, signal):
    """Fourier transform.

    Parameters
    ----------
    time : array
        The time points.
    signal : array
        Signal amplitude.

    Returns
    -------
    frequencies: array
        Frequencies (in Hz).
    fourier_transform: array
        Fourier transform of the signal.

    """
    fourier_transform = npo.fft.rfft(signal)
    N = len(fourier_transform)
    fs = 1 / npo.gradient(time)
    frequencies = fs[0:N] * npo.linspace(0, 1 / 2, N)
    return frequencies, fourier_transform


def restrict(x, min=None, max=None):
    if max is None:
        index_max = len(x)
    else:
        index_max = npo.argmin(abs(x - max))
    if min is None:
        index_min = 0
    else:
        index_min = npo.argmin(abs(x - min))
    x_restrict = x[index_min:index_max]
    return x_restrict, index_min, index_max
