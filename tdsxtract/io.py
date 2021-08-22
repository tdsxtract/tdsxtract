#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import numpy as npo


def load(filename):
    """Load measurement data.

    Parameters
    ----------
    filename : str
        Name of the file to load.

    Returns
    -------
    position: array
        position of the delay stage (in microns)
    voltage: array
        signal amplitude (in mV).

    """
    data = npo.loadtxt(filename, skiprows=13)
    position = data[:, 0]
    voltage = data[:, 1]
    return position, voltage
