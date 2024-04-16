#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import pathlib

import matplotlib.pyplot as plt
import numpy as npo

from tdsxtract import *

testdir = pathlib.Path(__file__).parent.absolute()


def test():
    sample_thickness = 500.0e-6
    # get the data
    pos_ref, v_ref = load(f"{testdir}/data/reference.txt")
    pos_samp, v_samp = load(f"{testdir}/data/sample.txt")
    assert npo.all(pos_ref == pos_samp)

    position = pos_ref * 1e-6
    p, _, _ = restrict(position)
    print(p)
    assert np.all(p == position)

    padding = False
    if padding:
        # padding
        expo_fft = 10
        extra0s = 2 ** (expo_fft + 3) - 2**expo_fft
        dx = position[1] - position[0]
        extra_position = position[-1] + npo.cumsum(npo.ones(extra0s) * dx)

        padding = npo.zeros(extra0s)
        position = npo.hstack((position, extra_position))
        v_samp = npo.hstack((v_samp, padding))
        v_ref = npo.hstack((v_ref, padding))

    time = pos2time(position)

    # get estimate of permittivity
    eps_guess = get_epsilon_estimate(v_ref, v_samp, time, sample_thickness)

    freqs_ref, fft_ref = fft(time, v_ref)
    freqs_samp, fft_samp = fft(time, v_samp)

    assert npo.all(freqs_ref == freqs_samp)

    freqs_THz = freqs_ref * 1e-12

    transmission = fft_samp / fft_ref

    layers = OrderedDict(
        {
            "superstrate": {"epsilon": 1, "mu": 1},
            "layer": {"epsilon": 3.4, "mu": 1, "thickness": 200},
            "substrate": {"epsilon": 1.5, "mu": 1.0},
        }
    )

    wave = {
        "lambda0": 100,
        "theta0": 0.0 * pi,
        "phi0": 0 * pi,
        "psi0": 0 * pi,
    }

    config = dict(layers=layers, wave=wave)

    out = get_coeffs_stack(config)
    # print(out)

    sample = Sample(
        {
            "unknown": {"epsilon": 3.0, "mu": 1.0, "thickness": 500e-6},
        }
    )

    freqs = freqs_ref

    freqs_restrict, index_min, index_max = restrict(freqs, 0.2e12, 1e12)
    t_exp = transmission[index_min:index_max]
    wavelengths = c / freqs_restrict

    epsilon_opt, h_opt, opt = extract(
        sample,
        wavelengths,
        t_exp,
        epsilon_initial_guess=eps_guess,
        eps_re_min=1,
        eps_re_max=100,
        eps_im_min=-10,
        eps_im_max=0,
        thickness_tol=1e-10,
        opt_thickness=False,
        weight=1.0,
    )

    eps_smooth = smooth(epsilon_opt)
