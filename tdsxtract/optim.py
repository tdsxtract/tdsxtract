#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import numpy as npo
from jax import vmap
from scipy.optimize import minimize

from .stack import *


def sample_transmission(epsilons, thickness, sample=None, wavelengths=None):
    unknown = "unknown"
    t = []
    sample[unknown]["thickness"] = thickness

    def _t(params):
        lambda0, eps = params
        sample[unknown]["epsilon"] = eps
        t = get_transmission(sample, lambda0)
        return t

    params = wavelengths, epsilons
    t = vmap(_t)(params)
    return np.array(t)


@jit
def _fmini(
    x,
    sample=None,
    t_exp=None,
    wavelengths=None,
    weights=(1, 0),
    opt_thickness=None,
):
    unknown = "unknown"
    nl = len(wavelengths)
    thickness = sample[unknown]["thickness"]
    if opt_thickness != None:
        thickness *= x[-1]
        sample[unknown]["thickness"] = thickness
        x_ = x[:-1]
    else:
        x_ = x
    epsilon = x_[:nl] + 1j * x_[nl:]
    gamma = 2 * pi / wavelengths
    thickness_tot = np.sum(np.array([k["thickness"] for lay, k in sample.items()]))
    phasor = np.exp(-1j * gamma * thickness_tot)
    t_exp_phased = t_exp * phasor

    t_model = sample_transmission(
        epsilon, thickness, sample=sample, wavelengths=wavelengths
    )
    mse_func = np.mean(
        np.abs(t_exp_phased - t_model) ** 2
    )  # / np.mean(np.abs(t_exp_phased) ** 2)
    # epsmax,epsmin = np.max((epsilon)),np.min((epsilon))
    # deps= np.abs(epsmax-epsmin)
    #
    # no = np.max(np.array([deps,1e-3]))
    # mse_grad = np.mean(np.abs(np.gradient(epsilon) ) ** 2)/ no**2
    mse_grad = np.mean(np.abs(np.gradient(epsilon)) ** 2) / np.mean(
        np.abs(epsilon) ** 2
    )  # * np.mean(np.abs( freq/ epsilon)**2)
    mse = weights[0] * mse_func + weights[1] * mse_grad

    return mse


_jit_gfunc = jit(grad(_fmini))


def _jac(x, **kwargs):
    return npo.array(_jit_gfunc(x, **kwargs))


def extract(
    sample,
    wavelengths,
    t_exp,
    epsilon_initial_guess=None,
    eps_re_min=None,
    eps_re_max=None,
    eps_im_min=None,
    eps_im_max=None,
    thickness_tol=0,
    opt_thickness=False,
    weight=1,
):

    h = sample["unknown"]["thickness"]
    nl = len(wavelengths)
    ones = npo.ones_like(wavelengths)

    # passiv = 1 - float(force_passive)

    if epsilon_initial_guess == None:
        epsilon_initial_guess = 1 + 0j
    eps_re0 = epsilon_initial_guess.real * ones
    eps_im0 = epsilon_initial_guess.imag * ones

    eps_re_min = float(eps_re_min) if eps_re_min is not None else None
    eps_re_max = float(eps_re_max) if eps_re_max is not None else None
    eps_im_min = float(eps_im_min) if eps_im_min is not None else None
    eps_im_max = float(eps_im_max) if eps_im_max is not None else None

    hmin, hmax = (1 - thickness_tol), (1 + thickness_tol)
    x0 = [eps_re0, eps_im0]
    if opt_thickness:
        x0.append(1)
    initial_guess = npo.float64(npo.hstack(x0))
    bounds = [(eps_re_min, eps_re_max) for i in range(nl)]
    bounds += [(eps_im_min, eps_im_max) for i in range(nl)]
    if opt_thickness:
        bounds += [(hmin, hmax)]
    bounds = npo.float64(bounds)

    weights = (float(weight), float(1 - weight))

    optthick = True if opt_thickness else None

    fmini_opt = lambda x: _fmini(
        x,
        sample=sample,
        weights=weights,
        t_exp=t_exp,
        wavelengths=wavelengths,
        opt_thickness=optthick,
    )
    jac_opt = lambda x: _jac(
        x,
        sample=sample,
        weights=weights,
        t_exp=t_exp,
        wavelengths=wavelengths,
        opt_thickness=optthick,
    )

    options = {
        "disp": True,
        "maxcor": 250,
        "ftol": 1e-16,
        "gtol": 1e-16,
        "eps": 1e-11,
        "maxfun": 15000,
        "maxiter": 15000,
        "iprint": 1,
        "maxls": 200,
        "finite_diff_rel_step": None,
    }

    opt = minimize(
        fmini_opt,
        initial_guess,
        bounds=bounds,
        tol=1e-16,
        options=options,
        jac=jac_opt,
        method="L-BFGS-B",
    )
    if opt_thickness:
        h_opt = opt.x[-1] * h
        _epsopt = opt.x[:-1]
    else:
        h_opt = h
        _epsopt = opt.x

    epsilon_opt = _epsopt[:nl] + 1j * _epsopt[nl:]
    return epsilon_opt, h_opt, opt
