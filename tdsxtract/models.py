#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import jax
import jax.numpy as np
import numpy as npo
from jax import grad, jit, vjp
from scipy.constants import c
from scipy.optimize import differential_evolution, minimize

jax.config.update("jax_enable_x64", True)


def _cole_cole(omega, eps_inf, eps_static, tau, alpha):
    return eps_inf + (eps_static - eps_inf) / (1 + (1j * omega * tau) ** (1 - alpha))


@jit
def _fmini(x, epsilon=None, wavelengths=None):
    omega = 2 * np.pi * c / wavelengths * 1e-12
    mod = _cole_cole(omega, *x)
    return np.mean(np.abs(mod - epsilon) ** 2)


_jit_gfunc = jit(grad(_fmini))


def _jac(x, **kwargs):
    return npo.array(_jit_gfunc(x, **kwargs))


class ColeCole:
    def __init__(self):
        pass

    def model(self, omega, eps_inf, eps_static, tau, alpha):
        return _cole_cole(omega, eps_inf, eps_static, tau, alpha)

    def fit(
        self,
        epsilon,
        wavelengths,
        bounds,
        x0=None,
        type="de",
    ):
        bounds = npo.float64(bounds)

        fmini_opt = lambda x: _fmini(
            x,
            epsilon=epsilon,
            wavelengths=wavelengths,
        )
        jac_opt = lambda x: _jac(
            x,
            epsilon=epsilon,
            wavelengths=wavelengths,
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

        if type == "de":
            opt = differential_evolution(fmini_opt, bounds)
        else:

            opt = minimize(
                fmini_opt,
                x0,
                bounds=bounds,
                tol=1e-16,
                options=options,
                jac=jac_opt,
                method="L-BFGS-B",
            )

        return opt
