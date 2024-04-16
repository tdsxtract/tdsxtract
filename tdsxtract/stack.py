#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from collections import OrderedDict

import jax
import jax.numpy as np
from jax import grad, jit, vjp
from scipy.constants import c, epsilon_0, mu_0

jax.config.update("jax_enable_x64", True)

inv = np.linalg.inv
pi = np.pi
Sample = OrderedDict


def get_coeffs_stack(config):

    layers, wave = config["layers"], config["wave"]

    lambda0 = wave["lambda0"]
    theta0 = wave["theta0"]
    phi0 = wave["phi0"]
    psi0 = wave["psi0"]

    k0 = 2 * pi / lambda0
    omega = k0 * c

    eps = [d["epsilon"] for d in layers.values()]
    eps = [e if not callable(e) else e(lambda0) for e in eps]

    for ie, e in enumerate(eps):
        if isinstance(e, tuple):
            if len(e) == 2:
                wl, eps_ = e
                eps[ie] = np.interp(lambda0, wl, eps_)
            else:
                raise (ValueError)
        else:
            eps[ie] = e
    mu = [d["mu"] for d in layers.values()]
    thicknesses = [d["thickness"] for d in layers.values() if "thickness" in d.keys()]

    alpha0 = k0 * np.sin(theta0) * np.cos(phi0)
    beta0 = k0 * np.sin(theta0) * np.sin(phi0)
    gamma0 = k0 * np.cos(theta0)
    Ex0 = np.cos(psi0) * np.cos(theta0) * np.cos(phi0) - np.sin(psi0) * np.sin(phi0)
    Ey0 = np.cos(psi0) * np.cos(theta0) * np.sin(phi0) + np.sin(psi0) * np.cos(phi0)
    Ez0 = -np.cos(psi0) * np.sin(theta0)

    def _matrix_pi(M, gamma):
        q = gamma * M
        return np.array(
            [
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [q[0, 1], -q[0, 1], -q[0, 0], q[0, 0]],
                [q[1, 1], -q[1, 1], -q[1, 0], q[1, 0]],
            ],
            dtype=complex,
        )

    def _matrix_t(gamma, e):
        pp = np.exp(1j * gamma * e)
        pm = np.exp(-1j * gamma * e)
        return np.array(
            [
                [pp, 0.0, 0.0, 0.0],
                [0.0, pm, 0.0, 0.0],
                [0.0, 0.0, pp, 0.0],
                [0.0, 0.0, 0.0, pm],
            ]
        )

    def _matrix_B(eps, mu):
        eps *= epsilon_0
        mu *= mu_0
        return np.array(
            [
                [omega * mu, 0, beta0],
                [0, omega * mu, -alpha0],
                [-beta0, alpha0, -omega * eps],
            ]
        )

    gamma = [np.sqrt(k0**2 * e * m - alpha0**2 - beta0**2) for e, m in zip(eps, mu)]
    B = [_matrix_B(e, m) for e, m in zip(eps, mu)]
    M = [inv(b) for b in B]
    Pi = [_matrix_pi(m, g) for m, g in zip(M, gamma)]
    T_ = [_matrix_t(g, e) for g, e in zip(gamma[1:-1], thicknesses)]
    # T.append(np.eye(4))
    Tr = T_ + [np.eye(4)]
    # T = [np.eye(4)] + T_

    M1 = np.eye(4)
    p_prev = Pi[0]
    for p, t in zip(Pi[1:], Tr):
        M1 = inv(t) @ inv(p) @ p_prev @ M1
        p_prev = p

    K = inv(M1)
    Q = np.array(
        [
            [K[0, 0], 0, K[0, 2], 0],
            [K[1, 0], -1, K[1, 2], 0],
            [K[2, 0], 0, K[2, 2], 0],
            [K[3, 0], 0, K[3, 2], -1],
        ],
        dtype=complex,
    )

    ## solve
    U0 = np.array([Ex0, 0, Ey0, 0], dtype=complex)
    sol = inv(Q) @ U0

    # get coefficients by recurrence
    phi_0 = np.array([Ex0, sol[1], Ey0, sol[3]])
    phi_end = np.array([sol[0], 0, sol[2], 0])
    p_prev = Pi[0]
    phi_prev = phi_0
    phi = [phi_0]

    for p, t in zip(Pi[1:], Tr):
        phi_j = inv(t) @ inv(p) @ p_prev @ phi_prev
        phi.append(phi_j)
        p_prev = p
        phi_prev = phi_j

    # assert np.all(np.abs(phi_j - phi_end)<1e-14)

    ## Ez
    for i, (p, g, m) in enumerate(zip(phi.copy(), gamma, M)):
        phiz_plus = (m[2, 0] * p[2] - m[2, 1] * p[0]) * g
        phiz_minus = (m[2, 1] * p[1] - m[2, 0] * p[3]) * g
        phixh_plus = (m[0, 0] * p[2] - m[0, 1] * p[0]) * g
        phixh_minus = (m[0, 1] * p[1] - m[0, 0] * p[3]) * g
        phiyh_plus = (m[1, 0] * p[2] - m[1, 1] * p[0]) * g
        phiyh_minus = (m[1, 1] * p[1] - m[1, 0] * p[3]) * g
        phizh_plus = (m[2, 1] * phixh_plus - m[2, 0] * phiyh_plus) * g
        phizh_minus = (m[2, 0] * phiyh_minus - m[2, 1] * phixh_minus) * g
        phi[i] = np.append(
            phi[i],
            np.array(
                [
                    phiz_plus,
                    phiz_minus,
                    phixh_plus,
                    phixh_minus,
                    phiyh_plus,
                    phiyh_minus,
                    phizh_plus,
                    phizh_minus,
                ]
            ),
        )

    R = (
        1.0
        / gamma[0] ** 2
        * (
            (gamma[0] ** 2 + alpha0**2) * abs(phi[0][1]) ** 2
            + (gamma[0] ** 2 + beta0**2) * abs(phi[0][3]) ** 2
            + 2 * alpha0 * beta0 * np.real(phi[0][1] * phi[0][3].conjugate())
        )
    )
    T = (
        1.0
        / (gamma[0] * gamma[-1] * mu[-1])
        * (
            (gamma[-1] ** 2 + alpha0**2) * abs(phi[-1][0]) ** 2
            + (gamma[-1] ** 2 + beta0**2) * abs(phi[-1][2]) ** 2
            + 2 * alpha0 * beta0 * np.real(phi[-1][0] * phi[-1][2].conjugate())
        )
    )
    return phi, alpha0, beta0, gamma, R, T
    # return phi[-1][0], gamma


def _ordered_dict_insert(ordered_dict, index, key, value):
    if key in ordered_dict:
        raise KeyError("Key already exists")
    if index < 0 or index > len(ordered_dict):
        raise IndexError("Index out of range")

    keys = list(ordered_dict.keys())[index:]
    ordered_dict[key] = value
    for k in keys:
        ordered_dict.move_to_end(k)

    return ordered_dict


def _make_layers(layers, sample):
    new = layers.copy()
    for i, lays in enumerate(sample.items()):
        new = _ordered_dict_insert(layers, i + 1, *lays)
    return new


def get_transmission(sample, lambda0):
    wave = {
        "lambda0": lambda0,
        "theta0": 0.0,
        "phi0": 0.0,
        "psi0": 0.0,
    }

    layers = OrderedDict(
        {
            "input medium": {"epsilon": 1.0, "mu": 1.0},
            "output medium": {"epsilon": 1.0, "mu": 1.0},
        }
    )

    layers = _make_layers(layers, sample)

    config = dict(layers=layers, wave=wave)

    out = get_coeffs_stack(config)
    return out[0][-1][0]


#
# if __name__ == "__main__":
#
#     layers = OrderedDict(
#         {
#             "superstrate": {"epsilon": 1, "mu": 1},
#             "layer": {"epsilon": 1, "mu": 1, "thickness": 500},
#             "substrate": {"epsilon": 1, "mu": 1.0},
#         }
#     )
#
#     wave_params = {
#         "lambda0": 1.1,
#         "theta0": 0.0 * pi,
#         "phi0": 0 * pi / 3,
#         "psi0": 0 * pi,
#     }
#
#     config = layers, wave_params
#
#     def f(eps_re, eps_im, config):
#         eps = eps_re + 1j * eps_im
#         config[0]["layer"]["epsilon"] = eps
#         out, gamma = get_coeffs_stack(config)
#         return out
#
#     def fre(eps_re, eps_im, config):
#         return f(eps_re, eps_im, config).real
#
#     def fim(eps_re, eps_im, config):
#         return f(eps_re, eps_im, config).imag
#
#     # f(12)
#
#     dfre_epsre = jit(grad(fre, 0))
#     dfre_epsim = jit(grad(fre, 1))
#     dfim_epsre = jit(grad(fim, 0))
#     dfim_epsim = jit(grad(fim, 1))
#     dfre_epsre(12.3, 1.2, config)
#     dfre_epsim(12.3, 1.2, config)
#     dfim_epsre(12.3, 1.2, config)
#     dfim_epsim(12.3, 1.2, config)
#
#     def df(eps_re, eps_im, config):
#         df_epsre = dfre_epsre(eps_re, eps_im, config) + 1j * dfim_epsre(
#             eps_re, eps_im, config
#         )
#         df_epsim = dfre_epsim(eps_re, eps_im, config) + 1j * dfim_epsim(
#             eps_re, eps_im, config
#         )
#         return np.array([df_epsre, df_epsim])
#
#     #
#     # q = vjp(f,12.3,12.2)
#     #
#     # df_epe_re = grad(f,0)
#     # df_epe_im = grad(f,1)
