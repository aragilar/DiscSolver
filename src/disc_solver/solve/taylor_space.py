# -*- coding: utf-8 -*-
"""
Scan parameter space for taylor series values
"""

from enum import IntEnum, unique

import attr
import logbook

import numpy as np
from numpy import zeros
from numpy.random import uniform

import matplotlib.pyplot as plt

from spaceplot import spaceplots

from .config import define_conditions
from .deriv_funcs import (
    get_taylor_first_order, get_taylor_second_order, get_taylor_third_order,
)

from ..file_format import SolutionInput
from ..utils import ODEIndex

log = logbook.Logger(__name__)

TAYLOR_NUM_ORDERS = 3


@unique
class Parameters(IntEnum):
    """
    Variables for taylor series scan
    """
    γ = 0
    v_rin_on_c_s = 1
    v_a_on_c_s = 2
    c_s_on_v_k = 3


@attr.s(cmp=False, hash=False)
class TaylorSpace:
    """
    Data class for holding taylor information
    """
    # pylint: disable=too-few-public-methods
    γ_arr = attr.ib()
    v_rin_on_c_s_arr = attr.ib()
    v_a_on_c_s_arr = attr.ib()
    c_s_on_v_k_arr = attr.ib()
    taylor_orders = attr.ib()


def compute_taylor_space(
    *, γ, v_rin_on_c_s, v_a_on_c_s, c_s_on_v_k, η_O=0, η_A=0, η_H=0,
    num_samples=10000
):
    """
    Generate taylor space based on ranges
    """
    taylor_orders = zeros((num_samples, TAYLOR_NUM_ORDERS + 1, len(ODEIndex)))

    γ_arr = uniform(γ[0], γ[1], size=num_samples)
    v_rin_on_c_s_arr = uniform(
        v_rin_on_c_s[0], v_rin_on_c_s[1], size=num_samples
    )
    v_a_on_c_s_arr = uniform(v_a_on_c_s[0], v_a_on_c_s[1], size=num_samples)
    c_s_on_v_k_arr = uniform(c_s_on_v_k[0], c_s_on_v_k[1], size=num_samples)

    for i, inp in enumerate(solution_input_generator(
        γ=γ_arr, v_rin_on_c_s=v_rin_on_c_s_arr, v_a_on_c_s=v_a_on_c_s_arr,
        c_s_on_v_k=c_s_on_v_k_arr, η_O=η_O, η_A=η_A, η_H=η_H,
    )):
        taylor_orders[i] = compute_taylor_values(inp)

    return TaylorSpace(
        γ_arr=γ_arr, v_rin_on_c_s_arr=v_rin_on_c_s_arr,
        v_a_on_c_s_arr=v_a_on_c_s_arr, c_s_on_v_k_arr=c_s_on_v_k_arr,
        taylor_orders=taylor_orders,
    )


def compute_taylor_values(inp, *, use_E_r=False):
    """
    Compute values for taylor series given inputs
    """
    cons = define_conditions(inp, use_E_r=use_E_r)
    init_con = cons.init_con
    γ = cons.γ
    a_0 = cons.a_0
    return (
        init_con,
        get_taylor_first_order(init_con=init_con, γ=γ),
        get_taylor_second_order(
            γ=γ, a_0=a_0, init_con=init_con, η_derivs=False
        ),
        get_taylor_third_order(
            γ=γ, a_0=a_0, init_con=init_con, η_derivs=False
        )
    )


def solution_input_generator(
    *, γ, v_rin_on_c_s, v_a_on_c_s, c_s_on_v_k, η_O, η_A, η_H
):
    """
    Generate `SolutionInput`s with with generated inputs
    """
    num_samples = len(γ)
    if num_samples != len(v_rin_on_c_s):
        raise RuntimeError("incorrect number of samples for v_rin_on_c_s")
    if num_samples != len(v_a_on_c_s):
        raise RuntimeError("incorrect number of samples for v_a_on_c_s")
    if num_samples != len(c_s_on_v_k):
        raise RuntimeError("incorrect number of samples for c_s_on_v_k")
    for i in range(num_samples):
        yield SolutionInput(
            start=0,
            stop=0,
            taylor_stop_angle=None,
            max_steps=None,
            num_angles=0,
            relative_tolerance=None,
            absolute_tolerance=None,
            nwalkers=None,
            iterations=None,
            threads=None,
            target_velocity=None,
            split_method=None,
            γ=γ[i],
            v_rin_on_c_s=v_rin_on_c_s[i],
            v_a_on_c_s=v_a_on_c_s[i],
            c_s_on_v_k=c_s_on_v_k[i],
            η_O=η_O,
            η_H=η_A,
            η_A=η_H,
        )


def plot_taylor_space(taylor_space, **kwargs):
    """
    Plot the taylor space
    """
    # pylint: disable=unused-variable

    xs = np.array([
        taylor_space.γ_arr,
        taylor_space.v_rin_on_c_s_arr,
        taylor_space.v_a_on_c_s_arr,
        taylor_space.c_s_on_v_k_arr,
    ]).T

    zero_order, first_order, second_order, third_order = zip(
        *taylor_space.taylor_orders
    )
    ys = np.array(second_order)
    print(min(ys[:, ODEIndex.v_r]))
    print(min(ys[:, ODEIndex.v_φ]))

    in_names = [item.name for item in Parameters]
    out_names = [item.name for item in ODEIndex]

    for plot in spaceplots(
        xs, ys, input_names=in_names, output_names=out_names, **kwargs
    ):
        plt.show()
        plt.close(plot)


def main():
    """
    Main function of ds-plot-taylor-space
    """
    taylor_space = compute_taylor_space(
        γ=[1e-5, 1e-2], c_s_on_v_k=[0.05, 0.03], v_rin_on_c_s=[0.1, 1.0],
        v_a_on_c_s=[0.1, 1.0], η_O=0.005, η_H=0.008, η_A=0.001,
        num_samples=10000,
    )

    limits = np.array([[0, None]] * len(ODEIndex))
    limits[ODEIndex.v_r] = [-100, 0]
    limits[ODEIndex.v_φ] = [-100, 0]

    plot_taylor_space(
        taylor_space, limits=limits,
        scatter_args=dict(marker='.', alpha=0.5),
    )
