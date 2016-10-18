# -*- coding: utf-8 -*-
"""
Define input and environment for ode system
"""

from math import sqrt

import logbook

import numpy as np

from ..file_format import ConfigInput, InitialConditions, SolutionInput

from ..utils import (
    str_to_float, str_to_int, str_to_bool, CaseDependentConfigParser,
    ODEIndex,
)

log = logbook.Logger(__name__)


def define_conditions(inp):
    """
    Compute initial conditions based on input
    """
    ρ = 1
    B_θ = 1
    v_θ = 0
    B_r = 0
    B_φ = 0

    v_r = - inp.v_rin_on_c_s
    a_0 = inp.v_a_on_c_s ** 2
    norm_kepler_sq = 1 / inp.c_s_on_v_k ** 2

    γ = inp.γ
    η_O = inp.η_O
    η_A = inp.η_A
    η_H = inp.η_H

    v_φ = (
        - v_r * η_H / (4 * (η_O + η_A)) + sqrt(
            norm_kepler_sq - 5/2 + 2 * γ + v_r * (
                a_0 / (η_O + η_A) + v_r / 2 * (
                    η_H ** 2 / (8 * (η_O + η_A) ** 2) - 1
                )
            )
        )
    )

    B_φ_prime = v_φ * v_r / (2 * a_0)

    init_con = np.zeros(11)

    init_con[ODEIndex.B_r] = B_r
    init_con[ODEIndex.B_φ] = B_φ
    init_con[ODEIndex.B_θ] = B_θ
    init_con[ODEIndex.v_r] = v_r
    init_con[ODEIndex.v_φ] = v_φ
    init_con[ODEIndex.v_θ] = v_θ
    init_con[ODEIndex.ρ] = ρ
    init_con[ODEIndex.B_φ_prime] = B_φ_prime
    init_con[ODEIndex.η_O] = η_O
    init_con[ODEIndex.η_A] = η_A
    init_con[ODEIndex.η_H] = η_H

    angles = np.radians(np.linspace(inp.start, inp.stop, inp.num_angles))

    return InitialConditions(
        norm_kepler_sq=norm_kepler_sq, a_0=a_0, init_con=init_con,
        angles=angles, γ=γ
    )


def get_input_from_conffile(*, config_file):
    """
    Get input values
    """
    config = CaseDependentConfigParser()
    if config_file:
        with config_file.open("r") as f:
            config.read_file(f)

    return ConfigInput(
        start=config.get("config", "start", fallback="0"),
        stop=config.get("config", "stop", fallback="5"),
        taylor_stop_angle=config.get(
            "config", "taylor_stop_angle", fallback="0.001"
        ),
        max_steps=config.get("config", "max_steps", fallback="10000"),
        num_angles=config.get("config", "num_angles", fallback="10000"),
        label=config.get("config", "label", fallback="default"),
        relative_tolerance=config.get(
            "config", "relative_tolerance", fallback="1e-6"
        ),
        absolute_tolerance=config.get(
            "config", "absolute_tolerance", fallback="1e-10"
        ),
        jump_before_sonic=config.get(
            "config", "jump_before_sonic", fallback="None"
        ),
        η_derivs=config.get("config", "η_derivs", fallback="True"),
        γ=config.get("initial", "γ", fallback="0.001"),
        v_rin_on_c_s=config.get("initial", "v_rin_on_c_s", fallback="1"),
        v_a_on_c_s=config.get("initial", "v_a_on_c_s", fallback="1"),
        c_s_on_v_k=config.get("initial", "c_s_on_v_k", fallback="0.03"),
        η_O=config.get("initial", "η_O", fallback="0.001"),
        η_H=config.get("initial", "η_H", fallback="0.0001"),
        η_A=config.get("initial", "η_A", fallback="0.0005"),
    )


def config_input_to_soln_input(inp):
    """
    Convert user input into solver input
    """
    return SolutionInput(
        start=str_to_float(inp.start),
        stop=str_to_float(inp.stop),
        taylor_stop_angle=str_to_float(inp.taylor_stop_angle),
        max_steps=str_to_int(inp.max_steps),
        num_angles=str_to_int(inp.num_angles),
        relative_tolerance=str_to_float(inp.relative_tolerance),
        absolute_tolerance=str_to_float(inp.absolute_tolerance),
        jump_before_sonic=(
            None if inp.jump_before_sonic == "None"
            else str_to_float(inp.jump_before_sonic)
        ),
        η_derivs=str_to_bool(inp.η_derivs),
        γ=str_to_float(inp.γ),
        v_rin_on_c_s=str_to_float(inp.v_rin_on_c_s),
        v_a_on_c_s=str_to_float(inp.v_a_on_c_s),
        c_s_on_v_k=str_to_float(inp.c_s_on_v_k),
        η_O=str_to_float(inp.η_O),
        η_H=str_to_float(inp.η_H),
        η_A=str_to_float(inp.η_A),
    )
