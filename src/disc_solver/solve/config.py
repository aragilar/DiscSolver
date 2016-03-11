# -*- coding: utf-8 -*-
"""
Define input and environment for ode system
"""

import configparser
from math import pi, sqrt

import logbook

import numpy as np

from .stepper import StepperError

from ..file_format import ConfigInput, InitialConditions, SolutionInput

from ..utils import float_with_frac
from ..utils import allvars as vars

log = logbook.Logger(__name__)


class CaseDependentConfigParser(configparser.ConfigParser):
    # pylint: disable=too-many-ancestors
    """
    configparser.ConfigParser subclass that removes the case transform.
    """
    def optionxform(self, optionstr):
        return optionstr


def strdict(d):
    """ stringify a dictionary"""
    return {str(k): str(v) for k, v in d.items()}


def define_conditions(inp):
    """
    Compute initial conditions based on input
    """
    ρ = 1  # ρ is always normalised by itself
    c_s = 1  # velocities normalised by c_s, so c_s = 1

    v_θ = 0  # symmetry across disc
    B_r = 0  # symmetry across disc
    B_φ = 0  # symmetry across disc

    v_r = - inp.v_rin_on_c_s  # velocities normalised by c_s
    B_θ = inp.v_a_on_c_s

    β = inp.β
    norm_kepler_sq = 1 / inp.c_s_on_v_k ** 2
    η_O = inp.η_O
    η_A = inp.η_A
    η_H = inp.η_H

    # solution for A * v_φ**2 + B * v_φ + C = 0
    A_v_φ = 1
    B_v_φ = (v_r * η_H) / (2 * (η_O + η_A))
    C_v_φ = (
        v_r**2 / 2 + 2 * β * c_s**2 -
        norm_kepler_sq - B_θ**2 * (
            v_r / (η_O + η_A)
        ) / (4 * pi * ρ)
    )
    log.debug("A_v_φ: {}".format(A_v_φ))
    log.debug("B_v_φ: {}".format(B_v_φ))
    log.debug("C_v_φ: {}".format(C_v_φ))

    v_φ = - 1 / (2 * A_v_φ) * (
        B_v_φ - sqrt(B_v_φ**2 - 4 * A_v_φ * C_v_φ)
    )

    B_φ_prime = (
        v_φ * v_r * 2 * pi * ρ
    ) / B_θ

    init_con = np.zeros(8)

    init_con[0] = B_r
    init_con[1] = B_φ
    init_con[2] = B_θ
    init_con[3] = v_r
    init_con[4] = v_φ
    init_con[5] = v_θ
    init_con[6] = ρ
    init_con[7] = B_φ_prime

    angles = np.radians(np.linspace(inp.start, inp.stop, inp.num_angles))

    return InitialConditions(
        norm_kepler_sq=norm_kepler_sq, c_s=c_s, η_O=η_O, η_A=η_A, η_H=η_H,
        init_con=init_con, angles=angles, β=β
    )


def get_input_from_conffile(conffile=None):
    """
    Get input values
    """
    config = CaseDependentConfigParser()
    if conffile:
        config.read_file(open(conffile))

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
        β=config.get("initial", "β", fallback="1.249"),
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
        start=float_with_frac(inp.start),
        stop=float_with_frac(inp.stop),
        taylor_stop_angle=float_with_frac(inp.taylor_stop_angle),
        max_steps=int(inp.max_steps),
        num_angles=int(inp.num_angles),
        relative_tolerance=float_with_frac(inp.relative_tolerance),
        absolute_tolerance=float_with_frac(inp.absolute_tolerance),
        β=float_with_frac(inp.β),
        v_rin_on_c_s=float_with_frac(inp.v_rin_on_c_s),
        v_a_on_c_s=float_with_frac(inp.v_a_on_c_s),
        c_s_on_v_k=float_with_frac(inp.c_s_on_v_k),
        η_O=float_with_frac(inp.η_O),
        η_H=float_with_frac(inp.η_H),
        η_A=float_with_frac(inp.η_A),
    )


def step_input():
    """
    Create new input for next step
    """
    def step_func(soln, soln_type, step_size):
        """
        Return new input
        """
        inp_dict = vars(soln.solution_input)
        prev_v_rin_on_c_s = inp_dict["v_rin_on_c_s"]
        if soln_type == "diverge":
            inp_dict["v_rin_on_c_s"] -= step_size
        elif soln_type == "sign flip":
            inp_dict["v_rin_on_c_s"] += step_size
        else:
            raise StepperError("Solution type not known")
        if prev_v_rin_on_c_s == inp_dict["v_rin_on_c_s"]:
            raise StepperError("Hit numerical limit")
        return SolutionInput(**inp_dict)
    return step_func
