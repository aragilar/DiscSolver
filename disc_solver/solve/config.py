# -*- coding: utf-8 -*-
"""
Define input and environment for ode system
"""

import configparser
from math import pi, sqrt

import logbook

import numpy as np

from .stepper import StepperError

from ..file_format import LATEST_NAMESPACE as namespace
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

    norm_kepler_sq = 1 / inp.c_s_on_v_k ** 2
    η_O = inp.η_O
    η_A = inp.η_A
    η_H = inp.η_H

    # solution for A * v_φ**2 + B * v_φ + C = 0
    A_v_φ = 1
    B_v_φ = (v_r * η_H) / (2 * (η_O + η_A))
    C_v_φ = (
        v_r**2 / 2 + 2 * inp.β * c_s**2 -
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

    return namespace.initial_conditions(  # pylint: disable=no-member
        norm_kepler_sq=norm_kepler_sq, c_s=c_s, η_O=η_O, η_A=η_A, η_H=η_H,
        init_con=init_con, angles=angles, β=inp.β
    )


def get_input(conffile=None):
    """
    Get input values
    """
    config = CaseDependentConfigParser()
    if conffile:
        config.read_file(open(conffile))

    return namespace.config_input(  # pylint: disable=no-member
        start=float(config.get("config", "start", fallback=0)),
        stop=float(config.get("config", "stop", fallback=5)),
        taylor_stop_angle=float(config.get(
            "config", "taylor_stop_angle", fallback=0.001
        )),
        max_steps=int(config.get("config", "max_steps", fallback=10000)),
        num_angles=int(config.get("config", "num_angles", fallback=10000)),
        label=config.get("config", "label", fallback="default"),
        relative_tolerance=float(config.get(
            "config", "relative_tolerance", fallback=1e-6
        )),
        absolute_tolerance=float(config.get(
            "config", "absolute_tolerance", fallback=1e-10
        )),
        β=float_with_frac(config.get("initial", "β", fallback=1.249)),
        v_rin_on_c_s=float(config.get("initial", "v_rin_on_c_s", fallback=1)),
        v_a_on_c_s=float(config.get("initial", "v_a_on_c_s", fallback=1)),
        c_s_on_v_k=float(config.get("initial", "c_s_on_v_k", fallback=0.03)),
        η_O=float(config.get("initial", "η_O", fallback=0.001)),
        η_H=float(config.get("initial", "η_H", fallback=0.0001)),
        η_A=float(config.get("initial", "η_A", fallback=0.0005)),
    )


def step_input(inp):
    """
    Create new input for next step
    """
    inp_dict = vars(inp)

    def step_func(soln_type, step_size):
        """
        Return new input
        """
        prev_v_rin_on_c_s = inp_dict["v_rin_on_c_s"]
        if soln_type == "diverge":
            inp_dict["v_rin_on_c_s"] -= step_size
        elif soln_type == "sign flip":
            inp_dict["v_rin_on_c_s"] += step_size
        else:
            raise StepperError("Solution type not known")
        if prev_v_rin_on_c_s == inp_dict["v_rin_on_c_s"]:
            raise StepperError("Hit numerical limit")
        return namespace.soln_input(**inp_dict)  # pylint: disable=no-member
    return step_func
