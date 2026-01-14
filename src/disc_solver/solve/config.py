# -*- coding: utf-8 -*-
"""
Define input and environment for ode system
"""

import logbook

import numpy as np
from numpy import sqrt

from ..float_handling import float_type, FLOAT_TYPE
from ..file_format import ConfigInput, InitialConditions
from ..utils import CaseDependentConfigParser, ODEIndex

from .utils import SolverError, add_overrides

log = logbook.Logger(__name__)


def v_φ_boundary_func(*, v_r, η_H, η_P, norm_kepler_sq, a_0, γ):
    """
    Compute v_φ at the midplane
    """
    try:
        return v_r * η_H / (4 * η_P) + sqrt(
            norm_kepler_sq - 5/2 + 2 * γ + v_r * (
                a_0 / η_P + v_r / 2 * (
                    η_H ** 2 / (8 * η_P ** 2) - 1
                )
            )
        )
    except ValueError as e:
        raise SolverError("Input implies complex v_φ") from e


def B_φ_prime_boundary_func(*, v_r, v_φ, a_0):
    """
    Compute B_φ_prime at the midplane
    """
    return v_φ * v_r / (2 * a_0)


def E_r_boundary_func(*, v_r, v_φ, η_P, η_H, η_perp_sq, a_0):
    """
    Compute E_r at the midplane
    """
    return - v_φ - v_r / η_P * (η_H + η_perp_sq * v_φ / (2 * a_0))


def define_conditions(inp):
    """
    Compute initial conditions based on input
    """
    ρ = float_type(1)
    B_θ = float_type(1)
    v_θ = float_type(0)
    B_r = float_type(0)
    B_φ = float_type(0)

    v_r = - inp.v_rin_on_c_s
    a_0 = inp.v_a_on_c_s ** 2
    norm_kepler_sq = 1 / inp.c_s_on_v_k ** 2

    γ = inp.γ
    η_O = inp.η_O
    η_A = inp.η_A
    η_H = inp.η_H

    η_P = η_O + η_A
    η_perp_sq = η_P ** 2 + η_H ** 2

    v_φ = v_φ_boundary_func(
        v_r=v_r, η_H=η_H, η_P=η_P, norm_kepler_sq=norm_kepler_sq, a_0=a_0, γ=γ
    )

    init_con = np.zeros(11, dtype=FLOAT_TYPE)

    init_con[ODEIndex.B_r] = B_r
    init_con[ODEIndex.B_φ] = B_φ
    init_con[ODEIndex.B_θ] = B_θ
    init_con[ODEIndex.v_r] = v_r
    init_con[ODEIndex.v_φ] = v_φ
    init_con[ODEIndex.v_θ] = v_θ
    init_con[ODEIndex.ρ] = ρ
    init_con[ODEIndex.η_O] = η_O
    init_con[ODEIndex.η_A] = η_A
    init_con[ODEIndex.η_H] = η_H

    if inp.use_E_r:
        E_r = E_r_boundary_func(
            v_r=v_r, v_φ=v_φ, η_P=η_P, η_H=η_H, η_perp_sq=η_perp_sq, a_0=a_0
        )
        init_con[ODEIndex.E_r] = E_r
    else:
        B_φ_prime = B_φ_prime_boundary_func(v_r=v_r, v_φ=v_φ, a_0=a_0)
        init_con[ODEIndex.B_φ_prime] = B_φ_prime

    angles = np.radians(np.linspace(inp.start, inp.stop, inp.num_angles))
    if np.any(np.isnan(init_con)):
        nan_names = []
        for i in np.isnan(init_con).nonzero()[0]:
            nan_names.append(ODEIndex(i).name)
        raise SolverError("Input implies NaN: {}".format(nan_names))

    return InitialConditions(
        norm_kepler_sq=norm_kepler_sq, a_0=a_0, init_con=init_con,
        angles=angles, γ=γ
    )


def get_input_from_conffile(*, config_file, overrides=None):
    """
    Get input values
    """
    config = CaseDependentConfigParser()
    if config_file:
        with config_file.open("r") as f:
            config.read_file(f)

    return add_overrides(
        overrides=overrides, config_input=ConfigInput.from_configparser(config)
    )


def new_inputs_with_overrides(config_input, solution_input, overrides):
    """
    Merge possibly differing ConfigInput and SolutionInput with additional
    changes within overrides
    """
    if overrides is None:
        overrides = {}
    new_config_input = add_overrides(
        config_input=config_input, overrides=overrides,
    )
    new_soln_input = new_config_input.to_soln_input()

    initial_soln_input = config_input.to_soln_input()

    if solution_input == initial_soln_input:
        return new_config_input, new_soln_input

    conf_dict = initial_soln_input.asdict()
    soln_dict = solution_input.asdict()
    use_soln_keys = []
    for key, val in conf_dict.items():
        if key in overrides:
            continue
        elif val != soln_dict[key]:
            use_soln_keys.append(key)

    for key in use_soln_keys:
        setattr(new_soln_input, key, soln_dict[key])

    return new_config_input, new_soln_input
