# -*- coding: utf-8 -*-
"""
Define input and environment for ode system
"""

import logbook
import numpy as np
from numpy import sqrt

from ..file_format import ConfigInput, InitialConditions
from ..float_handling import FLOAT_TYPE, float_type
from ..utils import CaseDependentConfigParser, ODEIndex
from .utils import SolverError, add_overrides

log = logbook.Logger(__name__)


def v_φ_boundary_func(*, v_r, η_H, η_P, norm_kepler_sq, a_0, γ):
    """
    Compute v_φ at the midplane
    """
    try:
        return v_r * η_H / (4 * η_P) + sqrt(
            norm_kepler_sq
            - 5 / 2
            + 2 * γ
            + v_r * (a_0 / η_P + v_r / 2 * (η_H ** 2 / (8 * η_P ** 2) - 1))
        )
    except ValueError as e:
        raise SolverError("Input implies complex v_φ") from e
    except RuntimeWarning as w:
        log.error(w)
        raise SolverError("Cannot take sqrt") from w


def B_φ_prime_boundary_func(*, v_r, v_φ, a_0):
    """
    Compute B_φ_prime at the midplane
    """
    return v_φ * v_r / (2 * a_0)


def E_r_boundary_func(*, v_r, v_φ, η_P, η_H, η_perp_sq, a_0):
    """
    Compute E_r at the midplane
    """
    return -v_φ - v_r / η_P * (η_H + η_perp_sq * v_φ / (2 * a_0))


def define_conditions(inp):
    """
    Compute initial conditions based on input
    """
    ρ = float_type(1)
    B_θ = float_type(1)
    v_θ = float_type(0)
    B_r = float_type(0)
    B_φ = float_type(0)

    v_r = -inp.v_rin_on_c_s
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
        raise SolverError("Input implies NaN")

    return InitialConditions(
        norm_kepler_sq=norm_kepler_sq, a_0=a_0, init_con=init_con, angles=angles, γ=γ
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
        overrides=overrides,
        config_input=ConfigInput(
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
            v_θ_sonic_crit=config.get("config", "v_θ_sonic_crit", fallback="None"),
            after_sonic=config.get("config", "after_sonic", fallback="None"),
            interp_range=config.get("config", "interp_range", fallback="10"),
            sonic_interp_size=config.get(
                "config", "sonic_interp_size", fallback="None"
            ),
            interp_slice=config.get("config", "interp_slice", fallback="-1000,-100"),
            η_derivs=config.get("config", "η_derivs", fallback="True"),
            nwalkers=config.get("config", "nwalkers", fallback="8"),
            iterations=config.get("config", "iterations", fallback="3"),
            threads=config.get("config", "threads", fallback="1"),
            mcmc_vars=config.get("config", "mcmc_vars", fallback="v_r,v_a,v_k"),
            target_velocity=config.get("config", "target_velocity", fallback="0.9"),
            split_method=config.get("config", "split_method", fallback="v_θ_deriv"),
            use_taylor_jump=config.get("config", "use_taylor_jump", fallback="True"),
            use_E_r=config.get("config", "use_E_r", fallback="False"),
            γ=config.get("initial", "γ", fallback="0.001"),
            v_rin_on_c_s=config.get("initial", "v_rin_on_c_s", fallback="1"),
            v_a_on_c_s=config.get("initial", "v_a_on_c_s", fallback="1"),
            c_s_on_v_k=config.get("initial", "c_s_on_v_k", fallback="0.03"),
            η_O=config.get("initial", "η_O", fallback="0.001"),
            η_H=config.get("initial", "η_H", fallback="0.0001"),
            η_A=config.get("initial", "η_A", fallback="0.0005"),
        ),
    )


def new_inputs_with_overrides(config_input, solution_input, overrides):
    """
    Merge possibly differing ConfigInput and SolutionInput with additional
    changes within overrides
    """
    if overrides is None:
        overrides = {}
    new_config_input = add_overrides(config_input=config_input, overrides=overrides,)
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
