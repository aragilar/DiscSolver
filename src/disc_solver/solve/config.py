# -*- coding: utf-8 -*-
"""
Define input and environment for ode system
"""

import attr
import logbook

import numpy as np
from numpy import sqrt

from ..float_handling import float_type, FLOAT_TYPE
from ..file_format import (
    ConfigInput, InitialConditions, SolutionInput, MCMCVars,
)
from ..utils import (
    str_to_float, str_to_int, str_to_bool, CaseDependentConfigParser,
    ODEIndex,
)

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
    except ValueError:
        raise SolverError("Input implies complex v_φ")


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


def mcmc_vars_str_to_obj(mcmc_str):
    """
    Convert mcmc_vars string to MCMCVars
    """
    split_str = mcmc_str.strip().split(',')
    return MCMCVars(
        with_v_r="v_r" in split_str,
        with_v_a="v_a" in split_str,
        with_v_k="v_k" in split_str,
    )


def define_conditions(inp, *, use_E_r=False):
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

    if use_E_r:
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

    return add_overrides(overrides=overrides, config_input=ConfigInput(
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
        v_θ_sonic_crit=config.get(
            "config", "v_θ_sonic_crit", fallback="None"
        ),
        η_derivs=config.get("config", "η_derivs", fallback="True"),
        nwalkers=config.get("config", "nwalkers", fallback="8"),
        iterations=config.get("config", "iterations", fallback="3"),
        threads=config.get("config", "threads", fallback="1"),
        mcmc_vars=config.get("config", "mcmc_vars", fallback="v_r,v_a,v_k"),
        target_velocity=config.get(
            "config", "target_velocity", fallback="0.9"
        ),
        split_method=config.get(
            "config", "split_method", fallback="v_θ_deriv"
        ),
        use_taylor_jump=config.get(
            "config", "use_taylor_jump", fallback="True"
        ),
        γ=config.get("initial", "γ", fallback="0.001"),
        v_rin_on_c_s=config.get("initial", "v_rin_on_c_s", fallback="1"),
        v_a_on_c_s=config.get("initial", "v_a_on_c_s", fallback="1"),
        c_s_on_v_k=config.get("initial", "c_s_on_v_k", fallback="0.03"),
        η_O=config.get("initial", "η_O", fallback="0.001"),
        η_H=config.get("initial", "η_H", fallback="0.0001"),
        η_A=config.get("initial", "η_A", fallback="0.0005"),
    ))


def config_input_to_soln_input(inp):
    """
    Convert user input into solver input
    """
    return SolutionInput(
        start=float_type(str_to_float(inp.start)),
        stop=float_type(str_to_float(inp.stop)),
        taylor_stop_angle=float_type(str_to_float(inp.taylor_stop_angle)),
        max_steps=str_to_int(inp.max_steps),
        num_angles=str_to_int(inp.num_angles),
        relative_tolerance=float_type(str_to_float(inp.relative_tolerance)),
        absolute_tolerance=float_type(str_to_float(inp.absolute_tolerance)),
        jump_before_sonic=(
            None if inp.jump_before_sonic == "None"
            else float_type(str_to_float(inp.jump_before_sonic))
        ),
        v_θ_sonic_crit=(
            None if inp.v_θ_sonic_crit == "None" or inp.v_θ_sonic_crit is None
            else float_type(str_to_float(inp.v_θ_sonic_crit))
        ),
        η_derivs=str_to_bool(inp.η_derivs),
        nwalkers=str_to_int(inp.nwalkers),
        iterations=str_to_int(inp.iterations),
        threads=str_to_int(inp.threads),
        target_velocity=float_type(str_to_float(inp.target_velocity)),
        split_method=inp.split_method,
        use_taylor_jump=str_to_bool(inp.use_taylor_jump),
        mcmc_vars=mcmc_vars_str_to_obj(inp.mcmc_vars),
        γ=float_type(str_to_float(inp.γ)),
        v_rin_on_c_s=float_type(str_to_float(inp.v_rin_on_c_s)),
        v_a_on_c_s=float_type(str_to_float(inp.v_a_on_c_s)),
        c_s_on_v_k=float_type(str_to_float(inp.c_s_on_v_k)),
        η_O=float_type(str_to_float(inp.η_O)),
        η_H=float_type(str_to_float(inp.η_H)),
        η_A=float_type(str_to_float(inp.η_A)),
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
    new_soln_input = config_input_to_soln_input(new_config_input)

    initial_soln_input = config_input_to_soln_input(config_input)

    if solution_input == initial_soln_input:
        return new_config_input, new_soln_input

    conf_dict = attr.asdict(initial_soln_input)
    soln_dict = attr.asdict(solution_input)
    use_soln_keys = []
    for key, val in conf_dict.items():
        if key in overrides:
            continue
        elif val != soln_dict[key]:
            use_soln_keys.append(key)

    for key in use_soln_keys:
        setattr(new_soln_input, key, soln_dict[key])

    return new_config_input, new_soln_input


def config_input_to_conf_file(filename, config_input):
    """
    Convert ConfigInput to cfg file
    """
    cfg = CaseDependentConfigParser()

    cfg["config"]["start"] = config_input.start
    cfg["config"]["stop"] = config_input.stop
    cfg["config"]["taylor_stop_angle"] = config_input.taylor_stop_angle
    cfg["config"]["max_steps"] = config_input.max_steps
    cfg["config"]["num_angles"] = config_input.num_angles
    cfg["config"]["label"] = config_input.label
    cfg["config"]["relative_tolerance"] = config_input.relative_tolerance
    cfg["config"]["absolute_tolerance"] = config_input.absolute_tolerance
    cfg["config"]["nwalkers"] = config_input.nwalkers
    cfg["config"]["iterations"] = config_input.iterations
    cfg["config"]["threads"] = config_input.threads
    cfg["config"]["target_velocity"] = config_input.target_velocity
    cfg["config"]["split_method"] = config_input.split_method

    cfg["initial"]["γ"] = config_input.γ
    cfg["initial"]["v_rin_on_c_s"] = config_input.v_rin_on_c_s
    cfg["initial"]["v_a_on_c_s"] = config_input.v_a_on_c_s
    cfg["initial"]["c_s_on_v_k"] = config_input.c_s_on_v_k
    cfg["initial"]["η_O"] = config_input.η_O
    cfg["initial"]["η_H"] = config_input.η_H
    cfg["initial"]["η_A"] = config_input.η_A

    cfg["config"]["η_derivs"] = config_input.η_derivs
    cfg["config"]["use_taylor_jump"] = config_input.use_taylor_jump

    if config_input.jump_before_sonic is not None:
        cfg["config"]["jump_before_sonic"] = config_input.jump_before_sonic
    if config_input.mcmc_vars is not None:
        cfg["config"]["mcmc_vars"] = config_input.mcmc_vars
    if config_input.v_θ_sonic_crit is not None:
        cfg["config"]["v_θ_sonic_crit"] = config_input.v_θ_sonic_crit

    with open(filename, 'w') as f:
        cfg.write(f)
