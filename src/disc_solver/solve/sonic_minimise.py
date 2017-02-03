# -*- coding: utf-8 -*-
"""
Find sonic point using minimisation
"""
from enum import IntEnum
from math import sqrt, tan, radians

from numpy import zeros, linspace, errstate, isnan
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from .config import define_conditions
from .solution import solution
from .utils import velocity_stop_generator

from ..file_format import InitialConditions, SolutionInput
from ..utils import ODEIndex

METHOD = "SLSQP"


class SonicVars(IntEnum):
    """
    """
    sonic_point = 0
    v_r = 1
    B_θ = 2
    B_r = 3
    B_φ = 4
    ρ = 5
    B_φ_prime = 6


class TotalVars(IntEnum):
    """
    """
    γ = 0
    v_rin_on_c_s = 1
    v_a_on_c_s = 2
    c_s_on_v_k = 3
    sonic_point = 4
    v_r = 5
    B_θ = 6
    B_r = 7
    B_φ = 8
    ρ = 9
    B_φ_prime = 10


def solver(inp, run, *, store_internal=True):
    """
    Minimisation solver
    """
    rootfn, rootfn_args = velocity_stop_generator(inp)
    cons = define_conditions(inp)
    initial_solution = solution(
        inp, cons, store_internal=store_internal,
        root_func=rootfn, root_func_args=rootfn_args,
    )
    result = minimize(
        generate_solve_from_root(
            inp, cons, initial_solution, store_internal=store_internal
        ), get_sonic_initial_guess(initial_solution), tol=1e-4,
        bounds=get_sonic_bounds(), method=METHOD, options={"maxiter":100000},
    )
    if not result.success:
        raise RuntimeError(
            "Minimisation solver failed on initial run with message: {}".format(
                result.message
            )
        )
    final_result = minimize(
        generate_total_solve(inp, rootfn, rootfn_args, store_internal),
        get_total_initial_guess(inp, result), bounds=get_total_bounds(),
        method=METHOD,
    )
    if not final_result.success:
        raise RuntimeError(
            "Minimisation solver failed on final run with message: {}".format(
                final_result.message
            )
        )


def generate_solve_from_root(inp, cons, initial_solution, store_internal):
    """
    """
    def solve_from_root(guess):
        mod_cons = sonic_vars_to_init_con(cons, guess, initial_solution)
        if mod_cons is None:
            return float("inf")
        soln = solution(
            inp, cons, with_taylor=False, modified_initial_conditions=mod_cons,
            store_internal=store_internal
        )
        return get_minimiser_value(
            soln.solution[-1], initial_solution.solution[-1]
        )
    return solve_from_root


def get_minimiser_value(attempt, actual):
    return sum((attempt - actual)**2)


def get_sonic_initial_guess(initial_solution):
    """
    """
    initial_guess = zeros(len(SonicVars))
    initial_guess[SonicVars.sonic_point] = initial_solution.angles[-1]
    for var in SonicVars:
        if hasattr(ODEIndex, var.name):
            initial_guess[var] = initial_solution.solution[
                -1, ODEIndex[var.name]
            ]
    if initial_guess[SonicVars.ρ] <= 0:
        initial_guess[SonicVars.ρ] = initial_solution.solution[
            -1, ODEIndex.ρ
        ]

    return initial_guess


def get_value_at_sonic_point(v_θ, variable):
    fit = interp1d(v_θ, variable, fill_value="extrapolate")
    return fit(1.0)


def generate_total_solve(inp, rootfn, rootfn_args, store_internal):
    """
    """
    def total_solve(guess):
        mod_inp, cons = total_vars_to_ode(inp, guess)
        if cons is None:
            return float("inf")
        #try:
        soln = solution(
            mod_inp, cons, store_internal=store_internal,
            root_func=rootfn, root_func_args=rootfn_args,
        )
        #except RuntimeError:
        #    return float("inf")
        mod_cons = total_vars_to_mod_cons(cons, guess, soln)
        sonic_soln = solution(
            mod_inp, cons, with_taylor=False,
            modified_initial_conditions=mod_cons,
            store_internal=store_internal
        )
        return get_minimiser_value(
            sonic_soln.solution[-1], soln.solution[-1]
        )
    return total_solve


def get_total_initial_guess(inp, result):
    """
    """
    initial_guess = zeros(len(TotalVars))
    for var in TotalVars:
        if hasattr(SonicVars, var.name):
            initial_guess[var] = result.x[SonicVars[var.name]]
        elif hasattr(inp, var.name):
            initial_guess[var] = getattr(inp, var.name)
    return initial_guess


def sonic_vars_to_init_con(cons, guess, initial_solution):
    mod_cons = InitialConditions(**vars(cons))
    mod_cons.angles = linspace(
        guess[SonicVars.sonic_point], initial_solution.angles[-1],
        len(cons.angles)
    )
    mod_cons.init_con[ODEIndex.v_θ] = 1
    for var in SonicVars:
        if hasattr(ODEIndex, var.name):
            mod_cons.init_con[ODEIndex[var.name]] = guess[var]

    θ = guess[SonicVars.sonic_point]
    a_0 = mod_cons.a_0
    γ = mod_cons.γ
    B_r = mod_cons.init_con[ODEIndex.B_r]
    B_φ = mod_cons.init_con[ODEIndex.B_φ]
    B_θ = mod_cons.init_con[ODEIndex.B_θ]
    v_r = mod_cons.init_con[ODEIndex.v_r]
    v_θ = mod_cons.init_con[ODEIndex.v_θ]
    ρ = mod_cons.init_con[ODEIndex.ρ]
    B_φ_prime = mod_cons.init_con[ODEIndex.B_φ_prime]
    η_O = mod_cons.init_con[ODEIndex.η_O]
    η_A = mod_cons.init_con[ODEIndex.η_A]
    η_H = mod_cons.init_con[ODEIndex.η_H]

    B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)
    with errstate(invalid="ignore"):
        norm_B_r, norm_B_φ, norm_B_θ = (
            B_r/B_mag, B_φ/B_mag, B_θ/B_mag
        )

    deriv_B_r = (
        (
            v_θ * B_r - v_r * B_θ + B_φ_prime * (
                η_H * norm_B_θ -
                η_A * norm_B_r * norm_B_φ
            ) + B_φ * (
                η_A * norm_B_φ * (
                    norm_B_θ * (1/4 - γ) +
                    norm_B_r * tan(θ)
                ) - η_H * (
                    norm_B_r * (1/4 - γ) +
                    norm_B_θ * tan(θ)
                )
            )
        ) / (
            η_O + η_A * (1 - norm_B_φ**2)
        ) - B_θ * (1/4 - γ)
    )

    try:
        mod_cons.init_con[ODEIndex.v_φ] = sqrt(- 1 - (
            v_r / 2 * (1 - 4 * γ) + a_0 / ρ * (
                (1/4 - γ) * B_θ * B_r + B_r * deriv_B_r + B_φ * B_φ_prime -
                B_φ ** 2 * tan(θ)
            )
        ) / tan(θ))
    except ValueError:
        return None
    if any(isnan(mod_cons.init_con)):
        return None

    return mod_cons


def total_vars_to_ode(inp, guess):
    mod_inp = SolutionInput(**vars(inp))
    for var in TotalVars:
        if hasattr(inp, var.name):
            setattr(inp, var.name, guess[var])
    #try:
    cons = define_conditions(mod_inp)
    #except ValueError:
    #    cons = None
    return mod_inp, cons


def total_vars_to_mod_cons(cons, guess, initial_solution):
    mod_cons = InitialConditions(**vars(cons))
    mod_cons.angles = linspace(
        guess[TotalVars.sonic_point], initial_solution.angles[-1],
        len(cons.angles)
    )
    mod_cons.init_con[ODEIndex.v_θ] = 1
    for var in TotalVars:
        if hasattr(ODEIndex, var.name):
            mod_cons.init_con[ODEIndex[var.name]] = guess[var]
    return mod_cons


def total_vars_to_mod_cons_pre(cons, guess):
    mod_cons = InitialConditions(**vars(cons))
    mod_cons.init_con[ODEIndex.v_θ] = 1
    for var in TotalVars:
        if hasattr(ODEIndex, var.name):
            mod_cons.init_con[ODEIndex[var.name]] = guess[var]
    return mod_cons


def get_sonic_bounds():
    bounds = [[None, None],] * len(SonicVars)
    bounds[SonicVars.ρ][0] = 0
    bounds[SonicVars.B_θ][0] = 0
    bounds[SonicVars.B_r][0] = 0
    bounds[SonicVars.B_φ][1] = 0
    bounds[SonicVars.sonic_point] = (radians(0.5), radians(10))
    return bounds


def get_total_bounds():
    bounds = [[None, None],] * len(TotalVars)
    bounds[TotalVars.sonic_point] = (radians(0.5), radians(10))
    bounds[TotalVars.γ] = (0, 0.25)
    bounds[TotalVars.v_rin_on_c_s][0] = 0
    bounds[TotalVars.v_a_on_c_s] = (0.2, 5)
    bounds[TotalVars.c_s_on_v_k] = (0.01, 0.05)
    bounds[TotalVars.ρ][0] = 0
    bounds[TotalVars.B_θ][0] = 0
    bounds[TotalVars.B_r][0] = 0
    bounds[TotalVars.B_φ][1] = 0
    return bounds
