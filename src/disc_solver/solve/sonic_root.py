# -*- coding: utf-8 -*-
"""
Find sonic point using minimisation
"""
from enum import IntEnum
from math import sqrt, tan, log, exp
from sys import float_info

import logbook

import emcee

from numpy import (
    argmax, concatenate, diff, errstate, full, isnan, linspace, zeros,
)
from numpy.random import randn

from scipy.optimize import root

from scikits.odes.sundials.cvode import StatusEnum as StatusFlag

from .config import define_conditions
from .solution import solution
from .utils import velocity_stop_generator, SolverError

from ..file_format import InitialConditions, SolutionInput, Solution
from ..utils import ODEIndex

logger = logbook.Logger(__name__)
METHOD = "anderson"
θ_SCALE_INITIAL_STOP = 0.9
ERR_FLOAT = -sqrt(float_info.max) / 10
INITIAL_SPREAD = 0.1


class TotalVars(IntEnum):
    """
    Variables for root solver
    """
    γ = 0
    θ_scale = 1
    v_φ = 2
    B_θ = 3
    B_r = 4
    B_φ = 5
    log_ρ = 6
    B_φ_prime = 7


def solver(inp, run, *, store_internal=True):
    """
    Minimisation solver
    """
    rootfn, rootfn_args = velocity_stop_generator(inp)
    initial_solution = solution(
        inp, define_conditions(inp), store_internal=store_internal,
        root_func=rootfn, root_func_args=rootfn_args,
    )
    run.solutions.add_solution(initial_solution)

    root_solver_func = generate_root_func(
        inp=inp, run=run, store_internal=store_internal,
    )
    mcmc_func = wrap_root_with_mcmc(root_solver_func)
    root_func = wrap_root_catch_error(root_solver_func)

    nwalkers = max(len(TotalVars) * 2, inp.nwalkers)
    sampler = emcee.EnsembleSampler(
        nwalkers, len(TotalVars), mcmc_func, threads=inp.threads,
    )
    with errstate(invalid="ignore"):
        sampler.run_mcmc(
            generate_initial_positions(
                guess_root_vals(inp, initial_solution), nwalkers
            ), inp.iterations
        )
    best_index = argmax(sampler.flatlnprobability)
    best_guess = sampler.flatchain[best_index]
    result = root(
        root_func, best_guess, method=METHOD, options={
            "maxiter": 50, "line_search": None,
        },
    )

    if not result.success:
        logger.error("Root solver failed with message: {}".format(
            result.message
        ))

    run.final_solution = run.solutions[str(len(run.solutions) - 1)]


def generate_root_func(*, run, inp, store_internal):
    """
    Create function to pass to root solver
    """
    def root_func(guess):
        """
        Function for root solver
        """
        mod_inp, initial_cons, θ_scale = total_vars_to_ode(inp, guess)
        soln = solution(
            mod_inp, initial_cons, store_internal=store_internal,
            θ_scale=θ_scale,
        )

        sonic_cons = total_vars_to_mod_cons(initial_cons, guess)

        sonic_soln = solution(
            mod_inp, initial_cons, with_taylor=False,
            modified_initial_conditions=sonic_cons,
            store_internal=store_internal, θ_scale=θ_scale,
        )

        write_solution(run, soln, sonic_soln)
        return get_root_results(
            sonic_soln.solution, soln.solution
        )
    return root_func


def get_root_results(sonic_values, midplane_values):
    """
    Get value of solution for root solver
    """
    root_results = sonic_values[-1, :8] - midplane_values[-1, :8]
    root_results[ODEIndex.ρ] = log(
        sonic_values[-1, ODEIndex.ρ] / midplane_values[-1, ODEIndex.ρ]
    )
    return root_results


def guess_root_vals(inp, initial_solution):
    """
    Create initial guess for root solver
    """
    initial_guess = zeros(len(TotalVars))
    for var in TotalVars:
        if hasattr(inp, var.name):
            initial_guess[var] = getattr(inp, var.name)
        else:
            initial_guess[var] = get_sonic_point_value(
                initial_solution, var.name
            )
    return initial_guess


def generate_initial_positions(initial_guess, nwalkers):
    """
    Generate initial positions of walkers
    """
    return initial_guess * (
        INITIAL_SPREAD * randn(nwalkers, len(initial_guess)) + 1
    )


def get_sonic_point_value(soln, name):
    """
    Extrapolate the value at the sonic point
    """
    d_v_θ = (1 - soln.solution[-1, ODEIndex.v_θ])
    derivs = diff(soln.solution, axis=0)[-1]

    if name == "θ_scale":
        return soln.angles[-1] / θ_SCALE_INITIAL_STOP
    elif name == "log_ρ":
        ρ = soln.solution[:, ODEIndex.ρ]
        deriv_ρ = log(ρ[-1] / ρ[-2])
        return (
            log(soln.solution[-1, ODEIndex.ρ]) +
            deriv_ρ / derivs[ODEIndex.v_θ] * d_v_θ
        )
    return (
        soln.solution[-1, ODEIndex[name]] +
        derivs[ODEIndex[name]] / derivs[ODEIndex.v_θ] * d_v_θ
    )


def total_vars_to_ode(inp, guess):
    """
    Convert root solver variables to solution input
    """
    mod_inp = SolutionInput(**vars(inp))
    mod_inp.γ = guess[TotalVars.γ]
    θ_scale = guess[TotalVars.θ_scale]
    cons = define_conditions(mod_inp)
    cons.angles = linspace(0, θ_SCALE_INITIAL_STOP, mod_inp.num_angles)

    return mod_inp, cons, θ_scale


def total_vars_to_mod_cons(cons, guess):
    """
    Convert root solver variables to modified initial conditions
    """
    mod_cons = InitialConditions(**vars(cons))
    mod_cons.angles = linspace(1, θ_SCALE_INITIAL_STOP, len(cons.angles))

    mod_cons.init_con[ODEIndex.v_θ] = 1
    for var in TotalVars:
        if hasattr(ODEIndex, var.name):
            mod_cons.init_con[ODEIndex[var.name]] = guess[var]
        elif var.name == "log_ρ":
            mod_cons.init_con[ODEIndex.ρ] = exp(guess[var])

    θ_scale = guess[TotalVars.θ_scale]
    θ = θ_scale
    a_0 = mod_cons.a_0
    γ = mod_cons.γ
    B_r = mod_cons.init_con[ODEIndex.B_r]
    B_φ = mod_cons.init_con[ODEIndex.B_φ]
    B_θ = mod_cons.init_con[ODEIndex.B_θ]
    v_φ = mod_cons.init_con[ODEIndex.v_φ]
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

    mod_cons.init_con[ODEIndex.v_r] = ρ * (
        η_O + η_A * (1 - norm_B_φ**2)
    ) / (
        B_r * B_θ * a_0 - (1 / 2 - 2 * γ) * ρ * (
            η_O + η_A * (1 - norm_B_φ**2)
        )
    ) * (
        tan(θ) * (v_φ ** 2 + 1) + a_0 / ρ * (
            B_r * (
                B_r + B_φ_prime * (
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
            ) + B_φ * B_φ_prime - B_φ ** 2 * tan(θ)
        )
    )

    if any(isnan(mod_cons.init_con)):
        raise SolverError("Initial conditions contains NaN: {}".format(
            mod_cons.init_con
        ))

    return mod_cons


def write_solution(run, initial_solution, sonic_solution):
    """
    Write solution created by root solver to file
    """
    if initial_solution.solution_input != sonic_solution.solution_input:
        raise SolverError("Input changed between initial and sonic")
    if (
        initial_solution.initial_conditions !=
        sonic_solution.initial_conditions
    ):
        raise SolverError(
            "Initial conditions changed between initial and sonic"
        )
    if initial_solution.coordinate_system != sonic_solution.coordinate_system:
        raise SolverError("Coordinates changed between initial and sonic")

    if initial_solution.flag != StatusFlag.ROOT_RETURN:
        logger.warn("Initial solution did not reach target velocity")
    if initial_solution.flag < 0:
        logger.warn("Initial solution failed with flag {}".format(
            initial_solution.flag.name
        ))
    if sonic_solution.flag < 0:
        logger.warn("Sonic solution failed with flag {}".format(
            sonic_solution.flag.name
        ))
    elif sonic_solution.flag != StatusFlag.SUCESS:
        logger.info("Sonic solution ended with flag {}".format(
            sonic_solution.flag.name
        ))
    final_flag = sonic_solution.flag

    joined_angles = concatenate((
        initial_solution.angles, sonic_solution.angles[::-1]
    ))
    joined_solution = concatenate((
        initial_solution.solution, sonic_solution.solution[::-1]
    ))
    if initial_solution.internal_data is None and (
        sonic_solution.internal_data is None
    ):
        joined_internal_data = None
    elif initial_solution.internal_data is None:
        joined_internal_data = sonic_solution.internal_data.flip()
    elif sonic_solution.internal_data is None:
        joined_internal_data = initial_solution.internal_data
    else:
        joined_internal_data = (
            initial_solution.internal_data +
            sonic_solution.internal_data.flip()
        )

    run.solutions.add_solution(Solution(
        solution_input=initial_solution.solution_input,
        coordinate_system=initial_solution.coordinate_system,
        initial_conditions=initial_solution.initial_conditions,
        angles=joined_angles,
        solution=joined_solution,
        flag=final_flag,
        internal_data=joined_internal_data,
        t_roots=initial_solution.t_roots,
        y_roots=initial_solution.y_roots,
        sonic_point=None, sonic_point_values=None,
    ))


def wrap_root_catch_error(root_func):
    """
    Catch exception in root func and return correct flag value
    """
    def new_root_func(*args, **kwargs):
        """
        Wrapper of root func
        """
        try:
            root_func(*args, **kwargs)
        except SolverError as e:
            logger.exception(e)
            return full(len(TotalVars), ERR_FLOAT)
    return new_root_func


def wrap_root_with_mcmc(root_solver_func):
    """
    Convert root solver function to one useable by emcee
    """
    def mcmc_func(*args, **kwargs):
        """
        Wrapper function for mcmc
        """
        try:
            return - sum(root_solver_func(*args, **kwargs) ** 2)
        except SolverError as e:
            logger.exception(e)
            return - float("inf")

    return mcmc_func
