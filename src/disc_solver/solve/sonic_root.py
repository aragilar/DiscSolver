# -*- coding: utf-8 -*-
"""
Find sonic point using minimisation
"""
from enum import IntEnum

import logbook

from numpy import (
    concatenate, diff, full, isnan, linspace, zeros, array, sqrt, log, exp,
)

from scipy.optimize import root

from scikits.odes.sundials.cvode import StatusEnum as StatusFlag

from .config import define_conditions
from .solution import solution
from .utils import velocity_stop_generator, SolverError

from ..float_handling import float_type, FLOAT_TYPE_INFO
from ..file_format import InitialConditions, Solution
from ..utils import ODEIndex

logger = logbook.Logger(__name__)
ROOT_METHOD = "anderson"
θ_SCALE_INITIAL_STOP = float_type(0.9)
ERR_FLOAT = -sqrt(FLOAT_TYPE_INFO.max) / 10
INITIAL_SPREAD = 0.1
LINE_SEARCH_METHOD = None
COMPARING_INDICES = [
    ODEIndex.v_r,
    ODEIndex.v_θ,
    ODEIndex.v_φ,
    ODEIndex.B_r,
    ODEIndex.B_θ,
    ODEIndex.B_φ,
]


class TotalVars(IntEnum):
    """
    Variables for root solver
    """
    θ_sonic = 0
    v_φ = 1
    v_r = 2
    B_θ = 3
    B_r = 4
    B_φ = 5
    log_ρ = 6
    B_φ_prime = 7


def solver(inp, run, *, store_internal=True):
    """
    Minimisation solver
    """
    cons = define_conditions(inp)
    initial_solution = solution(
        inp, cons, store_internal=store_internal,
        root_func=velocity_stop_generator(inp.target_velocity),
        root_func_args=1,
    )
    run.solutions.add_solution(initial_solution)

    root_solver_func = generate_root_func(
        initial_conditions=cons, run=run, store_internal=store_internal,
        initial_solution=initial_solution, solution_input=inp,
    )
    root_func = wrap_root_catch_error(root_solver_func)
    best_guess = guess_root_vals(inp, initial_solution)

    result = root(
        root_func, best_guess, method=ROOT_METHOD, options={
            "maxiter": inp.iterations, "line_search": LINE_SEARCH_METHOD,
        },
    )

    if not result.success:
        logger.error("Root solver failed with message: {}".format(
            result.message
        ))

    run.final_solution = run.solutions[str(len(run.solutions) - 1)]
    return result.success


def generate_root_func(
    *, run, solution_input, initial_conditions, initial_solution,
    store_internal
):
    """
    Create function to pass to root solver
    """
    initial_cons = initial_conditions
    initial_soln = initial_solution
    sonic_stop = get_sonic_stop(initial_solution)
    inp = solution_input

    def root_func(guess):
        """
        Function for root solver
        """
        sonic_cons = total_vars_to_mod_cons(
            initial_conditions=initial_cons, guess=guess, sonic_stop=sonic_stop
        )

        sonic_soln = solution(
            inp, initial_cons, with_taylor=False,
            modified_initial_conditions=sonic_cons,
            store_internal=store_internal,
        )

        write_solution(run, initial_soln, sonic_soln)
        return get_root_results(
            sonic_soln=sonic_soln,
            midplane_soln=initial_soln
        )
    return root_func


def get_root_results(*, sonic_soln, midplane_soln):
    """
    Get value of solution for root solver
    """
    sonic_values = sonic_soln.solution
    midplane_values = midplane_soln.solution

    if sonic_values.size == 0:
        raise SolverError("Sonic solution failed")

    root_results = list(
        sonic_values[-1, COMPARING_INDICES] -
        midplane_values[-1, COMPARING_INDICES]
    )
    if not root_results:
        raise SolverError("Subtraction of solutions failed")
    root_results.append(log(
        sonic_values[-1, ODEIndex.ρ] / midplane_values[-1, ODEIndex.ρ]
    ))
    root_results.append(sonic_soln.angles[-1] - midplane_soln.angles[-1])
    logger.info("Root diff is {}".format(root_results))
    return array(root_results)


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


def get_sonic_point_value(soln, name):
    """
    Extrapolate the value at the sonic point
    """
    d_v_θ = (1 - soln.solution[-1, ODEIndex.v_θ])
    derivs = diff(soln.solution, axis=0)[-1]

    if name == "θ_sonic":
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


def total_vars_to_mod_cons(*, initial_conditions, guess, sonic_stop):
    """
    Convert root solver variables to modified initial conditions
    """
    θ_sonic = guess[TotalVars.θ_sonic]
    mod_cons = InitialConditions(**vars(initial_conditions))
    mod_cons.angles = linspace(
        θ_sonic, sonic_stop, len(initial_conditions.angles)
    )

    mod_cons.init_con[ODEIndex.v_θ] = float_type(1)
    for var in TotalVars:
        if hasattr(ODEIndex, var.name):
            mod_cons.init_con[ODEIndex[var.name]] = guess[var]
        elif var.name == "log_ρ":
            mod_cons.init_con[ODEIndex.ρ] = exp(guess[var])

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
            return root_func(*args, **kwargs)
        except SolverError as e:
            logger.exception(e)
            return full(len(TotalVars), ERR_FLOAT)
    return new_root_func


def get_sonic_stop(initial_solution):
    """
    Get point at where to stop solution from sonic point
    """
    return initial_solution.angles[-1]
