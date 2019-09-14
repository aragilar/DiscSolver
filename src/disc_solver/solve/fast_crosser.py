# -*- coding: utf-8 -*-
"""
fast-crosser: combines different solvers to cross fast speed
"""
import logbook

from numpy import max as np_max
from scipy.optimize import minimize

from .single import solver as single_solver
from .stepper import solver as step_solver

from ..analyse.utils import get_mach_numbers
from ..file_format import SolutionInput

V_R_BOUNDS = ((0.25, 1.0),)

log = logbook.Logger(__name__)


def compute_minimiser_value(soln):
    """
    Compute value for minimiser to cross fast speed
    """
    _, _, _, fast_mach = get_mach_numbers(soln)
    return - np_max(fast_mach)


def minimiser_generator(
    *, soln_input, run, final_stop, final_steps, store_internal
):
    """
    Create function to pass to minimiser
    """
    v_θ_sonic_crit = soln_input.v_θ_sonic_crit
    jump_before_sonic = soln_input.jump_before_sonic
    after_sonic = soln_input.after_sonic
    get_last_solution = run.solutions.get_last_solution

    def minimiser(v_rin_on_c_s):
        """
        Function for minimiser
        """
        soln_input_dict = soln_input.asdict()
        soln_input_dict.update(
            v_rin_on_c_s=v_rin_on_c_s,
            v_θ_sonic_crit=None,
            after_sonic=None,
            jump_before_sonic=None,
        )

        step_solver(
            SolutionInput(**soln_input_dict), run,
            store_internal=store_internal, no_final_solution=True
        )
        final_solution_dict = get_last_solution().solution_input.asdict()
        final_solution_dict.update(
            v_θ_sonic_crit=v_θ_sonic_crit, after_sonic=after_sonic,
            jump_before_sonic=jump_before_sonic,
            stop=final_stop, num_angles=final_steps,
        )
        single_solver(
            SolutionInput(**final_solution_dict), run,
            store_internal=store_internal, no_final_solution=True,
        )
        log.info("Minimise ran at v_rin_on_c_s = {}".format(v_rin_on_c_s))
        return compute_minimiser_value(get_last_solution())

    return minimiser


def solver(
    soln_input, run, *, store_internal=True, final_stop=90, final_steps=900000
):
    """
    fast-crosser solver

    uses step solver and single solver plus minimisation to cross fast speed
    """
    minimiser = minimiser_generator(
        soln_input=soln_input, run=run, final_stop=final_stop,
        final_steps=final_steps, store_internal=store_internal,
    )
    res = minimize(minimiser, soln_input.v_rin_on_c_s, bounds=V_R_BOUNDS)
    log.info("Minimiser stopped: {}".format(res.message))
    log.notice("Adding best solution")

    v_θ_sonic_crit = soln_input.v_θ_sonic_crit
    after_sonic = soln_input.after_sonic
    get_last_solution = run.solutions.get_last_solution

    soln_input_dict = soln_input.asdict()
    soln_input_dict.update(
        v_rin_on_c_s=res.x,
        v_θ_sonic_crit=None,
        after_sonic=None,
        jump_before_sonic=None,
    )

    step_solver(
        SolutionInput(**soln_input_dict), run,
        store_internal=store_internal, no_final_solution=True
    )
    final_solution_dict = get_last_solution().solution_input.asdict()
    final_solution_dict.update(
        v_θ_sonic_crit=v_θ_sonic_crit, after_sonic=after_sonic,
        stop=final_stop, num_angles=final_steps,
    )
    single_solver(
        SolutionInput(**final_solution_dict), run,
        store_internal=store_internal,
    )
