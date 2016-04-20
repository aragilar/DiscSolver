# -*- coding: utf-8 -*-
"""
Stepper related logic
"""

import logbook

from numpy import diff

from .config import define_conditions
from .solution import solution
from .utils import validate_solution, onroot_continue
from ..file_format import Solution, SolutionInput
from ..utils import allvars as vars

log = logbook.Logger(__name__)


class StepperError(RuntimeError):
    """
    Base class for exceptions related to the stepper
    """
    pass


def binary_searcher(
    func, pass_func, fail_func, initial_input, num_attempts=100
):
    """
    Search for correct solution via a binary search
    """
    inp = initial_input
    try:
        for attempt in range(0, num_attempts):
            log.notice("Attempt {}:".format(attempt))
            out, sucess = func(inp)
            if sucess:
                return pass_func(out, attempt)
            else:
                inp = fail_func(out, attempt)
    except StepperError as e:
        log.critical(str(e))
    else:
        log.critical("Failed to find with {} attempts.".format(num_attempts))


def stepper_creator(
    soln_writer, step_func, get_soln_type, initial_step_size=1e-4,
    max_search_steps=10,
):
    """
    Create stepper
    """
    search_steps = 0
    binary_started = False
    prev_soln_type = None
    step_size = initial_step_size

    def stepper(soln, attempt):
        """
        Stepper func
        """
        nonlocal binary_started, search_steps, prev_soln_type, step_size
        soln_writer(soln, attempt)
        soln_type = get_soln_type(soln)
        log.notice("solution type: {}".format(soln_type))
        if not binary_started:
            search_steps += 1
            if search_steps > max_search_steps:
                raise StepperError(
                    "Failed to start binary search after {} attempts.".format(
                        search_steps
                    )
                )

            if prev_soln_type is not None:
                if prev_soln_type != soln_type:
                    binary_started = True
                    step_size /= 2
                return step_func(soln, soln_type, step_size)
            else:
                prev_soln_type = soln_type
                return step_func(soln, soln_type, step_size)

        step_size /= 2
        return step_func(soln, soln_type, step_size)

    return stepper


def writer_generator(run):
    """
    adds soln to file
    """
    def writer(soln, attempt):
        """
        writer
        """
        run.solutions[str(attempt)] = soln

    return writer


def cleanup_generator(run, writer):
    """
    creates symlink to correct solution
    """
    def cleanup(soln, attempt):
        """
        cleanup
        """
        writer(soln, attempt)
        run.final_solution = run.solutions[str(attempt)]

    return cleanup


def create_soln_splitter(method):
    """
    Create func to see split in solution
    """
    def v_θ_deriv(soln, num_check=10, start=-10):
        """
        Use derivative of v_θ to determine type of solution
        """
        v_θ = soln.solution[:, 5]
        problems = soln.internal_data.problems
        if any("negative velocity" in pl for pl in problems.values()):
            return "sign flip"
        if (num_check + start > 0) and start < 0:
            log.info("Using fewer samples than requested.")
            d_v_θ = diff(v_θ[start:])
        elif num_check + start == 0:
            d_v_θ = diff(v_θ[start:])
        else:
            d_v_θ = diff(v_θ[start:start + num_check])

        if all(d_v_θ > 0):
            return "diverge"
        elif all(d_v_θ < 0):
            return "sign flip"
        return "unknown"
    method_dict = {"v_θ_deriv": v_θ_deriv}

    return method_dict.get(method) or v_θ_deriv


def solver_generator():
    """
    Generate solver func
    """
    def solver(inp):
        """
        solver
        """
        inp = SolutionInput(**vars(inp))
        cons = define_conditions(inp)
        soln, internal_data, coords = solution(
            cons.angles, cons.init_con, cons.β, cons.c_s, cons.norm_kepler_sq,
            relative_tolerance=inp.relative_tolerance,
            absolute_tolerance=inp.absolute_tolerance,
            max_steps=inp.max_steps, taylor_stop_angle=inp.taylor_stop_angle,
            onroot_func=onroot_continue, find_sonic_point=True
        )
        soln = Solution(
            solution_input=inp, initial_conditions=cons, flag=soln.flag,
            coordinate_system=coords, internal_data=internal_data,
            angles=soln.values.t, solution=soln.values.y,
            t_roots=soln.roots.t, y_roots=soln.roots.y,
        )
        return validate_solution(soln)

    return solver


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
