# -*- coding: utf-8 -*-
"""
fast-crosser: combines different solvers to cross fast speed
"""
from enum import Enum

import logbook
from numpy import max as np_max
from numpy import sqrt

from ..analyse.utils import get_mach_numbers
from ..file_format import Solution, SolutionInput
from .single import solver as single_solver
from .solution import get_known_broken_solution
from .stepper import (StepperError, StepperStop, alternate_cleanup_generator,
                      binary_searcher, cleanup_generator)
from .stepper import solver as step_solver
from .stepper import writer_generator
from .utils import SolverError

log = logbook.Logger(__name__)

DEFAULT_NUM_ATTEMPTS = 1000
DEFAULT_MAX_SEARCH_STEPS = 200
DEFAULT_FINAL_STOP = 90
DEFAULT_FINAL_STEPS = 900000
GOLDEN_RATIO = (1 + sqrt(5)) / 2


class Phase(Enum):
    """
    The phases in which the stepper can be in
    """

    STARTING = "starting"
    FINDING_MAX = "finding max"
    FINDING_MIN = "finding min"
    BISECTING = "bisecting"


def compute_quality(soln):
    """
    Compute the value for comparing the solutions
    """
    _, _, _, fast_mach = get_mach_numbers(soln)
    return np_max(fast_mach)


def stepper_creator(soln_writer, *, step_size, solution_input):
    """
    Create stepper
    """

    def get_next_input(current_input, new_v_r):
        current_input.update(
            stop=solution_input.stop,
            num_angles=solution_input.num_angles,
            v_rin_on_c_s=new_v_r,
        )
        return current_input

    limits = {
        "min": None,
        "min_val": None,
        "max": None,
        "max_val": None,
        "mid": None,
        "mid_val": None,
    }
    phase = Phase.STARTING

    def stepper(soln, attempt):
        """
        Stepper func
        """
        nonlocal phase
        soln_writer(soln, attempt)
        inp_dict = vars(soln.solution_input)
        current_v_r = inp_dict["v_rin_on_c_s"]
        current_quality = compute_quality(soln)
        log.notice("v_r is {}", current_v_r)

        if phase == Phase.STARTING:
            log.notice("starting search for limits")
            limits["min"] = current_v_r
            limits["max"] = current_v_r
            limits["min_val"] = current_quality
            limits["max_val"] = current_quality
            phase = Phase.FINDING_MAX
            log.notice("starting search for max limit")
            return SolutionInput(
                **get_next_input(inp_dict, inp_dict["v_rin_on_c_s"] + step_size)
            )

        if phase == Phase.FINDING_MAX:
            if current_quality > limits["max_val"]:
                limits["max"] = current_v_r
                limits["max_val"] = current_quality
                log.notice("searching for max limit on attempt {}", attempt)
                return SolutionInput(
                    **get_next_input(inp_dict, inp_dict["v_rin_on_c_s"] + step_size)
                )
            phase = Phase.FINDING_MIN
            log.notice("starting search for min limit")
            return SolutionInput(**get_next_input(inp_dict, limits["min"] - step_size))

        if phase == Phase.FINDING_MIN:
            if current_quality > limits["min_val"]:
                limits["min"] = current_v_r
                limits["min_val"] = current_quality
                log.notice("searching for min limit on attempt {}", attempt)
                return SolutionInput(
                    **get_next_input(inp_dict, inp_dict["v_rin_on_c_s"] - step_size)
                )
            phase = Phase.BISECTING
            log.notice("found limits {} {}, bisecting", limits["min"], limits["max"])
            if limits["min"] == limits["max"]:
                raise StepperError("Unable to step off value")
            # This sets up the golden-section search by starting with the
            # golden ratio
            return SolutionInput(
                **get_next_input(
                    inp_dict,
                    limits["min"]
                    + ((limits["min"] - limits["max"]) / (GOLDEN_RATIO - 1)),
                )
            )
        if phase == Phase.BISECTING:
            log.notice("bisecting on attempt {}", attempt)
            if limits["mid"] is None:
                limits["mid"] = current_v_r
                limits["mid_val"] = current_quality
                return SolutionInput(
                    **get_next_input(
                        inp_dict, limits["min"] + (limits["max"] - limits["mid"])
                    )
                )
            if limits["mid"] > current_v_r:
                lmid_lim = current_v_r
                hmid_lim = limits["mid"]
                lmid_val = current_quality
                hmid_val = limits["mid_val"]
            else:
                hmid_lim = current_v_r
                lmid_lim = limits["mid"]
                hmid_val = current_quality
                lmid_val = limits["mid_val"]
            if lmid_val > hmid_val:
                limits["mid"] = lmid_lim
                limits["mid_val"] = lmid_val
                limits["max"] = hmid_lim
                limits["max_val"] = hmid_val
            else:
                limits["min"] = lmid_lim
                limits["min_val"] = lmid_val
                limits["mid"] = hmid_lim
                limits["mid_val"] = hmid_val
            new_v_r = limits["min"] + (limits["max"] - limits["mid"])
            if new_v_r == current_v_r:
                StepperStop("Hit numerical limit")
            return SolutionInput(**get_next_input(inp_dict, new_v_r))
        else:
            raise StepperError("Unknown phase")

    return stepper


def solution_generator(
    *,
    store_internal=True,
    run,
    final_stop=DEFAULT_FINAL_STOP,
    final_steps=DEFAULT_FINAL_STEPS,
):
    """
    Generate solution func
    """
    if store_internal is False:
        log.warning("Step split functions may need internal data to function")
    get_last_solution = run.solutions.get_last_solution

    def solution_func(inp):
        """
        solution func
        """
        try:
            step_solver(inp, run, store_internal=store_internal, no_final_solution=True)
        except SolverError as e:
            log.info(e)
            invalid_soln = get_known_broken_solution(inp)
            return invalid_soln, False
        final_solution_dict = get_last_solution().solution_input.asdict()
        final_solution_dict.update(
            stop=final_stop, num_angles=final_steps,
        )
        single_solver(
            SolutionInput(**final_solution_dict),
            run,
            store_internal=store_internal,
            no_final_solution=True,
        )
        return get_last_solution(), False

    return solution_func


def solver(
    soln_input,
    run,
    *,
    store_internal=True,
    final_stop=DEFAULT_FINAL_STOP,
    final_steps=DEFAULT_FINAL_STEPS,
    num_attempts=DEFAULT_NUM_ATTEMPTS,
):
    """
    fast-crosser solver

    uses step solver and single solver to cross fast speed
    """
    writer = writer_generator(run)
    cleanup = cleanup_generator(run, writer, no_final_solution=True,)
    best_solution = binary_searcher(
        solution_generator(
            store_internal=store_internal,
            run=run,
            final_stop=final_stop,
            final_steps=final_steps,
        ),
        cleanup,
        stepper_creator(
            writer, step_size=0.01 * soln_input.v_rin_on_c_s, solution_input=soln_input
        ),
        alternate_cleanup_generator(run, no_final_solution=True),
        soln_input,
        num_attempts=num_attempts,
    )
    if not isinstance(run.final_solution, Solution):
        if isinstance(best_solution, Solution):
            run.final_solution = best_solution
        else:
            run.final_solution = run.solutions.get_last_solution()

    return True
