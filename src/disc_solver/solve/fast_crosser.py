# -*- coding: utf-8 -*-
"""
fast-crosser: combines different solvers to cross fast speed
"""
from enum import Enum

import logbook

from numpy import max as np_max

from .single import solver as single_solver
from .stepper import (
    solver as step_solver, writer_generator, cleanup_generator,
    binary_searcher, stepper_creator, alternate_cleanup_generator,
    StepperStop, StepperError,
)

from ..analyse.utils import get_mach_numbers
from ..file_format import SolutionInput, Solution

log = logbook.Logger(__name__)

DEFAULT_NUM_ATTEMPTS = 1000
DEFAULT_MAX_SEARCH_STEPS = 200
DEFAULT_FINAL_STOP = 90
DEFAULT_FINAL_STEPS = 900000


class SplitterStatus(Enum):
    """
    Status Enum for which way to the best solution
    """
    STOP = "STOP"
    INCREASE = "increase"
    DECREASE = "decrease"


def step_input():
    """
    Create new input for next step
    """
    def step_func(soln, soln_type, step_size):
        """
        Return new input
        """
        inp_dict = vars(soln.solution_input)
        prev_v_r = inp_dict["v_rin_on_c_s"]
        if soln_type == SplitterStatus.DECREASE:
            inp_dict["v_rin_on_c_s"] -= step_size
        elif soln_type == SplitterStatus.INCREASE:
            inp_dict["v_rin_on_c_s"] += step_size
        elif soln_type == SplitterStatus.STOP:
            raise StepperStop("Stepper stopped")
        else:
            raise StepperError("Solution type not known")
        if prev_v_r == inp_dict["v_rin_on_c_s"]:
            raise StepperStop("Hit numerical limit")
        return SolutionInput(**inp_dict)
    return step_func


def solution_generator(
    *, store_internal=True, run, final_stop=DEFAULT_FINAL_STOP,
    final_steps=DEFAULT_FINAL_STEPS
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
        step_solver(
            inp, run, store_internal=store_internal, no_final_solution=True
        )
        final_solution_dict = get_last_solution().solution_input.asdict()
        final_solution_dict.update(
            stop=final_stop, num_angles=final_steps,
        )
        single_solver(
            SolutionInput(**final_solution_dict), run,
            store_internal=store_internal, no_final_solution=True,
        )
        return get_last_solution(), False

    return solution_func


class SolutionSplitter:
    """
    Work out which way to look for best solution
    """
    # pylint: disable=too-few-public-methods
    def __init__(self):
        self._previous_value = None
        self._previous_direction = None

    def _get_direction(self, soln):
        """
        Work out which way to look for best solution
        """
        _, _, _, fast_mach = get_mach_numbers(soln)
        current = np_max(fast_mach)

        if self._previous_value is None:
            self._previous_value = current
            self._previous_direction = SplitterStatus.INCREASE
            return self._previous_direction
        if self._previous_value > current and (
            self._previous_direction == SplitterStatus.DECREASE
        ):
            self._previous_value = current
            self._previous_direction = SplitterStatus.INCREASE
            return self._previous_direction
        if self._previous_value > current and (
            self._previous_direction == SplitterStatus.INCREASE
        ):
            self._previous_value = current
            self._previous_direction = SplitterStatus.DECREASE
            return self._previous_direction
        if self._previous_value < current and (
            self._previous_direction == SplitterStatus.DECREASE
        ):
            self._previous_value = current
            self._previous_direction = SplitterStatus.DECREASE
            return self._previous_direction
        if self._previous_value < current and (
            self._previous_direction == SplitterStatus.INCREASE
        ):
            self._previous_value = current
            self._previous_direction = SplitterStatus.INCREASE
            return self._previous_direction

        # increase by default
        self._previous_value = current
        self._previous_direction = SplitterStatus.INCREASE
        return self._previous_direction

    def __call__(self, soln):
        return soln, self._get_direction(soln)


def solver(
    soln_input, run, *, store_internal=True, final_stop=DEFAULT_FINAL_STOP,
    final_steps=DEFAULT_FINAL_STEPS, max_search_steps=DEFAULT_MAX_SEARCH_STEPS,
    num_attempts=DEFAULT_NUM_ATTEMPTS,
):
    """
    fast-crosser solver

    uses step solver and single solver to cross fast speed
    """
    step_func = step_input()
    writer = writer_generator(run)
    cleanup = cleanup_generator(
        run, writer, no_final_solution=True,
    )
    best_solution = binary_searcher(
        solution_generator(
            store_internal=store_internal, run=run, final_stop=final_stop,
            final_steps=final_steps,
        ), cleanup,
        stepper_creator(
            writer, step_func,
            SolutionSplitter(),
            max_search_steps=max_search_steps,
            initial_step_size=0.01 * soln_input.v_rin_on_c_s
        ),
        alternate_cleanup_generator(run, no_final_solution=True),
        soln_input, num_attempts=num_attempts,
    )
    if not isinstance(run.final_solution, Solution):
        if isinstance(best_solution, Solution):
            run.final_solution = best_solution
        else:
            run.final_solution = run.solutions.get_last_solution()

    return True
