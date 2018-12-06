# -*- coding: utf-8 -*-
"""
Stepper related logic
"""
from enum import Enum
from warnings import warn

import logbook

from numpy import diff, any as np_any, all as np_all, logical_and as np_and

from .config import define_conditions
from .solution import solution
from .utils import validate_solution, SolverError

from ..analyse.diverge_plot import diverge_plot

from ..float_handling import float_type
from ..file_format import SolutionInput, Solution
from ..utils import ODEIndex

log = logbook.Logger(__name__)

DEFAULT_SPLITTER_CUTOFF = 0.9
DEFAULT_NUM_ATTEMPTS = 100
INITIAL_STEP_SIZE = float_type(1e-4)
DEFAULT_MAX_SEARCH_STEPS = 20


class SplitterStatus(Enum):
    """
    Status Enum for which way to the best solution
    """
    STOP = "STOP"
    SIGN_FLIP = "sign flip"
    DIVERGE = "diverge"


class StepperError(SolverError):
    """
    Base class for exceptions related to the stepper
    """
    pass


def is_solution_best(soln):
    """
    Determine if solution is best possible solution
    """
    soln, valid = validate_solution(soln)
    if not valid:
        return soln, False

    v_θ = soln.solution[:, ODEIndex.v_θ]

    v_θ_near_sonic = np_and(v_θ > DEFAULT_SPLITTER_CUTOFF, v_θ < 1)
    if not np_any(v_θ_near_sonic):
        return soln, False

    if np_any(diff(v_θ[v_θ_near_sonic], n=2)) > 0:
        return soln, False

    return soln, True


def binary_searcher(
    func, pass_func, fail_func, initial_input, *,
    num_attempts=DEFAULT_NUM_ATTEMPTS
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
    return None


def stepper_creator(
    soln_writer, step_func, get_soln_type, initial_step_size=INITIAL_STEP_SIZE,
    max_search_steps=DEFAULT_NUM_ATTEMPTS,
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


def human_view_splitter_generator():
    """
    Generate splitter func which uses human input
    """
    solutions = []

    def human_view_splitter(soln, **kwargs):
        """
        Splitter func which uses human input
        """
        # pylint: disable=unused-argument
        solutions.append(soln)
        diverge_plot(*solutions, show=True)
        soln_type = input("What type: ").strip()
        while soln_type not in ("sign flip", "diverge", "STOP"):
            print("Solution type must be either 'sign flip' or 'diverge'")
            soln_type = input("What type: ").strip()
        return SplitterStatus(soln_type)
    return human_view_splitter


def v_θ_deriv_splitter(soln, cutoff=DEFAULT_SPLITTER_CUTOFF, **kwargs):
    """
    Use derivative of v_θ to determine type of solution
    """
    # pylint: disable=unused-argument
    v_θ = soln.solution[:, ODEIndex.v_θ]
    if soln.internal_data is not None:
        problems = soln.internal_data.problems
        if any("negative velocity" in pl for pl in problems.values()):
            return SplitterStatus.SIGN_FLIP
    else:
        log.notice("Skipping checking problems due to no internal data")

    v_θ_near_sonic = np_and(v_θ > cutoff, v_θ < 1)
    if not np_any(v_θ_near_sonic):
        return SplitterStatus.SIGN_FLIP

    v_θ_above_sonic = v_θ > 1
    if np_any(v_θ_above_sonic):
        return SplitterStatus.DIVERGE

    d_v_θ = diff(v_θ[v_θ_near_sonic])
    if np_all(d_v_θ > 0):
        return SplitterStatus.DIVERGE
    return SplitterStatus.SIGN_FLIP


def create_soln_splitter(method):
    """
    Create func to see split in solution
    """
    method_dict = {
        "v_θ_deriv": v_θ_deriv_splitter,
        "human": human_view_splitter_generator(),
    }

    return method_dict.get(method) or v_θ_deriv_splitter


def solution_generator(*, store_internal=True, run):
    """
    Generate solution func
    """
    if store_internal is False:
        warn("Step split functions may need internal data to function")

    def solution_func(inp):
        """
        solution func
        """
        inp = SolutionInput(**vars(inp))
        soln = solution(
            inp, define_conditions(inp, use_E_r=run.use_E_r),
            store_internal=store_internal, use_E_r=run.use_E_r,
        )
        return is_solution_best(soln)

    return solution_func


def step_input():
    """
    Create new input for next step
    """
    def step_func(soln, soln_type, step_size):
        """
        Return new input
        """
        inp_dict = vars(soln.solution_input)
        prev_γ = inp_dict["γ"]
        if soln_type == SplitterStatus.DIVERGE:
            inp_dict["γ"] -= step_size
        elif soln_type == SplitterStatus.SIGN_FLIP:
            inp_dict["γ"] += step_size
        elif soln_type == SplitterStatus.STOP:
            raise StepperError("Stepper stopped")
        else:
            raise StepperError("Solution type not known")
        if prev_γ == inp_dict["γ"]:
            raise StepperError("Hit numerical limit")
        return SolutionInput(**inp_dict)
    return step_func


def solver(
    soln_input, run, store_internal=True, num_attempts=DEFAULT_NUM_ATTEMPTS,
    max_search_steps=DEFAULT_MAX_SEARCH_STEPS,
):
    """
    Stepping solver
    """
    step_func = step_input()
    writer = writer_generator(run)
    cleanup = cleanup_generator(run, writer)
    binary_searcher(
        solution_generator(store_internal=store_internal, run=run), cleanup,
        stepper_creator(
            writer, step_func,
            create_soln_splitter(soln_input.split_method),
            max_search_steps=max_search_steps,
        ), soln_input, num_attempts=num_attempts,
    )
    if not isinstance(run.final_solution, Solution):
        run.final_solution = None
