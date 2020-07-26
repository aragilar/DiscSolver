# -*- coding: utf-8 -*-
"""
Stepper related logic
"""
from enum import Enum

import logbook

from numpy import diff, any as np_any, all as np_all, logical_and as np_and

from .config import define_conditions
from .solution import solution
from .utils import validate_solution, SolverError

from ..file_format import SolutionInput, Solution
from ..utils import ODEIndex

log = logbook.Logger(__name__)

DEFAULT_SPLITTER_CUTOFF = 0.9
DEFAULT_NUM_ATTEMPTS = 100
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


class StepperStop(SolverError):
    """
    Class to signal stepper has stopped
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
    func, pass_func, fail_func, final_fail_func, initial_input, *,
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
    except StepperStop as e:
        log.info(str(e))
        return final_fail_func(attempt)
    else:
        log.critical("Failed to find with {} attempts.".format(num_attempts))
    return None


def stepper_creator(
    soln_writer, step_func, get_soln_type, *, initial_step_size,
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
        log.debug(f"Writing attempt {attempt}")
        run.solutions.add_solution(soln)

    return writer


def cleanup_generator(run, writer, no_final_solution):
    """
    creates symlink to correct solution
    """
    def cleanup(soln, attempt):
        """
        cleanup
        """
        writer(soln, attempt)
        if no_final_solution:
            return None
        run.final_solution = run.solutions.get_last_solution()
        return run.final_solution

    return cleanup


def alternate_cleanup_generator(run, no_final_solution):
    """
    creates symlink for final solution if no perfect solution
    """
    def cleanup(attempt):
        """
        alternate cleanup
        """
        if no_final_solution:
            return None
        run.final_solution = run.solutions[str(attempt)]
        return run.final_solution

    return cleanup


def human_view_splitter_generator():
    """
    Generate splitter func which uses human input
    """
    from ..analyse.diverge_plot import diverge_plot

    solutions = []

    def human_view_splitter(soln, **kwargs):
        """
        Splitter func which uses human input
        """
        # pylint: disable=unused-argument
        solutions.append(soln)
        diverge_plot(enumerate(solutions), show=True)
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
        "v_θ_deriv": lambda: v_θ_deriv_splitter,
        "human": human_view_splitter_generator,
    }

    return method_dict.get(method)() or v_θ_deriv_splitter


def solution_generator(*, store_internal=True):
    """
    Generate solution func
    """
    if store_internal is False:
        log.warning("Step split functions may need internal data to function")

    def solution_func(inp):
        """
        solution func
        """
        inp = SolutionInput(**vars(inp))
        soln = solution(
            inp, define_conditions(inp), store_internal=store_internal,
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
            raise StepperStop("Stepper stopped")
        else:
            raise StepperError("Solution type not known")
        if prev_γ == inp_dict["γ"]:
            # we've hit the numerical limit, now find a diverging solution
            if soln_type == SplitterStatus.DIVERGE:
                raise StepperStop("Hit numerical limit")

            # we've got a sign flip solution then
            bigger_step = 0
            while prev_γ == inp_dict["γ"]:
                bigger_step += step_size
                inp_dict["γ"] += bigger_step
        return SolutionInput(**inp_dict)
    return step_func


def solver(
    soln_input, run, store_internal=True, num_attempts=DEFAULT_NUM_ATTEMPTS,
    max_search_steps=DEFAULT_MAX_SEARCH_STEPS, no_final_solution=False
):
    """
    Stepping solver
    """
    step_func = step_input()
    writer = writer_generator(run)
    cleanup = cleanup_generator(
        run, writer, no_final_solution=no_final_solution,
    )
    best_solution = binary_searcher(
        solution_generator(store_internal=store_internal), cleanup,
        stepper_creator(
            writer, step_func,
            create_soln_splitter(soln_input.split_method),
            max_search_steps=max_search_steps,
            initial_step_size=0.1 * soln_input.γ
        ),
        alternate_cleanup_generator(run, no_final_solution=no_final_solution),
        soln_input, num_attempts=num_attempts,
    )
    if not isinstance(run.final_solution, Solution) and not no_final_solution:
        run.final_solution = None

    if best_solution is None and not no_final_solution:
        return False
    return True
