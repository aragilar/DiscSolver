# -*- coding: utf-8 -*-
"""
Stepper related logic
"""

import logbook

log = logbook.Logger(__name__)


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
                return step_func(soln_type, step_size)
            else:
                prev_soln_type = soln_type
                return step_func(soln_type, step_size)

        step_size /= 2
        return step_func(soln_type, step_size)

    return stepper


def writer_generator(soln_file):
    """
    adds soln to file
    """
    def writer(soln, attempt):
        """
        writer
        """
        soln_file.root.solutions[str(attempt)] = soln

    return writer


def cleanup_generator(soln_file, writer):
    """
    creates symlink to correct solution
    """
    def cleanup(soln, attempt):
        """
        cleanup
        """
        writer(soln, attempt)
        soln_file.root.final_solution = soln_file.root.solutions[str(attempt)]

    return cleanup


class StepperError(RuntimeError):
    """
    Base class for exceptions related to the stepper
    """
    pass
