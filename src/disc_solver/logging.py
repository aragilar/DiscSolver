# -*- coding: utf-8 -*-
"""
Logging configuration functions
"""
import multiprocessing
from os import environ

from logbook import NullHandler, FileHandler, NestedSetup
from logbook.more import ColorizedStderrHandler
from logbook.queues import (
    ThreadedWrapperHandler, MultiProcessingHandler, MultiProcessingSubscriber,
)

from stringtopy import str_to_bool_converter

str_to_bool = str_to_bool_converter()


def logging_options(parser):
    """Add cli options for logging to parser"""
    LOG_LEVELS = ("critical", "error", "warning", "notice", "info", "debug")
    parser.add_argument("--log-file")
    parser.add_argument(
        "--log-file-level", choices=LOG_LEVELS, default="debug"
    )
    stderr_parser = parser.add_mutually_exclusive_group()
    stderr_parser.add_argument(
        "--stderr-level", choices=LOG_LEVELS, default="notice"
    )
    stderr_parser.add_argument(
        "--quiet", "-q", default=False, action="store_true",
    )
    stderr_parser.add_argument(
        "--verbose", "-v", default=False, action="store_true",
    )


def log_handler(args, thread_wrapping=True):
    """
    Return log handler with given config
    """
    if not isinstance(args, dict):
        args = vars(args)
    if args.get("quiet"):
        stderr_handler = ColorizedStderrHandler(level="ERROR")
    elif args.get("verbose"):
        stderr_handler = ColorizedStderrHandler(level="DEBUG")
    else:
        stderr_handler = ColorizedStderrHandler(
            level=args.get("stderr_level", "NOTICE").upper(), bubble=True
        )
    if args.get("log_file"):
        file_handler = FileHandler(
            args.get("log_file"),
            level=args.get("log_file_level", "DEBUG").upper(), bubble=True
        )
    else:
        file_handler = NullHandler()

    if thread_wrapping:
        disable_thread_wrapping = environ.get(
            "DISC_SOLVER_DISABLE_THREADING", None
        )
        if disable_thread_wrapping is not None:
            disable_thread_wrapping = str_to_bool(disable_thread_wrapping)

        if not disable_thread_wrapping:
            file_handler = ThreadedWrapperHandler(file_handler)
            stderr_handler = ThreadedWrapperHandler(stderr_handler)

    return NestedSetup([
        NullHandler(),  # catch everything else
        file_handler, stderr_handler
    ])


def enable_multiprocess_log_handing(args):
    """
    Set up logging when using multiprocessing
    """
    normal_handler = log_handler(args, thread_wrapping=False)
    manager = multiprocessing.Manager()
    queue = manager.Queue(-1)
    mp_handler = MultiProcessingHandler(queue)
    mp_handler.push_application()
    mp_sub = MultiProcessingSubscriber(queue)
    mp_sub.dispatch_in_background(normal_handler)
    return queue


def start_logging_in_process(queue):
    """
    Start logging in multiprocessing process
    """
    if queue is not None:
        mp_handler = MultiProcessingHandler(queue)
        mp_handler.push_application()
