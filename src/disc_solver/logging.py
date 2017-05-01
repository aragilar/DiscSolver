# -*- coding: utf-8 -*-
"""
Logging configuration functions
"""

from logbook import NullHandler, StderrHandler, FileHandler, NestedSetup
from logbook.queues import ThreadedWrapperHandler


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


def log_handler(args):
    """
    Return log handler with given config
    """
    if args.get("quiet"):
        stderr_handler = StderrHandler(level="ERROR")
    elif args.get("verbose"):
        stderr_handler = StderrHandler(level="DEBUG")
    else:
        stderr_handler = StderrHandler(
            level=args.get("stderr_level", "NOTICE").upper()
        )
    if args.get("log_file"):
        file_handler = FileHandler(
            args.get("log_file"),
            level=args.get("log_file_level", "DEBUG").upper()
        )
    else:
        file_handler = NullHandler()
    return NestedSetup([
        NullHandler(),  # catch everything else
        ThreadedWrapperHandler(file_handler),
        ThreadedWrapperHandler(stderr_handler)
    ])
