# -*- coding: utf-8 -*-
"""
Entry point to run with logging
"""

import sys

import logbook
from logbook.compat import redirected_warnings, redirected_logging

from disc_solver import main

file_handler = logbook.FileHandler('ode.log', mode="w", level=logbook.DEBUG)
stdout_handler = logbook.StreamHandler(sys.stdout, level=logbook.WARNING)
null_handler = logbook.NullHandler()
with redirected_warnings(), redirected_logging():
    with null_handler.applicationbound():
        with file_handler.applicationbound():
            with stdout_handler.applicationbound():
                main()
