import sys

import logbook
from logbook.compat import redirected_warnings, redirected_logging

from disc_solver import main

file_handler = logbook.FileHandler('ode.log', mode="w", level=logbook.DEBUG)
stdout_handler = logbook.StreamHandler(sys.stdout, level=logbook.WARNING)
null_handler = logbook.NullHandler()
with redirected_warnings(), redirected_logging():
    with null_handler.applicationbound(), file_handler.applicationbound(), stdout_handler.applicationbound():
        main()

