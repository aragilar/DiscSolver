#!/bin/sh

set -e # exit if any of the following command fail

PYTHONOPTIMIZE=1 tox # run the normal tests

#pylint disc_solver
#flake8 disc_solver # validate code


