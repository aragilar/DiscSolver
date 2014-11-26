#!/bin/sh

set -e # exit if any of the following command fail

PYTHONOPTIMIZE=1 tox # run tests with less debug output
