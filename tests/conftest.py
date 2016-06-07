from functools import partial
from pathlib import Path

import pytest

from disc_solver.solve import solve

ALL_SOLUTIONS = [
    #"single_solution_default",
    #"jump_solution_default",
    "step_solution_default",
]

@pytest.fixture(scope="session")
def single_solution_default(tmpdir_factory):
    method = "single"
    tmpdir = tmpdir_factory.mktemp(method)
    return solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None
    )

@pytest.fixture(scope="session")
def jump_solution_default(tmpdir_factory):
    method = "jump"
    tmpdir = tmpdir_factory.mktemp(method)
    return solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None
    )

@pytest.fixture(scope="session")
def step_solution_default(tmpdir_factory):
    method = "step"
    tmpdir = tmpdir_factory.mktemp(method)
    return solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None
    )

@pytest.fixture(scope="session", params=ALL_SOLUTIONS)
def solution(request):
    return request.getfuncargvalue(request.param)
