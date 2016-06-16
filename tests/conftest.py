from functools import partial
from pathlib import Path

import pytest

from disc_solver.solve import solve

PLOT_FILE = "plot.png"
ALL_SOLUTIONS = [
    "single_solution_default",
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

@pytest.fixture()
def mpl_interactive():
    import matplotlib.pyplot as plt
    plt.ion()

@pytest.fixture
def tmp_text_stream(request):
    from io import StringIO
    stream = StringIO()
    def fin():
        stream.close()
    request.addfinalizer(fin)
    return stream

@pytest.fixture
def plot_file(tmpdir):
    return Path(Path(str(tmpdir)), PLOT_FILE)
