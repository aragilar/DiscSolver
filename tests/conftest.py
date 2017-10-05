from functools import partial
from pathlib import Path

import pytest

from disc_solver.solve import solve

PLOT_FILE = "plot.png"
DEFAULT_SOLUTIONS = [
    "single_solution_default",
    "dae_single_solution_default",
    #"mcmc_solution_default",
    "sonic_root_solution_default",
    "step_solution_default",
]
NO_INTERNAL_SOLUTIONS = [
    "single_solution_no_internal",
    "dae_single_solution_no_internal",
    #"mcmc_solution_no_internal",
    "sonic_root_solution_no_internal",
]
ALL_SOLUTIONS = DEFAULT_SOLUTIONS + NO_INTERNAL_SOLUTIONS
MULTI_SOLUTIONS = [
    "mcmc_solution_no_internal",
    "sonic_root_solution_no_internal",
]

@pytest.fixture(scope="session")
def single_solution_default(tmpdir_factory):
    method = "single"
    tmpdir = tmpdir_factory.mktemp(method)
    return solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=True,
    )

@pytest.fixture(scope="session")
def dae_single_solution_default(tmpdir_factory):
    method = "dae_single"
    tmpdir = tmpdir_factory.mktemp(method)
    return solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=True,
    )

@pytest.fixture(scope="session")
def mcmc_solution_default(tmpdir_factory):
    method = "mcmc"
    tmpdir = tmpdir_factory.mktemp(method)
    return solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=True,
    )

@pytest.fixture(scope="session")
def step_solution_default(tmpdir_factory):
    method = "step"
    tmpdir = tmpdir_factory.mktemp(method)
    return solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=True,
    )

@pytest.fixture(scope="session")
def sonic_root_solution_default(tmpdir_factory):
    method = "sonic_root"
    tmpdir = tmpdir_factory.mktemp(method)
    return solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=True,
    )

@pytest.fixture(scope="session")
def single_solution_no_internal(tmpdir_factory):
    method = "single"
    tmpdir = tmpdir_factory.mktemp(method)
    return solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=False,
    )

@pytest.fixture(scope="session")
def dae_single_solution_no_internal(tmpdir_factory):
    method = "dae_single"
    tmpdir = tmpdir_factory.mktemp(method)
    return solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=False,
    )

@pytest.fixture(scope="session")
def mcmc_solution_no_internal(tmpdir_factory):
    method = "mcmc"
    tmpdir = tmpdir_factory.mktemp(method)
    return solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=False,
    )

@pytest.fixture(scope="session")
def sonic_root_solution_no_internal(tmpdir_factory):
    method = "sonic_root"
    tmpdir = tmpdir_factory.mktemp(method)
    return solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=False,
    )

@pytest.fixture(scope="session", params=ALL_SOLUTIONS)
def solution(request):
    return request.getfixturevalue(request.param)

@pytest.fixture(scope="session", params=DEFAULT_SOLUTIONS)
def solution_default(request):
    return request.getfixturevalue(request.param)

@pytest.fixture(scope="session", params=NO_INTERNAL_SOLUTIONS)
def solution_no_internal(request):
    return request.getfixturevalue(request.param)

@pytest.fixture(scope="session", params=MULTI_SOLUTIONS)
def solutions_many(request):
    return request.getfixturevalue(request.param)

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
