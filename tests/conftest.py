from pathlib import Path

import pytest

from disc_solver.solve import solve
from disc_solver.float_handling import float_type as FLOAT_TYPE

PLOT_FILE = "plot.png"

NON_INTERACTIVE_BACKENDS = [
    "agg", "cairo", "pdf", "pgf", "ps", "svg", "template"
]


DEFAULT_SOLUTIONS = [
    "single_solution_default",
    "sonic_root_solution_default",
    "step_solution_default",
    "hydrostatic_solution_default",
    "mod_hydro_solution_default",
]
NO_INTERNAL_SOLUTIONS = [
    "single_solution_no_internal",
    "sonic_root_solution_no_internal",
    "hydrostatic_solution_no_internal",
    "mod_hydro_solution_no_internal",
]
ALL_SOLUTIONS = DEFAULT_SOLUTIONS + NO_INTERNAL_SOLUTIONS
MULTI_SOLUTIONS = [
    "mcmc_solution_no_internal",
    "sonic_root_solution_no_internal",
]
TAYLOR_SOLUTIONS = [
    "single_solution_default",
    "sonic_root_solution_default",
    "step_solution_default",
]
DERIV_SOLUTIONS = [
    "single_solution_default",
    "sonic_root_solution_default",
    "step_solution_default",
]
DERIV_NO_INTERNAL_SOLUTIONS = [
    "single_solution_no_internal",
    "sonic_root_solution_no_internal",
]
USE_E_R_SOLUTIONS = [
    "mod_hydro_solution_use_E_r",
]


@pytest.fixture(scope="session")
def single_solution_default(tmpdir_factory):
    method = "single"
    tmpdir = tmpdir_factory.mktemp(method)
    return str(solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=True,
    )[0])


@pytest.fixture(scope="session")
def mcmc_solution_default(tmpdir_factory):
    method = "mcmc"
    tmpdir = tmpdir_factory.mktemp(method)
    return str(solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=True,
    )[0])


@pytest.fixture(scope="session")
def step_solution_default(tmpdir_factory):
    method = "step"
    tmpdir = tmpdir_factory.mktemp(method)
    return str(solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=True,
        max_search_steps=3, num_attempts=5,
    )[0])


@pytest.fixture(scope="session")
def sonic_root_solution_default(tmpdir_factory):
    method = "sonic_root"
    tmpdir = tmpdir_factory.mktemp(method)
    return str(solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=True,
    )[0])


@pytest.fixture(scope="session")
def hydrostatic_solution_default(tmpdir_factory):
    method = "hydrostatic"
    tmpdir = tmpdir_factory.mktemp(method)
    return str(solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=True,
    )[0])


@pytest.fixture(scope="session")
def mod_hydro_solution_default(tmpdir_factory):
    method = "mod_hydro"
    tmpdir = tmpdir_factory.mktemp(method)
    return str(solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=True,
    )[0])


@pytest.fixture(scope="session")
def single_solution_no_internal(tmpdir_factory):
    method = "single"
    tmpdir = tmpdir_factory.mktemp(method)
    return str(solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=False,
    )[0])


@pytest.fixture(scope="session")
def mcmc_solution_no_internal(tmpdir_factory):
    method = "mcmc"
    tmpdir = tmpdir_factory.mktemp(method)
    return str(solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=False,
    )[0])


@pytest.fixture(scope="session")
def sonic_root_solution_no_internal(tmpdir_factory):
    method = "sonic_root"
    tmpdir = tmpdir_factory.mktemp(method)
    return str(solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=False,
    )[0])


@pytest.fixture(scope="session")
def hydrostatic_solution_no_internal(tmpdir_factory):
    method = "hydrostatic"
    tmpdir = tmpdir_factory.mktemp(method)
    return str(solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=False,
    )[0])


@pytest.fixture(scope="session")
def mod_hydro_solution_no_internal(tmpdir_factory):
    method = "mod_hydro"
    tmpdir = tmpdir_factory.mktemp(method)
    return str(solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=False,
    )[0])


@pytest.fixture(scope="session")
def mod_hydro_solution_use_E_r(tmpdir_factory):
    method = "mod_hydro"
    tmpdir = tmpdir_factory.mktemp(method)
    return str(solve(
        sonic_method=method, output_dir=Path(str(tmpdir)),
        output_file=None, config_file=None, store_internal=True,
        overrides={"use_E_r": "True"},
    )[0])


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


@pytest.fixture(scope="session", params=TAYLOR_SOLUTIONS)
def solution_taylor(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session", params=DERIV_SOLUTIONS)
def solution_deriv(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session", params=DERIV_NO_INTERNAL_SOLUTIONS)
def solution_deriv_no_internal(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session", params=USE_E_R_SOLUTIONS)
def solution_use_E_r(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def mpl_interactive(request):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if mpl.get_backend() in NON_INTERACTIVE_BACKENDS:
        pytest.skip(
            "Skip interactive plots when using non-interactive backend used"
        )
    is_interactive = mpl.is_interactive()

    def fin():
        plt.close()
        mpl.interactive(is_interactive)

    request.addfinalizer(fin)
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


@pytest.fixture
def test_id(request):
    return str(request.node) + str(FLOAT_TYPE)


@pytest.fixture
def ds_csv_file_header(shared_datadir):
    return shared_datadir / "header.csv"


@pytest.fixture
def ds_csv_file_no_header(shared_datadir):
    return shared_datadir / "noheader.csv"
