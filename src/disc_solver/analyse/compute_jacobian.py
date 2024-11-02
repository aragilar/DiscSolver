# -*- coding: utf-8 -*-
"""
Analysis functions related to the jacobians of existing solutions
"""
from functools import wraps

from numpy import (
    degrees, zeros, logical_not as npnot, isfinite, nan, full,
    array as nparray, dot, sum as npsum,
)
from numpy.linalg import inv
from scipy.linalg import eigvals, eig
from scipy.optimize import least_squares
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..float_handling import float_type
from ..solve.solution import ode_system
from ..critical_points import get_critical_point_indices

from .utils import (
    single_solution_plotter, common_plotting_options, analyse_main_wrapper,
    get_common_plot_args, analysis_func_wrapper, plot_output_wrapper,
    DEFAULT_MPL_STYLE,
)

mpl.rcParams['agg.path.chunksize'] = 1000000
COLOURS = plt.get_cmap("tab20").colors
CRITICAL_POINT_MAP = {
    "slow": 0,
    "sonic": 1,
    "alfven": 2,
    "fast": 3,
}


def plot_parser(parser):
    """
    Add arguments for plot command to parser
    """
    common_plotting_options(parser)
    parser.add_argument(
        "--eps", action='store', default=1e-10, type=float
    )
    return parser


def get_plot_args(args):
    """
    Parse plot args
    """
    return {
        "eps": float(args.get("eps", 1e-10)),
    }


@analyse_main_wrapper(
    "Plot eigenvalues of jacobians for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
        "plot_args": get_plot_args,
    }
)
def jacobian_eigenvalues_main(
    soln, *, soln_range, common_plot_args, plot_args
):
    """
    Entry point for ds-derivs-plot
    """
    return jacobian_eigenvalues_plot(
        soln, soln_range=soln_range, **common_plot_args, **plot_args
    )


@analyse_main_wrapper(
    "Plot eigenvectors of jacobians for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
        "plot_args": get_plot_args,
    }
)
def jacobian_eigenvectors_main(
    soln, *, soln_range, common_plot_args, plot_args
):
    """
    Entry point for ds-derivs-plot
    """
    return jacobian_eigenvectors_plot(
        soln, soln_range=soln_range, **common_plot_args, **plot_args
    )


def eigen_parser(parser):
    """
    Add arguments for plot command to parser
    """
    common_plotting_options(parser)
    parser.add_argument(
        "--eps", action='store', default=1e-10, type=float
    )
    parser.add_argument(
        "--critical-point", action='store', default="sonic",
        choices=CRITICAL_POINT_MAP.keys(),
    )
    local_group = parser.add_mutually_exclusive_group()
    local_group.add_argument(
        "--local-eigvecs", action='store_true', dest="local_eigvecs",
    )
    local_group.add_argument(
        "--global-eigvecs", action='store_false', dest="local_eigvecs",
    )
    return parser


def get_eigen_args(args):
    """
    Parse plot args
    """
    return {
        "eps": float(args.get("eps", 1e-10)),
        "critical_point": args.get("critical_point", "sonic"),
        "local_eigvecs": args.get("local_eigvecs", True),
    }


@analyse_main_wrapper(
    "Plot eigenvectors of jacobians for DiscSolver",
    eigen_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
        "plot_args": get_eigen_args,
    }
)
def jacobian_eigenvector_coef_main(
    soln, *, soln_range, common_plot_args, plot_args
):
    """
    Entry point for ds-derivs-plot
    """
    return jacobian_eigenvector_coef_plot(
        soln, soln_range=soln_range, **common_plot_args, **plot_args
    )


@analyse_main_wrapper(
    "Plot eigenvectors of jacobians for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
        "plot_args": get_plot_args,
    }
)
def jacobian_eigenvector_norm_main(
    soln, *, soln_range, common_plot_args, plot_args
):
    """
    Entry point for ds-derivs-plot
    """
    return jacobian_eigenvector_norm_plot(
        soln, soln_range=soln_range, **common_plot_args, **plot_args
    )


def compute_jacobian(
    *, γ, a_0, norm_kepler_sq, init_con, θ_scale, η_derivs, use_E_r, θ, params,
    eps,
):
    """
    Compute jacobian from solution
    """
    rhs_eq, _ = ode_system(
        γ=γ, a_0=a_0, norm_kepler_sq=norm_kepler_sq, init_con=init_con,
        θ_scale=θ_scale, with_taylor=False, η_derivs=η_derivs,
        store_internal=False, use_E_r=use_E_r, v_θ_sonic_crit=None,
        after_sonic=None, deriv_v_θ_func=None, derivs_post_solution=True,
    )

    solution_length = params.shape[0]
    ode_size = params.shape[1]

    J = zeros([solution_length, ode_size, ode_size], dtype=float_type)

    # compute column for each param
    for i in range(ode_size):
        derivs_h = zeros([solution_length, ode_size], dtype=float_type)
        derivs_l = zeros([solution_length, ode_size], dtype=float_type)
        params_h = params.copy()
        params_l = params.copy()

        # offset only the value associated with this column
        params_h[:, i] += eps
        params_l[:, i] -= eps

        # we don't check the validity of inputs as we have these from the
        # solution
        rhs_eq(θ, params_h, derivs_h)
        rhs_eq(θ, params_l, derivs_l)

        J[:, :, i] = (derivs_h - derivs_l) / (2 * eps)
    return J


def compute_jacobian_from_solution(soln, *, eps, θ_scale=float_type(1)):
    """
    Compute jacobian from solution
    """
    solution = soln.solution
    angles = soln.angles
    cons = soln.initial_conditions
    soln_input = soln.solution_input

    init_con = cons.init_con
    γ = cons.γ
    a_0 = cons.a_0
    norm_kepler_sq = cons.norm_kepler_sq

    η_derivs = soln_input.η_derivs
    use_E_r = soln_input.use_E_r

    return compute_jacobian(
        γ=γ, a_0=a_0, norm_kepler_sq=norm_kepler_sq, init_con=init_con,
        θ_scale=θ_scale, η_derivs=η_derivs, use_E_r=use_E_r, θ=angles,
        params=solution, eps=eps,
    )


def jacobian_single_solution_plot_wrapper(func):
    """
    Helper wrapper for solutions working with jacobians
    """
    @wraps(func)
    def jacobian_wrapper(
        fig, soln, *args, eps, use_E_r, θ_scale=float_type(1), **kwargs
    ):
        # pylint: disable=unused-argument
        jacobians = compute_jacobian_from_solution(
            soln, eps=eps, θ_scale=θ_scale
        )
        return func(
            fig, *args, jacobians=jacobians, angles=soln.angles, **kwargs
        )

    return jacobian_wrapper


@analysis_func_wrapper
def jacobian_eigenvalues_plot(
    soln, *, soln_range=None, plot_filename=None, show=False, start=0, stop=90,
    figargs=None, linestyle='.', title=None, close=True, filename,
    mpl_style=DEFAULT_MPL_STYLE, with_version=True, eps
):
    """
    Plot eigenvalues of jacobians of a solution
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    # pylint: disable=missing-kwoa
    fig = plot_jacobian_eigenvalues(
        soln, soln_range, linestyle=linestyle, start=start, stop=stop,
        figargs=figargs, title=title, filename=filename,
        mpl_style=mpl_style, with_version=with_version, eps=eps
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@analysis_func_wrapper
def jacobian_eigenvectors_plot(
    soln, *, soln_range=None, plot_filename=None, show=False, start=0, stop=90,
    figargs=None, linestyle='.', title=None, close=True, filename,
    mpl_style=DEFAULT_MPL_STYLE, with_version=True, eps
):
    """
    Show derivatives
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    # pylint: disable=missing-kwoa
    fig = plot_jacobian_eigenvectors(
        soln, soln_range, linestyle=linestyle, start=start, stop=stop,
        figargs=figargs, title=title, filename=filename,
        mpl_style=mpl_style, with_version=with_version, eps=eps
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@analysis_func_wrapper
def jacobian_eigenvector_coef_plot(
    soln, *, soln_range=None, plot_filename=None, show=False, start=0, stop=90,
    figargs=None, linestyle='.', title=None, close=True, filename,
    mpl_style=DEFAULT_MPL_STYLE, with_version=True, eps, local_eigvecs=True,
    critical_point="sonic",
):
    """
    Show derivatives
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    # pylint: disable=missing-kwoa
    fig = plot_eigenvector_coef_around_sonic_point(
        soln, soln_range, linestyle=linestyle, start=start, stop=stop,
        figargs=figargs, title=title, filename=filename,
        mpl_style=mpl_style, with_version=with_version, eps=eps,
        local_eigvecs=local_eigvecs, critical_point=critical_point,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@analysis_func_wrapper
def jacobian_eigenvector_norm_plot(
    soln, *, soln_range=None, plot_filename=None, show=False, start=0, stop=90,
    figargs=None, linestyle='.', title=None, close=True, filename,
    mpl_style=DEFAULT_MPL_STYLE, with_version=True, eps
):
    """
    Show derivatives
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    # pylint: disable=missing-kwoa
    fig = plot_jacobian_eigenvector_norm(
        soln, soln_range, linestyle=linestyle, start=start, stop=stop,
        figargs=figargs, title=title, filename=filename,
        mpl_style=mpl_style, with_version=with_version, eps=eps
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


def compute_eigenvalues(jacobian):
    """
    Compute eigenvalues of a jacobian matrix
    """
    if not isfinite(jacobian).all():
        return full(jacobian.shape[0], nan)
    return eigvals(jacobian)


def compute_eigenvalues_and_eigenvectors(jacobian):
    """
    Compute eigenvalues and eigenvectors of a jacobian matrix
    """
    if not isfinite(jacobian).all():
        M = jacobian.shape[0]
        return full(M, nan), full([M, M], nan)
    return eig(jacobian)


def plot_log_complex_data(ax, *, x_data, y_data, **kwargs):
    """
    Plot complex data logarithmically
    """
    pos_real_data_slice = y_data.real >= 0
    pos_imag_data_slice = y_data.imag >= 0
    ax.plot(
        x_data[pos_real_data_slice],
        y_data[pos_real_data_slice].real,
        marker="^",
        linestyle='None',
        **kwargs
    )
    ax.plot(
        x_data[npnot(pos_real_data_slice)],
        - y_data[npnot(pos_real_data_slice)].real,
        marker="v",
        linestyle='None',
        **kwargs
    )
    ax.plot(
        x_data[pos_imag_data_slice],
        y_data[pos_imag_data_slice].imag,
        marker="2",
        linestyle='None',
        **kwargs
    )
    ax.plot(
        x_data[npnot(pos_imag_data_slice)],
        - y_data[npnot(pos_imag_data_slice)].imag,
        marker="1",
        linestyle='None',
        **kwargs
    )


@single_solution_plotter
@jacobian_single_solution_plot_wrapper
def plot_jacobian_eigenvalues(
    fig, *, jacobians, angles, start=0, stop=90, linestyle=',',
):
    """
    Generate plot of jacobians
    """
    # pylint: disable=unused-argument
    indexes = (start <= degrees(angles)) & (degrees(angles) <= stop)

    data = nparray([compute_eigenvalues(j) for j in jacobians])

    ax = fig.subplots()
    ax.set_xlabel("angle from plane (°)")
    ax.set_yscale("log")
    for i in range(data.shape[1]):
        plot_log_complex_data(
            ax, x_data=degrees(angles[indexes]),
            y_data=data[indexes, i],
            color=COLOURS[i],
            alpha=.5,
            label=str(i)
        )
    ax.legend(ncol=11)
    return fig


@single_solution_plotter
@jacobian_single_solution_plot_wrapper
def plot_jacobian_eigenvectors(
    fig, *, jacobians, angles, start=0, stop=90, linestyle=',',
):
    """
    Generate plot of jacobians
    """
    # pylint: disable=unused-argument
    indexes = (start <= degrees(angles)) & (degrees(angles) <= stop)

    eigvalues, eigvecs = zip(*[
        compute_eigenvalues_and_eigenvectors(j) for j in jacobians
    ])
    eigvalues = nparray(eigvalues)
    eigvecs = nparray(eigvecs)

    axes = fig.subplots(
        nrows=2, ncols=6, sharex=True, gridspec_kw=dict(hspace=0),
    )

    # only add label to bottom plots
    for ax in axes[1]:
        ax.set_xlabel("angle from plane (°)")

    axes.shape = 12
    for j in range(eigvalues.shape[1]):
        ax = axes[j]
        ax.set_yscale("log")
        plot_log_complex_data(
            ax, x_data=degrees(angles[indexes]),
            y_data=eigvalues[indexes, j],
            color="black",
            alpha=.5,
        )
        for i in range(eigvecs.shape[2]):
            plot_log_complex_data(
                ax, x_data=degrees(angles[indexes]),
                y_data=eigvecs[indexes, i, j],
                color=COLOURS[i],
                alpha=.5,
                label=str(i)
            )
        ax.legend()
    return fig


def compute_norm(v, *, axis):
    """
    Compute norms
    """
    return npsum(v * v.conj(), axis=axis)


@single_solution_plotter
@jacobian_single_solution_plot_wrapper
def plot_jacobian_eigenvector_norm(
    fig, *, jacobians, angles, start=0, stop=90, linestyle=',',
):
    """
    Generate plot of jacobians
    """
    # pylint: disable=unused-argument
    indexes = (start <= degrees(angles)) & (degrees(angles) <= stop)

    eigvecs = nparray([
        compute_eigenvalues_and_eigenvectors(j)[1] for j in jacobians
    ])

    norms = compute_norm(eigvecs, axis=1)

    ax = fig.subplots()

    ax.set_xlabel("angle from plane (°)")

    ax.plot(degrees(angles[indexes]), norms[indexes])
    return fig


def compute_coefs_of_subtracted_solution(subtracted_soln, eigvecs):
    """
    Compute coefficients of subtracted solution
    """
    def opt_func(x, *, soln):
        """
        Function for least squares
        """
        complex_diff = dot(eigvecs, x).sum(axis=0) - soln
        return nparray([complex_diff.real, complex_diff.imag]).flatten()

    initial_coefs = zeros(eigvecs.shape[0])
    res = []
    fun = []
    for soln in subtracted_soln:
        out = least_squares(opt_func, initial_coefs, kwargs={'soln': soln})
        if out.success:
            res.append(out.x)
            fun.append(out.fun)
        else:
            res.append(full(eigvecs.shape[0], nan))
            fun.append(full(eigvecs.shape[0], nan))
    return nparray(res), nparray(fun)


def compute_jacobian_at_index(soln, *, eps, θ_scale=float_type(1), index):
    """
    Compute jacobian from solution at index
    """
    s = slice(index, index+1)
    solution = soln.solution[s]
    angles = soln.angles[s]
    cons = soln.initial_conditions
    soln_input = soln.solution_input

    init_con = cons.init_con
    γ = cons.γ
    a_0 = cons.a_0
    norm_kepler_sq = cons.norm_kepler_sq

    η_derivs = soln_input.η_derivs
    use_E_r = soln_input.use_E_r

    return compute_jacobian(
        γ=γ, a_0=a_0, norm_kepler_sq=norm_kepler_sq, init_con=init_con,
        θ_scale=θ_scale, η_derivs=η_derivs, use_E_r=use_E_r, θ=angles,
        params=solution, eps=eps,
    )[0]


def get_eigenvector_coef_around_critical_point(
    soln, *, critical_point_index, eps, local_eigvecs
):
    """
    Get coefficients of eigenvectors around a critical point
    """
    solution = soln.solution
    critical_vals = solution[(critical_point_index,)]
    subtracted_soln = solution - critical_vals

    if local_eigvecs:
        jacobians = compute_jacobian_from_solution(
            soln, eps=eps,
        )
        eigvecs = nparray([
            compute_eigenvalues_and_eigenvectors(j)[1]
            for j in jacobians
        ])
        inv_eigvecs = inv(eigvecs)
        coefs = nparray([
            dot(inv_eigvecs[i], subtracted_soln[i])
            for i in range(inv_eigvecs.shape[0])
        ]).real

    else:
        # find eigenvectors at critical point
        critical_jac = compute_jacobian_at_index(
            soln, index=critical_point_index, eps=eps
        )
        _, eigvecs = compute_eigenvalues_and_eigenvectors(critical_jac)

        inv_eigvecs = inv(eigvecs)

        coefs = dot(inv_eigvecs, subtracted_soln.T).T

    return coefs


@single_solution_plotter
def plot_eigenvector_coef_around_sonic_point(
    fig, soln, *, start=0, stop=90, linestyle='.', eps, local_eigvecs=True,
    use_E_r, critical_point="sonic", **kwargs
):
    """
    Plot coefficients of eigenvectors around the sonic point
    """
    angles = soln.angles
    indexes = (start <= degrees(angles)) & (degrees(angles) <= stop)
    if use_E_r:
        print("E_r used")

    crit_indexes = get_critical_point_indices(soln)

    critical_point_index = crit_indexes[CRITICAL_POINT_MAP[critical_point]]
    critical_point_label = critical_point.capitalize() + " point"

    coefs = get_eigenvector_coef_around_critical_point(
        soln, critical_point_index=critical_point_index, eps=eps,
        local_eigvecs=local_eigvecs,
    )

    # coefs, err = compute_coefs_of_subtracted_solution(
    #     subtracted_soln[indexes], eigvecs
    # )

    ax = fig.subplots(**kwargs)
    for i in range(coefs.shape[1]):
        ax.plot(
            degrees(angles[indexes]),
            coefs[indexes, i],
            linestyle,
            color=COLOURS[i],
            label=str(i),
        )
        # ax.plot(
        #     degrees(angles[indexes]),
        #     err[:, i],
        #     'x',
        #     color=COLOURS[i],
        #     label=str(i),
        # )
    ax.axvline(
        degrees(angles[critical_point_index]), label=critical_point_label
    )
    ax.legend()
    ax.set_xlabel("angle from plane (°)")

    return fig
