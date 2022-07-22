# -*- coding: utf-8 -*-
"""
"""
from functools import wraps

from numpy import (
    degrees, zeros, logical_not as npnot, logical_and as npand, isfinite, nan,
    full, array as nparray
)
from scipy.linalg import eigvals
import matplotlib.pyplot as plt

from ..float_handling import float_type
from ..solve.solution import ode_system
from ..utils import get_solutions
from ..utils import ODEIndex
from .utils import (
    single_solution_plotter, common_plotting_options, analyse_main_wrapper,
    get_common_plot_args, analysis_func_wrapper, plot_output_wrapper,
    DEFAULT_MPL_STYLE,
)


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
def jacobian_eigenvalues_main(soln, *, soln_range, common_plot_args, plot_args):
    """
    Entry point for ds-derivs-plot
    """
    return jacobian_eigenvalues_plot(
        soln, soln_range=soln_range, **common_plot_args, **plot_args
    )


def compute_jacobian(
    *, γ, a_0, norm_kepler_sq, init_con, θ_scale, η_derivs, use_E_r, θ, params,
    eps,
):
    rhs_eq, _ = ode_system(
        γ=γ, a_0=a_0, norm_kepler_sq=norm_kepler_sq, init_con=init_con,
        θ_scale=θ_scale, with_taylor=False, η_derivs=η_derivs,
        store_internal=False, use_E_r=use_E_r, v_θ_sonic_crit=None,
        after_sonic=None, deriv_v_θ_func=None, check_params=False
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


@analysis_func_wrapper
def compute_jacobian_from_file(
    soln_file, *, soln_range=None, eps, θ_scale=float_type(1), **kwargs
):

    soln = get_solutions(soln_file, soln_range)
    return compute_jacobian_from_solution(soln, eps=eps, θ_scale=θ_scale)


def jacobian_single_solution_plot_wrapper(func):
    @wraps(func)
    def jacobian_wrapper(
        fig, soln, *args, eps, θ_scale=float_type(1), **kwargs
    ):
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
    Show derivatives
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = plot_jacobian_eigenvalues(
        soln, soln_range, linestyle=linestyle, start=start, stop=stop,
        figargs=figargs, title=title, filename=filename,
        mpl_style=mpl_style, with_version=with_version, eps=eps
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


def compute_eigenvalues(jacobian):
    if not isfinite(jacobian).all():
        return full(jacobian.shape[0], nan)
    return eigvals(jacobian)


def plot_log_complex_data(ax, *, x_data, y_data, **kwargs):
    pos_real_data_slice = y_data.real >= 0
    pos_imag_data_slice = y_data.imag >= 0
    ax.plot(
        x_data[pos_real_data_slice],
        y_data[pos_real_data_slice].real,
        marker="triangle_up",
        **kwargs
    )
    ax.plot(
        x_data[npnot(pos_real_data_slice)],
        y_data[npnot(pos_real_data_slice)].real,
        marker="triangle_up",
        **kwargs
    )
    ax.plot(
        x_data[pos_imag_data_slice],
        y_data[pos_imag_data_slice].imag,
        marker="tri_up",
        **kwargs
    )
    ax.plot(
        x_data[npnot(pos_imag_data_slice)],
        y_data[npnot(pos_imag_data_slice)].imag,
        marker="tri_up",
        **kwargs
    )


@single_solution_plotter
@jacobian_single_solution_plot_wrapper
def plot_jacobian_eigenvalues(
    fig, *, jacobians, angles, use_E_r, linestyle='.', figargs=None, start=0,
    stop=90
):
    """
    Generate plot of jacobians
    """
    if figargs is None:
        figargs = {}

    indexes = degrees(angles) <= stop

    data = nparray([compute_eigenvalues(j) for j in jacobians])

    ax = fig.subplots(**figargs)
    ax.set_xlabel("angle from plane (°)")
    ax.set_yscale("log")
    with plt.style.context({
        "axes.prop_cycle": cycler.cycler(color=plt.get_cmap("tab20").colors)
    }):
        for i in range(data.shape[1]):
            plot_log_complex_data(
                ax, x_data=degrees(angles[indexes]),
                y_data=data[indexes, i],
                color="C" + str(i),
            )
    return fig