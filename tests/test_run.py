# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

from disc_solver.analyse.acc_plot import acc_plot
from disc_solver.analyse.combine_plot import combine_plot
from disc_solver.analyse.compare_plot import compare_plot
from disc_solver.analyse.component_plot import plot as component_plot
from disc_solver.analyse.compute_jacobian import (
    jacobian_eigenvalues_plot, jacobian_eigenvectors_plot,
    jacobian_eigenvector_coef_plot,
)
from disc_solver.analyse.conserve_plot import conserve_main
from disc_solver.analyse.derivs_plot import derivs_plot
from disc_solver.analyse.diverge_plot import diverge_main
from disc_solver.analyse.dump_csv import dump_csv
from disc_solver.analyse.dump_config import dump_cfg
from disc_solver.analyse.hydro_check_plot import hydro_check_plot
from disc_solver.analyse.info import info
from disc_solver.analyse.j_e_plot import j_e_plot
from disc_solver.analyse.params_plot import params_plot
from disc_solver.analyse.phys_ratio_plot import plot as phys_ratio_plot
from disc_solver.analyse.plot import plot
from disc_solver.analyse.sonic_ratio_plot import plot as sonic_ratio_plot
from disc_solver.analyse.stats import stats_main, dump_csv_inputs_main
from disc_solver.analyse.surface_density_plot import surface_density_plot
from disc_solver.analyse.trajectory import plot as trajectory_plot
from disc_solver.analyse.trajectory import save_trajectory
from disc_solver.analyse.taylor_plot import taylor_plot
from disc_solver.analyse.utils import AnalysisError
from disc_solver.analyse.validate_plot import validate_plot
from disc_solver.analyse.v_deriv_cmp import plot as v_θ_deriv_cmp
from disc_solver.analyse.vert_plot import plot as vert_plot
from disc_solver.solve.csvrunner import csvrunner
from disc_solver.solve.hdf5runner import hdf5runner
from disc_solver.solve import solve
from disc_solver.solve.resolve import resolve
from disc_solver.solve.taylor_space import main as taylor_space_main
from disc_solver.solve.utils import SolverError


class TestSolve:
    def test_single_default_taylor_solution(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="single",
            config_file=None, output_file=None, store_internal=True,
            overrides={"use_taylor_jump": "False"},
        )

    def test_single_no_internal_taylor_solution(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="single",
            config_file=None, output_file=None, store_internal=False,
            overrides={"use_taylor_jump": "False"},
        )

    def test_single_default_step_sonic(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="single",
            config_file=None, output_file=None, store_internal=True,
            overrides={"use_taylor_jump": "True"},
        )

    def test_single_no_internal_step_sonic(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="single",
            config_file=None, output_file=None, store_internal=False,
            overrides={"use_taylor_jump": "True"},
        )

    def test_single_default_use_E_r(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="single",
            config_file=None, output_file=None, store_internal=True,
            overrides={"use_E_r": "True"},
        )

    def test_step_default(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="step",
            config_file=None, output_file=None, store_internal=True,
            max_search_steps=3, num_attempts=5,
        )

    def test_mcmc_default(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="mcmc",
            config_file=None, output_file=None, store_internal=True,
        )

    def test_mcmc_no_internal(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="mcmc",
            config_file=None, output_file=None, store_internal=False,
        )

    def test_sonic_root_default(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="sonic_root",
            config_file=None, output_file=None, store_internal=True,
        )

    def test_sonic_root_no_internal(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="sonic_root",
            config_file=None, output_file=None, store_internal=False,
        )

    def test_hydrostatic_default(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="hydrostatic",
            config_file=None, output_file=None, store_internal=True,
        )

    def test_hydrostatic_no_internal(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="hydrostatic",
            config_file=None, output_file=None, store_internal=False,
        )

    @pytest.mark.filterwarnings("ignore::h5preserve._utils.H5PreserveWarning")
    def test_hydrostatic_use_E_r(self, tmpdir):
        with pytest.raises(SolverError):
            solve(
                output_dir=Path(str(tmpdir)), sonic_method="hydrostatic",
                config_file=None, output_file=None, store_internal=True,
                overrides={"use_E_r": "True"},
            )

    def test_mod_hydro_default(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="mod_hydro",
            config_file=None, output_file=None, store_internal=True,
        )

    def test_mod_hydro_no_internal(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="mod_hydro",
            config_file=None, output_file=None, store_internal=False,
        )

    def test_mod_hydro_use_E_r(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="mod_hydro",
            config_file=None, output_file=None, store_internal=True,
            overrides={"use_E_r": "True"},
        )

    def test_fast_crosser_use_E_r(self, tmpdir, log_with_logbook):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="fast_crosser",
            config_file=None, output_file=None, store_internal=False,
            overrides={"use_E_r": "True"},
        )


class TestReSolve:
    def test_single_default(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="single",
            soln_filename=solution, soln_range=None, output_file=None,
            store_internal=True,
        )

    def test_single_no_internal(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="single",
            soln_filename=solution, soln_range=None, output_file=None,
            store_internal=False,
        )

    def test_single_default_step_sonic(self, tmpdir, single_solution_default):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="single",
            soln_filename=single_solution_default, soln_range=None,
            output_file=None, store_internal=True,
            overrides={"use_taylor_jump": "True"},
        )

    def test_step_default(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="step",
            soln_filename=solution, soln_range=None, output_file=None,
            store_internal=True, max_search_steps=3, num_attempts=5,
        )

    def test_mcmc_default(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="mcmc",
            soln_filename=solution, soln_range=None, output_file=None,
            store_internal=True,
        )

    def test_mcmc_no_internal(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="mcmc",
            soln_filename=solution, soln_range=None, output_file=None,
            store_internal=False,
        )

    def test_sonic_root_default(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="sonic_root",
            soln_filename=solution, soln_range=None, output_file=None,
            store_internal=True,
        )

    def test_sonic_root_no_internal(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="sonic_root",
            soln_filename=solution, soln_range=None, output_file=None,
            store_internal=False,
        )

    def test_hydrostatic_default(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="hydrostatic",
            soln_filename=solution, soln_range=None, output_file=None,
            store_internal=True,
        )

    def test_hydrostatic_no_internal(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="hydrostatic",
            soln_filename=solution, soln_range=None, output_file=None,
            store_internal=False,
        )


class TestAnalysis:
    def test_info_run(self, solution, tmp_text_stream):
        info(
            solution, group="run", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_info_status(self, solution, tmp_text_stream):
        info(
            solution, group="status", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_info_input(self, solution, tmp_text_stream):
        info(
            solution, group="input", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_info_initial_conditions(self, solution, tmp_text_stream):
        info(
            solution, group="initial-conditions", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_info_sonic_points(self, solution, tmp_text_stream):
        info(
            solution, group="sonic-points", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_info_crosses_points(self, solution, tmp_text_stream):
        info(
            solution, group="crosses-points", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_info_sonic_on_scale_single(
        self, single_solution_default, tmp_text_stream
    ):
        info(
            single_solution_default, group="sonic-on-scale", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_info_sonic_on_scale_single_no_internal(
        self, single_solution_no_internal, tmp_text_stream
    ):
        info(
            single_solution_no_internal, group="sonic-on-scale",
            soln_range=None, output_file=tmp_text_stream,
        )

    def test_info_sonic_on_scale_step(
        self, step_solution_default, tmp_text_stream
    ):
        info(
            step_solution_default, group="sonic-on-scale", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_info_solutions(self, solution, tmp_text_stream):
        info(
            solution, group="solutions", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_dump(self, solution, tmp_text_stream):
        dump_csv(
            solution, with_header=False, soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_dump_with_header(self, solution, tmp_text_stream):
        dump_csv(
            solution, with_header=True, soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_dump_cfg(self, solution, tmp_text_stream):
        dump_cfg(
            solution, soln_range=None, output_file=tmp_text_stream,
        )

    def test_plot_show(self, solution, mpl_interactive):
        plot(
            solution, show=True, with_slow=True, with_alfven=True,
            with_fast=True, with_sonic=True,
        )

    def test_plot_file(self, solution, plot_file):
        return plot(solution, plot_filename=plot_file)

    def test_plot_show_use_E_r(self, solution_use_E_r, mpl_interactive):
        plot(
            solution_use_E_r, show=True, with_slow=True, with_alfven=True,
            with_fast=True, with_sonic=True,
        )

    def test_compare_plot_file(self, solution, plot_file):
        solutions = [[solution, None], [solution, None]]
        return compare_plot(solutions, plot_filename=plot_file)

    def test_compare_plot_show(self, solution, mpl_interactive):
        solutions = [[solution, None], [solution, None]]
        compare_plot(solutions, show=True)

    def test_v_θ_deriv_cmp_file(self, solution_deriv, plot_file):
        solutions = [[solution_deriv, None], [solution_deriv, None]]
        return v_θ_deriv_cmp(solutions, plot_filename=plot_file)

    def test_v_θ_deriv_cmp_show(self, solution_deriv, mpl_interactive):
        solutions = [[solution_deriv, None], [solution_deriv, None]]
        v_θ_deriv_cmp(solutions, show=True)

    def test_sonic_ratio_plot_file(self, solution, plot_file):
        solutions = [[solution, None], [solution, None]]
        return sonic_ratio_plot(solutions, plot_filename=plot_file)

    def test_sonic_ratio_plot_show(self, solution, mpl_interactive):
        solutions = [[solution, None], [solution, None]]
        sonic_ratio_plot(solutions, show=True)

    def test_phys_ratio_plot_file(self, solution, plot_file):
        solutions = [[solution, None], [solution, None]]
        return phys_ratio_plot(solutions, plot_filename=plot_file)

    def test_phys_ratio_plot_show(self, solution, mpl_interactive):
        solutions = [[solution, None], [solution, None]]
        phys_ratio_plot(solutions, show=True)

    def test_derivs_show(self, solution_deriv, mpl_interactive):
        derivs_plot(solution_deriv, show=True)

    def test_derivs_file(self, solution_deriv, plot_file):
        return derivs_plot(
            solution_deriv, plot_filename=plot_file
        )

    def test_derivs_show_no_internal(
        self, solution_deriv_no_internal, mpl_interactive
    ):
        with pytest.raises(AnalysisError):
            derivs_plot(solution_deriv_no_internal, show=True)

    def test_derivs_file_no_internal(
        self, solution_deriv_no_internal, plot_file
    ):
        with pytest.raises(AnalysisError):
            derivs_plot(solution_deriv_no_internal, plot_filename=plot_file)

    def test_params_show(self, solution_default, mpl_interactive):
        params_plot(solution_default, show=True)

    def test_params_file(self, solution_default, plot_file):
        return params_plot(
            solution_default, plot_filename=plot_file
        )

    def test_params_show_no_internal(
        self, solution_no_internal, mpl_interactive
    ):
        with pytest.raises(AnalysisError):
            params_plot(solution_no_internal, show=True)

    def test_params_file_no_internal(self, solution_no_internal, plot_file):
        with pytest.raises(AnalysisError):
            params_plot(solution_no_internal, plot_filename=plot_file)

    def test_taylor_show(self, solution_taylor, mpl_interactive):
        taylor_plot(solution_taylor, show=True)

    def test_taylor_file(self, solution_taylor, plot_file):
        return taylor_plot(
            solution_taylor, plot_filename=plot_file
        )

    def test_taylor_show_values(self, solution_taylor, mpl_interactive):
        taylor_plot(solution_taylor, show=True, show_values=True)

    def test_taylor_file_values(self, solution_taylor, plot_file):
        return taylor_plot(
            solution_taylor, plot_filename=plot_file,
            show_values=True
        )

    def test_taylor_show_no_internal(
        self, solution_no_internal, mpl_interactive
    ):
        with pytest.raises(AnalysisError):
            taylor_plot(solution_no_internal, show=True)

    def test_taylor_file_no_internal(self, solution_no_internal, plot_file):
        with pytest.raises(AnalysisError):
            taylor_plot(solution_no_internal, plot_filename=plot_file)

    def test_combine_show(self, solution, mpl_interactive):
        combine_plot(
            solution, show=True, with_slow=True, with_alfven=True,
            with_fast=True, with_sonic=True,
        )

    def test_combine_file(self, solution, plot_file):
        return combine_plot(solution, plot_filename=plot_file)

    def test_component_show(self, solution, mpl_interactive):
        component_plot(solution, show=True)

    def test_component_file(self, solution, plot_file):
        return component_plot(solution, plot_filename=plot_file)

    def test_acc_show(self, solution, mpl_interactive):
        acc_plot(solution, show=True)

    def test_acc_file(self, solution, plot_file):
        return acc_plot(solution, plot_filename=plot_file)

    def test_surface_density_show(self, solution, mpl_interactive):
        surface_density_plot(solution, show=True)

    def test_surface_density_file(self, solution, plot_file):
        return surface_density_plot(solution, plot_filename=plot_file)

    def test_validate_show(self, solution_default, mpl_interactive):
        validate_plot(solution_default, show=True)

    def test_validate_file(self, solution_default, plot_file):
        return validate_plot(
            solution_default, plot_filename=plot_file
        )

    def test_validate_show_no_internal(
        self, solution_no_internal, mpl_interactive
    ):
        with pytest.raises(AnalysisError):
            validate_plot(solution_no_internal, show=True)

    def test_validate_file_no_internal(self, solution_no_internal, plot_file):
        with pytest.raises(AnalysisError):
            validate_plot(solution_no_internal, plot_filename=plot_file)

    def test_hydro_check_show(self, solution_default, mpl_interactive):
        hydro_check_plot(solution_default, show=True)

    def test_hydro_check_file(self, solution_default, plot_file):
        return hydro_check_plot(
            solution_default, plot_filename=plot_file
        )

    def test_hydro_check_show_no_internal(
        self, solution_no_internal, mpl_interactive
    ):
        with pytest.raises(AnalysisError):
            hydro_check_plot(solution_no_internal, show=True)

    def test_hydro_check_file_no_internal(
        self, solution_no_internal, plot_file
    ):
        with pytest.raises(AnalysisError):
            hydro_check_plot(solution_no_internal, plot_filename=plot_file)

    def test_diverge_show(self, solution, mpl_interactive):
        diverge_main([solution, '--show'])

    def test_diverge_file(self, solution, plot_file):
        diverge_main([solution, '--filename', str(plot_file)])

    def test_conserve_show(self, solution, mpl_interactive):
        conserve_main([solution, '--show'])

    def test_conserve_file(self, solution, plot_file):
        conserve_main([solution, '--filename', str(plot_file)])

    def test_j_e_plot_show(self, solution, mpl_interactive):
        j_e_plot(solution, show=True)

    def test_j_e_plot_file(self, solution, plot_file):
        return j_e_plot(solution, plot_filename=plot_file)

    def test_vert_plot_show(self, solution, mpl_interactive):
        vert_plot(
            solution, show=True, with_sonic=True,
        )

    def test_vert_plot_file(self, solution, plot_file):
        return vert_plot(solution, plot_filename=plot_file)

    def test_stats(self, solution, tmpdir):
        return stats_main([solution, '--file', str(tmpdir / 'stats.csv')])

    def test_dump_csv_inputs_main(self, solution, tmpdir):
        return dump_csv_inputs_main([
            solution, '--file', str(tmpdir / 'inputs.csv')
        ])

    def test_jacobian_eigenvalues_show(
        self, solution_not_approx, mpl_interactive
    ):
        jacobian_eigenvalues_plot(solution_not_approx, show=True, eps=1e-10)

    def test_jacobian_eigenvalues_file(self, solution_not_approx, plot_file):
        return jacobian_eigenvalues_plot(
            solution_not_approx, plot_filename=plot_file, eps=1e-10
        )

    def test_jacobian_eigenvectors_show(
        self, solution_not_approx, mpl_interactive
    ):
        jacobian_eigenvectors_plot(solution_not_approx, show=True, eps=1e-10)

    def test_jacobian_eigenvectors_file(self, solution_not_approx, plot_file):
        return jacobian_eigenvectors_plot(
            solution_not_approx, plot_filename=plot_file, eps=1e-10
        )

    def test_jacobian_eigenvector_coef_show(
        self, solution_not_approx, mpl_interactive
    ):
        jacobian_eigenvector_coef_plot(
            solution_not_approx, show=True, eps=1e-10
        )

    def test_jacobian_eigenvector_coef_file(
        self, solution_not_approx, plot_file
    ):
        return jacobian_eigenvector_coef_plot(
            solution_not_approx, plot_filename=plot_file, eps=1e-10
        )

    def test_trajectory_plot_show(self, solution, mpl_interactive):
        trajectory_plot(solution, show=True, v_start_position=(2, 0.00001))

    def test_trajectory_plot_file(self, solution, plot_file):
        return trajectory_plot(
            solution, plot_filename=plot_file, v_start_position=(2, 0.00001),
        )

    def test_save_trajectory(self, solution, tmp_path):
        stats_output_file = tmp_path / "stats.csv"
        B_output_file = tmp_path / "B.csv"
        v_output_file = tmp_path / "v.csv"
        return save_trajectory(
            solution, v_start_position=(2, 0.00001),
            stats_output_file=stats_output_file, B_output_file=B_output_file,
            v_output_file=v_output_file
        )


def test_taylor_space(mpl_interactive):
    taylor_space_main()


def test_csvrunner(tmp_text_stream, ds_csv_file_header):
    csvrunner(
        output_file=tmp_text_stream, input_file=ds_csv_file_header,
        sonic_method='single', store_internal=False, overrides={},
    )


def test_hdf5runner(tmpdir, ds_csv_file_header):
    hdf5runner(
        output_dir=tmpdir, input_file=ds_csv_file_header,
        store_internal=False, sonic_method='single', overrides={},
    )
