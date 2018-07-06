# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

from disc_solver.analyse.info import info
from disc_solver.analyse.plot import plot
from disc_solver.analyse.derivs_plot import derivs_plot
from disc_solver.analyse.params_plot import params_plot
from disc_solver.analyse.taylor_plot import taylor_plot
from disc_solver.analyse.combine_plot import combine_plot
from disc_solver.analyse.acc_plot import acc_plot
from disc_solver.analyse.validate_plot import validate_plot
from disc_solver.analyse.diverge_plot import diverge_main
from disc_solver.analyse.conserve_plot import conserve_main
from disc_solver.analyse.utils import AnalysisError
from disc_solver.solve import solve
from disc_solver.solve.resolve import resolve
from disc_solver.analyse.j_e_plot import j_e_plot
from disc_solver.solve.taylor_space import main as taylor_space_main


class TestSolve:
    def test_single_default(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="single",
            config_file=None, output_file=None, store_internal=True,
        )

    def test_single_no_internal(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="single",
            config_file=None, output_file=None, store_internal=False,
        )

    def test_step_default(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="step",
            config_file=None, output_file=None, store_internal=True,
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

    def test_step_default(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="step",
            soln_filename=solution, soln_range=None, output_file=None,
            store_internal=True,
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

    def test_plot_show(self, solution, mpl_interactive):
        plot(solution, show=True)

    @pytest.mark.mpl_image_compare
    def test_plot_file(self, solution, plot_file):
        return plot(solution, plot_filename=plot_file, close=False)

    def test_derivs_show(self, solution_deriv, mpl_interactive):
        derivs_plot(solution_deriv, show=True)

    @pytest.mark.mpl_image_compare
    def test_derivs_file(self, solution_deriv, plot_file):
        return derivs_plot(
            solution_deriv, plot_filename=plot_file, close=False
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

    @pytest.mark.mpl_image_compare
    def test_params_file(self, solution_default, plot_file):
        return params_plot(
            solution_default, plot_filename=plot_file, close=False
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

    @pytest.mark.mpl_image_compare
    def test_taylor_file(self, solution_taylor, plot_file):
        return taylor_plot(
            solution_taylor, plot_filename=plot_file, close=False
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
        combine_plot(solution, show=True)

    @pytest.mark.mpl_image_compare
    def test_combine_file(self, solution, plot_file):
        return combine_plot(solution, plot_filename=plot_file, close=False)

    def test_acc_show(self, solution, mpl_interactive):
        acc_plot(solution, show=True)

    @pytest.mark.mpl_image_compare
    def test_acc_file(self, solution, plot_file):
        return acc_plot(solution, plot_filename=plot_file, close=False)

    def test_validate_show(self, solution_default, mpl_interactive):
        validate_plot(solution_default, show=True)

    @pytest.mark.mpl_image_compare
    def test_validate_file(self, solution_default, plot_file):
        return validate_plot(
            solution_default, plot_filename=plot_file, close=False
        )

    def test_validate_show_no_internal(
        self, solution_no_internal, mpl_interactive
    ):
        with pytest.raises(AnalysisError):
            validate_plot(solution_no_internal, show=True)

    def test_validate_file_no_internal(self, solution_no_internal, plot_file):
        with pytest.raises(AnalysisError):
            validate_plot(solution_no_internal, plot_filename=plot_file)

    def test_jacobian_show(self, solution, mpl_interactive):
        plot(solution, show=True)

    @pytest.mark.mpl_image_compare
    def test_jacobian_file(self, solution, plot_file):
        return plot(solution, plot_filename=plot_file, close=False)

    def test_diverge_show(self, solution, mpl_interactive):
        diverge_main([str(solution), '--show'])

    def test_diverge_file(self, solution, plot_file):
        diverge_main([str(solution), '--filename', str(plot_file)])

    def test_conserve_show(self, solution, mpl_interactive):
        conserve_main([str(solution), '--show'])

    def test_conserve_file(self, solution, plot_file):
        conserve_main([str(solution), '--filename', str(plot_file)])

    def test_j_e_plot_show(self, solution, mpl_interactive):
        j_e_plot(solution, show=True)

    @pytest.mark.mpl_image_compare
    def test_j_e_plot_file(self, solution, plot_file):
        return j_e_plot(solution, plot_filename=plot_file, close=False)


def test_taylor_space(mpl_interactive):
    taylor_space_main()
