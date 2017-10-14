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
from disc_solver.analyse.jacobian_plot import jacobian_plot
from disc_solver.analyse.diverge_plot import diverge_main
from disc_solver.analyse.conserve_plot import conserve_main
from disc_solver.analyse.utils import AnalysisError
from disc_solver.solve import solve
from disc_solver.solve.resolve import resolve
from disc_solver.filter_files import filter_files



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

    def test_dae_single_default(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="dae_single",
            config_file=None, output_file=None, store_internal=True,
        )

    def test_dae_single_no_internal(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="dae_single",
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
            overrides={"use_taylor_jump": "True",},
        )

    def test_single_no_internal_step_sonic(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="single",
            config_file=None, output_file=None, store_internal=False,
            overrides={"use_taylor_jump": "True",},
        )

class TestReSolve:
    def test_single_default(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="single",
            soln_filename=solution, soln_range=None, output_file=None, store_internal=True,
        )

    def test_single_no_internal(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="single",
            soln_filename=solution, soln_range=None, output_file=None, store_internal=False,
        )

    def test_dae_single_default(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="dae_single",
            soln_filename=solution, soln_range=None, output_file=None, store_internal=True,
        )

    def test_dae_single_no_internal(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="dae_single",
            soln_filename=solution, soln_range=None, output_file=None, store_internal=False,
        )

    def test_step_default(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="step",
            soln_filename=solution, soln_range=None, output_file=None, store_internal=True,
        )

    def test_mcmc_default(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="mcmc",
            soln_filename=solution, soln_range=None, output_file=None, store_internal=True,
        )

    def test_mcmc_no_internal(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="mcmc",
            soln_filename=solution, soln_range=None, output_file=None, store_internal=False,
        )

    def test_sonic_root_default(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="sonic_root",
            soln_filename=solution, soln_range=None, output_file=None, store_internal=True,
        )

    def test_sonic_root_no_internal(self, tmpdir, solution):
        resolve(
            output_dir=Path(str(tmpdir)), sonic_method="sonic_root",
            soln_filename=solution, soln_range=None, output_file=None, store_internal=False,
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

    def test_info_sonic_on_scale(self, single_solution_default, tmp_text_stream):
        info(
            single_solution_default, group="sonic-on-scale", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_info_sonic_on_scale(self, single_solution_no_internal, tmp_text_stream):
        info(
            single_solution_no_internal, group="sonic-on-scale", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_info_sonic_on_scale(self, step_solution_default, tmp_text_stream):
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

    def test_plot_file(self, solution, plot_file):
        plot(solution, plot_filename=plot_file)

    def test_derivs_show(self, solution_default, mpl_interactive):
        derivs_plot(solution_default, show=True)

    def test_derivs_file(self, solution_default, plot_file):
        derivs_plot(solution_default, plot_filename=plot_file)

    def test_derivs_show(self, solution_no_internal, mpl_interactive):
        with pytest.raises(AnalysisError):
            derivs_plot(solution_no_internal, show=True)

    def test_derivs_file(self, solution_no_internal, plot_file):
        with pytest.raises(AnalysisError):
            derivs_plot(solution_no_internal, plot_filename=plot_file)

    def test_params_show(self, solution_default, mpl_interactive):
        params_plot(solution_default, show=True)

    def test_params_file(self, solution_default, plot_file):
        params_plot(solution_default, plot_filename=plot_file)

    def test_params_show(self, solution_no_internal, mpl_interactive):
        with pytest.raises(AnalysisError):
            params_plot(solution_no_internal, show=True)

    def test_params_file(self, solution_no_internal, plot_file):
        with pytest.raises(AnalysisError):
            params_plot(solution_no_internal, plot_filename=plot_file)

    def test_taylor_show(self, solution_default, mpl_interactive):
        taylor_plot(solution_default, show=True)

    def test_taylor_file(self, solution_default, plot_file):
        taylor_plot(solution_default, plot_filename=plot_file)

    def test_taylor_show(self, solution_no_internal, mpl_interactive):
        with pytest.raises(AnalysisError):
            taylor_plot(solution_no_internal, show=True)

    def test_taylor_file(self, solution_no_internal, plot_file):
        with pytest.raises(AnalysisError):
            taylor_plot(solution_no_internal, plot_filename=plot_file)

    def test_combine_show(self, solution, mpl_interactive):
        combine_plot(solution, show=True)

    def test_combine_file(self, solution, plot_file):
        combine_plot(solution, plot_filename=plot_file)

    def test_acc_show(self, solution, mpl_interactive):
        acc_plot(solution, show=True)

    def test_acc_file(self, solution, plot_file):
        acc_plot(solution, plot_filename=plot_file)

    def test_jacobian_show(self, solution, mpl_interactive):
        plot(solution, show=True)

    def test_jacobian_file(self, solution, plot_file):
        plot(solution, plot_filename=plot_file)

    def test_diverge_show(self, solution, mpl_interactive):
        diverge_main([str(solution), '--show'])

    def test_diverge_file(self, solution, plot_file):
        diverge_main([str(solution), '--filename', str(plot_file)])

    def test_conserve_show(self, solution, mpl_interactive):
        conserve_main([str(solution), '--show'])

    def test_conserve_file(self, solution, plot_file):
        conserve_main([str(solution), '--filename', str(plot_file)])

class TestFilter:
    pass
