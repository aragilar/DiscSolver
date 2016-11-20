# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

from disc_solver.config_generator import config_generator
from disc_solver.analyse.info import info
from disc_solver.analyse.plot import plot
from disc_solver.analyse.derivs_plot import derivs_plot
from disc_solver.analyse.params_plot import params_plot
from disc_solver.analyse.taylor_plot import taylor_plot
#from disc_solver.analyse.combine_plot import combine_plot
#from disc_solver.analyse.acc_plot import acc_plot
from disc_solver.analyse.jacobian_plot import jacobian_plot
from disc_solver.solve import solve
from disc_solver.filter_files import filter_files



class TestConfigGeneration:
    def test_default(self, tmpdir):
        for path in config_generator(
            input_file=None, output_path=Path(str(tmpdir))
        ):
            pass


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

    @pytest.mark.xfail
    def test_jump_default(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="jump",
            config_file=None, output_file=None, store_internal=True,
        )

    @pytest.mark.xfail
    def test_jump_no_internal(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="jump",
            config_file=None, output_file=None, store_internal=False,
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
            solution, group="initial_conditions", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_info_sonic_points(self, solution, tmp_text_stream):
        info(
            solution, group="sonic_points", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_plot_show(self, solution, mpl_interactive):
        plot(solution, show=True)

    def test_plot_file(self, solution, plot_file):
        plot(solution, plot_filename=plot_file)

    @pytest.mark.xfail
    def test_derivs_show(self, solution, mpl_interactive):
        derivs_plot(soln_file=solution, show=True)

    @pytest.mark.xfail
    def test_derivs_file(self, solution, plot_file):
        derivs_plot(soln_file=solution, plot_filename=plot_file)

    @pytest.mark.xfail
    def test_params_show(self, solution, mpl_interactive):
        params_plot(soln_file=solution, show=True)

    @pytest.mark.xfail
    def test_params_file(self, solution, plot_file):
        params_plot(soln_file=solution, plot_filename=plot_file)

    @pytest.mark.xfail
    def test_taylor_show(self, solution, mpl_interactive):
        taylor_plot(soln_file=solution, show=True)

    @pytest.mark.xfail
    def test_taylor_file(self, solution, plot_file):
        taylor_plot(soln_file=solution, plot_filename=plot_file)

    #def test_combine_show(self, solution, mpl_interactive):
    #    combine_plot(soln_file=solution, show=True)

    #def test_combine_file(self, solution, plot_file):
    #    combine_plot(soln_file=solution, plot_filename=plot_file)

    #def test_acc_show(self, solution, mpl_interactive):
    #    acc_plot(soln_file=solution, show=True)

    #def test_acc_file(self, solution, plot_file):
    #    acc_plot(soln_file=solution, plot_filename=plot_file)

    def test_jacobian_show(self, solution, mpl_interactive):
        plot(solution, show=True)

    def test_jacobian_file(self, solution, plot_file):
        plot(solution, plot_filename=plot_file)

class TestFilter:
    pass
