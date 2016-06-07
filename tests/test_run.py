# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

from disc_solver.analyse import analyse_main
from disc_solver.config_generator import config_generator
from disc_solver.solve import solve

PLOT_FILE = "plot.png"


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
            config_file=None, output_file=None
        )

    def test_step_default(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="step",
            config_file=None, output_file=None
        )

    @pytest.mark.xfail
    def test_jump_default(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), sonic_method="jump",
            config_file=None, output_file=None
        )

class TestAnalysis:
    def test_show(self, solution):
        import matplotlib.pyplot as plt
        plt.ion()
        analyse_main(output_file=solution, command="show")


    def test_plot(self, solution, tmpdir):
        analyse_main(
            output_file=solution, command="plot",
            plot_filename=Path(Path(str(tmpdir)), PLOT_FILE),
        )

    def test_info(self, solution):
        analyse_main(
            output_file=solution, command="info", input=True,
            initial_conditions=True, sound_ratio=True, sonic_points=True,
        )

    def test_derivs_show(self, solution):
        import matplotlib.pyplot as plt
        plt.ion()
        analyse_main(output_file=solution, command="deriv_show")

    def test_check_taylor(self, solution):
        import matplotlib.pyplot as plt
        plt.ion()
        analyse_main(output_file=solution, command="check_taylor")

    def test_params_show(self, solution):
        import matplotlib.pyplot as plt
        plt.ion()
        analyse_main(output_file=solution, command="params_show")


class TestFindSolutions:
    pass
