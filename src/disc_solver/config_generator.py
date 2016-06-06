"""
Generate config files for solver given ranges for values
"""
import argparse
from itertools import product
from math import sqrt
import operator
from pathlib import Path

import logbook
from logbook.compat import redirected_warnings, redirected_logging

from .logging import logging_options, log_handler
from .utils import CaseDependentConfigParser, expanded_path

log = logbook.Logger(__name__)


def range_generator(range_obj):
    """
    Generator which yields items based on range-like object
    """
    if range_obj.start is not None:
        num = range_obj.start
    else:
        num = 0
    stop = range_obj.stop
    if range_obj.step is not None:
        step = range_obj.step
    else:
        step = 1

    if num <= stop and step > 0:
        cmp = operator.le
    elif num >= stop and step < 0:
        cmp = operator.ge
    else:
        raise ValueError("Invalid range object")

    while cmp(num, stop):
        print(num, step)
        yield num
        num += step


def yield_configs(
    *, label_format, start, stop, taylor_stop_angle, jump_before_sonic,
    max_steps, num_angles, η_derivs, β, v_rin_on_c_s, v_a_on_c_s, c_s_on_v_k,
    η_O, η_H, η_A
):
    """
    Generate config settings based on ranges given
    """
    for β, v_rin_on_c_s, v_a_on_c_s, c_s_on_v_k, η_O, η_H, η_A in product(
        range_generator(β), range_generator(v_rin_on_c_s),
        range_generator(v_a_on_c_s), range_generator(c_s_on_v_k),
        range_generator(η_O), range_generator(η_H), range_generator(η_A)
    ):
        is_valid, case = is_valid_conditions(
            β, v_rin_on_c_s, v_a_on_c_s, c_s_on_v_k, η_O, η_H, η_A
        )
        if is_valid:
            label = label_format.format(
                β=β, v_rin_on_c_s=v_rin_on_c_s, v_a_on_c_s=v_a_on_c_s,
                c_s_on_v_k=c_s_on_v_k, η_O=η_O, η_H=η_H, η_A=η_A, case=case
            )
            yield (
                label, start, stop, taylor_stop_angle, jump_before_sonic,
                max_steps, num_angles, η_derivs, β, v_rin_on_c_s, v_a_on_c_s,
                c_s_on_v_k, η_O, η_H, η_A, case
            )


def is_valid_conditions(
    β, v_rin_on_c_s, v_a_on_c_s, c_s_on_v_k, η_O, η_H, η_A,
):
    """
    Return if conditions are valid to launch wind
    """
    # pylint: disable=unused-argument,too-many-return-statements
    if η_O > η_H >= η_A:
        if η_H < 2 and η_O < 2:
            if (
                sqrt(η_A / 2) <=
                v_a_on_c_s <=
                2 <=
                (v_rin_on_c_s / η_O) <=
                (1 / (2 * c_s_on_v_k))
            ):
                return True, "Ohm case 1"
            return False, None
        elif η_H < 2 and η_O >= 2:
            if (
                sqrt(η_A / 2) <=
                v_a_on_c_s <=
                (2 / η_O) <=
                (v_rin_on_c_s / 2) <=
                (1 / (η_O * c_s_on_v_k))
            ):
                return True, "Ohm case 2"
            return False, None
        elif η_H >= 2 and η_O < (η_H / η_A):
            if (
                sqrt(η_A / η_H) <=
                v_a_on_c_s <=
                (2 * sqrt(η_H) / η_O) <=
                (v_rin_on_c_s / 2) <=
                (1 / (η_O * c_s_on_v_k))
            ):
                return True, "Ohm case 3"
            return False, None
        return False, None
    raise RuntimeError("Non-ohmic case not implemented: {}, {}, {}".format(
        η_O, η_A, η_H))


def config_writer(output_path):
    """
    Return a function which writes config files to a specific path
    """
    def write_config(
        label, start, stop, taylor_stop_angle, jump_before_sonic, max_steps,
        num_angles, η_derivs, β, v_rin_on_c_s, v_a_on_c_s, c_s_on_v_k, η_O,
        η_H, η_A, case,
    ):
        """
        Write config to a unique file
        """
        uniq_hash = hash((
            label, start, stop, taylor_stop_angle, jump_before_sonic,
            max_steps, num_angles, η_derivs, β, v_rin_on_c_s, v_a_on_c_s,
            c_s_on_v_k, η_O, η_H, η_A, case,
        ))
        label = label + "-" + hex(uniq_hash)
        parser = CaseDependentConfigParser()
        parser["config"] = {
            "label": label,
            "start": start,
            "stop": stop,
            "taylor_stop_angle": taylor_stop_angle,
            "jump_before_sonic": jump_before_sonic,
            "max_steps": max_steps,
            "num_angles": num_angles,
            "η_derivs": η_derivs,
        }
        parser["initial"] = {
            "β": β,
            "v_rin_on_c_s": v_rin_on_c_s,
            "v_a_on_c_s": v_a_on_c_s,
            "c_s_on_v_k": c_s_on_v_k,
            "η_O": η_O,
            "η_H": η_H,
            "η_A": η_A,
        }
        path = output_path / Path(label + ".cfg")
        with path.open("w") as f:
            parser.write(f)
        return path
    return write_config


def read_input(input_file):
    """
    Read input file
    """
    parser = CaseDependentConfigParser()
    with input_file.open("r") as f:
        parser.read_file(f)

    return {
        "label_format": parser.get(
            "common", "label_format", fallback="generated"
        ),
        "start": parser.get(
            "common", "start", fallback="0",
        ),
        "stop": parser.get(
            "common", "stop", fallback="10",
        ),
        "taylor_stop_angle": parser.get(
            "common", "taylor_stop_angle", fallback="0.001",
        ),
        "jump_before_sonic": parser.get(
            "common", "jump_before_sonic", fallback="1e-15",
        ),
        "max_steps": parser.get(
            "common", "max_steps", fallback="1000000",
        ),
        "num_angles": parser.get(
            "common", "num_angles", fallback="20000",
        ),
        "η_derivs": parser.get(
            "common", "η_derivs", fallback="false",
        ),
        "β": slice(
            parser.get("β", "start", fallback=1.24),
            parser.get("β", "stop", fallback=1.25),
            parser.get("β", "step", fallback=0.001),
        ),
        "v_rin_on_c_s": slice(
            parser.get("v_rin_on_c_s", "start", fallback=0.5),
            parser.get("v_rin_on_c_s", "stop", fallback=2.5),
            parser.get("v_rin_on_c_s", "step", fallback=0.05)
        ),
        "v_a_on_c_s": slice(
            parser.get("v_a_on_c_s", "start", fallback=0.5),
            parser.get("v_a_on_c_s", "stop", fallback=1),
            parser.get("v_a_on_c_s", "step", fallback=0.05)
        ),
        "c_s_on_v_k": slice(
            parser.get("c_s_on_v_k", "start", fallback=0.1),
            parser.get("c_s_on_v_k", "stop", fallback=1e-2),
            parser.get("c_s_on_v_k", "step", fallback=-1e-2)
        ),
        "η_O": slice(
            parser.get("η_O", "start", fallback=2.5e-4),
            parser.get("η_O", "stop", fallback=0.05),
            parser.get("η_O", "step", fallback=5e-4)
        ),
        "η_H": slice(
            parser.get("η_H", "start", fallback=0),
            parser.get("η_H", "stop", fallback=0),
            parser.get("η_H", "step", fallback=0)
        ),
        "η_A": slice(
            parser.get("η_A", "start", fallback=0),
            parser.get("η_A", "stop", fallback=0),
            parser.get("η_A", "step", fallback=0)
        ),
    }


def config_generator(*, input_file, output_path):
    """
    Generate config files for solver
    """
    for path in (
        config_writer(output_path)(*cfg)
        for cfg in yield_configs(*read_input(input_file))
    ):
        yield path


def main():
    """
    Entry point for ds-gen-config
    """
    parser = argparse.ArgumentParser(
        description='Config Generator for DiscSolver'
    )
    parser.add_argument("input_file")
    parser.add_argument("--output-path", default=".")
    logging_options(parser)
    args = vars(parser.parse_args())
    with log_handler(args), redirected_warnings(), redirected_logging():
        for path in config_generator(
            input_file=expanded_path(args["input_file"]),
            output_path=expanded_path(args["output_path"]),
        ):
            print(path)


if __name__ == '__main__':
    main()
