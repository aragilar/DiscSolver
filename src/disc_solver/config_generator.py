"""
Generate config files for solver given ranges for values
"""
from itertools import product
from math import sqrt
import operator

from .utils import CaseDependentConfigParser


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


def generate_config(
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


def write_config(
    label, start, stop, taylor_stop_angle, jump_before_sonic, max_steps,
    num_angles, η_derivs, β, v_rin_on_c_s, v_a_on_c_s, c_s_on_v_k, η_O, η_H,
    η_A, case,
):
    """
    Write config to a unique file
    """
    uniq_hash = hash((
        label, start, stop, taylor_stop_angle, jump_before_sonic, max_steps,
        num_angles, η_derivs, β, v_rin_on_c_s, v_a_on_c_s, c_s_on_v_k, η_O,
        η_H, η_A, case,
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
    filename = label + ".cfg"
    with open(filename, "w") as f:
        parser.write(f)
    return filename


def main():
    """Main func"""
    args = {
        "label_format": "generated-{β}-{η_O}",
        "start": "0",
        "stop": "10",
        "taylor_stop_angle": "0.001",
        "jump_before_sonic": "1e-15",
        "max_steps": "1000000",
        "num_angles": "20000",
        "η_derivs": "false",
        "β": slice(1.24, 1.25, 0.001),
        "v_rin_on_c_s": slice(0.5, 2.5, 0.05),
        "v_a_on_c_s": slice(0.5, 1, 0.05),
        "c_s_on_v_k": slice(0.1, 1e-2, -1e-2),
        "η_O": slice(2.5e-4, 0.05, 5e-4),
        "η_H": slice(0),
        "η_A": slice(0),
    }
    for filename in (write_config(*cfg) for cfg in generate_config(**args)):
        print(filename)


if __name__ == '__main__':
    main()
