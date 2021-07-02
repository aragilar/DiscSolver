# -*- coding: utf-8 -*-
"""
Calculating v_θ near the sonic point
"""
# DO NOT IMPORT MATH, BREAKS FLOAT SUPPORT
from numpy import tan, degrees
import logbook

from root_solver import solve_quadratic

from .deriv_funcs import B_unit_derivs, C_func, Z_5_func, deriv_E_r_func
from .j_e_funcs import E_r_func, J_r_func, J_θ_func, J_φ_func
from .utils import SolverError

log = logbook.Logger(__name__)


def deriv_v_θ_sonic(
    *, a_0, ρ, B_r, B_φ, B_θ, η_O, η_H, η_A, θ, v_r, v_φ, v_θ=1, deriv_v_r,
    deriv_v_φ, deriv_B_r, deriv_B_θ, deriv_B_φ, γ, η_O_0, η_A_0, η_H_0,
    η_derivs, E_r, b_r, b_θ, b_φ
):
    """
    Compute v_θ' across the sonic point
    """
    if E_r is None:
        E_r = E_r_func(
            v_θ=v_θ, v_φ=v_φ, B_θ=B_θ, B_φ=B_φ, η_O=η_O,
            η_A=η_A, η_H=η_H,
            J_r=J_r_func(θ=θ, B_φ=B_φ, deriv_B_φ=deriv_B_φ),
            J_θ=J_θ_func(γ=γ, B_φ=B_φ),
            J_φ=J_φ_func(γ=γ, B_θ=B_θ, deriv_B_r=deriv_B_r),
            b_r=b_r, b_φ=b_φ, b_θ=b_θ,
        )

    deriv_b_r, deriv_b_φ, deriv_b_θ = B_unit_derivs(
        B_r=B_r, B_φ=B_φ, B_θ=B_θ, deriv_B_r=deriv_B_r, deriv_B_φ=deriv_B_φ,
        deriv_B_θ=deriv_B_θ, b_r=b_r, b_θ=b_θ, b_φ=b_φ
    )

    J_r = J_r_func(θ=θ, B_φ=B_φ, deriv_B_φ=deriv_B_φ)
    J_θ = J_θ_func(γ=γ, B_φ=B_φ)
    J_φ = J_φ_func(γ=γ, B_θ=B_θ, deriv_B_r=deriv_B_r)

    deriv_E_r = deriv_E_r_func(
        γ=γ, v_r=v_r, v_φ=v_φ, B_r=B_r, B_φ=B_φ, η_O=η_O, η_A=η_A,
        η_H=η_H, b_r=b_r, b_θ=b_θ, b_φ=b_φ, J_r=J_r, J_φ=J_φ, J_θ=J_θ,
    )

    C = C_func(η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ)

    Z_5 = Z_5_func(η_O=η_O, η_A=η_A, η_H=η_H, b_r=b_r, b_θ=b_θ, b_φ=b_φ, C=C)

    S_1 = S_1_func(v_r=v_r, v_θ=v_θ, γ=γ)

    deriv_ρ_up = deriv_ρ_up_func(ρ=ρ, v_r=v_r, v_θ=v_θ, γ=γ, θ=θ)
    deriv_ρ_down = deriv_ρ_down_func(ρ=ρ, v_θ=v_θ)

    if η_derivs:
        deriv_η_scale_up = deriv_η_up_func(
            deriv_ρ_up=deriv_ρ_up, deriv_B_θ=deriv_B_θ, ρ=ρ, B_r=B_r, B_φ=B_φ,
            B_θ=B_θ, deriv_B_r=deriv_B_r, deriv_B_φ=deriv_B_φ,
        )

        deriv_η_scale_down = deriv_η_down_func(
            deriv_ρ_down=deriv_ρ_down, ρ=ρ, B_r=B_r, B_φ=B_φ, B_θ=B_θ,
        )

        deriv_η_O_up = deriv_η_scale_up * η_O_0
        deriv_η_A_up = deriv_η_scale_up * η_A_0
        deriv_η_H_up = deriv_η_scale_up * η_H_0

        deriv_η_O_down = deriv_η_scale_down * η_O_0
        deriv_η_A_down = deriv_η_scale_down * η_A_0
        deriv_η_H_down = deriv_η_scale_down * η_H_0

    else:
        deriv_η_O_up = 0
        deriv_η_A_up = 0
        deriv_η_H_up = 0
        deriv_η_O_down = 0
        deriv_η_A_down = 0
        deriv_η_H_down = 0

    A_up = A_up_func(
        η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ,
        deriv_η_O_up=deriv_η_O_up, deriv_η_A_up=deriv_η_A_up,
        deriv_η_H_up=deriv_η_H_up, deriv_b_θ=deriv_b_θ, deriv_b_r=deriv_b_r,
        deriv_b_φ=deriv_b_φ
    )

    A_down = A_down_func(
        η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ,
        deriv_η_O_down=deriv_η_O_down, deriv_η_A_down=deriv_η_A_down,
        deriv_η_H_down=deriv_η_H_down, deriv_b_φ=deriv_b_φ
    )

    Z_5_up = Z_5_up_func(
        η_H=η_H, η_A=η_A, b_r=b_r, b_φ=b_φ, b_θ=b_θ, A_up=A_up, C=C,
        deriv_η_O_up=deriv_η_O_up, deriv_η_A_up=deriv_η_A_up,
        deriv_η_H_up=deriv_η_H_up, deriv_b_θ=deriv_b_θ, deriv_b_r=deriv_b_r,
        deriv_b_φ=deriv_b_φ
    )
    Z_5_down = Z_5_down_func(
        η_H=η_H, η_A=η_A, b_r=b_r, b_φ=b_φ, b_θ=b_θ, A_down=A_down, C=C,
        deriv_η_O_down=deriv_η_O_down, deriv_η_A_down=deriv_η_A_down,
        deriv_η_H_down=deriv_η_H_down,
    )

    dderiv_B_φ_up = dderiv_B_φ_up_func(
        η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ,
        deriv_η_O_up=deriv_η_O_up, deriv_η_A_up=deriv_η_A_up,
        deriv_η_H_up=deriv_η_H_up, deriv_b_θ=deriv_b_θ, deriv_b_r=deriv_b_r,
        deriv_b_φ=deriv_b_φ, B_r=B_r, B_φ=B_φ, B_θ=B_θ, θ=θ, v_r=v_r, v_φ=v_φ,
        v_θ=v_θ, deriv_v_r=deriv_v_r, deriv_v_φ=deriv_v_φ, deriv_B_r=deriv_B_r,
        deriv_B_θ=deriv_B_θ, deriv_B_φ=deriv_B_φ, γ=γ, E_r=E_r, C=C, A_up=A_up,
        deriv_E_r=deriv_E_r, Z_5=Z_5, Z_5_up=Z_5_up
    )
    dderiv_B_φ_down = dderiv_B_φ_down_func(
        η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ,
        deriv_η_O_down=deriv_η_O_down, deriv_η_A_down=deriv_η_A_down,
        deriv_η_H_down=deriv_η_H_down, B_r=B_r, B_φ=B_φ, B_θ=B_θ, θ=θ, v_r=v_r,
        v_φ=v_φ, v_θ=v_θ, γ=γ, E_r=E_r, C=C, A_down=A_down, Z_5=Z_5,
        Z_5_down=Z_5_down
    )

    dderiv_B_r_up = dderiv_B_r_up_func(
        η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ,
        deriv_η_O_up=deriv_η_O_up, deriv_η_A_up=deriv_η_A_up,
        deriv_η_H_up=deriv_η_H_up, deriv_b_θ=deriv_b_θ, deriv_b_r=deriv_b_r,
        deriv_b_φ=deriv_b_φ, B_r=B_r, B_φ=B_φ, B_θ=B_θ, θ=θ, v_r=v_r, v_θ=v_θ,
        deriv_v_r=deriv_v_r, deriv_B_r=deriv_B_r, deriv_B_θ=deriv_B_θ,
        deriv_B_φ=deriv_B_φ, γ=γ, dderiv_B_φ_up=dderiv_B_φ_up
    )
    dderiv_B_r_down = dderiv_B_r_down_func(
        η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ,
        deriv_η_O_down=deriv_η_O_down, deriv_η_A_down=deriv_η_A_down,
        deriv_η_H_down=deriv_η_H_down, B_r=B_r, B_φ=B_φ, B_θ=B_θ, θ=θ, v_r=v_r,
        v_θ=v_θ, deriv_B_φ=deriv_B_φ, γ=γ, dderiv_B_φ_down=dderiv_B_φ_down
    )

    S_1_dash_up = S_1_dash_up_func(
        a_0=a_0, ρ=ρ, B_r=B_r, B_φ=B_φ, B_θ=B_θ, θ=θ, v_φ=v_φ,
        deriv_v_φ=deriv_v_φ, deriv_B_r=deriv_B_r, deriv_B_θ=deriv_B_θ,
        deriv_B_φ=deriv_B_φ, γ=γ, dderiv_B_r_up=dderiv_B_r_up,
        dderiv_B_φ_up=dderiv_B_φ_up, deriv_ρ_up=deriv_ρ_up
    )
    S_1_dash_down = S_1_dash_down_func(
        a_0=a_0, ρ=ρ, B_r=B_r, B_φ=B_φ, B_θ=B_θ, θ=θ, deriv_B_r=deriv_B_r,
        deriv_B_φ=deriv_B_φ, γ=γ, dderiv_B_r_down=dderiv_B_r_down,
        dderiv_B_φ_down=dderiv_B_φ_down, deriv_ρ_down=deriv_ρ_down
    )

    a = 2 * v_θ
    b = v_r * v_θ + S_1 + v_θ * S_1_dash_down
    c = deriv_v_r / 2 * (v_θ ** 2 - 4 * γ) + v_θ * S_1_dash_up

    x1, x2 = solve_quadratic(a, b, c)
    log.info(f"{a}, {b}, {c}")
    if (x1 > 0) and (x2 > 0):
        log.warning(f"{a}, {b}, {c}")
        raise SolverError(
            "Two positive roots at {}: {} {}".format(degrees(θ), x1, x2)
        )
    elif x1 > 0:
        return x1
    elif x2 > 0:
        return x2
    log.warning(f"{a}, {b}, {c}")
    raise SolverError(
        "Two negative roots at {}: {} {}".format(degrees(θ), x1, x2)
    )


def S_1_func(*, v_r, v_θ, γ):
    """
    Compute S_1 across the sonic point
    """
    return - v_r / (2 * v_θ) * (v_θ ** 2 - 4 * γ)


def S_1_dash_up_func(
    *, a_0, ρ, B_r, B_φ, B_θ, θ, v_φ, deriv_v_φ, deriv_B_r, deriv_B_θ,
    deriv_B_φ, γ, dderiv_B_r_up, dderiv_B_φ_up, deriv_ρ_up
):
    """
    Compute \\overline{S_1}' across the sonic point
    """
    return (
        (1 + tan(θ) ** 2) * (v_φ ** 2 + 1) +
        2 * v_φ * deriv_v_φ * tan(θ) + a_0 / ρ * (
            (1 / 4 - γ) * deriv_B_θ * B_r + (1 / 4 - γ) * B_θ * deriv_B_r +
            deriv_B_r ** 2 + B_r * dderiv_B_r_up + deriv_B_φ ** 2 +
            B_φ * dderiv_B_φ_up - 2 * B_φ * deriv_B_φ * tan(θ) -
            B_φ ** 2 * (1 + tan(θ) ** 2) - deriv_ρ_up / ρ * (
                (1 / 4 - γ) * B_θ * B_r + B_r * deriv_B_r + B_φ * deriv_B_φ -
                B_φ ** 2 * tan(θ)
            )
        )
    )


def S_1_dash_down_func(
    *, a_0, ρ, B_r, B_φ, B_θ, θ, deriv_B_r, deriv_B_φ, γ, dderiv_B_r_down,
    dderiv_B_φ_down, deriv_ρ_down
):
    """
    Compute \\underline{S_1}' across the sonic point
    """
    return a_0 / ρ * (
        B_r * dderiv_B_r_down + B_φ * dderiv_B_φ_down - deriv_ρ_down / ρ * (
            (1 / 4 - γ) * B_θ * B_r + B_r * deriv_B_r + B_φ * deriv_B_φ -
            B_φ ** 2 * tan(θ)
        )
    )


def deriv_ρ_up_func(*, ρ, v_r, v_θ, γ, θ):
    """
    Compute \\overline{ρ}' across the sonic point
    """
    return - ρ * (
        2 * γ * v_r / v_θ - tan(θ)
    )


def deriv_ρ_down_func(*, ρ, v_θ):
    """
    Compute \\underline{ρ}' across the sonic point
    """
    return - ρ / v_θ


def dderiv_B_r_up_func(
    *, η_O, η_A, η_H, b_θ, b_r, b_φ, deriv_η_O_up, deriv_η_A_up, deriv_η_H_up,
    deriv_b_θ, deriv_b_r, deriv_b_φ, B_r, B_φ, B_θ, θ, v_r, v_θ,
    deriv_v_r, deriv_B_r, deriv_B_θ, deriv_B_φ, γ, dderiv_B_φ_up
):
    """
    Compute \\overline{B_r}'' across the sonic point
    """
    return 1 / (η_O + η_A * (1 - b_φ ** 2)) * (
        v_θ * deriv_B_r - deriv_v_r * B_θ - v_r * deriv_B_θ - dderiv_B_φ_up * (
            η_H * b_θ + η_A * b_r * b_φ
        ) - deriv_B_φ * (
            deriv_η_H_up * b_θ + η_H * deriv_b_θ + deriv_η_A_up * b_r * b_φ +
            η_A * deriv_b_r * b_φ + η_A * b_r * deriv_b_φ - η_A * b_φ * (
                b_r * tan(θ) - b_θ * (1 / 4 - γ)
            ) - η_H * (
                b_r * (1 / 4 - γ) + b_θ * tan(θ)
            )
        ) + B_φ * (
            deriv_η_A_up * b_φ * (
                b_r * tan(θ) - b_θ * (1 / 4 - γ)
            ) + η_A * deriv_b_φ * (
                b_r * tan(θ) - b_θ * (1 / 4 - γ)
            ) + η_A * b_φ * (
                deriv_b_r * tan(θ) + b_r * (1 + tan(θ) ** 2) -
                deriv_b_θ * (1 / 4 - γ)
            ) + deriv_η_H_up * (
                b_r * (1 / 4 - γ) + b_θ * tan(θ)
            ) + η_H * (
                deriv_b_r * (1 / 4 - γ) + deriv_b_θ * tan(θ) +
                b_θ * (1 + tan(θ) ** 2)
            )
        )
    ) - (
        deriv_η_O_up + deriv_η_A_up * (1 - b_φ ** 2) -
        2 * η_A * b_φ * deriv_b_φ
    ) / (
        (η_O + η_A * (1 - b_φ ** 2)) ** 2
    ) * (
        v_θ * B_r - v_r * B_θ - deriv_B_φ * (
            η_H * b_θ + η_A * b_r * b_φ
        ) + B_φ * (
            η_A * b_φ * (
                b_r * tan(θ) - b_θ * (1 / 4 - γ)
            ) + η_H * (
                b_r * (1 / 4 - γ) + b_θ * tan(θ)
            )
        )
    ) - deriv_B_θ * (1 / 4 - γ)


def dderiv_B_r_down_func(
    *, η_O, η_A, η_H, b_θ, b_r, b_φ, deriv_η_O_down, deriv_η_A_down,
    deriv_η_H_down, B_r, B_φ, B_θ, θ, v_r, v_θ, deriv_B_φ, γ, dderiv_B_φ_down
):
    """
    Compute \\underline{B_r}'' across the sonic point
    """
    return 1 / (η_O + η_A * (1 - b_φ ** 2)) * (
        B_r - dderiv_B_φ_down * (
            η_H * b_θ + η_A * b_r * b_φ
        ) - deriv_B_φ * (
            deriv_η_H_down * b_θ + deriv_η_A_down * b_r * b_φ
        ) + B_φ * (
            deriv_η_A_down * b_φ * (
                b_r * tan(θ) - b_θ * (1 / 4 - γ)
            ) + deriv_η_H_down * (
                b_r * (1 / 4 - γ) + b_θ * tan(θ)
            )
        )
    ) - (
        deriv_η_O_down + deriv_η_A_down * (1 - b_φ ** 2)
    ) / (
        (η_O + η_A * (1 - b_φ ** 2)) ** 2
    ) * (
        v_θ * B_r - v_r * B_θ - deriv_B_φ * (
            η_H * b_θ + η_A * b_r * b_φ
        ) + B_φ * (
            η_A * b_φ * (
                b_r * tan(θ) - b_θ * (1 / 4 - γ)
            ) + η_H * (
                b_r * (1 / 4 - γ) + b_θ * tan(θ)
            )
        )
    )


def dderiv_B_φ_up_func(
    *, η_O, η_A, η_H, b_θ, b_r, b_φ, deriv_η_O_up, deriv_η_A_up, deriv_η_H_up,
    deriv_b_θ, deriv_b_r, deriv_b_φ, B_r, B_φ, B_θ, θ, v_r, v_φ, v_θ,
    deriv_v_r, deriv_v_φ, deriv_B_r, deriv_B_θ, deriv_B_φ, γ, E_r, C, A_up,
    deriv_E_r, Z_5, Z_5_up
):
    """
    Compute \\overline{B_φ}'' across the sonic point
    """
    return (
        C * (
            v_θ * deriv_B_r - deriv_v_r * B_θ - v_r * deriv_B_θ
        ) + A_up * (
            v_θ * B_r - v_r * B_θ
        ) - deriv_E_r - deriv_v_φ * B_θ - v_φ * deriv_B_θ + deriv_B_φ * (
            v_θ + tan(θ) * (
                η_O + η_A * (1 - b_r ** 2)
            ) + (1 / 4 - γ) * (
                η_H * b_φ + η_A * b_r * b_θ
            ) - C * (
                η_A * b_φ * (
                    b_r * tan(θ) - b_θ * (1 / 4 - γ)
                ) + η_H * (
                    b_r * (1 / 4 - γ) + b_θ * tan(θ)
                )
            )
        ) + B_φ * (
            (1 + tan(θ) ** 2) * (η_O + η_A * (1 - b_r ** 2)) + tan(θ) * (
                deriv_η_O_up + deriv_η_A_up * (1 - b_r ** 2) -
                2 * η_A * b_r * deriv_b_r
            ) + (1 / 4 - γ) * (
                deriv_η_H_up * b_φ + η_H * deriv_b_φ +
                deriv_η_A_up * b_r * b_θ + η_A * deriv_b_r * b_θ +
                η_A * b_r * deriv_b_θ
            ) - A_up * (
                η_A * b_φ * (
                    b_r * tan(θ) - b_θ * (1 / 4 - γ)
                ) + η_H * (
                    b_r * (1 / 4 - γ) + b_θ * tan(θ)
                )
            ) - C * (
                deriv_η_A_up * b_φ * (
                    b_r * tan(θ) - b_θ * (1 / 4 - γ)
                ) + η_A * deriv_b_φ * (
                    b_r * tan(θ) - b_θ * (1 / 4 - γ)
                ) + η_A * b_φ * (
                    deriv_b_r * tan(θ) + b_r * (1 + tan(θ) ** 2) -
                    deriv_b_θ * (1 / 4 - γ)
                ) + η_H * (
                    deriv_b_r * (1 / 4 - γ) + deriv_b_θ * tan(θ) +
                    b_θ * (1 + tan(θ) ** 2)
                ) + deriv_η_H_up * (
                    b_r * (1 / 4 - γ) + b_θ * tan(θ)
                )
            )
        )
    ) / Z_5 - Z_5_up / (Z_5 ** 2) * (
        C * (
            v_θ * B_r - v_r * B_θ
        ) - E_r - v_φ * B_θ + B_φ * (
            v_θ + tan(θ) * (η_O + η_A * (1 - b_r ** 2)) + (1 / 4 - γ) * (
                η_H * b_φ + η_A * b_r * b_θ
            ) - C * (
                η_A * b_φ * (
                    b_r * tan(θ) - b_θ * (1 / 4 - γ)
                ) + η_H * (
                    b_r * (1 / 4 - γ) + b_θ * tan(θ)
                )
            )
        )
    )


def dderiv_B_φ_down_func(
    *, η_O, η_A, η_H, b_θ, b_r, b_φ, deriv_η_O_down, deriv_η_A_down,
    deriv_η_H_down, B_r, B_φ, B_θ, θ, v_r, v_φ, v_θ, γ, E_r, C, A_down,
    Z_5, Z_5_down
):
    """
    Compute \\underline{B_φ}'' across the sonic point
    """
    return (
        A_down * (
            v_θ * B_r - v_r * B_θ
        ) + B_φ * (
            1 + tan(θ) * (
                deriv_η_O_down + deriv_η_A_down * (1 - b_r ** 2)
            ) + (1 / 4 - γ) * (
                deriv_η_H_down * b_φ + deriv_η_A_down * b_r * b_θ
            ) - A_down * (
                η_A * b_φ * (
                    b_r * tan(θ) - b_θ * (1 / 4 - γ)
                ) + η_H * (
                    b_r * (1 / 4 - γ) + b_θ * tan(θ)
                )
            ) - C * (
                deriv_η_A_down * b_φ * (
                    b_r * tan(θ) - b_θ * (1 / 4 - γ)
                ) + deriv_η_H_down * (
                    b_r * (1 / 4 - γ) + b_θ * tan(θ)
                )
            )
        )
    ) / Z_5 - Z_5_down / (Z_5 ** 2) * (
        C * (
            v_θ * B_r - v_r * B_θ
        ) - E_r - v_φ * B_θ + B_φ * (
            v_θ + tan(θ) * (η_O + η_A * (1 - b_r ** 2)) + (1 / 4 - γ) * (
                η_H * b_φ + η_A * b_r * b_θ
            ) - C * (
                η_A * b_φ * (
                    b_r * tan(θ) - b_θ * (1 / 4 - γ)
                ) + η_H * (
                    b_r * (1 / 4 - γ) + b_θ * tan(θ)
                )
            )
        )
    )


def A_up_func(
    *, η_O, η_A, η_H, b_θ, b_r, b_φ, deriv_η_O_up, deriv_η_A_up, deriv_η_H_up,
    deriv_b_θ, deriv_b_r, deriv_b_φ
):
    """
    Compute \\overline{A} across the sonic point
    """
    return (
        deriv_η_H_up * b_θ + η_H * deriv_b_θ - deriv_η_A_up * b_r * b_φ -
        η_A * deriv_b_r * b_φ - η_A * b_r * deriv_b_φ
    ) / (
        η_O + η_A * (1 - b_φ ** 2)
    ) - (
        (
            deriv_η_O_up + deriv_η_A_up * (1 - b_φ ** 2) -
            2 * η_A * b_φ * deriv_b_φ
        ) * (η_H * b_θ - η_A * b_r * b_φ)
    ) / (
        (η_O + η_A * (1 - b_φ ** 2)) ** 2
    )


def A_down_func(
    *, η_O, η_A, η_H, b_θ, b_r, b_φ, deriv_η_O_down, deriv_η_A_down,
    deriv_η_H_down, deriv_b_φ
):
    """
    Compute \\underline{A} across the sonic point
    """
    return (
        deriv_η_H_down * b_θ - deriv_η_A_down * b_r * b_φ
    ) / (
        η_O + η_A * (1 - b_φ ** 2)
    ) - (
        (
            deriv_η_O_down + deriv_η_A_down * (1 - b_φ ** 2) -
            2 * η_A * b_φ * deriv_b_φ
        ) * (η_H * b_θ - η_A * b_r * b_φ)
    ) / (
        (η_O + η_A * (1 - b_φ ** 2)) ** 2
    )


def deriv_η_up_func(
    *, ρ, B_r, B_φ, B_θ, deriv_ρ_up, deriv_B_r, deriv_B_φ, deriv_B_θ
):
    """
    Compute the scaled part of the derivative of η assuming the same form as in
    SKW near the sonic point (\\overline{η}')
    """
    B_sq = B_r ** 2 + B_φ ** 2 + B_θ ** 2
    return (
        2 * (B_r * deriv_B_r + B_φ * deriv_B_φ + B_θ * deriv_B_θ) -
        B_sq * deriv_ρ_up / ρ
    ) / ρ


def deriv_η_down_func(*, ρ, B_r, B_φ, B_θ, deriv_ρ_down):
    """
    Compute the scaled part of the derivative of η assuming the same form as in
    SKW near the sonic point (\\underline{η}')
    """
    B_sq = B_r ** 2 + B_φ ** 2 + B_θ ** 2
    return - deriv_ρ_down * B_sq / (ρ ** 2)


def Z_5_up_func(
    *, η_H, η_A, b_r, b_φ, b_θ, A_up, C, deriv_η_O_up, deriv_η_A_up,
    deriv_η_H_up, deriv_b_θ, deriv_b_r, deriv_b_φ
):
    """
    Compute \\overline{Z_5} across the sonic point
    """
    return (
        deriv_η_O_up + deriv_η_A_up * (1 - b_r ** 2) - η_A * deriv_b_r * b_r +
        A_up * (η_H * b_θ - η_A * b_r * b_φ) + C * (
            deriv_η_H_up * b_θ + η_H * deriv_b_θ - deriv_η_A_up * b_r * b_φ -
            η_A * deriv_b_r * b_φ - η_A * b_r * deriv_b_φ
        )
    )


def Z_5_down_func(
    *, η_H, η_A, b_r, b_φ, b_θ, A_down, C, deriv_η_O_down, deriv_η_A_down,
    deriv_η_H_down,
):
    """
    Compute \\underline{Z_5} across the sonic point
    """
    return (
        deriv_η_O_down + deriv_η_A_down * (1 - b_r ** 2) +
        A_down * (η_H * b_θ - η_A * b_r * b_φ) + C * (
            deriv_η_H_down * b_θ - deriv_η_A_down * b_r * b_φ
        )
    )
