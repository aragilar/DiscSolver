# -*- coding: utf-8 -*-
"""
Handle setting precision of floating point numbers.

Defines the floating point type to use `FLOAT_TYPE`, and conversion function
`float_type`.
"""

import numpy as _np
from scikits.odes.sundials.common_defs import DTYPE as _DTYPE

FLOAT_TYPE = _DTYPE
float_type = FLOAT_TYPE
FLOAT_TYPE_INFO = _np.finfo(FLOAT_TYPE)
